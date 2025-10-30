import os, time, threading, signal
import cv2, numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "reconnect;1|reconnect_streamed;1|reconnect_delay_max;2|"
    "rw_timeout;5000000|stimeout;5000000|timeout;5000000"
)

app = FastAPI(title="live-road-ai")

RTMP = os.getenv("RTMP_URL")
if not RTMP:
    raise RuntimeError("Set RTMP_URL to your RTMP endpoint")

MODEL_CFG = {
    "police": {
        "path": os.getenv("POLICE_MODEL"),
        "conf": float(os.getenv("POLICE_CONF", "0.35")),
        "color": (60, 170, 255),
        "fallback_label": "police",
    },
    "pothole": {
        "path": os.getenv("POTHOLE_MODEL"),
        "conf": float(os.getenv("POTHOLE_CONF", "0.35")),
        "color": (90, 220, 100),
        "fallback_label": "pothole",
    },
    "roadwork": {
        "path": os.getenv("ROADWORK_MODEL"),
        "conf": float(os.getenv("ROADWORK_CONF", "0.35")),
        "color": (255, 120, 60),
        "fallback_label": "roadwork",
    },
}

YOLO_IOU    = float(os.getenv("YOLO_IOU",  "0.45"))
FRAME_MAX_W = int(os.getenv("FRAME_MAX_W", "960"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def open_capture():
    cap = cv2.VideoCapture(RTMP, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    return cap

cap = open_capture()
cap_lock = threading.Lock()

models = {}
for name, cfg in MODEL_CFG.items():
    path = cfg["path"]
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Model '{name}' not found at: {path}")
    m = YOLO(path)
    try:
        if list(getattr(m, "names", {}).values()) == ["0"]:
            m.names = {0: cfg["fallback_label"]}
    except Exception:
        pass
    try:
        m.fuse()
    except Exception:
        pass
    models[name] = m

last_dets = []
last_dets_lock = threading.Lock()
stop_event = threading.Event()

def _graceful_exit(*_):
    stop_event.set()
    with cap_lock:
        if cap and cap.isOpened():
            cap.release()

signal.signal(signal.SIGINT, _graceful_exit)
signal.signal(signal.SIGTERM, _graceful_exit)

@app.on_event("shutdown")
def on_shutdown():
    _graceful_exit()

def _maybe_resize(frame):
    if FRAME_MAX_W and FRAME_MAX_W > 0:
        h, w = frame.shape[:2]
        if w > FRAME_MAX_W:
            scale = FRAME_MAX_W / w
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return frame

def _infer_all(frame):
    """Run every model on the frame and merge detections."""
    merged = []
    ts = time.time()
    for name, mdl in models.items():
        cfg = MODEL_CFG[name]
        res = mdl.predict(frame, conf=cfg["conf"], iou=YOLO_IOU, verbose=False)
        if not res:
            continue
        r = res[0]
        if r.boxes is None or not hasattr(r.boxes, "data"):
            continue
        data = r.boxes.data.cpu().numpy()
        names = r.names or {}
        for row in data:
            x1, y1, x2, y2, conf, cls_id = row.tolist()
            label = names.get(int(cls_id), cfg["fallback_label"])
            merged.append({
                "model": name,
                "label": label,
                "cls": int(cls_id),
                "conf": float(conf),
                "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                "color": cfg["color"],
                "ts": ts,
            })
    return merged

def _draw_boxes(image_bgr, dets):
    for d in dets:
        x1, y1, x2, y2 = map(int, [d["x1"], d["y1"], d["x2"], d["y2"]])
        label = f'{d["label"]} {d["conf"]:.2f}'
        color = tuple(map(int, d.get("color", (0, 255, 0))))
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_bgr, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return image_bgr

@app.get("/health", response_class=PlainTextResponse)
def health():
    with cap_lock:
        is_open = cap.isOpened()
    return f"cap_is_open={is_open}, url={RTMP}"

@app.get("/models")
def models_info():
    return {
        k: {"path": v["path"], "conf": v["conf"]}
        for k, v in MODEL_CFG.items()
    }

@app.get("/last-dets")
def last_detections():
    with last_dets_lock:
        return JSONResponse(last_dets)

def mjpeg_frames():
    global cap
    backoff = 0.5
    try:
        while not stop_event.is_set():
            with cap_lock:
                if not cap.isOpened():
                    cap.release()
                    time.sleep(backoff)
                    if stop_event.is_set():
                        break
                    cap = open_capture()
                    backoff = min(backoff * 2, 5.0)
                    black = np.zeros((360, 640, 3), dtype=np.uint8)
                    ok, jpg = cv2.imencode(".jpg", black)
                    if ok:
                        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
                    continue

                ok, frame = cap.read()
            if not ok:
                with cap_lock:
                    cap.release()
                time.sleep(0.05)
                continue

            frame = _maybe_resize(frame)

            dets = _infer_all(frame)
            if dets:
                with last_dets_lock:
                    last_dets[:] = dets
                frame = _draw_boxes(frame, dets)

            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                backoff = 0.5
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
    except (GeneratorExit, BrokenPipeError):
        pass
    finally:
        with cap_lock:
            if cap and cap.isOpened():
                cap.release()

@app.get("/video-feed")
def video_feed():
    return StreamingResponse(mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame")

# run:  python -m uvicorn main:app --host 0.0.0.0 --port 8000