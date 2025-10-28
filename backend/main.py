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

app = FastAPI(title="live-road-ai-min")

RTMP = os.getenv("RTMP_URL")  # e.g. rtmp://127.0.0.1/live/stream # 18.216.45.42
if not RTMP:
    raise RuntimeError("Set RTMP_URL to your RTMP endpoint")

YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
YOLO_CONF   = float(os.getenv("YOLO_CONF", "0.25"))
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

model = YOLO(YOLO_MODEL)

#cache last detections so frontend can pull as JSON
last_dets = []
last_dets_lock = threading.Lock()

stop_event = threading.Event()

#graceful shutdown on backend shutdown
def _graceful_exit(*_):
    stop_event.set()
    with cap_lock:
        if cap and cap.isOpened():
            cap.release()

signal.signal(signal.SIGINT, _graceful_exit)
signal.signal(signal.SIGTERM, _graceful_exit)

@app.on_event("shutdown")
def on_shutdown():
    _graceful_exit

def _draw_boxes(image_bgr, boxes, names_lookup):
    for row in boxes:
        x1, y1, x2, y2, conf, cls_id = row.tolist()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = names_lookup.get(int(cls_id), str(int(cls_id)))
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(image_bgr, f"{label} {conf:.2f}", (x1, max(0, y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
    return image_bgr

def _maybe_resize(frame):
    if FRAME_MAX_W and FRAME_MAX_W > 0:
        h, w = frame.shape[:2]
        if w > FRAME_MAX_W:
            scale = FRAME_MAX_W / w
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return frame


@app.get("/health", response_class=PlainTextResponse)
def health():
    with cap_lock:
        is_open = cap.isOpened()
    return f"cap_is_open={cap.isOpened()}, url={RTMP}"

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
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
                    continue

                ok, frame = cap.read()
            if not ok:
                with cap_lock:
                    cap.release()
                time.sleep(0.05)
                continue

            frame = _maybe_resize(frame)

            results = model.predict(source=frame, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)
            if results and results[0].boxes is not None and hasattr(results[0].boxes, "data"):
                r = results[0]
                data = r.boxes.data.cpu().numpy()
                names = r.names

                tmp = []
                for row in data:
                    x1, y1, x2, y2, conf, cls_id = row.tolist()
                    tmp.append({
                        "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                        "conf": float(conf), "cls": int(cls_id), "label": names.get(int(cls_id), str(int(cls_id)))
                    })
                with last_dets_lock:
                    last_dets[:] = tmp

                frame = _draw_boxes(frame, data, names)

            ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                backoff = 0.5
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
            else:
                continue
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


#python -m uvicorn main:app --host 0.0.0.0 --port 8000