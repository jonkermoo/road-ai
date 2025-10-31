import os, time, threading, signal
from collections import defaultdict
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
    raise RuntimeError("Set RTMP_URL in backend/.env")

MODEL_CFG = {
    "police": {
         "path": os.getenv("POLICE_MODEL"),
        "conf": float(os.getenv("POLICE_CONF", "0.75")), 
        "emit_conf": float(os.getenv("POLICE_EMIT_CONF", "0.98")),
        "color": (60, 170, 255),
        "fallback_label": "police",
        "ar_min": float(os.getenv("POLICE_AR_MIN", "1.40")),
        "ar_max": float(os.getenv("POLICE_AR_MAX", "3.50")),
        "min_box_px": int(os.getenv("POLICE_MIN_BOX_PX", "70000")),
        "min_w_px":   int(os.getenv("POLICE_MIN_W_PX",   "180")),
        "min_h_px":   int(os.getenv("POLICE_MIN_H_PX",   "80")),
        "persist_frames": int(os.getenv("POLICE_PERSIST_FRAMES", "20")),
        "cooldown_s": float(os.getenv("POLICE_COOLDOWN_S", "25")),
    },
    "pothole": {
        "path": os.getenv("POTHOLE_MODEL"),
        "conf": float(os.getenv("POTHOLE_CONF", "0.75")),
        "emit_conf": float(os.getenv("POTHOLE_EMIT_CONF", "0.90")),
        "color": (90, 220, 100),
        "fallback_label": "pothole",
        "ar_min": float(os.getenv("POTHOLE_AR_MIN", "0.0")),
        "ar_max": float(os.getenv("POTHOLE_AR_MAX", "9.0")),
        "min_box_px": int(os.getenv("POTHOLE_MIN_BOX_PX", str(os.getenv("MIN_BOX_PX", "40000")))),
        "persist_frames": int(os.getenv("POTHOLE_PERSIST_FRAMES", str(os.getenv("PERSIST_FRAMES", "15")))),
        "cooldown_s": float(os.getenv("POTHOLE_COOLDOWN_S", str(os.getenv("COOLDOWN_S", "15")))),
    },
    "roadwork": {
        "path": os.getenv("ROADWORK_MODEL"),
        "conf": float(os.getenv("ROADWORK_CONF", "0.75")),
        "emit_conf": float(os.getenv("ROADWORK_EMIT_CONF", "0.90")),
        "color": (255, 120, 60),
        "fallback_label": "roadwork",
        "ar_min": float(os.getenv("ROADWORK_AR_MIN", "0.75")),
        "ar_max": float(os.getenv("ROADWORK_AR_MAX", "1.33")),
        "min_box_px": int(os.getenv("ROADWORK_MIN_BOX_PX", str(os.getenv("MIN_BOX_PX", "40000")))),
        "persist_frames": int(os.getenv("ROADWORK_PERSIST_FRAMES", str(os.getenv("PERSIST_FRAMES", "15")))),
        "cooldown_s": float(os.getenv("ROADWORK_COOLDOWN_S", str(os.getenv("COOLDOWN_S", "15")))),
    },
}

for _name, _cfg in MODEL_CFG.items():
    _cfg.setdefault("emit_conf", _cfg.get("conf", 0.9))
    _cfg.setdefault("ar_min", 0.0)
    _cfg.setdefault("ar_max", 9.0)

YOLO_IOU    = float(os.getenv("YOLO_IOU",  "0.45"))
FRAME_MAX_W = int(os.getenv("FRAME_MAX_W", "960"))

PERSIST_FRAMES = int(os.getenv("PERSIST_FRAMES", "6"))
PERSIST_IOU    = float(os.getenv("PERSIST_IOU", "0.30"))
COOLDOWN_S     = float(os.getenv("COOLDOWN_S", "8"))
MIN_BOX_PX     = int(os.getenv("MIN_BOX_PX", "9000"))

ROI_YMIN_FRAC  = float(os.getenv("ROI_YMIN_FRAC", "0.20"))
ROI_YMAX_FRAC  = float(os.getenv("ROI_YMAX_FRAC", "0.98"))


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
_events = [] 
_tracks = [] 
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

def _draw_boxes(image_bgr, dets):
    for d in dets:
        x1, y1, x2, y2 = map(int, [d["x1"], d["y1"], d["x2"], d["y2"]])
        label = f'{d["label"]} {d["conf"]:.2f}'
        color = tuple(map(int, d.get("color", (0, 255, 0))))
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_bgr, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return image_bgr

def _infer_all(frame):
    """Run each model and return list of detections above its DRAW threshold."""
    merged = []
    ts = time.time()
    H, W = frame.shape[:2]
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
                "_img_h": H,
                "_img_w": W,
            })
    return merged

def _area(b):
    return max(0, b[2]-b[0]) * max(0, b[3]-b[1])

def _aspect_ratio(b):
    """Return width/height ratio of a bounding box tuple (x1, y1, x2, y2)."""
    w = max(1, b[2] - b[0])
    h = max(1, b[3] - b[1])
    return w / h

def _iou(a,b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    ua = _area(a) + _area(b) - inter
    return inter/ua if ua > 0 else 0.0

def _match_track(d):
    """Return index of a matching track in _tracks or -1."""
    best_i, best_iou = -1, 0.0
    cand = (int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"]))
    for i, t in enumerate(_tracks):
        if t["model"] != d["model"] or t["label"] != d["label"]:
            continue
        iou = _iou(t["box"], cand)
        if iou > best_iou:
            best_i, best_iou = i, iou
    return best_i if best_iou >= PERSIST_IOU else -1

def _passes_emit_rules(d):
    cfg = MODEL_CFG[d["model"]]

    # conf gate (use emit_conf if present, else fall back to conf, else 0.9)
    emit_conf = float(cfg.get("emit_conf", cfg.get("conf", 0.9)))
    if d["conf"] < emit_conf:
        return False

    # size gates (area + w/h)
    box = (int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"]))
    min_box_px = int(cfg.get("min_box_px", MIN_BOX_PX))
    if _area(box) < min_box_px:
        return False
    min_w = int(cfg.get("min_w_px", 0))
    min_h = int(cfg.get("min_h_px", 0))
    w = max(0, box[2] - box[0]); h = max(0, box[3] - box[1])
    if (min_w and w < min_w) or (min_h and h < min_h):
        return False

    # aspect-ratio gate (use wide-open defaults if not provided)
    ar_min = float(cfg.get("ar_min", 0.0))
    ar_max = float(cfg.get("ar_max", 9.0))
    ar = _aspect_ratio(box)
    if not (ar_min <= ar <= ar_max):
        return False
    
    # ROI gate (ignore boxes too high/low in the frame)
    ymid = (box[1] + box[3]) * 0.5
    # store image height on the detection once per frame in _infer_all
    H = d.get("_img_h", None)
    if H:
        yf = ymid / float(H)
        if not (ROI_YMIN_FRAC <= yf <= ROI_YMAX_FRAC):
            return False
    

    return True

def _update_tracks_and_emit(detections):
    """Update persistence tracker with current detections and emit stable events."""
    now = time.time()

    # 1) decay stale tracks a bit so they drop off if they vanish
    for t in _tracks:
        if now - t["last_ts"] > 2.0:
            t["hits"] = max(0, t["hits"] - 1)

    # 2) incorporate detections that PASS strict event rules
    for d in detections:
        if not _passes_emit_rules(d):
            continue
        box = (int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"]))
        idx = _match_track(d)
        if idx == -1:
            _tracks.append({
                "model": d["model"],
                "label": d["label"],
                "box": box,
                "hits": 1,
                "last_ts": now,
                "sent_ts": 0.0,
                "conf": d["conf"],
            })
        else:
            t = _tracks[idx]
            # light smoothing so boxes are stable
            t["box"] = tuple(int(0.7*a + 0.3*b) for a, b in zip(t["box"], box))
            t["hits"] += 1
            t["last_ts"] = now
            t["conf"] = max(t["conf"], d["conf"])

    # 3) emit events only for stable tracks that respect cooldown
    for t in _tracks:
        cfg = MODEL_CFG.get(t["model"], {})
        need_hits = int(cfg.get("persist_frames", PERSIST_FRAMES))
        cooldown  = float(cfg.get("cooldown_s", COOLDOWN_S))
        if t["hits"] >= need_hits and (now - t["sent_ts"]) >= cooldown:
            payload = {
                "ts": now,
                "type": t["label"],
                "model": t["model"],
                "conf": round(float(t["conf"]), 4),
                "box": {"x1": t["box"][0], "y1": t["box"][1], "x2": t["box"][2], "y2": t["box"][3]},
            }
            _events.append(payload)
            t["sent_ts"] = now

    # 4) prune tracks that went stale
        _tracks[:] = [
        t for t in _tracks
        if (now - t["last_ts"] < 3.0) or (t["hits"] >= 2)
    ]



@app.get("/health", response_class=PlainTextResponse)
def health():
    with cap_lock:
        is_open = cap.isOpened()
    return f"cap_is_open={is_open}, url={RTMP}"

@app.get("/models")
def models_info():
    return {k: {"path": v["path"], "conf": v["conf"]} for k, v in MODEL_CFG.items()}

@app.get("/last-dets")
def last_detections():
    with last_dets_lock:
        return JSONResponse(last_dets)

@app.get("/events")
def events():
    return JSONResponse(_events[-100:])

def mjpeg_frames():
    global cap
    backoff = 0.5
    try:
        while not stop_event.is_set():
            # --- read frame (with reconnects) ---
            with cap_lock:
                if not cap.isOpened():
                    cap.release()
                    time.sleep(backoff)
                    if stop_event.is_set():
                        break
                    cap = open_capture()
                    backoff = min(backoff * 2, 5.0)
                    # yield a black frame while reconnecting
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

            # --- inference ---
            dets = _infer_all(frame)

            # Only keep detections above the model's DRAW threshold
            strong_for_draw = [
                d for d in dets
                if d["conf"] >= MODEL_CFG[d["model"]]["conf"]
                and _area((int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"]))) >= MIN_BOX_PX
            ]

            # Update persistence & possibly emit events (extra strict)
            if strong_for_draw:
                _update_tracks_and_emit(strong_for_draw)

            # expose latest drawn dets to /last-dets (for debugging/UI)
            with last_dets_lock:
                last_dets[:] = strong_for_draw

            # --- draw and send ---
            if strong_for_draw:
                frame = _draw_boxes(frame, strong_for_draw)

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