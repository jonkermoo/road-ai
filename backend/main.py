import os, time, threading, signal, traceback
from collections import deque
from typing import Optional, List, Dict, Any
import cv2, numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, PlainTextResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from ultralytics import YOLO
from supabase_io import upload_jpeg, insert_event

load_dotenv()

# OpenCV reconnect flags so RTMP auto-recovers if the stream drops.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "reconnect;1|reconnect_streamed;1|reconnect_delay_max;2|"
    "rw_timeout;5000000|stimeout;5000000|timeout;5000000"
)


# FastAPI
app = FastAPI(title="live-road-ai")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stream / Model configurations
RTMP = os.getenv("RTMP_URL")
if not RTMP:
    raise RuntimeError("Set RTMP_URL in backend/.env")

# Frame skipping for performance: process YOLO every N frames
PROCESS_EVERY_N_FRAMES = int(os.getenv("PROCESS_EVERY_N_FRAMES", "3"))

# Combined YOLO model configuration
# Single model detects all 3 classes: police, pothole, roadwork
COMBINED_MODEL_PATH = os.getenv("COMBINED_MODEL", "../ml/runs/detect/combined_road_ai/weights/best.pt")

# Per-class detection configs (conf, emit_conf, colors, filters)
# Class IDs: 0=police, 1=pothole, 2=roadwork (from combined model training)
CLASS_CFG: Dict[str, Dict[str, Any]] = {
    "police": {
        "conf": float(os.getenv("POLICE_CONF", "0.75")),
        "emit_conf": float(os.getenv("POLICE_EMIT_CONF", "0.98")),
        "color": (60, 170, 255),
        "ar_min": float(os.getenv("POLICE_AR_MIN", "1.40")),
        "ar_max": float(os.getenv("POLICE_AR_MAX", "3.50")),
        "min_box_px": int(os.getenv("POLICE_MIN_BOX_PX", "70000")),
        "min_w_px":   int(os.getenv("POLICE_MIN_W_PX",   "180")),
        "min_h_px":   int(os.getenv("POLICE_MIN_H_PX",   "80")),
        "persist_frames": int(os.getenv("POLICE_PERSIST_FRAMES", "20")),
        "cooldown_s": float(os.getenv("POLICE_COOLDOWN_S", "25")),
    },
    "pothole": {
        "conf": float(os.getenv("POTHOLE_CONF", "0.75")),
        "emit_conf": float(os.getenv("POTHOLE_EMIT_CONF", "0.90")),
        "color": (90, 220, 100),
        "ar_min": float(os.getenv("POTHOLE_AR_MIN", "0.0")),
        "ar_max": float(os.getenv("POTHOLE_AR_MAX", "9.0")),
        "min_box_px": int(os.getenv("POTHOLE_MIN_BOX_PX", str(os.getenv("MIN_BOX_PX", "40000")))),
        "persist_frames": int(os.getenv("POTHOLE_PERSIST_FRAMES", str(os.getenv("PERSIST_FRAMES", "15")))),
        "cooldown_s": float(os.getenv("POTHOLE_COOLDOWN_S", str(os.getenv("COOLDOWN_S", "15")))),
    },
    "roadwork": {
        "conf": float(os.getenv("ROADWORK_CONF", "0.75")),
        "emit_conf": float(os.getenv("ROADWORK_EMIT_CONF", "0.90")),
        "color": (255, 120, 60),
        "ar_min": float(os.getenv("ROADWORK_AR_MIN", "0.75")),
        "ar_max": float(os.getenv("ROADWORK_AR_MAX", "1.33")),
        "min_box_px": int(os.getenv("ROADWORK_MIN_BOX_PX", str(os.getenv("MIN_BOX_PX", "40000")))),
        "persist_frames": int(os.getenv("ROADWORK_PERSIST_FRAMES", str(os.getenv("PERSIST_FRAMES", "15")))),
        "cooldown_s": float(os.getenv("ROADWORK_COOLDOWN_S", str(os.getenv("COOLDOWN_S", "15")))),
    },
}
for _name, _cfg in CLASS_CFG.items():
    _cfg.setdefault("emit_conf", _cfg.get("conf", 0.9))
    _cfg.setdefault("ar_min", 0.0)
    _cfg.setdefault("ar_max", 9.0)

# Backward compatibility: MODEL_CFG is now CLASS_CFG
MODEL_CFG = CLASS_CFG

YOLO_IOU    = float(os.getenv("YOLO_IOU",  "0.45"))
FRAME_MAX_W = int(os.getenv("FRAME_MAX_W", "960"))

PERSIST_FRAMES = int(os.getenv("PERSIST_FRAMES", "6"))
PERSIST_IOU    = float(os.getenv("PERSIST_IOU", "0.30"))
COOLDOWN_S     = float(os.getenv("COOLDOWN_S", "8"))
MIN_BOX_PX     = int(os.getenv("MIN_BOX_PX", "9000"))

ROI_YMIN_FRAC  = float(os.getenv("ROI_YMIN_FRAC", "0.20"))
ROI_YMAX_FRAC  = float(os.getenv("ROI_YMAX_FRAC", "0.98"))

# Capture setup, self-healing
def open_capture() -> cv2.VideoCapture:
    """
    Open the upstream stream with FFMPEG backend.
    Set environment variables to help FFMPEG connect.
    """
    # Set FFMPEG options via environment variable
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtmp_transport;tcp|rtmp_buffer;1000000'

    # Try opening without specifying backend (auto-detect)
    cap = cv2.VideoCapture(RTMP)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        return cap

    # Fallback: try with explicit FFMPEG backend
    cap = cv2.VideoCapture(RTMP, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    return cap

cap = open_capture()
cap_lock = threading.Lock()


# Load single combined YOLO model
if not os.path.exists(COMBINED_MODEL_PATH):
    raise FileNotFoundError(f"Combined model not found at: {COMBINED_MODEL_PATH}")

combined_model = YOLO(COMBINED_MODEL_PATH)
try:
    combined_model.fuse()
except Exception:
    pass

# Verify model has expected class names
expected_classes = ["police", "pothole", "roadwork"]
model_classes = list(combined_model.names.values())
print(f"Loaded combined model with classes: {model_classes}")
if model_classes != expected_classes:
    print(f"WARNING: Expected {expected_classes}, got {model_classes}")


# Shared State: Detections, Tracks, Frames
# These are for UI/debug endpoints
last_dets: List[Dict[str, Any]] = []
last_dets_lock = threading.Lock()
_events: List[Dict[str, Any]] = []
_events_lock = threading.Lock()

# Simple track store for persistence gating (not full MOT)
_tracks: List[Dict[str, Any]] = []

# Graceful shutdown flag
stop_event = threading.Event()

# Keep a copy of the most recent frame for the event sinker snapshots
latest_frame_lock = threading.Lock()
latest_frame_bgr: Optional[np.ndarray] = None


# Session Management & Location Tracking
# Tracks the active device session and current GPS location
active_session: Optional[Dict[str, Any]] = None
active_session_lock = threading.Lock()
current_location: Optional[Dict[str, float]] = None
location_lock = threading.Lock()


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


# utility helpers
def _maybe_resize(frame: np.ndarray) -> np.ndarray:
    """Optionally downscale width to FRAME_MAX_W for stable throughput."""
    if FRAME_MAX_W and FRAME_MAX_W > 0:
        h, w = frame.shape[:2]
        if w > FRAME_MAX_W:
            scale = FRAME_MAX_W / w
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return frame

def _draw_boxes(image_bgr: np.ndarray, dets: List[Dict[str, Any]]) -> np.ndarray:
    """Draw labeled boxes (for /video-feed debugging)."""
    for d in dets:
        x1, y1, x2, y2 = map(int, [d["x1"], d["y1"], d["x2"], d["y2"]])
        label = f'{d["label"]} {d["conf"]:.2f}'
        color = tuple(map(int, d.get("color", (0, 255, 0))))
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_bgr, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return image_bgr

def _area(b):
    return max(0, b[2]-b[0]) * max(0, b[3]-b[1])

def _aspect_ratio(b):
    """Return width/height ratio of (x1, y1, x2, y2)."""
    w = max(1, b[2] - b[0]); h = max(1, b[3] - b[1])
    return w / h

def _iou(a,b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    ua = _area(a) + _area(b) - inter
    return inter/ua if ua > 0 else 0.0

# post-processing
def _infer_all(frame: np.ndarray) -> List[Dict[str, Any]]:
    """Run combined model inference and categorize detections by class."""
    detections: List[Dict[str, Any]] = []
    ts = time.time()
    H, W = frame.shape[:2]

    # Run single combined model (detects all 3 classes at once)
    # Use minimum confidence across all classes
    min_conf = min(cfg["conf"] for cfg in CLASS_CFG.values())
    res = combined_model.predict(frame, conf=min_conf, iou=YOLO_IOU, verbose=False)

    if not res:
        return detections

    r = res[0]
    if r.boxes is None or not hasattr(r.boxes, "data"):
        return detections

    data = r.boxes.data.cpu().numpy()
    names = r.names or {}

    # Process each detection and route to correct class
    for row in data:
        x1, y1, x2, y2, conf, cls_id = row.tolist()
        label = names.get(int(cls_id), "unknown")

        # Get class-specific config (use label name to look up config)
        if label not in CLASS_CFG:
            continue  # Skip unknown classes

        cfg = CLASS_CFG[label]

        # Apply class-specific confidence threshold
        if conf < cfg["conf"]:
            continue

        detections.append({
            "model": label,  # Use label as "model" for backward compatibility
            "label": label,
            "cls": int(cls_id),
            "conf": float(conf),
            "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
            "color": cfg["color"],
            "ts": ts,
            "_img_h": H, "_img_w": W,
        })

    return detections

def _match_track(d) -> int:
    best_i, best_iou = -1, 0.0
    cand = (int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"]))
    for i, t in enumerate(_tracks):
        if t["model"] != d["model"] or t["label"] != d["label"]:
            continue
        iou = _iou(t["box"], cand)
        if iou > best_iou:
            best_i, best_iou = i, iou
    return best_i if best_iou >= PERSIST_IOU else -1

def _passes_emit_rules(d) -> bool:
    cfg = MODEL_CFG[d["model"]]

    # 1) Confidence gate (stricter emit_conf than draw/conf)
    emit_conf = float(cfg.get("emit_conf", cfg.get("conf", 0.9)))
    if d["conf"] < emit_conf:
        return False

    # 2) Size gates (area + w/h)
    box = (int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"]))
    min_box_px = int(cfg.get("min_box_px", MIN_BOX_PX))
    if _area(box) < min_box_px:
        return False
    min_w = int(cfg.get("min_w_px", 0))
    min_h = int(cfg.get("min_h_px", 0))
    w = max(0, box[2] - box[0]); h = max(0, box[3] - box[1])
    if (min_w and w < min_w) or (min_h and h < min_h):
        return False

    # 3) Aspect-ratio gate
    ar_min = float(cfg.get("ar_min", 0.0))
    ar_max = float(cfg.get("ar_max", 9.0))
    ar = _aspect_ratio(box)
    if not (ar_min <= ar <= ar_max):
        return False

    # 4) ROI gate (reject too high/low boxes)
    ymid = (box[1] + box[3]) * 0.5
    H = d.get("_img_h", None)
    if H:
        yf = ymid / float(H)
        if not (ROI_YMIN_FRAC <= yf <= ROI_YMAX_FRAC):
            return False

    return True

# emits events only after certain requirements met
def _update_tracks_and_emit(detections: List[Dict[str, Any]]) -> None:
    with _events_lock:
        _events.append(payload)

    now = time.time()

    # 1) Decay stale tracks (so they drop if they vanish)
    for t in _tracks:
        if now - t["last_ts"] > 2.0:
            t["hits"] = max(0, t["hits"] - 1)

    # 2) Incorporate detections that PASS strict emit rules
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

    # emit events only for stable tracks that respect cooldown
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

            with eventq_lock:
                event_queue.append({"label": t["label"]})

    _tracks[:] = [
        t for t in _tracks
        if (now - t["last_ts"] < 3.0) or (t["hits"] >= 2)
    ]

# Event Sync, decouple network I/O (Supabase) from real-time loop.
event_queue = deque(maxlen=1000)
eventq_lock = threading.Lock()

# dequeues emitted items and uploads latest frame to Supabase
def _event_sink_worker():
    while not stop_event.is_set():
        item = None
        with eventq_lock:
            if event_queue:
                item = event_queue.popleft()
        if item is None:
            time.sleep(0.02)
            continue

        label = item["label"]
        try:
            snap = None
            with latest_frame_lock:
                if latest_frame_bgr is not None:
                    snap = latest_frame_bgr.copy()

            img_url = None
            if snap is not None:
                ok, jpg = cv2.imencode(".jpg", snap, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    img_url = upload_jpeg(jpg.tobytes())

            # get current location if available
            lat, lng = None, None
            with location_lock:
                if current_location:
                    lat = current_location.get("lat")
                    lng = current_location.get("lng")

            insert_event(evt_type=label, img_url=img_url, lat=lat, lng=lng)

        except Exception as e:
            print("[warn] event_sink failed:", e)
            traceback.print_exc()

sink_thread = threading.Thread(target=_event_sink_worker, daemon=True)
sink_thread.start()

# video endpoint, draws boxes and keeps safe copy of latest frame
def mjpeg_frames():
    global cap, latest_frame_bgr
    backoff = 0.5
    frame_count = 0
    last_detections = []  # Cache detections from last processed frame

    try:
        while not stop_event.is_set():
            # read frame (with reconnects)
            with cap_lock:
                if not cap.isOpened():
                    cap.release()
                    time.sleep(backoff)
                    if stop_event.is_set():
                        break
                    cap = open_capture()
                    backoff = min(backoff * 2, 5.0)
                    # yield a black frame while reconnecting (keeps clients connected)
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
            frame_count += 1

            with latest_frame_lock:
                latest_frame_bgr = frame.copy()

            # Frame skipping: only run YOLO every N frames
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                dets = _infer_all(frame)

                strong_for_draw = [
                    d for d in dets
                    if d["conf"] >= MODEL_CFG[d["model"]]["conf"]
                    and _area((int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"]))) >= MIN_BOX_PX
                ]

                if strong_for_draw:
                    _update_tracks_and_emit(strong_for_draw)

                last_detections = strong_for_draw  # Cache for skipped frames

                with last_dets_lock:
                    last_dets[:] = strong_for_draw
            else:
                # Reuse cached detections from last processed frame
                strong_for_draw = last_detections

            out = frame
            if strong_for_draw:
                out = _draw_boxes(out, strong_for_draw)

            ok, jpg = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                backoff = 0.5
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
    except (GeneratorExit, BrokenPipeError):
        pass
    finally:
        with cap_lock:
            if cap and cap.isOpened():
                cap.release()


# Pydantic Models for API
class SessionClaim(BaseModel):
    device_id: str

class LocationUpdate(BaseModel):
    lat: float
    lng: float


# API Endpoints
# session management endpoints
@app.post("/session/claim")
def claim_session(claim: SessionClaim):
    """Claim the active session. Only one device can be active at a time."""
    global active_session
    with active_session_lock:
        now = time.time()
        # check if there's already an active session
        if active_session:
            # allow reclaim on device if stale
            if active_session["device_id"] == claim.device_id:
                active_session["last_heartbeat"] = now
                return {"success": True, "message": "Session renewed"}
            elif now - active_session["last_heartbeat"] > 30:
                # stale session, allow takeover
                active_session = {"device_id": claim.device_id, "claimed_at": now, "last_heartbeat": now}
                return {"success": True, "message": "Session claimed (previous session expired)"}
            else:
                raise HTTPException(status_code=409, detail="Another device is already streaming")

        # no active session, claim it
        active_session = {"device_id": claim.device_id, "claimed_at": now, "last_heartbeat": now}
        return {"success": True, "message": "Session claimed"}

@app.post("/session/heartbeat")
def session_heartbeat(claim: SessionClaim):
    """Send heartbeat to keep session alive."""
    global active_session
    with active_session_lock:
        if not active_session or active_session["device_id"] != claim.device_id:
            raise HTTPException(status_code=403, detail="No active session or device mismatch")
        active_session["last_heartbeat"] = time.time()
        return {"success": True}

@app.post("/session/release")
def release_session(claim: SessionClaim):
    """Release the active session."""
    global active_session, current_location
    with active_session_lock:
        if active_session and active_session["device_id"] == claim.device_id:
            active_session = None
            with location_lock:
                current_location = None
            return {"success": True, "message": "Session released"}
        return {"success": False, "message": "No active session or device mismatch"}

@app.get("/session/status")
def session_status():
    """Check if there's an active session."""
    with active_session_lock:
        if active_session:
            return {
                "active": True,
                "device_id": active_session["device_id"],
                "claimed_at": active_session["claimed_at"],
                "last_heartbeat": active_session["last_heartbeat"]
            }
        return {"active": False}

# location update endpoint
@app.post("/update-location")
def update_location(location: LocationUpdate, device_id: str):
    """Receive location updates from the active device."""
    global current_location
    with active_session_lock:
        if not active_session or active_session["device_id"] != device_id:
            raise HTTPException(status_code=403, detail="Not the active device")

    with location_lock:
        current_location = {"lat": location.lat, "lng": location.lng}

    return {"success": True}

@app.get("/current-location")
def get_current_location():
    """Get the current GPS location."""
    with location_lock:
        if current_location:
            return current_location
        return {"lat": None, "lng": None}

# existing API endpoints
@app.get("/health", response_class=PlainTextResponse)
def health():
    with cap_lock:
        is_open = cap.isOpened()
    return f"cap_is_open={is_open}, url={RTMP}"

@app.get("/models")
def models_info():
    return {
        "model_type": "combined",
        "model_path": COMBINED_MODEL_PATH,
        "classes": list(CLASS_CFG.keys()),
        "class_configs": {k: {"conf": v["conf"], "emit_conf": v["emit_conf"]} for k, v in CLASS_CFG.items()}
    }

@app.get("/last-dets")
def last_detections():
    with last_dets_lock:
        return JSONResponse(last_dets)

@app.get("/events")
def events():
    with _events_lock:
        return JSONResponse(_events[-100:])

@app.get("/video-feed")
def video_feed():
    return StreamingResponse(mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame")


# run:  python -m uvicorn main:app --host 0.0.0.0 --port 8000
