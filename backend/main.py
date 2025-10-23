import os, time, threading
import cv2, numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware   
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

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
    return cv2.VideoCapture(RTMP, cv2.CAP_FFMPEG)

cap = open_capture()
cap_lock = threading.Lock()

@app.get("/health", response_class=PlainTextResponse)
def health():
    return f"cap_is_open={cap.isOpened()}, url={RTMP}"

def mjpeg_frames():
    global cap
    backoff = 0.5
    while True:
        if not cap.isOpened():
            cap.release()
            time.sleep(backoff)
            cap = open_capture()
            backoff = min(backoff * 2, 5.0)
            black = np.zeros((360, 640, 3), dtype=np.uint8)
            ok, jpg = cv2.imencode(".jpg", black)
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
            continue

        ok, frame = cap.read()
        if not ok:
            cap.release()
            continue

        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue

        backoff = 0.5
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")

@app.get("/video-feed")
def video_feed():
    return StreamingResponse(mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame")


#python -m uvicorn main:app --host 0.0.0.0 --port 8000