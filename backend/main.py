import os, time
import cv2, numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, PlainTextResponse
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="live-road-ai-min")

RTMP = os.getenv("RTMP_URL")  # e.g. rtmp://127.0.0.1/live/stream
if not RTMP:
    raise RuntimeError("Set RTMP_URL to your RTMP endpoint")

def open_capture():
    # RTMP needs FFMPEG backend
    return cv2.VideoCapture(RTMP, cv2.CAP_FFMPEG)

cap = open_capture()

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