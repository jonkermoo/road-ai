import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()
cap = cv2.VideoCapture(0)

def mjpeg_frames():
    while True:
        ok, frame = cap.read()
        if not ok:
            # yield a small black frame and stop
            black = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, jpg = cv2.imencode(".jpg", black, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ret:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                       jpg.tobytes() + b"\r\n")
            break
        ret, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               jpg.tobytes() + b"\r\n")
        
def video_feed():
    return StreamingResponse(mjpeg_frames(),
      media_type="multipart/x-mixed-replace; boundary=frame")