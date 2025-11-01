# RoadAI

### DJI Camera (RTMP Publisher)

### NGINX RTMP Server

### FastAPI

### React Frontend

### YoloV8

This project sets up a real-time video streaming pipeline using a DJI Camera as
an RTMP publisher and NGINX as an RTMP server. It will do live road assessments
and mark down events such as potholes, construction, and police sightings.

## Notes

# Background Event Sink System

thread-safe event queue that offloads all Supabase uploads and databases and
writes to a dedicated background worker thread.

- Prevents network latency from blocking the real-time video stream
- Ensures smooth frame rates even under heavy network traffic
- YOLO runs everything on the main thread, having a multi-threaded, concurrency
  safe pipeline prevents two threads modifying the same frame at once.

# Self-healing RTMP capture

OpenCV capture connection automatically detects stream drops and recovers
without a manual restart

- added a placeholder frame while reconnecting
- prevented full application crashes when the live feed stops
