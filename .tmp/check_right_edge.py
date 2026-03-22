import cv2
import numpy as np
import os

os.makedirs("d:/project/video_check/.tmp/debug", exist_ok=True)

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 270)
ret, frame = cap.read()
frame = cv2.rotate(frame, cv2.ROTATE_180)
h, w = frame.shape[:2]
cap.release()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# The right edge strip is at x ~ 450-464
# Zoom into right edge at the two seam y positions
for sy in [200, 210, 220, 230, 240, 410, 420, 430, 440, 450]:
    crop = frame[sy-15:sy+15, w-30:w]
    big = cv2.resize(crop, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f"d:/project/video_check/.tmp/debug/redge_y{sy}.png", big)
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    print(f"y={sy} right edge: min={g.min()} max={g.max()} mean={g.mean():.1f}")

# Also check if the left edge (x=0..30) has markers
for sy in [200, 210, 220, 230, 240, 410, 420, 430, 440, 450]:
    crop = frame[sy-15:sy+15, 0:30]
    big = cv2.resize(crop, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f"d:/project/video_check/.tmp/debug/ledge_y{sy}.png", big)
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    print(f"y={sy} left  edge: min={g.min()} max={g.max()} mean={g.mean():.1f}")
