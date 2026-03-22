import cv2
import numpy as np
import os

os.makedirs("d:/project/video_check/.tmp", exist_ok=True)

video_path = "d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video:", video_path)
    exit(1)

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Total frames: {total}, FPS: {fps}")

best_frame = None
best_count = 0
best_idx = 0

step = max(1, total // 60)

for i in range(0, total, step):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
        continue

    # Rotate 180 as per project convention
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    # Detect red in HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
    mask = mask1 | mask2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = [c for c in contours if cv2.contourArea(c) > 100]

    if len(circles) > best_count:
        best_count = len(circles)
        best_frame = frame.copy()
        best_idx = i

if best_frame is not None:
    print(f"Best frame: {best_idx} with {best_count} red regions")
    hsv = cv2.cvtColor(best_frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
    mask = mask1 | mask2
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    debug = best_frame.copy()
    for c in contours:
        area = cv2.contourArea(c)
        if area > 100:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(debug, (x, y), (x+w, y+h), (0, 255, 0), 2)
            print(f"  Red region at ({x},{y}) size {w}x{h} area={area:.0f}")
    cv2.imwrite("d:/project/video_check/.tmp/marker_debug.png", debug)
    cv2.imwrite("d:/project/video_check/.tmp/marker_frame.png", best_frame)
    print("Saved: .tmp/marker_debug.png and .tmp/marker_frame.png")
else:
    print("No red regions found in any frame.")

cap.release()
