import cv2
import numpy as np
import os

os.makedirs("d:/project/video_check/.tmp/debug", exist_ok=True)

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Sample several frames and save annotated debug images
for idx, frame_no in enumerate(range(200, min(total, 200 + 10))):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold for dark marks
    _, dark = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    # Find contours of dark regions
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    debug = frame.copy()

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if 50 < area < 3000:
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / max(h, 1)
            # Roughly square or circular
            if 0.3 < aspect < 3.0:
                cx, cy = x + w//2, y + h//2
                candidates.append((cx, cy, w, h, area))
                cv2.rectangle(debug, (x, y), (x+w, y+h), (0, 255, 0), 1)

    # Save annotated frame
    out = f"d:/project/video_check/.tmp/debug/frame_{frame_no:04d}_candidates.png"
    cv2.imwrite(out, debug)
    print(f"Frame {frame_no}: {len(candidates)} dark candidates -> {out}")
    if idx == 0:
        # Also save clean frame
        cv2.imwrite("d:/project/video_check/.tmp/debug/clean_frame.png", frame)

cap.release()
print("Done. Open .tmp/debug/ to inspect.")
