import cv2
import numpy as np
import os

os.makedirs("d:/project/video_check/.tmp/debug", exist_ok=True)

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")

# Look at the midpoint capture frame for label_001U (frame 9)
# and label_005U (frame 113) to see where the label content actually is
for fn in [9, 113, 200, 270]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    h, w = frame.shape[:2]

    # Draw horizontal lines every 50px so we can see Y coordinates
    debug = frame.copy()
    for y in range(0, h, 50):
        cv2.line(debug, (0, y), (w, y), (0, 0, 255), 1)
        cv2.putText(debug, f"y={y}", (5, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # Draw vertical lines for column boundaries
    cv2.line(debug, (185, 0), (185, h), (0, 255, 0), 1)
    cv2.line(debug, (225, 0), (225, h), (0, 255, 0), 1)

    cv2.imwrite(f"d:/project/video_check/.tmp/debug/grid_f{fn}.png", debug)
    print(f"Frame {fn}: saved grid overlay")

cap.release()
