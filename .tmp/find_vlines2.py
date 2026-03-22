import cv2
import numpy as np
import os

os.makedirs("d:/project/video_check/.tmp/debug", exist_ok=True)

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
ret, frame = cap.read()
frame = cv2.rotate(frame, cv2.ROTATE_180)
h, w = frame.shape[:2]
cap.release()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Try multiple brightness thresholds to find subtle white marks
for thresh_val in [140, 150, 160, 170, 180, 190, 200]:
    bright = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)[1]
    # Vertical line kernel — narrow and tall
    vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    vert = cv2.morphologyEx(bright, cv2.MORPH_OPEN, vkernel)

    contours, _ = cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    debug = frame.copy()
    count = 0
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        aspect = ch / max(cw, 1)
        # Filter: narrow (width < 15), tall enough (height 15-80), not at edges
        if cw < 15 and 15 < ch < 80 and aspect > 3 and 10 < x < (w-10):
            cv2.rectangle(debug, (x, y), (x+cw, y+ch), (0, 0, 255), 2)
            cv2.putText(debug, f"{x},{y}", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
            count += 1

    cv2.imwrite(f"d:/project/video_check/.tmp/debug/vlines_t{thresh_val}.png", debug)
    print(f"Threshold {thresh_val}: {count} markers found")

print("Done — check .tmp/debug/vlines_t*.png")
