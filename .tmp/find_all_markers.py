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

# Look for markers EVERYWHERE (no x filter) using the same approach
blur = cv2.GaussianBlur(gray, (51, 51), 0)
local_bright = cv2.subtract(gray, blur)
_, local_mask = cv2.threshold(local_bright, 10, 255, cv2.THRESH_BINARY)
_, abs_mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
combined = cv2.bitwise_and(local_mask, abs_mask)

vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 8))
vert = cv2.morphologyEx(combined, cv2.MORPH_OPEN, vkernel)

contours, _ = cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

debug = frame.copy()
all_markers = []
for c in contours:
    x, y, cw, ch = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    if cw < 10 and 10 < ch < 70 and ch/max(cw,1) > 2.5 and area > 8:
        all_markers.append((x, y, cw, ch, area))
        cv2.rectangle(debug, (x, y), (x+cw, y+ch), (0, 0, 255), 2)
        cv2.putText(debug, f"{x},{y}", (x+5, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)

all_markers.sort(key=lambda m: (m[0], m[1]))
print(f"All vertical bright marks found: {len(all_markers)}")
for x, y, cw, ch, area in all_markers:
    region = "LEFT" if x < 100 else "CENTER" if 150 < x < 250 else "RIGHT" if x > 350 else "MID"
    print(f"  {region:6s} ({x:3d},{y:3d}) {cw}x{ch} area={area:.0f}")

cv2.imwrite("d:/project/video_check/.tmp/debug/all_markers_270.png", debug)

# Zoom into right-edge seam areas
for sy in [220, 430]:
    crop = frame[sy-40:sy+40, w-80:w]
    big = cv2.resize(crop, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f"d:/project/video_check/.tmp/debug/right_seam_{sy}.png", big)
