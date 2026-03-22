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

# Detect markers: vertical white slits on center strip
blur = cv2.GaussianBlur(gray, (51, 51), 0)
local_bright = cv2.subtract(gray, blur)
_, local_mask = cv2.threshold(local_bright, 12, 255, cv2.THRESH_BINARY)
_, abs_mask = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY)
combined = cv2.bitwise_and(local_mask, abs_mask)

vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 8))
vert = cv2.morphologyEx(combined, cv2.MORPH_OPEN, vkernel)

contours, _ = cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter for real markers on center strip (x ~ 180-230)
markers = []
for c in contours:
    x, y, cw, ch = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    if cw < 10 and 12 < ch < 70 and ch/max(cw,1) > 3 and area > 10:
        if 170 < x < 240:
            markers.append((x, y, cw, ch))

markers.sort(key=lambda m: m[1])

# Draw result — big red circles + arrows pointing to each marker
output = frame.copy()
for i, (x, y, cw, ch) in enumerate(markers):
    cx = x + cw // 2
    cy = y + ch // 2
    # Red circle around marker
    cv2.circle(output, (cx, cy), 30, (0, 0, 255), 3)
    # Label
    cv2.putText(output, f"MARKER {i+1}", (cx + 35, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # Arrow pointing to marker
    cv2.arrowedLine(output, (cx + 33, cy), (cx + 5, cy), (0, 0, 255), 2, tipLength=0.3)

cv2.imwrite("d:/project/video_check/.tmp/markers_spotted.png", output)
print(f"Found {len(markers)} markers:")
for i, (x, y, cw, ch) in enumerate(markers):
    print(f"  MARKER {i+1}: x={x}, y={y}, size={cw}x{ch}")
print("Saved: .tmp/markers_spotted.png")
