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

# Same approach as markers_v3 which produced the correct M2, M5, M9
blur = cv2.GaussianBlur(gray, (51, 51), 0)
local_bright = cv2.subtract(gray, blur)
_, local_mask = cv2.threshold(local_bright, 12, 255, cv2.THRESH_BINARY)
_, abs_mask = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY)
combined = cv2.bitwise_and(local_mask, abs_mask)

vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 8))
vert = cv2.morphologyEx(combined, cv2.MORPH_OPEN, vkernel)

contours, _ = cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Collect candidates on center strip (x ~ 180-230)
candidates = []
for c in contours:
    x, y, cw, ch = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    if cw < 15 and 8 < ch < 60 and ch/max(cw,1) > 2.5 and area > 10:
        if 180 < x < 240:
            candidates.append((x, y, cw, ch, area))

candidates.sort(key=lambda m: m[1])

# Group nearby candidates into single markers (M2+M5 were the same marker split)
markers = []
used = [False] * len(candidates)
for i in range(len(candidates)):
    if used[i]:
        continue
    group = [candidates[i]]
    used[i] = True
    for j in range(i+1, len(candidates)):
        if used[j]:
            continue
        # Same marker if y positions are within 40px and x within 15px
        if abs(candidates[j][1] - candidates[i][1]) < 50 and abs(candidates[j][0] - candidates[i][0]) < 15:
            group.append(candidates[j])
            used[j] = True
    # Merge group into one bounding box
    min_x = min(g[0] for g in group)
    min_y = min(g[1] for g in group)
    max_x = max(g[0] + g[2] for g in group)
    max_y = max(g[1] + g[3] for g in group)
    markers.append((min_x, min_y, max_x - min_x, max_y - min_y))

markers.sort(key=lambda m: m[1])

# Draw clean output
output = frame.copy()
colors = [(0, 0, 255), (0, 180, 0)]  # red, green
for i, (x, y, mw, mh) in enumerate(markers):
    cx = x + mw // 2
    cy = y + mh // 2
    color = colors[i % len(colors)]

    # Circle
    cv2.circle(output, (cx, cy), 35, color, 3)

    # Arrow from left
    cv2.arrowedLine(output, (cx - 80, cy), (cx - 38, cy), color, 2, tipLength=0.3)

    # Label on left side
    label = f"MARKER {i+1}"
    cv2.putText(output, label, (cx - 170, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Coordinates below
    coord = f"({x},{y}) {mw}x{mh}"
    cv2.putText(output, coord, (cx - 170, cy + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

cv2.imwrite("d:/project/video_check/.tmp/markers_confirmed.png", output)
print(f"Confirmed {len(markers)} markers:")
for i, (x, y, mw, mh) in enumerate(markers):
    print(f"  MARKER {i+1}: ({x},{y}) size {mw}x{mh}")
print("Saved: .tmp/markers_confirmed.png")
