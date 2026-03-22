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

# Find the center strip — it's the bright vertical band near the MIDDLE (not at edges)
col_means = np.mean(gray, axis=0)
# Exclude edges (first/last 15% of width) to avoid picking up the bright right edge
margin = int(w * 0.15)
center_zone = col_means[margin:w-margin]
strip_center = margin + int(np.argmax(center_zone))
print(f"Frame: {w}x{h}, Center strip at x={strip_center}")
print(f"Column brightness around center: {col_means[strip_center-5:strip_center+5]}")

# Now also check: the markers might NOT be on the center strip
# They might be on the label edges (left/right side of each label column)
# Let me scan the ENTIRE frame for small vertical white marks

# Strategy: look for local brightness peaks that form small vertical segments
# Use Laplacian or edge detection to find thin bright features

# Approach: threshold on absolute brightness, then filter for vertical shape
# But also check RELATIVE brightness (brighter than local neighborhood)

# Create local mean image
blur = cv2.GaussianBlur(gray, (51, 51), 0)
# Pixels brighter than their neighborhood
local_bright = cv2.subtract(gray, blur)

# Threshold the local brightness difference
_, local_mask = cv2.threshold(local_bright, 12, 255, cv2.THRESH_BINARY)

# Also need absolute brightness (markers are white, not just relatively bright)
_, abs_mask = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY)

# Combine: must be both locally bright AND absolutely bright
combined = cv2.bitwise_and(local_mask, abs_mask)

# Morphological: keep only vertical shapes
vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 8))
vert = cv2.morphologyEx(combined, cv2.MORPH_OPEN, vkernel)

cv2.imwrite("d:/project/video_check/.tmp/debug/local_bright.png", local_mask)
cv2.imwrite("d:/project/video_check/.tmp/debug/combined_mask.png", combined)
cv2.imwrite("d:/project/video_check/.tmp/debug/vert_final.png", vert)

contours, _ = cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

debug = frame.copy()
markers = []

for c in contours:
    x, y, cw, ch = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    aspect = ch / max(cw, 1)
    # Marker criteria: narrow width, reasonable height, vertically oriented
    if cw < 15 and 10 < ch < 70 and aspect > 2.5:
        markers.append((x, y, cw, ch, area))

# Sort by y position
markers.sort(key=lambda m: m[1])

for i, (x, y, cw, ch, area) in enumerate(markers):
    cv2.rectangle(debug, (x, y), (x+cw, y+ch), (0, 0, 255), 2)
    cv2.putText(debug, f"M{i}({x},{y})", (x+8, y+ch//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,255), 1)
    print(f"  M{i}: ({x},{y}) size {cw}x{ch} area={area:.0f}")

cv2.imwrite("d:/project/video_check/.tmp/debug/markers_v3.png", debug)
print(f"\nTotal: {len(markers)} markers")

# Save zoomed crops
for i, (mx, my, mw, mh, _) in enumerate(markers):
    pad = 25
    crop = frame[max(0,my-pad):min(h,my+mh+pad), max(0,mx-pad):min(w,mx+mw+pad)]
    crop_big = cv2.resize(crop, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f"d:/project/video_check/.tmp/debug/mzoom_{i}.png", crop_big)
