"""Dump ALL detected bright vertical features to understand what are the real markers."""
import cv2
import numpy as np

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
ret, frame = cap.read()
frame = cv2.rotate(frame, cv2.ROTATE_180)
H, W = frame.shape[:2]
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cap.release()

print(f"Frame size: {W}x{H}")

# Detect ALL vertical features, broad detection
blur = cv2.GaussianBlur(gray, (51, 51), 0)
local_bright = cv2.subtract(gray, blur)
_, local_mask = cv2.threshold(local_bright, 10, 255, cv2.THRESH_BINARY)
_, abs_mask = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)
combined = cv2.bitwise_and(local_mask, abs_mask)
vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
vert = cv2.morphologyEx(combined, cv2.MORPH_OPEN, vkernel)
contours, _ = cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

all_features = []
for c in contours:
    x, y, cw, ch = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    if ch > 5 and area > 5:
        all_features.append((x, y, cw, ch, area, ch/max(cw,1)))

# Sort by area descending
all_features.sort(key=lambda f: -f[4])

print(f"\nAll vertical features ({len(all_features)} total), sorted by area:")
print(f"{'x':>4} {'y':>4} {'w':>3} {'h':>3} {'area':>6} {'h/w':>5}  zone")
for x, y, w, h, area, ratio in all_features[:50]:
    zone = "LEFT" if x < W//3 else ("CENTER" if x < 2*W//3 else "RIGHT")
    print(f"{x:4d} {y:4d} {w:3d} {h:3d} {area:6.0f} {ratio:5.1f}  {zone}")

# Save annotated with ALL features
ann = frame.copy()
for x, y, w, h, area, ratio in all_features[:50]:
    color = (0, 255, 255)  # yellow
    if ratio > 2.0 and w < 15:
        color = (0, 0, 255)  # red = likely marker
    cv2.rectangle(ann, (x, y), (x+w, y+h), color, 1)
    cv2.putText(ann, f"{area:.0f}", (x, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)

cv2.imwrite("d:/project/video_check/.tmp/all_features.png", ann)
print("\nSaved all_features.png")
