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

print(f"Frame size: {w}x{h}")

# Crop and save 4 corner regions so we can see what's actually there
margin = 120  # px from each edge

# Define label zone (exclude the bright strip and bottom background)
# From the frame, labels occupy roughly top 75% of the frame
label_h = int(h * 0.75)

corners = {
    "top_left":     (0, 0, margin*2, margin*2),
    "top_right":    (w - margin*2, 0, w, margin*2),
    "mid_left":     (0, label_h//2 - margin, margin*2, label_h//2 + margin),
    "mid_right":    (w - margin*2, label_h//2 - margin, w, label_h//2 + margin),
    "center_top":   (w//2 - margin, 0, w//2 + margin, margin*2),
    "center_mid":   (w//2 - margin, label_h//2 - margin, w//2 + margin, label_h//2 + margin),
}

for name, (x1, y1, x2, y2) in corners.items():
    crop = frame[y1:y2, x1:x2]
    # Scale up 3x for visibility
    crop_big = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f"d:/project/video_check/.tmp/debug/corner_{name}.png", crop_big)

    # Analyze brightness in this region
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    print(f"{name}: min={gray_crop.min()} max={gray_crop.max()} mean={gray_crop.mean():.1f}")

# Also save full frame with grid overlay showing where labels likely are
debug = frame.copy()
# Draw horizontal lines where label boundaries might be
for y in range(0, label_h, label_h // 4):
    cv2.line(debug, (0, y), (w, y), (0, 255, 0), 1)
cv2.line(debug, (w//2, 0), (w//2, label_h), (0, 255, 0), 1)
cv2.imwrite("d:/project/video_check/.tmp/debug/grid_overlay.png", debug)
print("Saved corner crops and grid overlay")
