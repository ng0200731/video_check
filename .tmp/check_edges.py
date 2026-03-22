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

# Zoom into all 4 quadrants at the seam lines
# Seam at y~220: check left edge, center-left, center-right, right edge
# Seam at y~420: same

seam_y = 220
regions = {
    "left_edge":   (0, seam_y-40, 60, seam_y+40),
    "center_left": (180, seam_y-40, 230, seam_y+40),
    "center_right":(230, seam_y-40, 280, seam_y+40),
    "right_edge":  (w-60, seam_y-40, w, seam_y+40),
}

for name, (x1, y1, x2, y2) in regions.items():
    crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
    big = cv2.resize(crop, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f"d:/project/video_check/.tmp/debug/seam220_{name}.png", big)
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    print(f"seam220 {name}: min={gray_crop.min()} max={gray_crop.max()} mean={gray_crop.mean():.1f}")

# Same for seam at y~420
seam_y = 420
regions2 = {
    "left_edge":   (0, seam_y-40, 60, seam_y+40),
    "center_left": (180, seam_y-40, 230, seam_y+40),
    "center_right":(230, seam_y-40, 280, seam_y+40),
    "right_edge":  (w-60, seam_y-40, w, seam_y+40),
}

for name, (x1, y1, x2, y2) in regions2.items():
    crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
    big = cv2.resize(crop, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f"d:/project/video_check/.tmp/debug/seam420_{name}.png", big)
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    print(f"seam420 {name}: min={gray_crop.min()} max={gray_crop.max()} mean={gray_crop.mean():.1f}")
