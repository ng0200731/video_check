"""Re-examine: find ALL bright vertical features across full width, not just center strip."""
import cv2
import numpy as np

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 249)
ret, frame = cap.read()
frame = cv2.rotate(frame, cv2.ROTATE_180)
H, W = frame.shape[:2]
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cap.release()

# Detect ALL bright vertical bars across FULL width
blur = cv2.GaussianBlur(gray, (51, 51), 0)
local_bright = cv2.subtract(gray, blur)
_, mask = cv2.threshold(local_bright, 10, 255, cv2.THRESH_BINARY)
_, abs_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
combined = cv2.bitwise_and(mask, abs_mask)
vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 8))
vert = cv2.morphologyEx(combined, cv2.MORPH_OPEN, vkernel)
contours, _ = cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

all_marks = []
for c in contours:
    x, y, cw, ch = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    if ch > 10 and cw < 25 and ch / max(cw, 1) > 1.5 and area > 15:
        all_marks.append((x, y, cw, ch, area))

all_marks.sort(key=lambda m: (m[1], m[0]))
print(f"Frame 249: {len(all_marks)} vertical features")
print(f"{'x':>4} {'y':>4} {'w':>3} {'h':>3} {'area':>6} {'zone'}")
for x, y, w, h, area in all_marks:
    zone = f"x={x:3d}"
    print(f"{x:4d} {y:4d} {w:3d} {h:3d} {area:6.0f}  {zone}")

# Annotate frame with ALL features and labels
ann = frame.copy()
for i, (x, y, w, h, area) in enumerate(all_marks):
    cv2.rectangle(ann, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.putText(ann, f"{i}:({x},{y}) a={area:.0f}", (x+w+2, y+h//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1)

cv2.imwrite("d:/project/video_check/.tmp/all_marks_fullwidth.png", ann)
print("\nSaved all_marks_fullwidth.png")
