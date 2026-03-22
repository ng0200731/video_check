"""Diagnostic: extract frame 4050, examine the blue registration marks on left/right edges."""
import cv2
import numpy as np

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 4050)
ret, frame = cap.read()
frame = cv2.rotate(frame, cv2.ROTATE_180)
cap.release()

h, w = frame.shape[:2]
print(f"Frame size: {w}x{h}")

# Convert to HSV to look for colored marks (blue markers should have distinct hue)
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Blue in HSV: H=100-130, S>50, V>50
lower_blue = np.array([90, 30, 80])
upper_blue = np.array([140, 255, 255])
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Also check for any saturated color (markers vs gray labels)
# Saturation channel - labels are gray (low sat), markers should pop
sat = hsv[:,:,1]
_, sat_mask = cv2.threshold(sat, 25, 255, cv2.THRESH_BINARY)

# Find contours on blue mask
contours_blue, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"\nBlue mask contours: {len(contours_blue)}")
for i, c in enumerate(contours_blue):
    x, y, cw, ch = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    if area > 5:
        print(f"  Blue #{i}: x={x}, y={y}, w={cw}, h={ch}, area={area}")

# Find contours on saturation mask (any colored mark)
contours_sat, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"\nSaturation mask contours (area>20): {len([c for c in contours_sat if cv2.contourArea(c)>20])}")
for i, c in enumerate(contours_sat):
    x, y, cw, ch = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    if area > 20:
        ratio = ch / max(cw, 1)
        print(f"  Sat #{i}: x={x}, y={y}, w={cw}, h={ch}, area={area}, h/w={ratio:.1f}")

# Draw annotations on frame copy
annotated = frame.copy()

# Draw all blue contours
cv2.drawContours(annotated, [c for c in contours_blue if cv2.contourArea(c) > 5], -1, (255, 0, 0), 2)

# Draw all saturated contours
for c in contours_sat:
    if cv2.contourArea(c) > 20:
        x, y, cw, ch = cv2.boundingRect(c)
        cv2.rectangle(annotated, (x, y), (x+cw, y+ch), (0, 255, 0), 1)
        cv2.putText(annotated, f"{x},{y}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

cv2.imwrite("d:/project/video_check/.tmp/debug/blue_markers.png", annotated)
cv2.imwrite("d:/project/video_check/.tmp/debug/blue_mask.png", blue_mask)
cv2.imwrite("d:/project/video_check/.tmp/debug/sat_mask.png", sat_mask)
print("\nSaved: blue_markers.png, blue_mask.png, sat_mask.png")
