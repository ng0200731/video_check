"""Capture single Calvin Klein: from left marker to right edge."""
import cv2

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 249)
ret, frame = cap.read()
frame = cv2.rotate(frame, cv2.ROTATE_180)
H, W = frame.shape[:2]
cap.release()

# Upper label: y=131 to y=338
# Right-side single label: x=155 (left marker) to x=464 (right edge)
# Markers at x≈155 (left) and x≈430 (right)
y_top = 131
y_bot = 338
x_left = 140   # slightly before left marker to include it fully
x_right = W    # full right edge, includes right marker

single = frame[y_top:y_bot, x_left:x_right]
cv2.imwrite("d:/project/video_check/.tmp/upper_label.png", single)
print(f"upper_label.png: {single.shape[1]}x{single.shape[0]}")
