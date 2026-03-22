"""Capture single Calvin Klein: width confirmed, height +30% each side."""
import cv2

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 249)
ret, frame = cap.read()
frame = cv2.rotate(frame, cv2.ROTATE_180)
H, W = frame.shape[:2]
cap.release()

# Confirmed width: x=140 to x=464
# Height: y=131 to y=338 (207px) + 30% each side = 62px each
y_top = 131 - 62   # 69
y_bot = 338 + 62   # 400
x_left = 140
x_right = W

single = frame[y_top:y_bot, x_left:x_right]
cv2.imwrite("d:/project/video_check/.tmp/upper_label.png", single)
print(f"upper_label.png: {single.shape[1]}x{single.shape[0]}")
print(f"Crop: x={x_left}..{x_right}, y={y_top}..{y_bot}")
