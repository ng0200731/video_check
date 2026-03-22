"""Capture upper-right single Calvin Klein label from frame 249."""
import cv2
import numpy as np

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 249)
ret, frame = cap.read()
frame = cv2.rotate(frame, cv2.ROTATE_180)
H, W = frame.shape[:2]
cap.release()

# Known from analysis:
# Seams: 38, 131, 235, 338, 440, 541, 590
# Marker1 center y=234, Marker2 center y=439
# Upper label = y=131 to y=338 (contains marker1)
# Right column = x=240 to x=464

y_top = 131
y_bot = 338
x_left = 240
x_right = W

upper_right = frame[y_top:y_bot, x_left:x_right]
cv2.imwrite("d:/project/video_check/.tmp/upper_label.png", upper_right)
print(f"upper_label.png: {upper_right.shape[1]}x{upper_right.shape[0]}")

# Annotated
ann = frame.copy()
cv2.rectangle(ann, (x_left, y_top), (x_right, y_bot), (0, 0, 255), 3)
for sy in [38, 131, 235, 338, 440, 541, 590]:
    cv2.line(ann, (0, sy), (W, sy), (0, 255, 255), 1)
# Markers
cv2.rectangle(ann, (242, 220), (260, 249), (255, 0, 0), 2)
cv2.rectangle(ann, (244, 425), (260, 453), (255, 0, 0), 2)
cv2.imwrite("d:/project/video_check/.tmp/annotated_frame.png", ann)
print("Done.")
