"""Capture upper label FULL WIDTH from frame 249 — both markers visible."""
import cv2

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 249)
ret, frame = cap.read()
frame = cv2.rotate(frame, cv2.ROTATE_180)
H, W = frame.shape[:2]
cap.release()

# Seams from analysis: 38, 131, 235, 338, 440, 541
# Upper label = y=131 to y=338, full width
y_top = 131
y_bot = 338

upper = frame[y_top:y_bot, 0:W]
cv2.imwrite("d:/project/video_check/.tmp/upper_label.png", upper)
print(f"upper_label.png: {upper.shape[1]}x{upper.shape[0]}")
