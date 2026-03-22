"""Capture upper and lower label from frame 200 using confirmed seam boundaries."""
import cv2

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
ret, frame = cap.read()
frame = cv2.rotate(frame, cv2.ROTATE_180)
cap.release()

H, W = frame.shape[:2]

# Confirmed boundaries
upper = frame[128:335, 0:W]   # y=128 to y=335
lower = frame[335:538, 0:W]   # y=335 to y=538

cv2.imwrite("d:/project/video_check/.tmp/upper_label.png", upper)
cv2.imwrite("d:/project/video_check/.tmp/lower_label.png", lower)

# Annotated frame
ann = frame.copy()
cv2.rectangle(ann, (0, 128), (W, 335), (0, 0, 255), 3)   # RED upper
cv2.rectangle(ann, (0, 335), (W, 538), (0, 255, 0), 3)   # GREEN lower
cv2.imwrite("d:/project/video_check/.tmp/annotated_frame.png", ann)

print(f"upper_label.png: {upper.shape[1]}x{upper.shape[0]}")
print(f"lower_label.png: {lower.shape[1]}x{lower.shape[0]}")
print("Done.")
