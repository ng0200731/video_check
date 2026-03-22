import cv2
import os

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")
print("Opened:", cap.isOpened())
print("Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Total frames:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
print("FPS:", cap.get(cv2.CAP_PROP_FPS))
print("Codec fourcc:", int(cap.get(cv2.CAP_PROP_FOURCC)))

# Try reading first frame without rotation
ret, frame = cap.read()
print("Read success:", ret)
if ret:
    print("Frame shape:", frame.shape)
    print("Frame min/max:", frame.min(), frame.max())
    cv2.imwrite("d:/project/video_check/.tmp/raw_frame0.png", frame)
    print("Saved raw_frame0.png")

# Try frame 100
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
ret, frame = cap.read()
print("Frame 100 read:", ret)
if ret:
    print("Frame 100 shape:", frame.shape)
    print("Frame 100 min/max:", frame.min(), frame.max())
    cv2.imwrite("d:/project/video_check/.tmp/raw_frame100.png", frame)
    print("Saved raw_frame100.png")

cap.release()
