import cv2
import os

os.makedirs("d:/project/video_check/.tmp/samples", exist_ok=True)

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for i, frame_idx in enumerate([0, total//6, total//3, total//2, 2*total//3, 5*total//6]):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    path = f"d:/project/video_check/.tmp/samples/frame_{i:02d}_{frame_idx}.png"
    cv2.imwrite(path, frame)
    print(f"Saved: {path}")

cap.release()
print("Done.")
