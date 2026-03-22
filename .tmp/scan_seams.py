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

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Save column brightness profile to understand frame structure
col_means = np.mean(gray, axis=0).astype(int)
# Print brightness profile at key x positions
for x in range(0, w, 20):
    print(f"  x={x:3d}: brightness={col_means[x]}")

# Save enlarged horizontal strips at the label seam lines
# From M5 at y=227 and M9 at y=418, seams are around y~220 and y~420
seam_ys = [220, 420]
for sy in seam_ys:
    strip = frame[sy-5:sy+5, :, :]
    strip_big = cv2.resize(strip, (w*3, 60), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f"d:/project/video_check/.tmp/debug/seam_y{sy}.png", strip_big)

    # Also save a wider band zoomed
    band = frame[sy-30:sy+30, :, :]
    band_big = cv2.resize(band, (w*2, 240), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f"d:/project/video_check/.tmp/debug/band_y{sy}.png", band_big)
    print(f"Seam at y={sy}: saved strip and band")
