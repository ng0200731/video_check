import cv2
import numpy as np
import os

os.makedirs("d:/project/video_check/.tmp/debug", exist_ok=True)

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
ret, frame = cap.read()
frame = cv2.rotate(frame, cv2.ROTATE_180)
cv2.imwrite("d:/project/video_check/.tmp/debug/clean200.png", frame)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect bright vertical lines using morphological approach
# Kernel: tall and narrow to isolate vertical lines
vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]
vert_lines = cv2.morphologyEx(bright, cv2.MORPH_OPEN, vkernel)

cv2.imwrite("d:/project/video_check/.tmp/debug/vert_mask.png", vert_lines)

contours, _ = cv2.findContours(vert_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
debug = frame.copy()

print(f"Vertical bright line candidates: {len(contours)}")
for c in contours:
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    aspect = h / max(w, 1)
    if aspect > 3:  # taller than wide = vertical line
        cv2.rectangle(debug, (x, y), (x+w, y+h), (0, 0, 255), 2)
        print(f"  Vertical line at ({x},{y}) size {w}x{h} area={area:.0f}")

cv2.imwrite("d:/project/video_check/.tmp/debug/vlines_detected.png", debug)
print("Saved: vlines_detected.png and vert_mask.png")
cap.release()
