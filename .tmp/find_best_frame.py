"""Quick scan: find first frame with clearly visible markers."""
import cv2
import numpy as np

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
ret, f0 = cap.read()
f0 = cv2.rotate(f0, cv2.ROTATE_180)
H, W = f0.shape[:2]
print(f"Video: {W}x{H}, {total} frames, {fps} fps")

# Scan first 100 frames, detect markers using proven algorithm
def detect_markers_wide(gray):
    blur = cv2.GaussianBlur(gray, (51, 51), 0)
    local_bright = cv2.subtract(gray, blur)
    _, local_mask = cv2.threshold(local_bright, 12, 255, cv2.THRESH_BINARY)
    _, abs_mask = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY)
    combined = cv2.bitwise_and(local_mask, abs_mask)
    vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 8))
    vert = cv2.morphologyEx(combined, cv2.MORPH_OPEN, vkernel)
    contours, _ = cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marks = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if cw < 20 and ch > 10 and ch / max(cw, 1) > 2.0 and area > 10:
            marks.append((x, y, cw, ch, area))
    return marks

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
best_frame = 0
best_count = 0
best_total_area = 0

for fno in range(min(200, total)):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marks = detect_markers_wide(gray)
    total_area = sum(m[4] for m in marks)
    # Want frame with most/largest markers visible
    if total_area > best_total_area:
        best_total_area = total_area
        best_count = len(marks)
        best_frame = fno
    if fno % 20 == 0:
        print(f"  f{fno:4d}: {len(marks)} marks, total_area={total_area:.0f}")

print(f"\nBest frame: {best_frame} with {best_count} marks, area={best_total_area:.0f}")

# Also check frames around the best
for fno in range(max(0, best_frame-5), best_frame+6):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marks = detect_markers_wide(gray)
    total_area = sum(m[4] for m in marks)
    print(f"  f{fno}: {len(marks)} marks, area={total_area:.0f}")
    for m in sorted(marks, key=lambda m: (m[1], m[0])):
        print(f"    x={m[0]:3d} y={m[1]:3d} w={m[2]:2d} h={m[3]:2d} area={m[4]:.0f}")

cap.release()
