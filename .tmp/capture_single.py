"""Find frame with 4 clearly visible markers, capture upper-left single label."""
import cv2
import numpy as np

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
H0, W0 = 832, 464  # known frame size after rotation

def detect_markers(gray):
    """Detect markers in center strip."""
    strip = gray[:, 180:240]
    blur = cv2.GaussianBlur(strip, (31, 31), 0)
    bright = cv2.subtract(strip, blur)
    _, mask = cv2.threshold(bright, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marks = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if ch > 15 and cw < 15 and area > 30:
            marks.append((x + 180, y, cw, ch, area))
    return marks

# Scan all frames, find the one with strongest 4 markers
best_frame = -1
best_score = 0

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
for fno in range(total):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marks = detect_markers(gray)

    if len(marks) >= 2:
        # Score = total area of markers (bigger = more clearly visible)
        total_area = sum(m[4] for m in marks)
        if total_area > best_score:
            best_score = total_area
            best_frame = fno

    if fno % 500 == 0:
        print(f"  scanned {fno}/{total}...")

print(f"\nBest frame: {best_frame} (marker area score: {best_score:.0f})")

# Now capture that frame
cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame)
ret, frame = cap.read()
frame = cv2.rotate(frame, cv2.ROTATE_180)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
marks = detect_markers(gray)

print(f"Frame {best_frame}: {len(marks)} markers")
for m in sorted(marks, key=lambda m: m[1]):
    print(f"  x={m[0]}, y={m[1]}, w={m[2]}, h={m[3]}, area={m[4]:.0f}")

# Find seams using brightness dips
row_avg = np.mean(gray, axis=1)
from scipy.signal import find_peaks
inverted = -row_avg
peaks, props = find_peaks(inverted, distance=30, prominence=1.5)
seams = [(p, row_avg[p], props['prominences'][list(peaks).index(p)]) for p in peaks if p < 600]
print(f"\nSeams:")
for y, avg, prom in seams:
    print(f"  y={y}, brightness={avg:.0f}, prominence={prom:.1f}")

# Upper label boundaries: find the two seams surrounding the first marker
marker_ys = sorted([m[1] + m[3]//2 for m in marks])
first_marker_y = marker_ys[0]
seam_ys = sorted([s[0] for s in seams])

# Find seam above and below first marker
seam_above = max([s for s in seam_ys if s < first_marker_y], default=0)
seam_below = min([s for s in seam_ys if s > first_marker_y], default=H0)

print(f"\nFirst marker center y={first_marker_y}")
print(f"Upper label: y={seam_above} to y={seam_below}")

# Center strip x position (where marker is)
strip_x = 205  # approximate center of strip

# Crop upper label, LEFT column only (x=0 to strip center)
upper_left = frame[seam_above:seam_below, 0:strip_x]
cv2.imwrite("d:/project/video_check/.tmp/upper_label.png", upper_left)
print(f"Saved upper_label.png: {upper_left.shape[1]}x{upper_left.shape[0]}")

# Also save annotated full frame
ann = frame.copy()
# Red box = upper left label
cv2.rectangle(ann, (0, seam_above), (strip_x, seam_below), (0, 0, 255), 3)
# Mark all markers in blue
for m in marks:
    cv2.rectangle(ann, (m[0], m[1]), (m[0]+m[2], m[1]+m[3]), (255, 0, 0), 2)
cv2.imwrite("d:/project/video_check/.tmp/annotated_frame.png", ann)
print("Saved annotated_frame.png")

cap.release()
