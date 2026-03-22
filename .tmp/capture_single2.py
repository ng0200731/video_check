"""Capture upper-right single Calvin Klein label from frame with clearest markers.
Label = full seam-to-seam height, right column only (strip to right edge)."""
import cv2
import numpy as np
from scipy.signal import find_peaks

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def detect_markers(gray):
    strip = gray[:, 180:260]
    blur = cv2.GaussianBlur(strip, (31, 31), 0)
    bright = cv2.subtract(strip, blur)
    _, mask = cv2.threshold(bright, 12, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marks = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if ch > 10 and cw < 20 and ch / max(cw, 1) > 1.5 and area > 20:
            marks.append((x + 180, y, cw, ch, area))
    return marks

def find_seams(gray):
    row_avg = np.mean(gray, axis=1)
    inverted = -row_avg
    peaks, props = find_peaks(inverted, distance=30, prominence=1.5)
    return [(int(p), float(row_avg[p]), float(props['prominences'][i]))
            for i, p in enumerate(peaks) if p < 600]

# Use frame 200 (known good from earlier analysis)
# Check a range of frames to find one with 4 clearly visible markers
print("Scanning for frames with clearest markers...")
results = []
for fno in range(150, 300):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marks = detect_markers(gray)
    total_area = sum(m[4] for m in marks)
    if len(marks) >= 2:
        results.append((fno, len(marks), total_area, marks))

results.sort(key=lambda r: -r[2])
print(f"Top 5 frames:")
for fno, nm, ta, marks in results[:5]:
    print(f"  f{fno}: {nm} markers, total_area={ta:.0f}")
    for m in sorted(marks, key=lambda m: m[1]):
        print(f"    x={m[0]}, y={m[1]}, w={m[2]}, h={m[3]}, area={m[4]:.0f}")

# Use the best one
best_fno = results[0][0]
cap.set(cv2.CAP_PROP_POS_FRAMES, best_fno)
ret, frame = cap.read()
frame = cv2.rotate(frame, cv2.ROTATE_180)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
H, W = frame.shape[:2]
marks = detect_markers(gray)

# Find seams
seams = find_seams(gray)
seam_ys = sorted([s[0] for s in seams])
print(f"\nFrame {best_fno}, seams: {seam_ys}")

# Markers sorted by Y
marks_sorted = sorted(marks, key=lambda m: m[1])
marker1_cy = marks_sorted[0][1] + marks_sorted[0][3] // 2
print(f"First marker center y={marker1_cy}")

# Upper label = between the two faint seams that surround marker1
# Faint seams are spaced ~100px above and ~100px below the marker
# Find the seam pair that gives ~200px spacing and contains the marker
for i in range(len(seam_ys) - 1):
    s1, s2 = seam_ys[i], seam_ys[i+1]
    spacing = s2 - s1
    if 180 < spacing < 220 and s1 < marker1_cy < s2:
        y_top = s1
        y_bot = s2
        print(f"Upper label: y={y_top} to y={y_bot} (height={spacing})")
        break

# Right column: from center strip to right edge
# Strip center is ~220px, but we want just after the strip
x_left = 240  # right edge of strip
x_right = W

upper_right = frame[y_top:y_bot, x_left:x_right]
cv2.imwrite("d:/project/video_check/.tmp/upper_label.png", upper_right)
print(f"Saved upper_label.png: {upper_right.shape[1]}x{upper_right.shape[0]}")

# Also full-width upper for reference
upper_full = frame[y_top:y_bot, 0:W]
cv2.imwrite("d:/project/video_check/.tmp/upper_label_full.png", upper_full)

# Annotated
ann = frame.copy()
cv2.rectangle(ann, (x_left, y_top), (x_right, y_bot), (0, 0, 255), 3)
for m in marks:
    cv2.rectangle(ann, (m[0], m[1]), (m[0]+m[2], m[1]+m[3]), (255, 0, 0), 2)
for s in seam_ys:
    cv2.line(ann, (0, s), (W, s), (0, 255, 255), 1)
cv2.imwrite("d:/project/video_check/.tmp/annotated_frame.png", ann)
print("Saved annotated_frame.png")

cap.release()
