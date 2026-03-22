"""Capture upper and lower single Calvin Klein labels from first frame with 4 clear markers.
Output:
  label_1_U.png  — upper label (single CK, with both markers)
  label_2_L.png  — lower label (single CK, with both markers)
  frame_capture.png — the full frame used for capture
"""
import cv2
import numpy as np
from scipy.signal import find_peaks

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def detect_markers(gray):
    """Detect tall vertical bars across full width."""
    blur = cv2.GaussianBlur(gray, (51, 51), 0)
    local_bright = cv2.subtract(gray, blur)
    _, mask = cv2.threshold(local_bright, 10, 255, cv2.THRESH_BINARY)
    _, abs_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    combined = cv2.bitwise_and(mask, abs_mask)
    vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 8))
    vert = cv2.morphologyEx(combined, cv2.MORPH_OPEN, vkernel)
    contours, _ = cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marks = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        # Real markers: tall (h>25), narrow (w<10), high aspect ratio
        if ch > 25 and cw < 10 and ch / max(cw, 1) > 5.0 and area > 40:
            marks.append((x, y, cw, ch, area))
    return marks

def cluster_y(marks, gap=50):
    """Group markers into Y-rows."""
    if not marks:
        return []
    centers = sorted(m[1] + m[3]//2 for m in marks)
    rows = [[centers[0]]]
    for y in centers[1:]:
        if y - rows[-1][-1] < gap:
            rows[-1].append(y)
        else:
            rows.append([y])
    return [int(np.mean(r)) for r in rows]

def find_seams(gray):
    """Find horizontal seam lines via brightness dips."""
    row_avg = np.mean(gray, axis=1)
    inverted = -row_avg
    peaks, props = find_peaks(inverted, distance=30, prominence=1.5)
    return sorted([int(p) for p in peaks if p < 600])

# Scan frame by frame for first frame with 4 clear markers (2 rows × 2 markers each)
print("Scanning for first frame with 4 clear markers...")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
found_frame = -1

for fno in range(total):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marks = detect_markers(gray)
    y_rows = cluster_y(marks)

    # Need exactly 2 Y-rows, each with 2 markers (left + right)
    if len(y_rows) >= 2 and len(marks) >= 4:
        # Verify: each row should have a left marker (x<200) and right marker (x>350)
        row_marks = {r: [] for r in y_rows}
        for m in marks:
            cy = m[1] + m[3]//2
            for r in y_rows:
                if abs(cy - r) < 50:
                    row_marks[r].append(m)
                    break

        rows_with_pair = 0
        for r in y_rows[:2]:
            xs = [m[0] for m in row_marks[r]]
            has_left = any(x < 200 for x in xs)
            has_right = any(x > 350 for x in xs)
            if has_left and has_right:
                rows_with_pair += 1

        if rows_with_pair >= 2:
            found_frame = fno
            print(f"Found frame {fno}: {len(marks)} markers in {len(y_rows)} rows")
            for m in sorted(marks, key=lambda m: (m[1], m[0])):
                print(f"  x={m[0]:3d}, y={m[1]:3d}, w={m[2]}, h={m[3]}, area={m[4]:.0f}")
            break

    if fno % 500 == 0:
        print(f"  scanned {fno}/{total}...")

if found_frame < 0:
    print("No frame with 4 clear markers found!")
    cap.release()
    exit()

# Re-read the found frame
cap.set(cv2.CAP_PROP_POS_FRAMES, found_frame)
ret, frame = cap.read()
frame = cv2.rotate(frame, cv2.ROTATE_180)
H, W = frame.shape[:2]
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Find seams
seams = find_seams(gray)
print(f"\nSeams: {seams}")

# Get marker rows
marks = detect_markers(gray)
y_rows = cluster_y(marks)
print(f"Marker rows (center Y): {y_rows}")

# For each marker row, find the seam above and seam below
# Then expand height by 30% each side
# Width: x=140 to x=464 (confirmed)
x_left = 140
x_right = W

outdir = "d:/project/video_check/.tmp/"

for i, row_y in enumerate(y_rows[:2]):
    # Find seam above and below this marker row
    seam_above = max([s for s in seams if s < row_y], default=0)
    seam_below = min([s for s in seams if s > row_y], default=H)

    label_h = seam_below - seam_above
    expand = int(label_h * 0.30)

    y_top = max(0, seam_above - expand)
    y_bot = min(H, seam_below + expand)

    suffix = "U" if i == 0 else "L"
    idx = i + 1
    fname = f"label_{idx}_{suffix}.png"

    crop = frame[y_top:y_bot, x_left:x_right]
    cv2.imwrite(outdir + fname, crop)
    print(f"\n{fname}: {crop.shape[1]}x{crop.shape[0]}")
    print(f"  Seams: y={seam_above} to y={seam_below} (label_h={label_h})")
    print(f"  +30%: y={y_top} to y={y_bot} (total_h={y_bot-y_top})")

# Save the full frame
cv2.imwrite(outdir + "frame_capture.png", frame)
print(f"\nframe_capture.png: {W}x{H}")

cap.release()
print("\nDone.")
