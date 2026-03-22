"""Analyze marker geometry: find frames with clear markers, measure Y-rows."""
import cv2
import numpy as np

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def detect_markers(gray):
    """Detect small vertical bars (registration marks)."""
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

def cluster_y(marks, gap=40):
    """Group markers into Y-rows (markers within gap px are same row)."""
    if not marks:
        return []
    ys = sorted(m[1] + m[3]//2 for m in marks)  # center y of each mark
    rows = [[ys[0]]]
    for y in ys[1:]:
        if y - rows[-1][-1] < gap:
            rows[-1].append(y)
        else:
            rows.append([y])
    return [int(np.mean(r)) for r in rows]

# Scan every frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_data = []  # (frame_no, n_marks, y_rows, marks)

for fno in range(total):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marks = detect_markers(gray)
    y_rows = cluster_y(marks)
    frame_data.append((fno, len(marks), y_rows, marks))
    if fno % 500 == 0:
        print(f"  scanned {fno}/{total}...")

cap.release()
print(f"Scanned {len(frame_data)} frames")

# Find frames with 3 marker rows (upper boundary, shared boundary, lower boundary)
# These are the frames where both labels are fully visible
good_frames = []
for fno, nm, yrows, marks in frame_data:
    if len(yrows) >= 3 and nm >= 6:
        # Check spacing between rows is reasonable (150-250px = label height)
        spacings = [yrows[i+1] - yrows[i] for i in range(len(yrows)-1)]
        if all(120 < s < 280 for s in spacings):
            good_frames.append((fno, nm, yrows, spacings, marks))

print(f"\nFrames with 3+ marker rows and good spacing: {len(good_frames)}")
print(f"\nFirst 30 good frames:")
for fno, nm, yrows, sp, _ in good_frames[:30]:
    print(f"  f{fno:5d}: {nm:2d} marks, rows={yrows}, spacing={sp}")

# Also show distribution of Y-row positions
if good_frames:
    all_row0 = [f[2][0] for f in good_frames]
    all_row1 = [f[2][1] for f in good_frames]
    all_row2 = [f[2][2] for f in good_frames]
    print(f"\nRow 0 (top):    min={min(all_row0)}, max={max(all_row0)}, median={int(np.median(all_row0))}")
    print(f"Row 1 (middle): min={min(all_row1)}, max={max(all_row1)}, median={int(np.median(all_row1))}")
    print(f"Row 2 (bottom): min={min(all_row2)}, max={max(all_row2)}, median={int(np.median(all_row2))}")

    # Show spacing stats
    all_sp = [s for f in good_frames for s in f[3]]
    print(f"\nSpacing: min={min(all_sp)}, max={max(all_sp)}, median={int(np.median(all_sp))}")
