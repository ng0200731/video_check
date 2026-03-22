import cv2
import numpy as np
import os
from scipy.signal import find_peaks

os.makedirs("d:/project/video_check/.tmp/labels_v2", exist_ok=True)

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, f0 = cap.read()
f0 = cv2.rotate(f0, cv2.ROTATE_180)
FRAME_H, FRAME_W = f0.shape[:2]
print(f"Total frames: {total}, FPS: {fps}, Size: {FRAME_W}x{FRAME_H}")

# --- Marker detection (same proven algorithm) ---
def detect_markers(gray):
    blur = cv2.GaussianBlur(gray, (51, 51), 0)
    local_bright = cv2.subtract(gray, blur)
    _, local_mask = cv2.threshold(local_bright, 12, 255, cv2.THRESH_BINARY)
    _, abs_mask = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY)
    combined = cv2.bitwise_and(local_mask, abs_mask)
    vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 8))
    vert = cv2.morphologyEx(combined, cv2.MORPH_OPEN, vkernel)
    contours, _ = cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if cw < 15 and 8 < ch < 60 and ch / max(cw, 1) > 2.5 and area > 10:
            if 180 < x < 240:
                candidates.append((x, y, cw, ch))

    candidates.sort(key=lambda m: m[1])
    markers = []
    used = [False] * len(candidates)
    for i in range(len(candidates)):
        if used[i]:
            continue
        group = [candidates[i]]
        used[i] = True
        for j in range(i + 1, len(candidates)):
            if used[j]:
                continue
            if abs(candidates[j][1] - candidates[i][1]) < 50 and abs(candidates[j][0] - candidates[i][0]) < 15:
                group.append(candidates[j])
                used[j] = True
        min_x = min(g[0] for g in group)
        min_y = min(g[1] for g in group)
        max_x = max(g[0] + g[2] for g in group)
        max_y = max(g[1] + g[3] for g in group)
        cx = min_x + (max_x - min_x) // 2
        cy = min_y + (max_y - min_y) // 2
        markers.append((cx, cy, min_x, min_y, max_x - min_x, max_y - min_y))
    return markers


# --- Seam line detection ---
def detect_seam_lines(gray, x_start=30, x_end=170):
    """Find horizontal seam lines (dark bands where 'Calvin Klein' text is).
    Returns sorted list of y positions."""
    strip = gray[:, x_start:x_end]
    row_means = np.mean(strip, axis=1)
    # Seam lines are darker than surrounding label body
    inv = 255 - row_means
    peaks, _ = find_peaks(inv, height=20, distance=100, prominence=8)
    return sorted(peaks.tolist())


# --- Strategy ---
# 1. Scan all frames, track marker state (VISIBLE / GONE)
# 2. When markers disappear = label is fully visible
# 3. At that moment, detect seam lines to find label boundaries
# 4. Crop between seam lines

# The marker appears when the seam (fold line) crosses the center strip.
# The marker IS on the seam line. So:
# - Marker visible = seam is at the marker's Y position
# - Marker gone = seam has scrolled out, labels fill the frame
# - Capture when marker is gone (label is unobstructed by marker)

# Track upper marker (y < 350)
UPPER_ZONE = (150, 350)
LOWER_ZONE = (350, 550)

class MarkerTracker:
    def __init__(self, name, y_zone):
        self.name = name
        self.y_min, self.y_max = y_zone
        self.visible = False
        self.gone_count = 0
        self.GONE_THRESHOLD = 8
        self.last_marker_y = None
        self.events = []  # (frame_no, event_type, marker_y)

    def update(self, frame_no, markers):
        in_zone = [m for m in markers if self.y_min <= m[1] <= self.y_max]
        has_marker = len(in_zone) > 0

        if has_marker:
            self.last_marker_y = in_zone[0][1]
            if not self.visible:
                self.events.append((frame_no, "appear", self.last_marker_y))
            self.visible = True
            self.gone_count = 0
        else:
            if self.visible:
                self.gone_count += 1
                if self.gone_count >= self.GONE_THRESHOLD:
                    self.events.append((frame_no - self.GONE_THRESHOLD, "gone", self.last_marker_y))
                    self.visible = False

upper = MarkerTracker("upper", UPPER_ZONE)
lower = MarkerTracker("lower", LOWER_ZONE)

# Pass 1: track marker events
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_no = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    markers = detect_markers(gray)
    upper.update(frame_no, markers)
    lower.update(frame_no, markers)
    frame_no += 1

print(f"Upper events: {len(upper.events)}")
print(f"Lower events: {len(lower.events)}")

# Build capture windows: between consecutive "gone" and "appear" events
# for the UPPER tracker, a label U is visible from "gone" to next "appear"
def build_captures(tracker, label_suffix):
    captures = []
    label_num = 0
    for i in range(len(tracker.events) - 1):
        if tracker.events[i][1] == "gone" and tracker.events[i+1][1] == "appear":
            gone_frame = tracker.events[i][0]
            appear_frame = tracker.events[i+1][0]
            mid_frame = (gone_frame + appear_frame) // 2
            duration = appear_frame - gone_frame
            if duration > 3:  # skip tiny gaps (noise)
                label_num += 1
                captures.append((f"label_{label_num:03d}{label_suffix}", mid_frame, gone_frame, appear_frame))
    return captures

upper_captures = build_captures(upper, "U")
lower_captures = build_captures(lower, "B")

all_captures = upper_captures + lower_captures
all_captures.sort(key=lambda x: x[2])  # sort by start frame

print(f"\nUpper labels: {len(upper_captures)}")
print(f"Lower labels: {len(lower_captures)}")
print(f"Total: {len(all_captures)}")

# Show first 10
print(f"\n{'Name':>12s}  {'Mid':>5s}  {'Start':>5s}  {'End':>5s}  {'Dur':>4s}")
for name, mid, start, end in all_captures[:15]:
    print(f"{name:>12s}  {mid:5d}  {start:5d}  {end:5d}  {end-start:4d}")

# Pass 2: capture label images
# Column boundaries (carrier strip at x~188-236, confirmed via R-B color diff)
LEFT_COL = (0, 180)
RIGHT_COL = (245, 454)

# Expected label height ~195-210 px (distance between seams)
EXPECTED_LABEL_H = 205

cap2 = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")

saved = 0
skipped = 0
for name, mid_frame, start, end in all_captures:
    cap2.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    ret, frame = cap2.read()
    if not ret:
        continue
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect seam lines at this frame (try both columns, merge results)
    seams_l = detect_seam_lines(gray, x_start=30, x_end=170)
    seams_r = detect_seam_lines(gray, x_start=260, x_end=420)
    # Merge and deduplicate (within 30px = same seam)
    all_seams = sorted(set(seams_l + seams_r))
    seams = []
    for s in all_seams:
        if not seams or s - seams[-1] > 30:
            seams.append(s)

    if len(seams) < 2:
        skipped += 1
        continue

    # Build label regions from consecutive seam pairs
    regions = []
    for i in range(len(seams) - 1):
        h = seams[i+1] - seams[i]
        if 150 < h < 260:  # plausible label height (~205 ± 50)
            regions.append((seams[i], seams[i+1]))

    if not regions:
        skipped += 1
        continue

    # Pick region based on label type:
    # U label: upper part of frame (region center < 350)
    # B label: lower part of frame (region center > 300)
    if name.endswith("U"):
        candidates = [r for r in regions if (r[0]+r[1])/2 < 400]
    else:
        candidates = [r for r in regions if (r[0]+r[1])/2 > 250]

    if not candidates:
        candidates = regions  # fallback

    # Among candidates, pick the one closest to expected position
    target_y = 330 if name.endswith("U") else 530
    best = min(candidates, key=lambda r: abs((r[0] + r[1]) / 2 - target_y))
    y1, y2 = best

    # Crop left and right columns (no margin — include the Calvin Klein text)
    left_label = frame[y1:y2, LEFT_COL[0]:LEFT_COL[1]]
    right_label = frame[y1:y2, RIGHT_COL[0]:RIGHT_COL[1]]

    cv2.imwrite(f"d:/project/video_check/.tmp/labels_v2/{name}_L.png", left_label)
    cv2.imwrite(f"d:/project/video_check/.tmp/labels_v2/{name}_R.png", right_label)
    saved += 1

cap2.release()
print(f"\nSaved {saved} label pairs ({saved*2} images) to .tmp/labels_v2/")
print(f"Skipped {skipped} (could not detect seam lines)")
