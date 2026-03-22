import cv2
import numpy as np
import os

os.makedirs("d:/project/video_check/.tmp/labels", exist_ok=True)

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Total frames: {total}, FPS: {fps}")

# Read first frame to get dimensions
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, f0 = cap.read()
f0 = cv2.rotate(f0, cv2.ROTATE_180)
FRAME_H, FRAME_W = f0.shape[:2]
print(f"Frame size: {FRAME_W}x{FRAME_H}")

# --- Marker detection ---
def detect_markers(gray):
    """Returns list of (cx, cy, x, y, w, h) for each vertical white slit marker on center strip."""
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
        w = max_x - min_x
        h = max_y - min_y
        cx = min_x + w // 2
        cy = min_y + h // 2
        markers.append((cx, cy, min_x, min_y, w, h))
    return markers

# --- Label capture regions ---
# Upper label (U): left half of frame, between the top seam and center
# Lower label (B): left half of frame, between the center and bottom seam
# The center strip is at x~205, so:
#   Left label column:  x = 0 to ~190
#   Right label column: x = ~220 to ~450
# For now capture the RIGHT column (it's more fully visible)
# Label region: between two consecutive marker Y positions

# Actually, from the frame layout:
# - Upper seam markers at y ~ 210-220
# - Lower seam markers at y ~ 418-437
# The label sits BETWEEN seams. So:
#   Label U = region from upper seam to lower seam (y ~ 220 to 418)
#   Label B = region from lower seam to next upper seam (wraps across frames)
# But since the roll scrolls, we capture at the moment the marker is detected.

# Strategy: state machine per seam
# IDLE -> marker detected -> TRACKING (save start frame) -> marker disappears ->
#   marker re-appears -> CAPTURE (label = frames between disappear and re-appear)

# Simpler approach: capture a snapshot of the label region at the midpoint
# between two consecutive marker appearances at the same seam.

# --- Scan all frames ---
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

UPPER_ZONE = (150, 300)  # y range for upper seam marker
LOWER_ZONE = (350, 500)  # y range for lower seam marker

class SeamTracker:
    def __init__(self, name, y_zone):
        self.name = name
        self.y_min, self.y_max = y_zone
        self.state = "WAITING"  # WAITING -> VISIBLE -> GONE -> WAITING
        self.marker_start_frame = None
        self.marker_gone_frame = None
        self.gone_count = 0
        self.labels = []  # (label_num, capture_frame, start_frame, end_frame)
        self.label_count = 0
        self.GONE_THRESHOLD = 5  # frames without marker = marker gone

    def update(self, frame_no, markers):
        """Update with detected markers for this frame. Returns capture_frame if label complete."""
        in_zone = [m for m in markers if self.y_min <= m[1] <= self.y_max]
        has_marker = len(in_zone) > 0

        if self.state == "WAITING":
            if has_marker:
                self.marker_start_frame = frame_no
                self.state = "VISIBLE"
                self.gone_count = 0
            return None

        elif self.state == "VISIBLE":
            if has_marker:
                self.gone_count = 0  # still visible
            else:
                self.gone_count += 1
                if self.gone_count >= self.GONE_THRESHOLD:
                    self.marker_gone_frame = frame_no - self.GONE_THRESHOLD
                    self.state = "GONE"
            return None

        elif self.state == "GONE":
            if has_marker:
                # Marker reappeared = new label boundary!
                self.label_count += 1
                # Capture frame = midpoint between gone and reappear
                capture_frame = (self.marker_gone_frame + frame_no) // 2
                self.labels.append((self.label_count, capture_frame,
                                    self.marker_start_frame, frame_no))
                # Reset for next label
                self.marker_start_frame = frame_no
                self.state = "VISIBLE"
                self.gone_count = 0
                return capture_frame
            return None

upper_tracker = SeamTracker("U", UPPER_ZONE)
lower_tracker = SeamTracker("B", LOWER_ZONE)

# First pass: find all label boundaries
frame_no = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    markers = detect_markers(gray)

    upper_tracker.update(frame_no, markers)
    lower_tracker.update(frame_no, markers)
    frame_no += 1

print(f"\nUpper seam (U) labels: {upper_tracker.label_count}")
print(f"Lower seam (B) labels: {lower_tracker.label_count}")

# Combine and sort all labels
all_labels = []
for lnum, cap_frame, start, end in upper_tracker.labels:
    all_labels.append((f"label_{lnum:03d}U", cap_frame, start, end))
for lnum, cap_frame, start, end in lower_tracker.labels:
    all_labels.append((f"label_{lnum:03d}B", cap_frame, start, end))
all_labels.sort(key=lambda x: x[2])  # sort by start frame

print(f"\nTotal labels: {len(all_labels)}")
print(f"{'Name':>12s}  {'Capture':>7s}  {'Start':>5s}  {'End':>5s}  {'Duration':>8s}")
for name, cap_f, start, end in all_labels[:20]:
    print(f"{name:>12s}  {cap_f:7d}  {start:5d}  {end:5d}  {end-start:8d}")

# Second pass: capture label images at the midpoint frames
cap2 = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")

# Determine crop regions for left and right label columns
# Center strip at x~205, width ~20
LEFT_COL = (0, 185)     # x range for left column
RIGHT_COL = (225, FRAME_W - 10)  # x range for right column

for name, cap_frame, start, end in all_labels:
    cap2.set(cv2.CAP_PROP_POS_FRAMES, cap_frame)
    ret, frame = cap2.read()
    if not ret:
        continue
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    # Determine Y crop based on seam type
    # From grid overlay: upper label content at y=30-210, lower at y=240-410
    # Calvin Klein text: upper at ~y=250, lower at ~y=460
    # Label U sits ABOVE the upper seam marker (y < 210)
    # Label B sits BETWEEN upper and lower seam markers (y=240 to y=410)
    if name.endswith("U"):
        y1, y2 = 30, 210
    else:
        y1, y2 = 240, 415

    # Crop left column label
    left_label = frame[y1:y2, LEFT_COL[0]:LEFT_COL[1]]
    cv2.imwrite(f"d:/project/video_check/.tmp/labels/{name}_L.png", left_label)

    # Crop right column label
    right_label = frame[y1:y2, RIGHT_COL[0]:RIGHT_COL[1]]
    cv2.imwrite(f"d:/project/video_check/.tmp/labels/{name}_R.png", right_label)

cap2.release()

print(f"\nSaved {len(all_labels) * 2} label crops to .tmp/labels/")
print("Format: label_NNNU_L.png (upper-left), label_NNNB_R.png (bottom-right), etc.")
