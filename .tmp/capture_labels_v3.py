import cv2
import numpy as np
import os

os.makedirs("d:/project/video_check/.tmp/labels_v3", exist_ok=True)

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, f0 = cap.read()
f0 = cv2.rotate(f0, cv2.ROTATE_180)
FRAME_H, FRAME_W = f0.shape[:2]
print(f"Total frames: {total}, FPS: {fps}, Size: {FRAME_W}x{FRAME_H}")

# --- Marker detection (proven algorithm) ---
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
        cy = min_y + (max_y - min_y) // 2
        markers.append(cy)
    return sorted(markers)


# --- Marker tracking ---
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
        self.events = []

    def update(self, frame_no, marker_ys):
        in_zone = [y for y in marker_ys if self.y_min <= y <= self.y_max]
        has_marker = len(in_zone) > 0

        if has_marker:
            self.last_marker_y = in_zone[0]
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

# Pass 1: scan all frames
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_no = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marker_ys = detect_markers(gray)
    upper.update(frame_no, marker_ys)
    lower.update(frame_no, marker_ys)
    frame_no += 1

print(f"Upper events: {len(upper.events)}")
print(f"Lower events: {len(lower.events)}")


# Build captures from gone->appear pairs
def build_captures(tracker, suffix):
    caps = []
    num = 0
    for i in range(len(tracker.events) - 1):
        e1 = tracker.events[i]
        e2 = tracker.events[i + 1]
        if e1[1] == "gone" and e2[1] == "appear":
            dur = e2[0] - e1[0]
            if dur > 3:
                num += 1
                mid = (e1[0] + e2[0]) // 2
                caps.append((f"label_{num:03d}{suffix}", mid, e1[0], e2[0]))
    return caps


upper_caps = build_captures(upper, "U")
lower_caps = build_captures(lower, "B")
all_captures = sorted(upper_caps + lower_caps, key=lambda x: x[2])

print(f"\nUpper labels: {len(upper_caps)}, Lower labels: {len(lower_caps)}, Total: {len(all_captures)}")

# --- Pass 2: capture with FIXED crop coordinates ---
# From analysis: seam lines at y≈38, 236, 441, 594 (±15px)
# Label height = 205px
# Full label regions:
#   Region 1 (upper): y=38 to y=236 -> use y=35 to y=240 with padding
#   Region 2 (middle): y=236 to y=441 -> use y=235 to y=445
# When only 3 seams visible (no top seam), regions shift:
#   Region 1: y=233 to y=438
#   Region 2: y=438 to y=593

# Strategy: at capture frame, find the strongest seam line and anchor from there.
# The "Calvin Klein" text creates a dark band ~10px tall at the seam.
# We just need ONE reliable seam to anchor both labels.

# Column boundaries (carrier strip at x~185-235, add 15px safety margin)
LEFT_COL = (0, 170)
RIGHT_COL = (255, 454)
LABEL_H = 205  # fixed label height

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

    # Find the strongest dark row in the label area (seam with Calvin Klein text)
    # Use right column (cleaner text) for seam detection
    strip = gray[:, 260:420]
    row_means = np.mean(strip, axis=1)

    # Background brightness (label body area, no text)
    bg = np.mean(row_means[280:380])

    # Find seam lines: local minimum that is at least 8 units below background
    seam_threshold = bg - 8

    # Zone A: y=200-260 (middle seam, where Calvin Klein text appears)
    # Zone B: y=410-470 (lower seam)
    zone_a = row_means[200:260]
    zone_b = row_means[410:470]

    seam_a_y = 200 + np.argmin(zone_a) if np.min(zone_a) < seam_threshold else None
    seam_b_y = 410 + np.argmin(zone_b) if np.min(zone_b) < seam_threshold else None

    # Fallback: if neither found, try wider zones
    if seam_a_y is None and seam_b_y is None:
        # Try finding ANY strong dark row in y=100-550
        wide = row_means[100:550]
        wide_min_y = 100 + np.argmin(wide)
        if np.min(wide) < seam_threshold:
            if wide_min_y < 350:
                seam_a_y = wide_min_y
            else:
                seam_b_y = wide_min_y

    if name.endswith("U"):
        # Upper label: between seam_a - LABEL_H and seam_a
        if seam_a_y is not None:
            y1 = seam_a_y - LABEL_H
            y2 = seam_a_y
        elif seam_b_y is not None:
            y1 = seam_b_y - LABEL_H
            y2 = seam_b_y
        else:
            skipped += 1
            continue
    else:
        # Lower label: between seam_a and seam_a + LABEL_H (or seam_b)
        if seam_a_y is not None and seam_b_y is not None:
            y1 = seam_a_y
            y2 = seam_b_y
        elif seam_a_y is not None:
            y1 = seam_a_y
            y2 = seam_a_y + LABEL_H
        elif seam_b_y is not None:
            y1 = seam_b_y - LABEL_H
            y2 = seam_b_y
        else:
            skipped += 1
            continue

    # Clamp to frame bounds
    y1 = max(0, y1)
    y2 = min(FRAME_H, y2)

    if y2 - y1 < 100:
        skipped += 1
        continue

    # Crop
    left_label = frame[y1:y2, LEFT_COL[0]:LEFT_COL[1]]
    right_label = frame[y1:y2, RIGHT_COL[0]:RIGHT_COL[1]]

    cv2.imwrite(f"d:/project/video_check/.tmp/labels_v3/{name}_L.png", left_label)
    cv2.imwrite(f"d:/project/video_check/.tmp/labels_v3/{name}_R.png", right_label)
    saved += 1

cap2.release()
print(f"\nSaved {saved} label pairs ({saved * 2} images) to .tmp/labels_v3/")
print(f"Skipped {skipped} (no reliable seam found)")
