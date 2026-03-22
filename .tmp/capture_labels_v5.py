import cv2
import numpy as np
import os

os.makedirs("d:/project/video_check/.tmp/labels_v5", exist_ok=True)

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret, f0 = cap.read()
f0 = cv2.rotate(f0, cv2.ROTATE_180)
FRAME_H, FRAME_W = f0.shape[:2]
print(f"Total frames: {total}, FPS: {fps}, Size: {FRAME_W}x{FRAME_H}")

LABEL_H = 205


# --- Original marker detection (proven, fixed x-range 180-240) ---
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
                cy = y + ch // 2
                candidates.append(cy)
    return sorted(set(candidates))


# --- Carrier strip detection (per-frame) ---
def find_strips(frame):
    """Find carrier strips using R-B color difference. Returns list of (x1, x2)."""
    h = frame.shape[0]
    sample_ys = range(50, min(h, 500), 50)
    rb_sum = np.zeros(frame.shape[1], dtype=np.float64)
    count = 0
    for y in sample_ys:
        row = frame[y, :, :]
        rb = row[:, 2].astype(np.float64) - row[:, 0].astype(np.float64)
        rb_sum += rb
        count += 1
    rb_avg = rb_sum / max(count, 1)

    warm = np.where(rb_avg > 10)[0]
    if len(warm) == 0:
        return []

    strips = []
    start = warm[0]
    for j in range(1, len(warm)):
        if warm[j] - warm[j - 1] > 10:
            strips.append((int(start), int(warm[j - 1])))
            start = warm[j]
    strips.append((int(start), int(warm[-1])))

    return [(s, e) for s, e in strips if e - s >= 20]


def find_label_columns(strips, frame_w):
    """Find label columns (gaps between/outside strips)."""
    edges = [0]
    for s, e in sorted(strips):
        edges.append(s)
        edges.append(e)
    edges.append(frame_w)

    columns = []
    margin = 10
    for i in range(0, len(edges) - 1, 2):
        x1 = edges[i] + margin
        x2 = edges[i + 1] - margin
        if x2 - x1 > 40:
            columns.append((x1, x2))

    columns.sort(key=lambda c: c[1] - c[0], reverse=True)
    return columns


# --- Marker tracking (same as v3) ---
UPPER_ZONE = (150, 350)
LOWER_ZONE = (350, 550)


class MarkerTracker:
    def __init__(self, y_zone):
        self.y_min, self.y_max = y_zone
        self.visible = False
        self.gone_count = 0
        self.GONE_THRESHOLD = 8
        self.last_marker_y = None
        self.events = []

    def update(self, frame_no, marker_ys):
        in_zone = [y for y in marker_ys if self.y_min <= y <= self.y_max]
        has = len(in_zone) > 0
        if has:
            self.last_marker_y = in_zone[0]
            if not self.visible:
                self.events.append((frame_no, "appear", self.last_marker_y))
            self.visible = True
            self.gone_count = 0
        else:
            if self.visible:
                self.gone_count += 1
                if self.gone_count >= self.GONE_THRESHOLD:
                    self.events.append(
                        (frame_no - self.GONE_THRESHOLD, "gone", self.last_marker_y)
                    )
                    self.visible = False


upper = MarkerTracker(UPPER_ZONE)
lower = MarkerTracker(LOWER_ZONE)

# Pass 1: scan markers
print("Pass 1: scanning markers...")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
for fn in range(total):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marker_ys = detect_markers(gray)
    upper.update(fn, marker_ys)
    lower.update(fn, marker_ys)

print(f"  Upper events: {len(upper.events)}, Lower events: {len(lower.events)}")


def build_captures(tracker, suffix):
    caps = []
    num = 0
    for i in range(len(tracker.events) - 1):
        e1, e2 = tracker.events[i], tracker.events[i + 1]
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

print(f"  Upper: {len(upper_caps)}, Lower: {len(lower_caps)}, Total: {len(all_captures)}")


# Pass 2: capture labels with per-frame column detection
print("Pass 2: capturing labels...")
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

    # Find carrier strips and label columns for THIS frame
    strips = find_strips(frame)
    columns = find_label_columns(strips, FRAME_W) if strips else []

    if not columns:
        skipped += 1
        continue

    # Use the widest column (main visible label)
    col_x1, col_x2 = columns[0]

    # Find seam Y for cropping (dark row = Calvin Klein text)
    strip_gray = gray[:, col_x1:col_x2]
    row_means = np.mean(strip_gray, axis=1)
    bg = np.mean(row_means[280:380])
    threshold = bg - 5

    # Search in zones
    def find_seam(y_start, y_end):
        zone = row_means[y_start:y_end]
        if len(zone) == 0:
            return None
        if np.min(zone) < threshold:
            return y_start + int(np.argmin(zone))
        return None

    seam_a = find_seam(200, 270)
    seam_b = find_seam(400, 470)

    # Fallback: wider search
    if seam_a is None:
        seam_a = find_seam(150, 300)
    if seam_b is None:
        seam_b = find_seam(370, 520)

    if name.endswith("U"):
        if seam_a is not None:
            y1 = max(0, seam_a - LABEL_H)
            y2 = seam_a
        elif seam_b is not None:
            y1 = max(0, seam_b - LABEL_H)
            y2 = seam_b
        else:
            skipped += 1
            continue
    else:
        if seam_a is not None and seam_b is not None:
            y1 = seam_a
            y2 = seam_b
        elif seam_a is not None:
            y1 = seam_a
            y2 = min(FRAME_H, seam_a + LABEL_H)
        elif seam_b is not None:
            y1 = max(0, seam_b - LABEL_H)
            y2 = seam_b
        else:
            skipped += 1
            continue

    if y2 - y1 < 100:
        skipped += 1
        continue

    # Crop the main label column
    label_img = frame[y1:y2, col_x1:col_x2]
    cv2.imwrite(f"d:/project/video_check/.tmp/labels_v5/{name}.png", label_img)

    # Also capture second column if available
    if len(columns) >= 2:
        col2_x1, col2_x2 = columns[1]
        if col2_x2 - col2_x1 > 60:
            label_img2 = frame[y1:y2, col2_x1:col2_x2]
            cv2.imwrite(
                f"d:/project/video_check/.tmp/labels_v5/{name}_col2.png", label_img2
            )

    saved += 1

cap2.release()
print(f"\nSaved {saved} labels to .tmp/labels_v5/")
print(f"Skipped {skipped}")
