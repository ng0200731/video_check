"""Label inspection processor.

Detects individual labels from a scrolling label-roll video,
crops them, compares each to an approved reference image, and
classifies as approved / rejected.
"""

import cv2
import numpy as np
import os
import sqlite3
import threading
import time

# ---------------------------------------------------------------------------
# Tuning parameters
# ---------------------------------------------------------------------------
MATCH_THRESHOLD = 0.18        # SSIM >= this → approved
GONE_THRESHOLD = 8            # frames without marker → marker gone
LABEL_H = 205                 # expected label height in pixels
UPPER_ZONE = (150, 350)       # y range for upper seam marker
LOWER_ZONE = (350, 550)       # y range for lower seam marker

# ---------------------------------------------------------------------------
# Paths (defaults, overridden by run())
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, ".tmp", "results.db")
SCHEMA_PATH = os.path.join(BASE_DIR, "sql", "schema.sql")
FRAMES_DIR = os.path.join(BASE_DIR, ".tmp", "frames")
APPROVED_DIR = os.path.join(BASE_DIR, "approved_labels")
REJECTED_DIR = os.path.join(BASE_DIR, "rejected_labels")


# ---------------------------------------------------------------------------
# State exposed to the web UI
# ---------------------------------------------------------------------------
class ProcessorState:
    def __init__(self):
        self.status = "idle"          # idle | scanning | capturing | done | error
        self.total_frames = 0
        self.current_frame = 0
        self.labels_found = 0
        self.labels_approved = 0
        self.labels_rejected = 0
        self.error = ""
        self.lock = threading.Lock()

    def to_dict(self):
        with self.lock:
            return {
                "status": self.status,
                "total_frames": self.total_frames,
                "current_frame": self.current_frame,
                "labels_found": self.labels_found,
                "labels_approved": self.labels_approved,
                "labels_rejected": self.labels_rejected,
                "error": self.error,
            }


state = ProcessorState()


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())
    conn.execute("DELETE FROM labels")  # fresh run
    conn.commit()
    return conn


def insert_label(conn, name, frame_no, start, end, seam_type, similarity,
                 status, image_path, col2_path=None):
    conn.execute(
        """INSERT INTO labels
           (name, frame_no, start_frame, end_frame, seam_type, similarity,
            status, image_path, col2_path)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (name, frame_no, start, end, seam_type, similarity,
         status, image_path, col2_path),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Image processing helpers
# ---------------------------------------------------------------------------
def detect_markers(gray):
    """Detect vertical white slit markers on the carrier strip (x 180-240)."""
    blur = cv2.GaussianBlur(gray, (51, 51), 0)
    local_bright = cv2.subtract(gray, blur)
    _, local_mask = cv2.threshold(local_bright, 12, 255, cv2.THRESH_BINARY)
    _, abs_mask = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY)
    combined = cv2.bitwise_and(local_mask, abs_mask)
    vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 8))
    vert = cv2.morphologyEx(combined, cv2.MORPH_OPEN, vkernel)
    contours, _ = cv2.findContours(
        vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if cw < 15 and 8 < ch < 60 and ch / max(cw, 1) > 2.5 and area > 10:
            if 180 < x < 240:
                cy = y + ch // 2
                candidates.append(cy)
    return sorted(set(candidates))


def find_strips(frame):
    """Find carrier strips via R-B colour difference. Returns [(x1, x2), ...]."""
    h = frame.shape[0]
    rb_sum = np.zeros(frame.shape[1], dtype=np.float64)
    count = 0
    for y in range(50, min(h, 500), 50):
        row = frame[y, :, :]
        rb_sum += row[:, 2].astype(np.float64) - row[:, 0].astype(np.float64)
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
    """Return label columns sorted by width (widest first)."""
    edges = [0]
    for s, e in sorted(strips):
        edges.append(s)
        edges.append(e)
    edges.append(frame_w)

    cols = []
    margin = 10
    for i in range(0, len(edges) - 1, 2):
        x1 = edges[i] + margin
        x2 = edges[i + 1] - margin
        if x2 - x1 > 40:
            cols.append((x1, x2))
    cols.sort(key=lambda c: c[1] - c[0], reverse=True)
    return cols


def find_seam_y(row_means, y_start, y_end, bg):
    """Find darkest row in zone (seam with Calvin Klein text)."""
    zone = row_means[y_start:y_end]
    if len(zone) == 0:
        return None
    if bg - np.min(zone) > 5:
        return y_start + int(np.argmin(zone))
    return None


def crop_reference(ref_img):
    """Auto-crop reference photo to extract just the label region.

    The reference is a high-res photo of a label on a dark background.
    We threshold to find the bright label area and crop to its bounding box.
    """
    gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ref_img
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    x, y, w, h = cv2.boundingRect(contours[0])
    return ref_img[y:y+h, x:x+w]


def compute_similarity(label_img, ref_img):
    """Compare a captured label to the reference using SSIM on center crop.

    Uses center 60% of each image to avoid seam-text misalignment,
    then computes a simplified structural similarity index.
    """
    h_lab, w_lab = label_img.shape[:2]
    ref_resized = cv2.resize(ref_img, (w_lab, h_lab))

    # Center-crop both to 60% to avoid edge seam text differences
    frac = 0.6
    ch, cw = int(h_lab * frac), int(w_lab * frac)
    y1 = (h_lab - ch) // 2
    x1 = (w_lab - cw) // 2

    crop_lab = label_img[y1:y1+ch, x1:x1+cw]
    crop_ref = ref_resized[y1:y1+ch, x1:x1+cw]

    gray_lab = cv2.cvtColor(crop_lab, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray_ref = cv2.cvtColor(crop_ref, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # SSIM constants
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(gray_lab, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(gray_ref, (11, 11), 1.5)

    sigma1_sq = cv2.GaussianBlur(gray_lab ** 2, (11, 11), 1.5) - mu1 ** 2
    sigma2_sq = cv2.GaussianBlur(gray_ref ** 2, (11, 11), 1.5) - mu2 ** 2
    sigma12 = cv2.GaussianBlur(gray_lab * gray_ref, (11, 11), 1.5) - mu1 * mu2

    num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / den
    return float(np.mean(ssim_map))


# ---------------------------------------------------------------------------
# Marker tracker
# ---------------------------------------------------------------------------
class MarkerTracker:
    def __init__(self, y_zone):
        self.y_min, self.y_max = y_zone
        self.visible = False
        self.gone_count = 0
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
                if self.gone_count >= GONE_THRESHOLD:
                    self.events.append(
                        (frame_no - GONE_THRESHOLD, "gone", self.last_marker_y)
                    )
                    self.visible = False


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


# ---------------------------------------------------------------------------
# Main processing function (runs in a background thread)
# ---------------------------------------------------------------------------
def run(video_path, reference_path):
    """Process the video end-to-end. Updates global `state`."""
    try:
        _run_impl(video_path, reference_path)
    except Exception as exc:
        with state.lock:
            state.status = "error"
            state.error = str(exc)
        raise


def _run_impl(video_path, reference_path):
    with state.lock:
        state.status = "scanning"
        state.error = ""
        state.labels_found = 0
        state.labels_approved = 0
        state.labels_rejected = 0
        state.current_frame = 0

    # Load reference
    ref_img = cv2.imread(reference_path)
    if ref_img is None:
        raise FileNotFoundError(f"Reference image not found: {reference_path}")
    ref_img = crop_reference(ref_img)

    # Ensure output dirs
    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(APPROVED_DIR, exist_ok=True)
    os.makedirs(REJECTED_DIR, exist_ok=True)

    conn = init_db()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with state.lock:
        state.total_frames = total

    upper = MarkerTracker(UPPER_ZONE)
    lower = MarkerTracker(LOWER_ZONE)

    # --- Pass 1: scan markers ---
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

        if fn % 100 == 0:
            with state.lock:
                state.current_frame = fn

    upper_caps = build_captures(upper, "U")
    lower_caps = build_captures(lower, "B")
    all_captures = sorted(upper_caps + lower_caps, key=lambda x: x[2])

    with state.lock:
        state.status = "capturing"
        state.labels_found = len(all_captures)
        state.current_frame = 0

    # --- Pass 2: capture and compare ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for idx, (name, mid_frame, start, end) in enumerate(all_captures):
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fw = frame.shape[1]

        # Per-frame strip and column detection
        strips = find_strips(frame)
        columns = find_label_columns(strips, fw) if strips else []
        if not columns:
            continue

        col_x1, col_x2 = columns[0]

        # Seam Y detection
        strip_gray = gray[:, col_x1:col_x2]
        row_means = np.mean(strip_gray, axis=1)
        bg = np.mean(row_means[280:380])

        seam_a = find_seam_y(row_means, 200, 270, bg)
        if seam_a is None:
            seam_a = find_seam_y(row_means, 150, 300, bg)
        seam_b = find_seam_y(row_means, 400, 470, bg)
        if seam_b is None:
            seam_b = find_seam_y(row_means, 370, 520, bg)

        seam_type = name[-1]  # "U" or "B"

        if seam_type == "U":
            if seam_a is not None:
                y1, y2 = max(0, seam_a - LABEL_H), seam_a
            elif seam_b is not None:
                y1, y2 = max(0, seam_b - LABEL_H), seam_b
            else:
                continue
        else:
            if seam_a is not None and seam_b is not None:
                y1, y2 = seam_a, seam_b
            elif seam_a is not None:
                y1, y2 = seam_a, min(frame.shape[0], seam_a + LABEL_H)
            elif seam_b is not None:
                y1, y2 = max(0, seam_b - LABEL_H), seam_b
            else:
                continue

        if y2 - y1 < 100:
            continue

        label_img = frame[y1:y2, col_x1:col_x2]

        # Similarity
        sim = compute_similarity(label_img, ref_img)
        approved = sim >= MATCH_THRESHOLD
        status_str = "approved" if approved else "rejected"

        sim_pct = sim * 100
        fname = f"{name}_sim{sim_pct:.1f}_{status_str.upper()}.png"

        dest_dir = APPROVED_DIR if approved else REJECTED_DIR
        dest_path = os.path.join(dest_dir, fname)
        cv2.imwrite(dest_path, label_img)

        # Col2
        col2_path = None
        if len(columns) >= 2:
            c2x1, c2x2 = columns[1]
            if c2x2 - c2x1 > 60:
                col2_img = frame[y1:y2, c2x1:c2x2]
                col2_fname = f"{name}_col2.png"
                col2_path = os.path.join(FRAMES_DIR, col2_fname)
                cv2.imwrite(col2_path, col2_img)

        insert_label(
            conn, name, mid_frame, start, end, seam_type,
            round(sim_pct, 1), status_str, dest_path, col2_path,
        )

        with state.lock:
            state.current_frame = idx + 1
            if approved:
                state.labels_approved += 1
            else:
                state.labels_rejected += 1

    cap.release()
    conn.close()

    with state.lock:
        state.status = "done"
