"""Capture upper and lower labels — fix seam selection.
Label spans from faint_seam ABOVE marker to faint_seam BELOW marker (~200px).
The strong seam at the marker row is IN THE MIDDLE of the label, not a boundary.
"""
import cv2
import numpy as np
from scipy.signal import find_peaks

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")

def detect_markers(gray):
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
        if ch > 25 and cw < 10 and ch / max(cw, 1) > 5.0 and area > 40:
            marks.append((x, y, cw, ch, area))
    return marks

def cluster_y(marks, gap=50):
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
    row_avg = np.mean(gray, axis=1)
    inverted = -row_avg
    peaks, props = find_peaks(inverted, distance=30, prominence=1.5)
    return sorted([int(p) for p in peaks if p < 650])

# Use frame 249 (confirmed good earlier with clear markers)
cap.set(cv2.CAP_PROP_POS_FRAMES, 249)
ret, frame = cap.read()
frame = cv2.rotate(frame, cv2.ROTATE_180)
H, W = frame.shape[:2]
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cap.release()

marks = detect_markers(gray)
y_rows = cluster_y(marks)
seams = find_seams(gray)

print(f"Frame 249")
print(f"Markers: {[(m[0], m[1], m[3]) for m in sorted(marks, key=lambda m: (m[1], m[0]))]}")
print(f"Marker rows (center Y): {y_rows}")
print(f"Seams: {seams}")

# Seams: [38, 131, 235, 338, 440, 541, 590]
# Marker rows: ~235, ~440
# Pattern: seams alternate faint/strong
#   38 (faint) - 131 (faint) - 235 (STRONG, marker row) - 338 (faint) - 440 (STRONG, marker row) - 541 (faint)
#
# Each label spans faint_seam → STRONG_seam → faint_seam
# Upper label: y=131 → 235 → 338  (total 207px, marker at center)
# Lower label: y=338 → 440 → 541  (total 203px, marker at center)

# Confirmed from earlier: upper = y=131 to y=338, lower = y=338 to y=541
# These are the correct full-label boundaries

x_left = 140
x_right = W
outdir = "d:/project/video_check/.tmp/"

# Upper label: y=131 to y=338 + 30% expansion
label_h_u = 338 - 131  # 207
expand_u = int(label_h_u * 0.30)  # 62
y_top_u = max(0, 131 - expand_u)   # 69
y_bot_u = min(H, 338 + expand_u)   # 400

upper = frame[y_top_u:y_bot_u, x_left:x_right]
cv2.imwrite(outdir + "label_1_U.png", upper)
print(f"\nlabel_1_U.png: {upper.shape[1]}x{upper.shape[0]}")
print(f"  Label: y=131..338 (207px), +30%: y={y_top_u}..{y_bot_u} ({y_bot_u-y_top_u}px)")

# Lower label: y=338 to y=541 + 30% expansion
label_h_l = 541 - 338  # 203
expand_l = int(label_h_l * 0.30)  # 61
y_top_l = max(0, 338 - expand_l)   # 277
y_bot_l = min(H, 541 + expand_l)   # 602

lower = frame[y_top_l:y_bot_l, x_left:x_right]
cv2.imwrite(outdir + "label_2_L.png", lower)
print(f"\nlabel_2_L.png: {lower.shape[1]}x{lower.shape[0]}")
print(f"  Label: y=338..541 (203px), +30%: y={y_top_l}..{y_bot_l} ({y_bot_l-y_top_l}px)")

# Save full frame
cv2.imwrite(outdir + "frame_capture.png", frame)
print(f"\nframe_capture.png: {W}x{H}")
print("Done.")
