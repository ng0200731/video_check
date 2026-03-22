"""Map exact coordinates: markers, text, seams on frame 200."""
import cv2
import numpy as np

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
ret, frame = cap.read()
frame = cv2.rotate(frame, cv2.ROTATE_180)
H, W = frame.shape[:2]
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cap.release()

# Find horizontal seams (thin dark/bright lines running across the frame)
# Seams appear as horizontal brightness transitions
# Compute horizontal gradient (row-by-row average brightness)
row_avg = np.mean(gray, axis=1)

# Find seams by looking for local minima in brightness (dark lines)
# or sharp transitions
row_diff = np.abs(np.diff(row_avg.astype(float)))

# Print column-averaged brightness for key Y ranges
print(f"Frame: {W}x{H}")
print(f"\n=== Row-average brightness (y=0..600) ===")
for y in range(0, min(600, H), 5):
    bar = "#" * int(row_avg[y] / 5)
    marker = ""
    if row_diff[y] > 3:
        marker = " <-- transition"
    print(f"y={y:3d}: avg={row_avg[y]:5.1f} {bar}{marker}")

# Detect the center-strip markers precisely
# They are at x≈203-206 in the center strip
print(f"\n=== Center strip marker detection ===")
# Focus on center strip x=180..240
strip = gray[:, 180:240]
strip_blur = cv2.GaussianBlur(strip, (31, 31), 0)
strip_bright = cv2.subtract(strip, strip_blur)
_, strip_mask = cv2.threshold(strip_bright, 15, 255, cv2.THRESH_BINARY)

# Find contours in strip
contours, _ = cv2.findContours(strip_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    x, y, cw, ch = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    if ch > 15 and cw < 15:
        real_x = x + 180
        print(f"  MARKER: x={real_x}, y={y}, w={cw}, h={ch}, area={area}")

# Also detect horizontal seams more precisely
# Look for rows where there's a sharp horizontal line
print(f"\n=== Horizontal seam detection ===")
# Use Canny edge detection, look for horizontal edges
edges = cv2.Canny(gray, 30, 80)
hkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
horiz_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, hkernel)

# Find Y positions of horizontal lines
row_edge_sum = np.sum(horiz_edges, axis=1)
seam_ys = []
for y in range(H):
    if row_edge_sum[y] > 1000:  # strong horizontal edge
        if not seam_ys or y - seam_ys[-1] > 20:
            seam_ys.append(y)
        elif row_edge_sum[y] > row_edge_sum[seam_ys[-1]]:
            seam_ys[-1] = y

print(f"Seam Y positions: {seam_ys}")
for s in seam_ys:
    print(f"  y={s}, edge_strength={row_edge_sum[s]}")

# Save debug image with seams and markers annotated with coordinates
ann = frame.copy()
for s in seam_ys:
    cv2.line(ann, (0, s), (W, s), (0, 255, 255), 1)
    cv2.putText(ann, f"seam y={s}", (5, s-3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

for c in contours:
    x, y, cw, ch = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    if ch > 15 and cw < 15:
        real_x = x + 180
        cv2.rectangle(ann, (real_x, y), (real_x+cw, y+ch), (0, 0, 255), 2)
        cv2.putText(ann, f"marker ({real_x},{y})-({real_x+cw},{y+ch})", (real_x+8, y+ch//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

cv2.imwrite("d:/project/video_check/.tmp/coord_map.png", ann)
print("\nSaved coord_map.png")
