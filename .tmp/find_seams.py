"""Find ALL seams precisely using brightness dips."""
import cv2
import numpy as np

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
ret, frame = cap.read()
frame = cv2.rotate(frame, cv2.ROTATE_180)
H, W = frame.shape[:2]
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cap.release()

row_avg = np.mean(gray, axis=1)

# Find all local minima in brightness (seams are dark lines)
from scipy.signal import find_peaks
# Invert to find minima as peaks
inverted = -row_avg
peaks, props = find_peaks(inverted, distance=30, prominence=1.5)
print("Seam candidates (brightness dips):")
for p in peaks:
    if p < 600:
        print(f"  y={p:3d}, brightness={row_avg[p]:.1f}, prominence={props['prominences'][list(peaks).index(p)]:.1f}")

# Draw ALL seams on frame
ann = frame.copy()
for p in peaks:
    if p < 600:
        cv2.line(ann, (0, p), (W, p), (0, 255, 255), 1)
        cv2.putText(ann, f"y={p} (avg={row_avg[p]:.0f})", (5, p-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

# Also draw markers
cv2.rectangle(ann, (203, 227), (207, 255), (0, 0, 255), 2)
cv2.putText(ann, "marker1", (210, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
cv2.rectangle(ann, (206, 431), (210, 458), (0, 0, 255), 2)
cv2.putText(ann, "marker2", (213, 448), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)

cv2.imwrite("d:/project/video_check/.tmp/all_seams.png", ann)
print("\nSaved all_seams.png")
