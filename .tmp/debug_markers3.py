"""Scan multiple frames to find where markers are most visible, try broader detection."""
import cv2
import numpy as np

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")

# The center strip markers (red in scan_4050) are already detected well.
# Let's look at what the user means by "blue markers" — scan a frame
# and look at the entire frame structure more carefully.

# Try several frames
for fno in [200, 270, 1080, 2000, 3000, 4050]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Look for the thin vertical marks using the existing proven algorithm
    # but with wider x-range to catch edge markers
    blur = cv2.GaussianBlur(gray, (51, 51), 0)
    local_bright = cv2.subtract(gray, blur)
    _, local_mask = cv2.threshold(local_bright, 12, 255, cv2.THRESH_BINARY)
    _, abs_mask = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY)
    combined = cv2.bitwise_and(local_mask, abs_mask)
    vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 8))
    vert = cv2.morphologyEx(combined, cv2.MORPH_OPEN, vkernel)

    contours, _ = cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotated = frame.copy()
    all_marks = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        # Wider filter: any thin vertical mark
        if cw < 20 and ch > 8 and ch / max(cw, 1) > 2.0 and area > 8:
            all_marks.append((x, y, cw, ch, area))
            cv2.rectangle(annotated, (x, y), (x+cw, y+ch), (0, 0, 255), 1)
            cv2.putText(annotated, f"{x},{y}", (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

    cv2.imwrite(f"d:/project/video_check/.tmp/debug/allmarks_f{fno}.png", annotated)

    # Print x positions to see distribution
    xs = sorted(set(m[0] for m in all_marks))
    print(f"Frame {fno}: {len(all_marks)} vertical marks at x positions: {xs}")
    for m in sorted(all_marks, key=lambda m: (m[0], m[1])):
        print(f"  x={m[0]:3d} y={m[1]:3d} w={m[2]:2d} h={m[3]:2d} area={m[4]:.0f}")

cap.release()
print("\nSaved allmarks_f*.png")
