import cv2
import numpy as np
import os

os.makedirs("d:/project/video_check/.tmp/debug", exist_ok=True)

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Total frames: {total}, FPS: {fps}")

def detect_marker(gray):
    """Detect vertical white slit marker on center strip.
    Returns list of (x, y, w, h) for each marker found."""
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
            if 180 < x < 240:  # center strip
                candidates.append((x, y, cw, ch))

    # Group nearby candidates into single markers
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
        markers.append((min_x, min_y, max_x - min_x, max_y - min_y))

    return markers

# Scan every frame and track MARKER 1 (upper region, y < 350)
# When marker1 disappears and reappears = new label
UPPER_Y_MAX = 350  # marker1 is in upper half of frame

marker_events = []  # (frame_no, marker_y)
prev_had_marker = False
cooldown = 0

for frame_no in range(total):
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.rotate(frame, cv2.ROTATE_180)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    markers = detect_marker(gray)

    # Check for upper marker (MARKER 1 zone)
    upper_markers = [m for m in markers if m[1] < UPPER_Y_MAX]

    if cooldown > 0:
        cooldown -= 1
        continue

    if upper_markers and not prev_had_marker:
        # New marker appeared in upper zone
        m = upper_markers[0]
        marker_events.append((frame_no, m[1]))
        cooldown = 10  # skip a few frames to avoid re-detecting same marker
        prev_had_marker = True
    elif not upper_markers:
        prev_had_marker = False

cap.release()

print(f"\nMarker events (upper zone): {len(marker_events)}")
print(f"{'Frame':>6s}  {'Y pos':>5s}  {'Gap':>5s}")
for i, (fn, y) in enumerate(marker_events):
    gap = fn - marker_events[i-1][0] if i > 0 else 0
    print(f"{fn:6d}  {y:5d}  {gap:5d}")

# Save annotated frames for first few marker events
cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")
for i, (fn, y) in enumerate(marker_events[:6]):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    markers = detect_marker(gray)

    output = frame.copy()
    for mx, my, mw, mh in markers:
        cx, cy = mx + mw // 2, my + mh // 2
        color = (0, 0, 255) if my < UPPER_Y_MAX else (0, 180, 0)
        cv2.circle(output, (cx, cy), 30, color, 3)

    gap = fn - marker_events[i-1][0] if i > 0 else 0
    cv2.putText(output, f"Frame {fn} | Event {i+1} | Gap={gap}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imwrite(f"d:/project/video_check/.tmp/debug/event_{i+1}_f{fn}.png", output)

cap.release()
print("\nSaved event frames to .tmp/debug/event_*.png")
