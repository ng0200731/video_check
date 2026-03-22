import cv2
import numpy as np
import os

os.makedirs("d:/project/video_check/.tmp/debug", exist_ok=True)

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Sample 20 frames across the video and find ALL vertical white slit markers
# Focus on the center strip region (x ~ 190-220 based on previous findings)
print(f"Total frames: {total}")

for fi, frame_no in enumerate(range(0, total, max(1, total//20))):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Local brightness approach — find pixels brighter than neighborhood
    blur = cv2.GaussianBlur(gray, (51, 51), 0)
    local_bright = cv2.subtract(gray, blur)
    _, local_mask = cv2.threshold(local_bright, 12, 255, cv2.THRESH_BINARY)
    _, abs_mask = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY)
    combined = cv2.bitwise_and(local_mask, abs_mask)

    vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 8))
    vert = cv2.morphologyEx(combined, cv2.MORPH_OPEN, vkernel)

    contours, _ = cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    markers = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        # Filter: narrow, vertical, not text (exclude areas near known text zones)
        # Center strip is around x=200-210
        if cw < 10 and 12 < ch < 70 and ch/max(cw,1) > 3 and area > 10:
            # Only keep if near center strip (x between 180 and 230)
            if 180 < x < 230:
                markers.append((x, y, cw, ch, area))

    if len(markers) >= 2:
        markers.sort(key=lambda m: m[1])
        debug = frame.copy()
        desc = []
        for i, (x, y, cw, ch, area) in enumerate(markers):
            cv2.rectangle(debug, (x, y), (x+cw, y+ch), (0, 0, 255), 2)
            desc.append(f"({x},{y} {cw}x{ch})")
        cv2.imwrite(f"d:/project/video_check/.tmp/debug/scan_{frame_no:04d}.png", debug)
        print(f"Frame {frame_no:4d}: {len(markers)} markers: {', '.join(desc)}")

cap.release()
print("Done")
