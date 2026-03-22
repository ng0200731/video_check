"""Save annotated sample frames with red/green rectangles to verify geometry."""
import cv2
import numpy as np

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")

# Pick a few good frames from each type
test_frames = [200, 667, 920, 1057, 1895]  # Type A: rows near [~40, ~235, ~440]

for fno in test_frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    blur = cv2.GaussianBlur(gray, (51, 51), 0)
    local_bright = cv2.subtract(gray, blur)
    _, local_mask = cv2.threshold(local_bright, 12, 255, cv2.THRESH_BINARY)
    _, abs_mask = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY)
    combined = cv2.bitwise_and(local_mask, abs_mask)
    vkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 8))
    vert = cv2.morphologyEx(combined, cv2.MORPH_OPEN, vkernel)
    contours, _ = cv2.findContours(vert, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marks = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if cw < 20 and ch > 10 and ch / max(cw, 1) > 2.0 and area > 10:
            marks.append((x, y, cw, ch, area))

    # Cluster Y rows
    if not marks:
        continue
    ys = sorted(m[1] + m[3]//2 for m in marks)
    rows_y = [[ys[0]]]
    for y in ys[1:]:
        if y - rows_y[-1][-1] < 40:
            rows_y[-1].append(y)
        else:
            rows_y.append([y])
    row_centers = [int(np.mean(r)) for r in rows_y]

    ann = frame.copy()

    # Draw markers as blue rectangles
    for x, y, cw, ch, _ in marks:
        cv2.rectangle(ann, (x, y), (x+cw, y+ch), (255, 150, 0), 2)

    # Draw row lines
    for ry in row_centers:
        cv2.line(ann, (0, ry), (ann.shape[1], ry), (255, 255, 0), 1)
        cv2.putText(ann, f"y={ry}", (5, ry-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # If 3 rows: draw red rectangle (upper label) and green rectangle (lower label)
    if len(row_centers) >= 3:
        x_margin = 5
        # Upper label: from row[0] to row[1], full width
        cv2.rectangle(ann, (x_margin, row_centers[0] - 10), (ann.shape[1] - x_margin, row_centers[1] + 10),
                      (0, 0, 255), 3)  # RED
        cv2.putText(ann, "UPPER (RED)", (10, row_centers[0] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Lower label: from row[1] to row[2], full width
        cv2.rectangle(ann, (x_margin, row_centers[1] - 10), (ann.shape[1] - x_margin, row_centers[2] + 10),
                      (0, 255, 0), 3)  # GREEN
        cv2.putText(ann, "LOWER (GREEN)", (10, row_centers[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(f"d:/project/video_check/.tmp/debug/capture_preview_f{fno}.png", ann)
    print(f"f{fno}: rows={row_centers}, saved preview")

cap.release()
print("Done. Check .tmp/debug/capture_preview_f*.png")
