"""Find first frame with 4 clear markers visible, capture upper + lower label."""
import cv2
import numpy as np

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

def detect_markers(gray):
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
    return marks

def cluster_y(marks, gap=40):
    if not marks:
        return []
    ys = sorted(m[1] + m[3]//2 for m in marks)
    rows = [[ys[0]]]
    for y in ys[1:]:
        if y - rows[-1][-1] < gap:
            rows[-1].append(y)
        else:
            rows.append([y])
    return [int(np.mean(r)) for r in rows]

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
for fno in range(total):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    marks = detect_markers(gray)
    y_rows = cluster_y(marks)

    # Need 3 Y-rows with ~200px spacing = 4 corner markers visible
    if len(y_rows) >= 3:
        spacings = [y_rows[i+1] - y_rows[i] for i in range(len(y_rows)-1)]
        # Both spacings should be ~200px (label height)
        if all(160 < s < 250 for s in spacings[:2]):
            print(f"Frame {fno}: {len(marks)} marks, rows={y_rows}, spacing={spacings}")

            # Crop upper label: row[0] to row[1]
            y_top = y_rows[0]
            y_mid = y_rows[1]
            y_bot = y_rows[2]
            W = frame.shape[1]

            upper = frame[y_top:y_mid, 0:W]
            lower = frame[y_mid:y_bot, 0:W]

            cv2.imwrite("d:/project/video_check/.tmp/upper_label.png", upper)
            cv2.imwrite("d:/project/video_check/.tmp/lower_label.png", lower)

            # Also save annotated frame for reference
            ann = frame.copy()
            cv2.rectangle(ann, (0, y_top), (W, y_mid), (0, 0, 255), 3)   # RED upper
            cv2.rectangle(ann, (0, y_mid), (W, y_bot), (0, 255, 0), 3)   # GREEN lower
            for x, y, cw, ch, _ in marks:
                cv2.rectangle(ann, (x, y), (x+cw, y+ch), (255, 150, 0), 2)  # markers
            cv2.imwrite("d:/project/video_check/.tmp/annotated_frame.png", ann)

            print(f"Saved: upper_label.png ({upper.shape[1]}x{upper.shape[0]})")
            print(f"Saved: lower_label.png ({lower.shape[1]}x{lower.shape[0]})")
            print(f"Saved: annotated_frame.png")
            break

cap.release()
