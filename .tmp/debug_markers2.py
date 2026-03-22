"""Zoom into the marker areas to understand their visual signature."""
import cv2
import numpy as np

cap = cv2.VideoCapture("d:/project/video_check/WhatsApp Video 2026-03-21 at 18.24.08.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 4050)
ret, frame = cap.read()
frame = cv2.rotate(frame, cv2.ROTATE_180)
cap.release()

# From the user's annotation, blue markers are at left and right edges of each label row
# The seam lines (horizontal gaps between labels) are at roughly y=220 and y=430
# Markers sit ON the seam lines, at the left edge (~x=70) and right edge (~x=350)

# Let's crop generous regions around these 4 marker positions
marker_regions = [
    ("left_upper",  50, 200, 110, 260),   # left edge, upper seam
    ("right_upper", 330, 200, 390, 260),   # right edge, upper seam
    ("left_lower",  50, 410, 110, 470),    # left edge, lower seam
    ("right_lower", 330, 410, 390, 470),   # right edge, lower seam
]

for name, x1, y1, x2, y2 in marker_regions:
    crop = frame[y1:y2, x1:x2]
    # Scale up 4x for visibility
    big = cv2.resize(crop, (crop.shape[1]*4, crop.shape[0]*4), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(f"d:/project/video_check/.tmp/debug/marker_edge_{name}.png", big)

    # Also analyze pixel stats in this region
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    print(f"\n{name} (x={x1}-{x2}, y={y1}-{y2}):")
    print(f"  Gray: min={gray.min()}, max={gray.max()}, mean={gray.mean():.1f}, std={gray.std():.1f}")
    print(f"  Hue:  min={hsv[:,:,0].min()}, max={hsv[:,:,0].max()}, mean={hsv[:,:,0].mean():.1f}")
    print(f"  Sat:  min={hsv[:,:,1].min()}, max={hsv[:,:,1].max()}, mean={hsv[:,:,1].mean():.1f}")
    print(f"  Val:  min={hsv[:,:,2].min()}, max={hsv[:,:,2].max()}, mean={hsv[:,:,2].mean():.1f}")

# Also let's look at a wider horizontal strip at the seam lines to find where markers actually are
# Check row intensity profile at y=225 and y=435
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
for yy in [215, 220, 225, 230, 430, 435, 440, 445]:
    row = gray[yy, :]
    # Find dark spots (markers should be darker than the surrounding label)
    dark_spots = np.where(row < 160)[0]
    if len(dark_spots) > 0:
        # Group consecutive dark pixels
        groups = np.split(dark_spots, np.where(np.diff(dark_spots) > 3)[0] + 1)
        interesting = [(g[0], g[-1], len(g), row[g].mean()) for g in groups if len(g) > 2]
        if interesting:
            print(f"\ny={yy} dark spots: {interesting}")

# Let's also try: look at the frame with enhanced contrast on the seam areas
# The marker might be a slightly darker or lighter notch
for seam_y in [220, 435]:
    strip = frame[seam_y-15:seam_y+15, :, :]
    # Enhance
    lab = cv2.cvtColor(strip, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    big = cv2.resize(enhanced, (enhanced.shape[1]*2, enhanced.shape[0]*4), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(f"d:/project/video_check/.tmp/debug/seam_enhanced_{seam_y}.png", big)

print("\nDone. Check marker_edge_*.png and seam_enhanced_*.png")
