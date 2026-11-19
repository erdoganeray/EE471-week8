"""MediaPipe Interactive Segmenter: segment the object under a user stroke/click.

The "user click" is simulated with a positive stroke at normalized (x, y).
Default: the man's torso in pose-3.jpg.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so we can import common.py

import cv2
import numpy as np
import mediapipe as mp
from common import ASSETS, base_options, load_mp_image, to_rgb, save_rgb

# --x / --y = where the "click" is, as a fraction of the image (0..1)
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--input", default=str(ASSETS / "images" / "pose-3.jpg"))
ap.add_argument("--x", type=float, default=0.5, help="normalized x of the click (0..1)")
ap.add_argument("--y", type=float, default=0.4, help="normalized y of the click (0..1)")
args = ap.parse_args()

# Step 1: choose the model
v = mp.tasks.vision
opts = v.InteractiveSegmenterOptions(base_options=base_options("interactive_segmentation.task"))

# Step 2: describe the click as a "positive" stroke (= "select the object here")
image = load_mp_image(args.input)
stroke = v.InteractiveSegmenterStroke(
    brush_mode=v.InteractiveSegmenterBrushMode.POSITIVE,
    points=[v.InteractiveSegmenterStrokePoint(args.x, args.y)],
    is_completed=True)

# Step 3: this task has no running mode: set the image once, then segment with strokes
with v.InteractiveSegmenter.create_from_options(opts) as seg:
    seg.set_image(image)
    mask_img = seg.segment([stroke])

# Step 4: mask values are 0..1; above 0.5 = selected object
mask = np.asarray(mask_img.numpy_view()).squeeze()
print(f"mask {mask.shape}, min={mask.min():.2f} max={mask.max():.2f}")
sel = mask > 0.5
print(f"selected pixels: {sel.mean() * 100:.1f}%")

# Step 5: paint the selected area red (50% blend) and mark the click in green
rgb = to_rgb(image)
overlay = rgb.copy()
overlay[sel] = (255, 0, 0)
out = cv2.addWeighted(rgb, 0.5, overlay, 0.5, 0)
h, w = rgb.shape[:2]
cv2.circle(out, (int(args.x * w), int(args.y * h)), 5, (0, 255, 0), -1)
save_rgb(out, "interactive_segmenter.png")  # saved to outputs/
