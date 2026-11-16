"""MediaPipe Interactive Segmenter: segment the object under a user stroke/click.

The "user click" is simulated with a positive stroke at normalized (x, y).
Default: the man's torso in pose-3.jpg.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import mediapipe as mp
from common import ASSETS, base_options, load_mp_image, to_rgb, save_rgb

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--input", default=str(ASSETS / "images" / "pose-3.jpg"))
ap.add_argument("--x", type=float, default=0.5, help="normalized x of the click (0..1)")
ap.add_argument("--y", type=float, default=0.4, help="normalized y of the click (0..1)")
args = ap.parse_args()

v = mp.tasks.vision
opts = v.InteractiveSegmenterOptions(base_options=base_options("interactive_segmentation.task"))
image = load_mp_image(args.input)
stroke = v.InteractiveSegmenterStroke(
    brush_mode=v.InteractiveSegmenterBrushMode.POSITIVE,
    points=[v.InteractiveSegmenterStrokePoint(args.x, args.y)],
    is_completed=True)

with v.InteractiveSegmenter.create_from_options(opts) as seg:
    seg.set_image(image)            # no running mode: set the image once ...
    mask_img = seg.segment([stroke])  # ... then segment with any number of strokes

mask = np.asarray(mask_img.numpy_view()).squeeze()
print(f"mask {mask.shape}, min={mask.min():.2f} max={mask.max():.2f}")
sel = mask > 0.5
print(f"selected pixels: {sel.mean() * 100:.1f}%")

rgb = to_rgb(image)
overlay = rgb.copy()
overlay[sel] = (255, 0, 0)
out = cv2.addWeighted(rgb, 0.5, overlay, 0.5, 0)
h, w = rgb.shape[:2]
cv2.circle(out, (int(args.x * w), int(args.y * h)), 5, (0, 255, 0), -1)
save_rgb(out, "interactive_segmenter.png")
