"""MediaPipe Image Segmenter: person mask (selfie segmenter) -> blurred background."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import mediapipe as mp
from common import base_options, default_args, load_mp_image, to_rgb, save_rgb

args = default_args(__doc__, "pose-3.jpg")
opts = mp.tasks.vision.ImageSegmenterOptions(
    base_options=base_options("selfie_segmenter.tflite"),
    output_category_mask=True, output_confidence_masks=False)
image = load_mp_image(args.input)
with mp.tasks.vision.ImageSegmenter.create_from_options(opts) as seg:
    result = seg.segment(image)

mask = result.category_mask.numpy_view().squeeze() == 0  # model has one label ("selfie") = category 0 = person
print(f"mask {mask.shape}, person pixels: {mask.mean() * 100:.1f}%")
rgb = to_rgb(image)
blurred = cv2.GaussianBlur(rgb, (0, 0), 15)
out = np.where(mask[..., None], rgb, blurred)
save_rgb(out.astype(np.uint8), "image_segmenter.png")
