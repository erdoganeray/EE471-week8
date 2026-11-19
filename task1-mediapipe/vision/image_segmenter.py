"""MediaPipe Image Segmenter: person mask (selfie segmenter) -> blurred background."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so we can import common.py

import cv2
import numpy as np
import mediapipe as mp
from common import base_options, default_args, load_mp_image, to_rgb, save_rgb

args = default_args(__doc__, "pose-3.jpg")  # --input <image>, default is a sample person

# Step 1: choose the model and ask for a category mask (one class id per pixel)
opts = mp.tasks.vision.ImageSegmenterOptions(
    base_options=base_options("selfie_segmenter.tflite"),
    output_category_mask=True, output_confidence_masks=False)

# Step 2: load the image and segment it
image = load_mp_image(args.input)
with mp.tasks.vision.ImageSegmenter.create_from_options(opts) as seg:
    result = seg.segment(image)

# Step 3: True where the pixel belongs to the person (the model's only label "selfie" = category 0)
mask = result.category_mask.numpy_view().squeeze() == 0
print(f"mask {mask.shape}, person pixels: {mask.mean() * 100:.1f}%")

# Step 4: keep the person sharp, blur everything else, and save
rgb = to_rgb(image)
blurred = cv2.GaussianBlur(rgb, (0, 0), 15)
out = np.where(mask[..., None], rgb, blurred)
save_rgb(out.astype(np.uint8), "image_segmenter.png")  # saved to outputs/
