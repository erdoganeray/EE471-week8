"""MediaPipe Object Detector: COCO boxes + labels (EfficientDet-Lite0)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so we can import common.py

import cv2
import mediapipe as mp
from common import base_options, default_args, load_mp_image, to_rgb, save_rgb

args = default_args(__doc__, "cat_and_dog.jpg")  # --input <image>, default has a cat and a dog

# Step 1: choose the model (80 COCO classes); keep at most 5 objects with score >= 0.4
opts = mp.tasks.vision.ObjectDetectorOptions(
    base_options=base_options("efficientdet_lite0.tflite"), max_results=5, score_threshold=0.4)

# Step 2: load the image and detect objects
image = load_mp_image(args.input)
with mp.tasks.vision.ObjectDetector.create_from_options(opts) as det:
    result = det.detect(image)

# Step 3: print label, score and box, and draw them on a copy of the image
rgb = to_rgb(image)
print(f"objects: {len(result.detections)}")
for d in result.detections:
    b, c = d.bounding_box, d.categories[0]
    print(f"  {c.category_name} {c.score:.2f} box=({b.origin_x},{b.origin_y},{b.width},{b.height})")
    cv2.rectangle(rgb, (b.origin_x, b.origin_y), (b.origin_x + b.width, b.origin_y + b.height), (0, 255, 0), 2)
    cv2.putText(rgb, f"{c.category_name} {c.score:.2f}", (b.origin_x, max(b.origin_y - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
save_rgb(rgb, "object_detector.png")  # saved to outputs/
