"""MediaPipe Face Detector: bounding boxes + 6 keypoints per face."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import mediapipe as mp
from common import base_options, default_args, load_mp_image, to_rgb, save_rgb

args = default_args(__doc__, "face-3.png")
opts = mp.tasks.vision.FaceDetectorOptions(
    base_options=base_options("blaze_face_short_range.tflite"),
    min_detection_confidence=0.5)
image = load_mp_image(args.input)
with mp.tasks.vision.FaceDetector.create_from_options(opts) as detector:
    result = detector.detect(image)

rgb = to_rgb(image)
print(f"faces found: {len(result.detections)}")
for d in result.detections:
    b = d.bounding_box
    print(f"  score={d.categories[0].score:.2f} box=({b.origin_x},{b.origin_y},{b.width},{b.height})")
    cv2.rectangle(rgb, (b.origin_x, b.origin_y), (b.origin_x + b.width, b.origin_y + b.height), (0, 255, 0), 2)
    for k in d.keypoints:  # normalized coordinates
        cv2.circle(rgb, (int(k.x * rgb.shape[1]), int(k.y * rgb.shape[0])), 3, (255, 0, 0), -1)
save_rgb(rgb, "face_detector.png")
