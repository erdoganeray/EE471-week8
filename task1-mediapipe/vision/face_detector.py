"""MediaPipe Face Detector: bounding boxes + 6 keypoints per face."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so we can import common.py

import cv2
import mediapipe as mp
from common import base_options, default_args, load_mp_image, to_rgb, save_rgb

args = default_args(__doc__, "face-3.png")  # --input <image>, default is a sample face

# Step 1: choose the model (BlazeFace) and the minimum confidence for a detection
opts = mp.tasks.vision.FaceDetectorOptions(
    base_options=base_options("blaze_face_short_range.tflite"),
    min_detection_confidence=0.5)

# Step 2: load the image and run the detector
image = load_mp_image(args.input)
with mp.tasks.vision.FaceDetector.create_from_options(opts) as detector:
    result = detector.detect(image)

# Step 3: print each face and draw the box + keypoints on a copy of the image
rgb = to_rgb(image)
print(f"faces found: {len(result.detections)}")
for d in result.detections:
    b = d.bounding_box
    print(f"  score={d.categories[0].score:.2f} box=({b.origin_x},{b.origin_y},{b.width},{b.height})")
    cv2.rectangle(rgb, (b.origin_x, b.origin_y), (b.origin_x + b.width, b.origin_y + b.height), (0, 255, 0), 2)
    for k in d.keypoints:  # keypoints are normalized (0..1), so scale to pixels
        cv2.circle(rgb, (int(k.x * rgb.shape[1]), int(k.y * rgb.shape[0])), 3, (255, 0, 0), -1)
save_rgb(rgb, "face_detector.png")  # saved to outputs/
