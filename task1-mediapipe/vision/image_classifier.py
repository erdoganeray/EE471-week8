"""MediaPipe Image Classifier: ImageNet top-k labels (EfficientNet-Lite0)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so we can import common.py

import mediapipe as mp
from common import base_options, default_args, load_mp_image

args = default_args(__doc__, "burger.jpg")  # --input <image>, default is a burger photo

# Step 1: choose the model (1000 ImageNet classes) and return the 5 best guesses
opts = mp.tasks.vision.ImageClassifierOptions(
    base_options=base_options("efficientnet_lite0.tflite"), max_results=5)

# Step 2: load the image and classify it
image = load_mp_image(args.input)
with mp.tasks.vision.ImageClassifier.create_from_options(opts) as clf:
    result = clf.classify(image)

# Step 3: print label + confidence, best first
for c in result.classifications[0].categories:
    print(f"  {c.category_name:25s} {c.score:.3f}")
