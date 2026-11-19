"""MediaPipe Hand Landmarker: 21 landmarks + handedness per hand."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so we can import common.py

import mediapipe as mp
from common import base_options, default_args, load_mp_image, to_rgb, draw_landmarks, save_rgb

args = default_args(__doc__, "woman_hands.jpg")  # --input <image>, default has two hands

# Step 1: choose the model and the max number of hands
opts = mp.tasks.vision.HandLandmarkerOptions(
    base_options=base_options("hand_landmarker.task"), num_hands=2)

# Step 2: load the image and run the landmarker
image = load_mp_image(args.input)
with mp.tasks.vision.HandLandmarker.create_from_options(opts) as hl:
    result = hl.detect(image)

# Step 3: print left/right label + score for each hand and draw the 21 points
rgb = to_rgb(image)
conns = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS  # finger bones
print(f"hands: {len(result.hand_landmarks)}")
for hand, handed in zip(result.hand_landmarks, result.handedness):
    print(f"  {handed[0].category_name} ({handed[0].score:.2f}), {len(hand)} landmarks")
    draw_landmarks(rgb, hand, conns)
save_rgb(rgb, "hand_landmarker.png")  # saved to outputs/
