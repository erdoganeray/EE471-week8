"""MediaPipe Face Landmarker: 478 landmarks, blendshapes, head-pose matrix."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so we can import common.py

import mediapipe as mp
from common import base_options, default_args, load_mp_image, to_rgb, draw_landmarks, save_rgb

args = default_args(__doc__, "face-3.png")  # --input <image>, default is a sample face

# Step 1: choose the model and ask for the extra outputs (expression scores + head pose)
opts = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=base_options("face_landmarker.task"),
    num_faces=1, output_face_blendshapes=True, output_facial_transformation_matrixes=True)

# Step 2: load the image and run the landmarker
image = load_mp_image(args.input)
with mp.tasks.vision.FaceLandmarker.create_from_options(opts) as lm:
    result = lm.detect(image)

# Step 3: draw the 478 points on a copy of the image
rgb = to_rgb(image)
print(f"faces: {len(result.face_landmarks)}")
for face in result.face_landmarks:
    print(f"  {len(face)} landmarks")
    draw_landmarks(rgb, face, r=1)

# Step 4: blendshapes = scores for facial expressions (smile, squint, ...); show the top 5
for blendshapes in result.face_blendshapes:
    top = sorted(blendshapes, key=lambda c: -c.score)[:5]
    print("  top blendshapes:", ", ".join(f"{c.category_name}={c.score:.2f}" for c in top))

# Step 5: 4x4 matrix = head pose (rotation + position); Task 3 uses it to get the yaw angle
if len(result.facial_transformation_matrixes):
    print("  head pose matrix:\n", result.facial_transformation_matrixes[0])
save_rgb(rgb, "face_landmarker.png")  # saved to outputs/
