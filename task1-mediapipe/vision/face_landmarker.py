"""MediaPipe Face Landmarker: 478 landmarks, blendshapes, head-pose matrix."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mediapipe as mp
from common import base_options, default_args, load_mp_image, to_rgb, draw_landmarks, save_rgb

args = default_args(__doc__, "face-3.png")
opts = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=base_options("face_landmarker.task"),
    num_faces=1, output_face_blendshapes=True, output_facial_transformation_matrixes=True)
image = load_mp_image(args.input)
with mp.tasks.vision.FaceLandmarker.create_from_options(opts) as lm:
    result = lm.detect(image)

rgb = to_rgb(image)
print(f"faces: {len(result.face_landmarks)}")
for face in result.face_landmarks:
    print(f"  {len(face)} landmarks")
    draw_landmarks(rgb, face, r=1)
for blendshapes in result.face_blendshapes:
    top = sorted(blendshapes, key=lambda c: -c.score)[:5]
    print("  top blendshapes:", ", ".join(f"{c.category_name}={c.score:.2f}" for c in top))
if len(result.facial_transformation_matrixes):
    print("  head pose matrix:\n", result.facial_transformation_matrixes[0])
save_rgb(rgb, "face_landmarker.png")
