"""MediaPipe Holistic Landmarker: face + pose + both hands in one pass."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mediapipe as mp
from common import base_options, default_args, load_mp_image, to_rgb, draw_landmarks, save_rgb

args = default_args(__doc__, "pose-3.jpg")
opts = mp.tasks.vision.HolisticLandmarkerOptions(
    base_options=base_options("holistic_landmarker.task"))
image = load_mp_image(args.input)
with mp.tasks.vision.HolisticLandmarker.create_from_options(opts) as hl:
    result = hl.detect(image)

rgb = to_rgb(image)
v = mp.tasks.vision
parts = [("face", result.face_landmarks, None, (0, 200, 255)),
         ("pose", result.pose_landmarks, v.PoseLandmarksConnections.POSE_LANDMARKS, (0, 255, 0)),
         ("left hand", result.left_hand_landmarks, v.HandLandmarksConnections.HAND_CONNECTIONS, (255, 0, 0)),
         ("right hand", result.right_hand_landmarks, v.HandLandmarksConnections.HAND_CONNECTIONS, (255, 0, 255))]
for name, lms, conns, color in parts:
    print(f"{name}: {len(lms)} landmarks")
    if lms:
        draw_landmarks(rgb, lms, conns, color, r=1 if name == "face" else 3)
save_rgb(rgb, "holistic_landmarker.png")
