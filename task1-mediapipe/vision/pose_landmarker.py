"""MediaPipe Pose Landmarker: 33 body landmarks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so we can import common.py

import mediapipe as mp
from common import base_options, default_args, load_mp_image, to_rgb, draw_landmarks, save_rgb

args = default_args(__doc__, "pose-3.jpg")  # --input <image>, default is a sample person

# Step 1: choose the model (lite = fastest) and the max number of people
opts = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=base_options("pose_landmarker_lite.task"), num_poses=2)

# Step 2: load the image and run the landmarker
image = load_mp_image(args.input)
with mp.tasks.vision.PoseLandmarker.create_from_options(opts) as pl:
    result = pl.detect(image)

# Step 3: print key points and draw the skeleton (points + connecting lines)
rgb = to_rgb(image)
conns = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS  # which points connect to which
print(f"poses: {len(result.pose_landmarks)}")
for pose in result.pose_landmarks:
    # index 0 = nose, 15 = left wrist, 16 = right wrist (x, y are normalized 0..1)
    print(f"  {len(pose)} landmarks; nose=({pose[0].x:.2f},{pose[0].y:.2f}) "
          f"l_wrist=({pose[15].x:.2f},{pose[15].y:.2f}) r_wrist=({pose[16].x:.2f},{pose[16].y:.2f})")
    draw_landmarks(rgb, pose, conns)
save_rgb(rgb, "pose_landmarker.png")  # saved to outputs/
