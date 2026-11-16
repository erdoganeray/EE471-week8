"""MediaPipe Pose Landmarker: 33 body landmarks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mediapipe as mp
from common import base_options, default_args, load_mp_image, to_rgb, draw_landmarks, save_rgb

args = default_args(__doc__, "pose-3.jpg")
opts = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=base_options("pose_landmarker_lite.task"), num_poses=2)
image = load_mp_image(args.input)
with mp.tasks.vision.PoseLandmarker.create_from_options(opts) as pl:
    result = pl.detect(image)

rgb = to_rgb(image)
conns = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
print(f"poses: {len(result.pose_landmarks)}")
for pose in result.pose_landmarks:
    print(f"  {len(pose)} landmarks; nose=({pose[0].x:.2f},{pose[0].y:.2f}) "
          f"l_wrist=({pose[15].x:.2f},{pose[15].y:.2f}) r_wrist=({pose[16].x:.2f},{pose[16].y:.2f})")
    draw_landmarks(rgb, pose, conns)
save_rgb(rgb, "pose_landmarker.png")
