"""Task 2: which arm is raised?  usage: python predict.py PATH-TO-INPUT-IMAGE
Prints exactly one of: left, right, both, None

"left"/"right" is the PERSON's own left/right arm (MediaPipe's convention), i.e. the
arm that appears on the image's right side is the person's left arm.
"""
import sys
from pathlib import Path

import mediapipe as mp

# the pose model sits next to this script (so Docker needs no download)
MODEL = Path(__file__).parent / "pose_landmarker_lite.task"

# MediaPipe pose landmark indices
L_SHOULDER, R_SHOULDER = 11, 12
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24

MIN_VISIBILITY = 0.5   # ignore landmarks the model is not confident about
RAISE_MARGIN = 0.05    # wrist must be this far above the shoulder (fraction of torso length)


def detect_pose(image_path: str):
    """Run the Pose Landmarker; return the 33 landmarks of the first person, or None."""
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)), num_poses=1)
    image = mp.Image.create_from_file(image_path)
    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(image)
    return result.pose_landmarks[0] if result.pose_landmarks else None


def is_raised(lm, shoulder: int, wrist: int, hip: int) -> bool:
    """Arm is raised if the wrist is clearly above the shoulder (image y grows downward)."""
    s, w, h = lm[shoulder], lm[wrist], lm[hip]
    if min(s.visibility, w.visibility) < MIN_VISIBILITY:  # not reliably visible -> not raised
        return False
    torso = abs(h.y - s.y) or 1.0          # scale margin by body size
    return (s.y - w.y) > RAISE_MARGIN * torso  # smaller y = higher in the image


def classify(image_path: str) -> str:
    """Check both arms and map the result to left / right / both / None."""
    lm = detect_pose(image_path)
    if lm is None:                          # nobody found in the image
        return "None"
    left = is_raised(lm, L_SHOULDER, L_WRIST, L_HIP)
    right = is_raised(lm, R_SHOULDER, R_WRIST, R_HIP)
    if left and right:
        return "both"
    if left:
        return "left"
    if right:
        return "right"
    return "None"


if __name__ == "__main__":
    # command line: python predict.py <image>  -> prints one word
    if len(sys.argv) != 2:
        sys.exit("usage: python predict.py PATH-TO-INPUT-IMAGE")
    print(classify(sys.argv[1]))
