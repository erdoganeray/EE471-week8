"""Task 3: which way is the face looking?  usage: python predict.py PATH-TO-INPUT-IMAGE
Prints exactly one of: left, right, straight

"left"/"right" is as seen in the image: a face turned towards the image's left edge is "left".
"""
import math
import sys
from pathlib import Path

import mediapipe as mp

# the face model sits next to this script (so Docker needs no download)
MODEL = Path(__file__).parent / "face_landmarker.task"
YAW_THRESHOLD_DEG = 15.0   # |yaw| below this counts as looking straight


def head_yaw_degrees(image_path: str):
    """Estimate head yaw with the Face Landmarker's facial transformation matrix.

    Returns None if no face is found. Negative yaw = turned towards the image's left,
    positive = towards the image's right.
    """
    # Step 1: run the Face Landmarker and ask for the head-pose matrix
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL)),
        num_faces=1,
        output_facial_transformation_matrixes=True)
    image = mp.Image.create_from_file(image_path)
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(image)
    if not result.face_landmarks:           # no face in the image
        return None
    # Step 2: the 4x4 matrix is the head pose; its top-left 3x3 part is the rotation
    m = result.facial_transformation_matrixes[0]
    # yaw (rotation around the vertical axis) from the rotation part of the matrix
    return math.degrees(math.atan2(-m[2][0], math.hypot(m[2][1], m[2][2])))


def classify(image_path: str) -> str:
    """Turn the yaw angle into left / right / straight."""
    yaw = head_yaw_degrees(image_path)
    if yaw is None:
        raise ValueError("no face detected")
    if yaw < -YAW_THRESHOLD_DEG:            # turned clearly towards the image's left
        return "left"
    if yaw > YAW_THRESHOLD_DEG:             # turned clearly towards the image's right
        return "right"
    return "straight"                       # within +-15 degrees


if __name__ == "__main__":
    # command line: python predict.py <image>  -> prints one word
    if len(sys.argv) != 2:
        sys.exit("usage: python predict.py PATH-TO-INPUT-IMAGE")
    try:
        print(classify(sys.argv[1]))
    except ValueError as e:
        sys.exit(f"error: {e}")
