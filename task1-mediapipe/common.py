"""Shared helpers for the Task 1 MediaPipe demos."""
import argparse
import sys
from pathlib import Path

# folders used by all scripts
ROOT = Path(__file__).parent
MODELS = ROOT / "models"      # downloaded model files (git-ignored)
ASSETS = ROOT / "assets"      # small sample images / audio
OUTPUTS = ROOT / "outputs"    # annotated images and generated audio


def model_path(filename: str) -> str:
    """Return a local model path, with a clear message instead of a surprise download."""
    p = MODELS / filename
    if not p.exists():
        sys.exit(f"Model missing: {p}\nRun: python download_models.py --only <name>  (resumable)")
    return str(p)


def base_options(filename: str):
    """BaseOptions tells a MediaPipe task which model file to load."""
    import mediapipe as mp
    return mp.tasks.BaseOptions(model_asset_path=model_path(filename))


def out_path(name: str) -> str:
    """Path inside outputs/ (the folder is created if needed)."""
    OUTPUTS.mkdir(exist_ok=True)
    return str(OUTPUTS / name)


def load_mp_image(path: str):
    """Load an image file as a MediaPipe Image."""
    import mediapipe as mp
    return mp.Image.create_from_file(path)


def to_rgb(mp_image):
    """mediapipe Image -> contiguous RGB uint8 numpy array (a copy we can draw on)."""
    import numpy as np
    return np.ascontiguousarray(mp_image.numpy_view()[..., :3]).copy()


def save_rgb(rgb_array, name: str) -> str:
    """Save an RGB numpy array to outputs/<name> (OpenCV wants BGR, so convert first)."""
    import cv2
    p = out_path(name)
    cv2.imwrite(p, cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))
    print(f"saved -> {p}")
    return p


def draw_landmarks(rgb, landmarks, connections=None, color=(0, 255, 0), r=3):
    """Draw normalized landmarks (+ optional connections) on an RGB array, in place."""
    import cv2
    h, w = rgb.shape[:2]
    pts = [(int(l.x * w), int(l.y * h)) for l in landmarks]  # normalized 0..1 -> pixels
    for c in connections or []:  # white lines between connected points
        a, b = (c.start, c.end) if hasattr(c, "start") else c
        if a < len(pts) and b < len(pts):
            cv2.line(rgb, pts[a], pts[b], (255, 255, 255), 1)
    for p in pts:  # colored dot on every point
        cv2.circle(rgb, p, r, color, -1)
    return rgb


def default_args(description, default_image):
    """Parse --input <image>; falls back to a sample image from assets/images."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--input", default=str(ASSETS / "images" / default_image))
    return ap.parse_args()
