"""MediaPipe Text Summarizer: on-device summaries in two modes (KEYPOINTS and TLDR)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so we can import common.py

import mediapipe as mp
from common import MODELS, base_options

# a short article to summarize (pass your own text as command-line arguments)
TEXT = (
    "MediaPipe is a cross-platform framework from Google for building machine learning pipelines. "
    "Its Tasks API wraps ready-made solutions such as face detection, pose estimation, object detection, "
    "text classification and audio classification behind a few lines of code. Models are small enough to run "
    "on phones, laptops and browsers without sending any data to a server, which keeps latency low and "
    "protects user privacy. Developers can start from the provided pretrained models and later customise "
    "them with MediaPipe Model Maker."
)
if len(sys.argv) > 1:
    TEXT = " ".join(sys.argv[1:])

cache = MODELS / "cache"  # files the model prepares on first run
cache.mkdir(exist_ok=True)

# Step 1: the same model has two modes: KEYPOINTS (bullet list) and TLDR (short paragraph)
Mode = mp.tasks.text.TextSummarizerMode
for mode in (Mode.KEYPOINTS, Mode.TLDR):
    opts = mp.tasks.text.TextSummarizerOptions(
        base_options=base_options("summarization_quant_200m_2modes.litertlm"),
        mode=mode, cache_dir=str(cache))
    # Step 2: summarize the text with this mode and print the result
    with mp.tasks.text.TextSummarizer.create_from_options(opts) as summarizer:
        res = summarizer.summarize(TEXT)
    print(f"--- {mode.name} ---\n{res.summary}")
