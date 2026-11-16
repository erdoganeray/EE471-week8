"""MediaPipe Text Summarizer: on-device summaries in two modes (KEYPOINTS and TLDR)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mediapipe as mp
from common import MODELS, base_options

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

cache = MODELS / "cache"
cache.mkdir(exist_ok=True)
Mode = mp.tasks.text.TextSummarizerMode
for mode in (Mode.KEYPOINTS, Mode.TLDR):
    opts = mp.tasks.text.TextSummarizerOptions(
        base_options=base_options("summarization_quant_200m_2modes.litertlm"),
        mode=mode, cache_dir=str(cache))
    with mp.tasks.text.TextSummarizer.create_from_options(opts) as summarizer:
        res = summarizer.summarize(TEXT)
    print(f"--- {mode.name} ---\n{res.summary}")
