"""MediaPipe Text Proofreader: on-device grammar/spelling correction (200M LiteRT-LM model)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so we can import common.py

import mediapipe as mp
from common import MODELS, base_options

# sentences with typos/grammar mistakes (pass your own as command-line arguments)
TEXTS = [
    "Their going to the libary tomorow to study for there exam.",
    "She dont like apples, but he like them a lot.",
]
if len(sys.argv) > 1:
    TEXTS = sys.argv[1:]

# Step 1: choose the model; cache_dir stores files the model prepares on first run
cache = MODELS / "cache"
cache.mkdir(exist_ok=True)
opts = mp.tasks.text.TextProofreaderOptions(
    base_options=base_options("proofread_quant_200m.litertlm"), cache_dir=str(cache))

# Step 2: proofread each text and print the corrected version
with mp.tasks.text.TextProofreader.create_from_options(opts) as proofreader:
    for t in TEXTS:
        res = proofreader.proofread(t)
        print(f"input : {t}\noutput: {res.proofread_text}")
        # res.corrections is a character-level diff (SAME / INSERTION / DELETION); count real edits
        edits = [c for c in res.corrections if c.type.name != "SAME"]
        print(f"  {len(edits)} character edits (insertions/deletions)")
