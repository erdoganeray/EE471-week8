"""MediaPipe Text Classifier: sentiment (positive/negative) with a BERT model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mediapipe as mp
from common import base_options

SENTENCES = [
    "I loved this course, the in-class exercises were great!",
    "The internet is so slow that nothing works, this is awful.",
    "It is a movie about a man who goes to the shop.",
]
if len(sys.argv) > 1:
    SENTENCES = sys.argv[1:]

opts = mp.tasks.text.TextClassifierOptions(base_options=base_options("bert_classifier.tflite"))
with mp.tasks.text.TextClassifier.create_from_options(opts) as clf:
    for s in SENTENCES:
        cats = clf.classify(s).classifications[0].categories
        top = max(cats, key=lambda c: c.score)
        print(f"{top.category_name:9s} {top.score:.3f} | {s}")
