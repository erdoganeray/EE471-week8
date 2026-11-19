"""MediaPipe Text Classifier: sentiment (positive/negative) with a BERT model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so we can import common.py

import mediapipe as mp
from common import base_options

# sample sentences: positive, negative, neutral (pass your own as command-line arguments)
SENTENCES = [
    "I loved this course, the in-class exercises were great!",
    "The internet is so slow that nothing works, this is awful.",
    "It is a movie about a man who goes to the shop.",
]
if len(sys.argv) > 1:
    SENTENCES = sys.argv[1:]

# Step 1: choose the model (BERT fine-tuned for positive/negative sentiment)
opts = mp.tasks.text.TextClassifierOptions(base_options=base_options("bert_classifier.tflite"))

# Step 2: classify each sentence and print the label with the highest score
with mp.tasks.text.TextClassifier.create_from_options(opts) as clf:
    for s in SENTENCES:
        cats = clf.classify(s).classifications[0].categories
        top = max(cats, key=lambda c: c.score)
        print(f"{top.category_name:9s} {top.score:.3f} | {s}")
