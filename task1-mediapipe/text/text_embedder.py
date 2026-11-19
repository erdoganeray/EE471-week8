"""MediaPipe Text Embedder: sentence vectors + cosine similarity (Universal Sentence Encoder)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so we can import common.py

import mediapipe as mp
from common import base_options

# A and B mean the same thing, C is unrelated (pass your own 3 sentences as arguments)
A = "I like to drink coffee in the morning."
B = "Every morning I enjoy a cup of coffee."
C = "The rocket launched into orbit at midnight."
if len(sys.argv) == 4:
    A, B, C = sys.argv[1:]

# Step 1: choose the model; l2_normalize makes vectors length 1
opts = mp.tasks.text.TextEmbedderOptions(
    base_options=base_options("universal_sentence_encoder.tflite"), l2_normalize=True)

# Step 2: turn each sentence into a vector (embedding)
with mp.tasks.text.TextEmbedder.create_from_options(opts) as emb:
    ea, eb, ec = (emb.embed(t).embeddings[0] for t in (A, B, C))
print("embedding size:", len(ea.embedding))

# Step 3: cosine similarity: closer to 1 = more similar meaning
sim = mp.tasks.text.TextEmbedder.cosine_similarity
print(f"A~B (similar meaning): {sim(ea, eb):.3f}\n  A: {A}\n  B: {B}")
print(f"A~C (unrelated)      : {sim(ea, ec):.3f}\n  C: {C}")
