"""MediaPipe Image Embedder: vector per image + cosine similarity between images."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mediapipe as mp
from common import ASSETS, base_options, load_mp_image

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--a", default=str(ASSETS / "images" / "cat.jpg"))
ap.add_argument("--b", default=str(ASSETS / "images" / "cats_and_dogs.jpg"))
ap.add_argument("--c", default=str(ASSETS / "images" / "burger.jpg"))
args = ap.parse_args()

opts = mp.tasks.vision.ImageEmbedderOptions(
    base_options=base_options("mobilenet_v3_small.tflite"), l2_normalize=True)
with mp.tasks.vision.ImageEmbedder.create_from_options(opts) as emb:
    ea, eb, ec = (emb.embed(load_mp_image(p)).embeddings[0] for p in (args.a, args.b, args.c))
print("embedding size:", len(ea.embedding))
sim = mp.tasks.vision.ImageEmbedder.cosine_similarity
print(f"cat vs cats_and_dogs: {sim(ea, eb):.3f}")
print(f"cat vs burger       : {sim(ea, ec):.3f}")
