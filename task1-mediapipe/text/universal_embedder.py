"""MediaPipe Universal Embedder: text and images embedded in ONE shared space (EmbeddingGemma 2, 440M text+vision).

A text query is compared with images: the matching caption should score highest.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so we can import common.py

import mediapipe as mp
from common import ASSETS, MODELS, base_options

r = mp.tasks.retrieval
# 3 images and 3 captions; caption i describes image i
IMAGES = ["cat.jpg", "burger.jpg", "pose-3.jpg"]
CAPTIONS = ["a photo of a cat", "a cheeseburger on a plate", "a man cheering with his arms up"]

cache = MODELS / "cache"  # files the model prepares on first run
cache.mkdir(exist_ok=True)

# Step 1: choose the multimodal model (understands both text and images)
opts = r.UniversalEmbedderOptions(
    base_options=base_options("embeddinggemma-2-text-vision-440m.litertlm"), cache_dir=str(cache))
with r.UniversalEmbedder.create_from_options(opts) as emb:
    # Step 2: embed every image (from raw bytes) and every caption into vectors
    img = {n: emb.embed_image((ASSETS / "images" / n).read_bytes()).embeddings[0] for n in IMAGES}
    txt = {c: emb.embed_text(c).embeddings[0] for c in CAPTIONS}
    print("embedding size:", len(next(iter(img.values())).embedding))

    # Step 3: similarity table: rows = captions, columns = images; the diagonal should be highest
    print(f"{'':36s}" + "".join(f"{n:>12s}" for n in IMAGES))
    for c, te in txt.items():
        print(f"{c:36s}" + "".join(f"{r.UniversalEmbedder.cosine_similarity(te, img[n]):12.3f}" for n in IMAGES))

    # Step 4: it also works text vs text
    t1, t2 = (emb.embed_text(s).embeddings[0] for s in ("I love dogs", "Puppies are my favourite animals"))
    print(f"\ntext~text similarity: {r.UniversalEmbedder.cosine_similarity(t1, t2):.3f}")
