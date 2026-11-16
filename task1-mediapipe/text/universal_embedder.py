"""MediaPipe Universal Embedder: text and images embedded in ONE shared space (EmbeddingGemma 2, 440M text+vision).

A text query is compared with images: the matching caption should score highest.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mediapipe as mp
from common import ASSETS, MODELS, base_options

r = mp.tasks.retrieval
IMAGES = ["cat.jpg", "burger.jpg", "pose-3.jpg"]
CAPTIONS = ["a photo of a cat", "a cheeseburger on a plate", "a man cheering with his arms up"]

cache = MODELS / "cache"
cache.mkdir(exist_ok=True)
opts = r.UniversalEmbedderOptions(
    base_options=base_options("embeddinggemma-2-text-vision-440m.litertlm"), cache_dir=str(cache))
with r.UniversalEmbedder.create_from_options(opts) as emb:
    img = {n: emb.embed_image((ASSETS / "images" / n).read_bytes()).embeddings[0] for n in IMAGES}
    txt = {c: emb.embed_text(c).embeddings[0] for c in CAPTIONS}
    print("embedding size:", len(next(iter(img.values())).embedding))
    print(f"{'':36s}" + "".join(f"{n:>12s}" for n in IMAGES))
    for c, te in txt.items():
        print(f"{c:36s}" + "".join(f"{r.UniversalEmbedder.cosine_similarity(te, img[n]):12.3f}" for n in IMAGES))
    t1, t2 = (emb.embed_text(s).embeddings[0] for s in ("I love dogs", "Puppies are my favourite animals"))
    print(f"\ntext~text similarity: {r.UniversalEmbedder.cosine_similarity(t1, t2):.3f}")
