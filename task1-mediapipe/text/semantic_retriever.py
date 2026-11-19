"""MediaPipe Semantic Retriever: index documents in a vector store and search them by meaning.

Uses a UniversalEmbedder (EmbeddingGemma 2, text 270M) as the encoder.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so we can import common.py

import mediapipe as mp
from common import MODELS, base_options

r = mp.tasks.retrieval
# the "database": 4 short documents about different topics
DOCS = {
    "doc-docker": "Docker packages an application and its dependencies into a portable container image.",
    "doc-git": "Git is a distributed version control system used to track changes in source code.",
    "doc-mediapipe": "MediaPipe offers ready-made on-device ML solutions such as pose and face landmark detection.",
    "doc-pasta": "To cook pasta, boil salted water, add the pasta and stir for about ten minutes.",
}
# search queries (pass your own as command-line arguments)
QUERIES = sys.argv[1:] or ["How do I ship my app in a container?", "recognise body pose in a picture", "dinner recipe"]

cache = MODELS / "cache"  # files the model prepares on first run
cache.mkdir(exist_ok=True)

# Step 1: create the embedder that turns text into vectors
embedder = r.UniversalEmbedder.create_from_options(r.UniversalEmbedderOptions(
    base_options=base_options("embeddinggemma-2-text-270m.litertlm"), cache_dir=str(cache)))

# Step 2: create the retriever on top of it (no database_path = kept in memory)
opts = r.SemanticRetrieverOptions(embedder=embedder, embedding_dimension=768)
with r.SemanticRetriever.create_from_options(opts) as retriever:
    # Step 3: insert (embed + store) every document
    for doc_id, text in DOCS.items():
        retriever.insert_document(doc_id, text, metadata={"source": "demo"})
    print("indexed:", retriever.get_all_record_ids())

    # Step 4: search by meaning; the best matches come back with a similarity score
    for q in QUERIES:
        print(f"\nquery: {q}")
        for hit in retriever.retrieve(q, limit=2, min_similarity=0.0):
            print(f"  {hit.score:.3f}  {hit.id}: {hit.content[:70]}")
embedder.close()
