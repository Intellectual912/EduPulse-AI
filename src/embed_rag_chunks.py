import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

INPUT = Path("data/processed/rag_chunks.jsonl")
OUTPUT = Path("data/processed/rag_embeddings.npy")

print("Loading RAG chunks...")

chunks = []
with INPUT.open(encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

texts = [chunk["text"] for chunk in chunks]

print("Total chunks:", len(texts))
print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings. Please wait...")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

np.save(OUTPUT, embeddings)

print("\n=== EMBEDDING GENERATION COMPLETE ===")
print("Chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)
print("Output:", OUTPUT)
print("All values finite:", bool(np.isfinite(embeddings).all()))
print("All vectors normalized:", bool(np.allclose(
    np.linalg.norm(embeddings, axis=1), 1.0, atol=0.001
)))
