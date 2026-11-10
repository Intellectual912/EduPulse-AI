import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

INPUT = Path("data/processed/scheme_catalog_chunks.jsonl")
OUTPUT = Path("data/processed/scheme_catalog_embeddings.npy")

with INPUT.open(encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [record["text"] for record in records]

vectors = model.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=False
)

np.save(OUTPUT, vectors)

print("=== SCHEME EMBEDDINGS COMPLETE ===")
print("Records:", len(records))
print("Vector dimensions:", vectors.shape)
print("All values finite:", bool(np.isfinite(vectors).all()))
print("Output:", OUTPUT)
