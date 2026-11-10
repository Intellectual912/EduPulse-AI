import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

BASE = Path("data/processed")
CHUNKS = BASE / "scheme_catalog_chunks.jsonl"
VECTORS = BASE / "scheme_catalog_embeddings.npy"

with CHUNKS.open(encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

vectors = np.load(VECTORS)

assert len(records) == len(vectors) == 15, "Record count mismatch"
assert vectors.shape[1] == 384, "Unexpected embedding dimensions"
assert np.isfinite(vectors).all(), "Invalid embedding values"

model = SentenceTransformer("all-MiniLM-L6-v2")

questions = [
    "List all the Semester 5 subjects.",
    "What is the course code for Computer Networks?",
    "Which professional elective subjects are available?"
]

print("=== SCHEME RETRIEVAL TEST ===")

for question in questions:
    query = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = vectors @ query
    best_indices = np.argsort(scores)[::-1][:3]

    print(f"\nQuestion: {question}")

    for rank, index in enumerate(best_indices, 1):
        record = records[index]
        print(
            f"{rank}. Score: {scores[index]:.3f} | "
            f"{record['chunk_id']} | "
            f"{record['text'][:130]}"
        )

print("\nValidation: PASSED")
