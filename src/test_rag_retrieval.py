import json
import numpy as np
from sentence_transformers import SentenceTransformer

with open("data/processed/rag_chunks.jsonl", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]

embeddings = np.load("data/processed/rag_embeddings.npy")
model = SentenceTransformer("all-MiniLM-L6-v2")

question = "List all course titles and subject names prescribed for semester five of the CSE AIML programme."

query = model.encode(question, normalize_embeddings=True)
scores = embeddings @ query

candidates = [
    i for i, chunk in enumerate(chunks)
    if chunk["subject"] == "AIML Scheme and Syllabus"
]

ranked = sorted(candidates, key=lambda i: scores[i], reverse=True)[:5]

print("\nQUESTION:", question)

for rank, i in enumerate(ranked, 1):
    chunk = chunks[i]
    passage = " ".join(chunk["text"].split())[:180]

    print(
        f"\n{rank}. Score: {scores[i]:.3f}"
        f"\n   Source: {chunk['source']}"
        f"\n   Page: {chunk['page']}"
        f"\n   Text: {passage}..."
    )
