import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path("data/processed/rag_chunks.jsonl")
EMBEDDINGS_FILE = Path("data/processed/rag_embeddings.npy")
TARGET_SUBJECT = "Computer Networks"
TOP_K = 3

QUESTIONS = [
    "Explain the OSI reference model and describe the functions of its seven layers.",
    "Explain the TCP/IP protocol suite and its layers.",
    "Differentiate between circuit switching and packet switching.",
    "Explain the three types of HDLC frames: information, supervisory and unnumbered frames."
]

def main():
    if not CHUNKS_FILE.exists() or not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError("RAG chunks or embeddings file is missing.")

    with CHUNKS_FILE.open(encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    embeddings = np.load(EMBEDDINGS_FILE)

    if len(chunks) != len(embeddings):
        raise ValueError("Chunk and embedding counts do not match.")

    if embeddings.ndim != 2 or embeddings.shape[1] != 384:
        raise ValueError("Unexpected embedding dimensions.")

    indices = [
        i for i, chunk in enumerate(chunks)
        if chunk["subject"] == TARGET_SUBJECT
        and chunk["source"] == "Computer Networks Textbook.pdf"
    ]

    if not indices:
        raise ValueError("No Computer Networks textbook chunks found.")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("=== COMPUTER NETWORKS RETRIEVAL EVALUATION ===")
    print("Available textbook chunks:", len(indices))

    for question in QUESTIONS:
        query_vector = model.encode(
            question,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores = embeddings[indices] @ query_vector
        ranked_positions = np.argsort(scores)[::-1][:TOP_K]

        print("\nQUESTION:", question)

        for rank, position in enumerate(ranked_positions, 1):
            index = indices[position]
            chunk = chunks[index]
            preview = " ".join(chunk["text"].split())[:300]

            print(
                f"\n{rank}. Score: {scores[position]:.3f}"
                f"\n   Page: {chunk['page']}"
                f"\n   Source: {chunk['source']}"
                f"\n   Passage: {preview}..."
            )

    print("\n=== EVALUATION FINISHED ===")

if __name__ == "__main__":
    main()
