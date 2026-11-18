import os
import json
from pathlib import Path
from functools import lru_cache

import numpy as np
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "rag_chunks.jsonl"
EMBEDDINGS_PATH = BASE_DIR / "data" / "processed" / "rag_embeddings.npy"


@lru_cache(maxsize=1)
def load_resources():
    load_dotenv(BASE_DIR / ".env")

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing from .env")

    if not CHUNKS_PATH.exists() or not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError("RAG chunks or embeddings file is missing.")

    with open(CHUNKS_PATH, encoding="utf-8") as file:
        chunks = [json.loads(line) for line in file if line.strip()]

    embeddings = np.load(EMBEDDINGS_PATH)

    if len(chunks) != embeddings.shape[0]:
        raise ValueError("RAG chunks and embeddings do not match.")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = genai.Client(api_key=api_key)

    return chunks, embeddings, model, client


def ask_question(question: str, subject: str = None, top_k: int = 5):
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    chunks, embeddings, model, client = load_resources()

    query_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    scores = embeddings @ query_embedding

    candidates = [
        i for i, chunk in enumerate(chunks)
        if subject is None or chunk.get("subject") == subject
    ]

    if not candidates:
        raise ValueError(f"No RAG passages found for subject: {subject}")

    ranked = sorted(
        candidates,
        key=lambda i: scores[i],
        reverse=True
    )[:max(1, min(top_k, 8))]

    context_parts = []
    sources = []

    for i in ranked:
        chunk = chunks[i]

        source = chunk.get("source", "Unknown source")
        page = chunk.get("page", "Unknown")
        text = chunk.get("text", "")

        context_parts.append(
            f"Source: {source}\n"
            f"PDF page: {page}\n"
            f"Passage:\n{text}"
        )

        sources.append({
            "source": source,
            "page": page,
            "subject": chunk.get("subject"),
            "similarity": round(float(scores[i]), 4)
        })

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are EduPulse AI, an academic learning assistant.

Answer the student's question using ONLY the supplied textbook
passages.

Rules:
1. Explain in clear, student-friendly language.
2. Do not invent facts that are absent from the passages.
3. If the passages do not contain enough information, say so.
4. Cite relevant sources using [Source: filename, PDF page: number].
5. Do not claim that you consulted any source not supplied below.

TEXTBOOK CONTEXT:
{context}

STUDENT QUESTION:
{question}

Provide the answer:
"""

    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        contents=prompt
    )

    answer = response.text

    if not answer:
        raise RuntimeError("Gemini returned an empty response.")

    return {
        "answer": answer.strip(),
        "sources": sources
    }
