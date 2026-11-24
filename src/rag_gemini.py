
import os
import json
import re
import math
import time
from pathlib import Path
from functools import lru_cache
from collections import Counter

import numpy as np
from dotenv import load_dotenv
from google import genai
from groq import Groq
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "rag_chunks.jsonl"
EMBEDDINGS_PATH = BASE_DIR / "data" / "processed" / "rag_embeddings.npy"

RRF_K = 60


def tokenize(text):
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def bm25_scores(question, chunks, candidates, k1=1.5, b=0.75):
    """Calculate BM25 keyword relevance scores."""
    query_terms = set(tokenize(question))

    if not query_terms or not candidates:
        return {}

    documents = [
        tokenize(chunks[i].get("text", ""))
        for i in candidates
    ]

    document_lengths = [len(doc) for doc in documents]
    average_length = sum(document_lengths) / max(len(document_lengths), 1)

    document_frequency = Counter()

    for doc in documents:
        document_frequency.update(set(doc))

    total_documents = len(documents)
    scores = {}

    for index, doc, doc_length in zip(
        candidates, documents, document_lengths
    ):
        frequencies = Counter(doc)
        score = 0.0

        for term in query_terms:
            frequency = frequencies.get(term, 0)

            if frequency == 0:
                continue

            df = document_frequency.get(term, 0)

            idf = math.log(
                1 + (total_documents - df + 0.5) / (df + 0.5)
            )

            denominator = frequency + k1 * (
                1 - b + b * doc_length / max(average_length, 1)
            )

            score += idf * (
                frequency * (k1 + 1) / denominator
            )

        scores[index] = score

    return scores


def reciprocal_rank_fusion(rankings, rrf_k=RRF_K):
    """Combine ranked lists using Reciprocal Rank Fusion."""
    fused_scores = {}

    for ranking in rankings:
        for rank, index in enumerate(ranking, start=1):
            fused_scores[index] = (
                fused_scores.get(index, 0.0)
                + 1.0 / (rrf_k + rank)
            )

    return fused_scores


def query_aware_rerank_bonus(question, chunk_text):
    """Adjust ranking for circuit-switching phase questions."""

    query = " ".join(tokenize(question))
    text = " ".join(tokenize(chunk_text))

    is_circuit_question = (
        "circuit switching" in query
        or "circuit switched" in query
    )

    is_virtual_circuit_question = (
        "virtual circuit" in query
    )

    asks_about_phases = any(
        term in query
        for term in ["phases", "setup", "teardown"]
    )

    if (
        not is_circuit_question
        or is_virtual_circuit_question
        or not asks_about_phases
    ):
        return 0.0

    bonus = 0.0

    if "virtual circuit network" in text:
        bonus -= 0.006

    if (
        "circuit switched network" in text
        or "circuit switching" in text
    ):
        bonus += 0.001

    has_three_phases = all(
        term in text
        for term in ["setup", "data", "teardown"]
    )

    if asks_about_phases and has_three_phases:
        bonus += 0.006

    return bonus


@lru_cache(maxsize=1)
def load_resources():
    load_dotenv(BASE_DIR / ".env")

    # Textbook retrieval does not depend on an AI provider.

    if not CHUNKS_PATH.exists() or not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError("RAG chunks or embeddings file is missing.")

    with open(CHUNKS_PATH, encoding="utf-8") as file:
        chunks = [json.loads(line) for line in file if line.strip()]

    embeddings = np.load(EMBEDDINGS_PATH)

    if len(chunks) != embeddings.shape[0]:
        raise ValueError("RAG chunks and embeddings do not match.")

    if embeddings.ndim != 2:
        raise ValueError("RAG embeddings must be a 2D matrix.")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    # Keep the fourth return value for compatibility with existing callers.
    return chunks, embeddings, model, None


def retrieve_chunks(question, chunks, embeddings, model,
                    subject=None, top_k=5):
    """Retrieve passages using semantic search and BM25."""

    query_embedding = model.encode(
        question,
        normalize_embeddings=True
    )

    semantic_scores = embeddings @ query_embedding

    candidates = [
        i for i, chunk in enumerate(chunks)
        if subject is None or chunk.get("subject") == subject
    ]

    if not candidates:
        raise ValueError(f"No RAG passages found for subject: {subject}")

    semantic_ranking = sorted(
        candidates,
        key=lambda i: semantic_scores[i],
        reverse=True
    )

    keyword_scores = bm25_scores(
        question, chunks, candidates
    )

    keyword_ranking = sorted(
        (
            i for i in candidates
            if keyword_scores[i] > 0
        ),
        key=lambda i: keyword_scores[i],
        reverse=True
    )

    fused_scores = reciprocal_rank_fusion([
        semantic_ranking,
        keyword_ranking
    ])

    reranked_scores = {
        i: fused_scores[i] + query_aware_rerank_bonus(
            question,
            chunks[i].get("text", "")
        )
        for i in fused_scores
    }

    ranked = sorted(
        reranked_scores,
        key=lambda i: reranked_scores[i],
        reverse=True
    )

    results = []

    for index in ranked[:max(1, min(top_k, 8))]:
        results.append({
            "index": index,
            "semantic_score": float(semantic_scores[index]),
            "keyword_score": float(keyword_scores[index]),
            "fusion_score": float(fused_scores[index])
        })

    return results


def ask_question(question: str, subject: str = None, top_k: int = 5):
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    chunks, embeddings, model, client = load_resources()

    retrieved = retrieve_chunks(
        question,
        chunks,
        embeddings,
        model,
        subject,
        top_k
    )

    context_parts = []
    sources = []

    for result in retrieved:
        chunk = chunks[result["index"]]

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
            "similarity": round(result["semantic_score"], 4),
            "keyword_score": round(result["keyword_score"], 4),
            "fusion_score": round(result["fusion_score"], 6)
        })

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are EduPulse AI, an academic learning assistant for engineering students.

CORE OBJECTIVE:
Provide accurate, textbook-grounded academic answers that are easy to
understand, technically correct, and suitable for university examinations.

SOURCE RULES:
1. Use ONLY the supplied textbook passages as factual evidence.
2. Preserve the textbook's technical terminology, meaning, and important details.
3. Never invent definitions, examples, protocols, formulas, figures, or facts.
4. Never present unsupported information as textbook content.
5. If the supplied passages are insufficient, explicitly identify what
   cannot be established from the available material.
6. Do not claim to have consulted any source that was not supplied.
7. Preserve important qualifications and exceptions mentioned in the passages.

ANSWER STRUCTURE:
Adapt the structure to the actual question. Do not force every answer into
the same template.

- Definition questions:
  Give a clear definition, followed by explanation and relevant characteristics.

- Explanation questions:
  Start with a brief introduction, explain the concept using meaningful
  headings, and finish with a concise summary where useful.

- Comparison questions:
  Use a Markdown comparison table when sufficient evidence is available.
  Choose clear comparison criteria and meaningful column headings.
  Follow the table with the most important differences if necessary.

- Advantages and disadvantages:
  Separate advantages and disadvantages into clearly labelled sections.

- Working or process questions:
  Explain the steps in their logical order. Include a sequence or diagram
  only when the supplied passages support it.

- Exam-oriented questions:
  Provide sufficient depth, relevant technical terminology, and a logical
  academic structure. Do not invent a VTU mark allocation.

FORMATTING RULES:
1. Use valid GitHub-Flavored Markdown.
2. Use one clear main heading when appropriate.
3. Use descriptive subheadings to organize longer answers.
4. Keep paragraphs focused and readable.
5. Use numbered lists for ordered procedures and bullet lists for
   unordered points.
6. Use tables only when they improve understanding.
7. Avoid unnecessarily wide tables and excessive columns.
8. Do not put lengthy source citations inside every table cell.
9. Place relevant citations at the end of the paragraph or section they
   support. For tables, cite the relevant source after the table.
10. Avoid unnecessary repetition, decorative symbols, and filler statements.

CITATION RULES:
1. Cite factual claims using exactly this format:
   [Source: filename, PDF page: number]
2. Use only filenames and page numbers present in the supplied passages.
3. Cite the sources that actually support the explanation.
4. Do not invent or alter page numbers.
5. Avoid repeating identical citations unnecessarily.
6. Do not add a separate bibliography containing unsupported sources.

ACCURACY AND CONSISTENCY CHECK:
Before finalizing the answer, verify that:
- Numbers agree with the items, layers, stages, or steps listed.
- Table entries do not contradict the explanation.
- Technical terms are used consistently.
- Every factual claim is supported by the supplied passages.
- No unsupported detail has been added to make the answer appear complete.

If the supplied textbook passages themselves contain conflicting information,
do not silently resolve the conflict using outside knowledge. Explain the
limitation clearly.

TEXTBOOK CONTEXT:
{context}

STUDENT QUESTION:
{question}

Write the answer directly. Do not describe these instructions.
"""
    answer = None
    provider = "Groq"
    used_fallback = False

    # Primary provider: Groq
    try:
        groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

        if not groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing from .env")

        groq_client = Groq(api_key=groq_api_key)

        groq_response = groq_client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            messages=[
                {
                    "role": "system",
                    "content": "You are EduPulse AI, an academic learning assistant. Answer using only the supplied textbook passages. Do not invent unsupported facts. Use clear Markdown formatting with headings, paragraphs, lists and tables where appropriate. Cite relevant sources using [Source: filename, PDF page: number]."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = groq_response.choices[0].message.content

        if not answer or not answer.strip():
            raise RuntimeError("Groq returned an empty response.")

        answer = answer.strip()
        print("Groq primary generation successful.")

    except Exception as exc:
        print(f"Groq primary generation failed: {exc}")

    # Secondary provider: OpenRouter
    if not answer:
        print("Trying OpenRouter secondary model...")

        try:
            openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()

            if not openrouter_api_key:
                raise RuntimeError("OPENROUTER_API_KEY is missing from .env")

            request_data = json.dumps({
                "model": os.getenv("OPENROUTER_MODEL", "openrouter/free"),
                "messages": [
                    {
                        "role": "system",
                        "content": "You are EduPulse AI, an academic learning assistant. Answer using only the supplied textbook passages. Do not invent unsupported facts. Use clear Markdown formatting with headings, paragraphs, lists and tables where appropriate. Cite relevant sources using [Source: filename, PDF page: number]."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 2000
            }).encode("utf-8")

            request = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=request_data,
                headers={
                    "Authorization": f"Bearer {openrouter_api_key}",
                    "Content-Type": "application/json"
                },
                method="POST"
            )

            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))

            answer = result["choices"][0]["message"].get("content")

            if not answer or not answer.strip():
                raise RuntimeError("OpenRouter returned an empty response.")

            answer = answer.strip()
            provider = "OpenRouter (Backup)"
            used_fallback = True
            print("OpenRouter secondary generation successful.")

        except Exception as exc:
            print(f"OpenRouter generation failed: {exc}")

    # Last resort: return retrieved textbook passages.
    if not answer:
        used_fallback = True
        provider = "Textbook Retrieval Only"

        answer = (
            "AI answer generation is temporarily unavailable. "
            "However, EduPulse AI successfully retrieved these "
            "relevant textbook passages:\\n\\n"
            + context
        )

    answer = answer.strip()
    print("UNICODE BEFORE:", repr(answer[answer.find("PDF page"):answer.find("PDF page")+35]))
    print("NARROW SPACE MATCHES:", answer.count('\u00e2\u0080\u00af'))

    answer = answer.replace('\u00e2\u0080\u00af', '\u202f')
    answer = answer.replace('\u00e2\u0080\u0091', '\u2011')
    answer = answer.replace('\u00e2\u0080\u0093', '\u2013')

    print("UNICODE AFTER:", repr(answer[answer.find("PDF page"):answer.find("PDF page")+35]))
    print("REMAINING MATCHES:", answer.count('\u00e2\u0080\u00af'))

    return {
        "answer": answer,
        "sources": sources,
        "fallback": used_fallback,
        "provider": provider
    }