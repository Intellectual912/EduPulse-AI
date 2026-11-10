import json
import re
from pathlib import Path

INPUT = Path("data/processed/rag_pages.jsonl")
OUTPUT = Path("data/processed/rag_chunks.jsonl")

CHUNK_SIZE = 1000
OVERLAP = 150
STEP = CHUNK_SIZE - OVERLAP

def clean_text(text):
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def get_subject(source):
    name = source.lower()

    if "scheme" in name or "syllabus" in name:
        return "AIML Scheme and Syllabus"
    if "computer networks" in name:
        return "Computer Networks"
    if "information retrieval" in name:
        return "Information Retrieval"
    if "toc" in name:
        return "Theory of Computation"

    return "Unclassified"

pages = 0
empty_pages = 0
chunk_count = 0

with INPUT.open(encoding="utf-8") as infile, \
     OUTPUT.open("w", encoding="utf-8") as outfile:

    for line in infile:
        row = json.loads(line)
        pages += 1

        text = clean_text(row.get("text", ""))

        if not text:
            empty_pages += 1
            continue

        start = 0

        while start < len(text):
            chunk_text = text[start:start + CHUNK_SIZE]

            if chunk_text.strip():
                chunk = {
                    "chunk_id": f"chunk_{chunk_count + 1:06d}",
                    "source": row["source"],
                    "page": row["page"],
                    "subject": get_subject(row["source"]),
                    "text": chunk_text.strip()
                }

                outfile.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                chunk_count += 1

            if start + CHUNK_SIZE >= len(text):
                break

            start += STEP

print("=== RAG CHUNKING COMPLETE ===")
print(f"Pages processed: {pages}")
print(f"Empty pages skipped: {empty_pages}")
print(f"Chunks created: {chunk_count}")
print(f"Chunk size: {CHUNK_SIZE}")
print(f"Overlap: {OVERLAP}")
print(f"Output: {OUTPUT}")
