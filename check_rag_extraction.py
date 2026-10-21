import json
from collections import defaultdict
from pathlib import Path

path = Path("data/processed/rag_pages.jsonl")
stats = defaultdict(lambda: {"pages": 0, "empty": 0, "ocr": 0})
samples = defaultdict(list)

with path.open(encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        name = row["source"]
        text = row.get("text", "").strip()

        stats[name]["pages"] += 1
        if not text:
            stats[name]["empty"] += 1
        if row["method"].startswith("ocr"):
            stats[name]["ocr"] += 1

        if text and len(samples[name]) < 2:
            samples[name].append(
                (row["page"], row["method"], text[:160])
            )

print("=== PER-PDF QUALITY SUMMARY ===")
for name, s in sorted(stats.items()):
    print(f"\n{name}")
    print(f"Pages: {s['pages']} | Empty: {s['empty']} | OCR pages: {s['ocr']}")
    for page, method, preview in samples[name]:
        print(f"  Page {page} [{method}]: {preview!r}")
