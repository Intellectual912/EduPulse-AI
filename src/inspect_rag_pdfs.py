from pathlib import Path
from pypdf import PdfReader

SOURCE_DIR = Path("data/rag_sources")
PDF_FILES = sorted(SOURCE_DIR.glob("*.pdf"))

if not PDF_FILES:
    raise SystemExit(f"No PDFs found in {SOURCE_DIR.resolve()}")

for pdf_path in PDF_FILES:
    print("\n" + "=" * 75)
    print(f"FILE: {pdf_path.name}")

    try:
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)
        pages_with_text = 0
        total_chars = 0
        previews = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages_with_text += 1
                total_chars += len(text)

            if page_number <= 3:
                previews.append((page_number, text[:250]))

        print(f"Total pages: {total_pages}")
        print(f"Pages containing text: {pages_with_text}/{total_pages}")
        print(f"Total extracted characters: {total_chars:,}")

        for page_number, preview in previews:
            print(f"\n--- Page {page_number} preview ---")
            print(preview if preview else "[No extractable text]")

    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")

print("\nInspection complete.")
