
import json
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "rag_sources"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "rag_pages.jsonl"

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

OCR_ALL = {
    "TOC Textbook.pdf",
    "Information Retrieval Textbook.pdf",
}


def ocr_page(page):
    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(2, 2),
        alpha=False,
    )
    image = Image.frombytes(
        "RGB",
        (pix.width, pix.height),
        pix.samples,
    )
    return pytesseract.image_to_string(
        image,
        lang="eng",
    ).strip()


def extract_page(page, filename):
    if filename in OCR_ALL:
        return ocr_page(page), "ocr"

    # Prefer native PDF text where it is readable.
    text = (page.get_text("text") or "").strip()

    # Use OCR when the page has little usable text.
    if len(text) < 80:
        ocr_text = ocr_page(page)
        if len(ocr_text) > len(text):
            return ocr_text, "ocr_fallback"

    return text, "native"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_pages = 0
    pages_with_text = 0
    methods = {"native": 0, "ocr": 0, "ocr_fallback": 0}
    errors = []

    with OUTPUT_FILE.open("w", encoding="utf-8") as output:
        for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
            print(f"Processing: {pdf_path.name}")

            try:
                with pymupdf.open(pdf_path) as pdf:
                    for page_index, page in enumerate(pdf):
                        total_pages += 1

                        try:
                            text, method = extract_page(
                                page, pdf_path.name
                            )
                            methods[method] += 1

                            record = {
                                "source": pdf_path.name,
                                "page": page_index + 1,
                                "method": method,
                                "text": text,
                            }
                            output.write(
                                json.dumps(
                                    record, ensure_ascii=False
                                ) + "\n"
                            )

                            if text.strip():
                                pages_with_text += 1

                        except Exception as exc:
                            errors.append({
                                "source": pdf_path.name,
                                "page": page_index + 1,
                                "error": str(exc),
                            })

            except Exception as exc:
                errors.append({
                    "source": pdf_path.name,
                    "page": None,
                    "error": str(exc),
                })

    report = {
        "total_pages": total_pages,
        "pages_with_text": pages_with_text,
        "methods": methods,
        "errors": errors,
        "output_file": str(OUTPUT_FILE),
    }

    report_path = OUTPUT_DIR / "rag_extraction_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nExtraction summary")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("\nFull page text saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
