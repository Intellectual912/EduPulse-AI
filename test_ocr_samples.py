from pathlib import Path
import pymupdf
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

folder = Path("data/rag_sources")
sample_pages = [1, 2, 5, 10, 20]

for filename in [
    "TOC Textbook.pdf",
    "Information Retrieval Textbook.pdf",
]:
    path = folder / filename
    print("\n" + "=" * 65)
    print(filename)

    with pymupdf.open(path) as pdf:
        for page_number in sample_pages:
            if page_number > len(pdf):
                continue

            page = pdf[page_number - 1]
            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(2, 2),
                alpha=False
            )
            image = Image.frombytes(
                "RGB",
                (pix.width, pix.height),
                pix.samples
            )

            text = pytesseract.image_to_string(
                image,
                lang="eng"
            ).strip()

            print(f"\n--- Page {page_number}: {len(text)} characters ---")
            print("Preview:", repr(text[:120]) if text else "[No readable text detected]")
