import importlib.util

packages = [
    "pypdf",
    "fitz",
    "pdfplumber",
    "sentence_transformers",
    "chromadb",
    "faiss",
    "google",
    "google.generativeai",
]

for package in packages:
    try:
        installed = importlib.util.find_spec(package) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        installed = False
    print(f"{package}: {'installed' if installed else 'not installed'}")
