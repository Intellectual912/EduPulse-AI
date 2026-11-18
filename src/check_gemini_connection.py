import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env")

api_key = os.getenv("GEMINI_API_KEY", "").strip()

if not api_key or api_key in {
    "PASTE_YOUR_NEW_API_KEY_HERE",
    "YOUR_NEW_API_KEY"
}:
    print("ERROR: A real API key is missing from .env.")
    raise SystemExit(1)

model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

try:
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents="Reply with exactly: EduPulse Gemini connection successful."
    )

    if response.text:
        print("SUCCESS: Gemini API connection is working.")
        print("Model:", model)
        print("Gemini response:", response.text.strip())
    else:
        print("ERROR: Gemini returned an empty response.")

except Exception as error:
    print("GEMINI CONNECTION FAILED")
    print("Error type:", type(error).__name__)
    print("Details:", str(error)[:500])
