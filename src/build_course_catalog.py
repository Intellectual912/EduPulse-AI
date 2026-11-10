import json
from pathlib import Path

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

catalogue = {
    "university": "Visvesvaraya Technological University",
    "programme": "B.E. Computer Science and Engineering (Artificial Intelligence and Machine Learning)",
    "scheme": "2022",
    "semester": 5,
    "source": "5th sem aiml scheme.pdf",
    "source_page": 1,

    "courses": [
        {"code": "BCS501", "title": "Software Engineering & Project Management", "type": "PCC", "credits": 3},
        {"code": "BCS502", "title": "Computer Networks", "type": "IPCC", "credits": 4},
        {"code": "BCS503", "title": "Theory of Computation", "type": "PCC", "credits": 4},
        {"code": "BAIL504", "title": "Data Visualization Lab", "type": "PCCL", "credits": 1},
        {"code": "BXX515x", "title": "Professional Elective Course", "type": "PEC", "credits": 3},
        {"code": "BCI586", "title": "Mini Project", "type": "PROJ", "credits": 2},
        {"code": "BRMK557", "title": "Research Methodology and IPR", "type": "AEC", "credits": 3},
        {"code": "BCS508", "title": "Environmental Studies and E-waste Management", "type": "HSMS", "credits": 2}
    ],

    "professional_electives": [
        {"code": "BAI515A", "title": "Computer Vision"},
        {"code": "BCS515C", "title": "Unix System Programming"},
        {"code": "BAI515B", "title": "Information Retrieval"},
        {"code": "BCI515D", "title": "Image and Video Processing"}
    ],

    "mandatory_course_choices": [
        {"code": "BNSK559", "title": "National Service Scheme (NSS)"},
        {"code": "BPEK559", "title": "Physical Education (PE) (Sports and Athletics)"},
        {"code": "BYOK559", "title": "Yoga"}
    ]
}

json_path = OUTPUT_DIR / "aiml_sem5_course_catalog.json"

with json_path.open("w", encoding="utf-8") as f:
    json.dump(catalogue, f, indent=4, ensure_ascii=False)

# Create searchable text records from the structured catalogue.
records = []

for course in catalogue["courses"]:
    records.append({
        "chunk_id": f"scheme_{course['code']}",
        "subject": "AIML Scheme",
        "source": catalogue["source"],
        "page": catalogue["source_page"],
        "text": (
            f"VTU B.E. CSE Artificial Intelligence and Machine Learning, "
            f"2022 scheme, Semester 5. Course code: {course['code']}. "
            f"Course title: {course['title']}. "
            f"Course category: {course['type']}. "
            f"Credits: {course['credits']}."
        )
    })

for elective in catalogue["professional_electives"]:
    records.append({
        "chunk_id": f"elective_{elective['code']}",
        "subject": "AIML Scheme",
        "source": catalogue["source"],
        "page": catalogue["source_page"],
        "text": (
            f"VTU Semester 5 professional elective option. "
            f"Course code: {elective['code']}. "
            f"Course title: {elective['title']}."
        )
    })

for choice in catalogue["mandatory_course_choices"]:
    records.append({
        "chunk_id": f"mandatory_{choice['code']}",
        "subject": "AIML Scheme",
        "source": catalogue["source"],
        "page": catalogue["source_page"],
        "text": (
            f"VTU Semester 5 mandatory course choice. "
            f"Course code: {choice['code']}. "
            f"Course title: {choice['title']}. "
            f"Students register for one of the listed alternatives."
        )
    })

chunks_path = OUTPUT_DIR / "scheme_catalog_chunks.jsonl"

with chunks_path.open("w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("=== COURSE CATALOGUE CREATED ===")
print("Core/semester course entries:", len(catalogue["courses"]))
print("Professional elective options:", len(catalogue["professional_electives"]))
print("Mandatory course choices:", len(catalogue["mandatory_course_choices"]))
print("Searchable records:", len(records))
print("JSON:", json_path)
print("Search chunks:", chunks_path)
