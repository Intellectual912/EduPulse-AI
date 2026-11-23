"""Diagnostic question service for VTU BCS502 Computer Networks."""

import json
from pathlib import Path

from src.cn_knowledge_tracing import ConceptMasteryTracker


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"


class CNDiagnosticService:
    """Connect Computer Networks diagnostic questions with BKT tracking."""

    def __init__(self):
        concept_path = DATA_DIR / "cn_concept_catalog.json"
        question_path = DATA_DIR / "cn_diagnostic_questions.json"

        with concept_path.open(encoding="utf-8-sig") as file:
            concept_data = json.load(file)

        with question_path.open(encoding="utf-8-sig") as file:
            question_data = json.load(file)

        self.concepts = {
            item["concept_id"]: item
            for item in concept_data["concepts"]
        }

        self.questions = question_data["questions"]
        self.tracker = ConceptMasteryTracker()
        self.attempted_questions = set()

        question_ids = [
            question["question_id"]
            for question in self.questions
        ]

        if len(question_ids) != len(set(question_ids)):
            raise ValueError("Duplicate question IDs found.")

        for question in self.questions:
            if question["concept_id"] not in self.concepts:
                raise ValueError(
                    f"Unknown concept: {question['concept_id']}"
                )

            if question["correct_option"] not in question["options"]:
                raise ValueError(
                    f"Invalid answer key: {question['question_id']}"
                )

    def get_questions(self, module=None):
        """Return questions without revealing their correct answers."""

        results = []

        for question in self.questions:
            concept = self.concepts[question["concept_id"]]

            if module is not None and concept["module"] != module:
                continue

            results.append({
                "question_id": question["question_id"],
                "concept_id": question["concept_id"],
                "concept_name": concept["concept_name"],
                "question": question["question"],
                "options": question["options"],
            })

        return results

    def submit_answer(self, question_id, selected_option):
        """Evaluate an answer and update the associated concept mastery."""

        question = next(
            (
                item for item in self.questions
                if item["question_id"] == question_id
            ),
            None,
        )

        if question is None:
            raise ValueError("Question ID not found.")

        if question_id in self.attempted_questions:
            raise ValueError("Question has already been attempted.")

        if selected_option not in question["options"]:
            raise ValueError("Invalid answer option.")

        correct = selected_option == question["correct_option"]
        concept_id = question["concept_id"]

        mastery = self.tracker.update(concept_id, correct)
        self.attempted_questions.add(question_id)

        return {
            "question_id": question_id,
            "concept_id": concept_id,
            "correct": correct,
            "correct_option": question["correct_option"],
            "explanation": question["explanation"],
            "mastery": mastery,
            "mastery_percentage": round(mastery * 100, 2),
        }

    def get_mastery_report(self):
        """Return the current mastery estimate for every concept."""

        report = []

        for concept_id, concept in self.concepts.items():
            mastery = self.tracker.get_mastery(concept_id)

            report.append({
                "concept_id": concept_id,
                "concept_name": concept["concept_name"],
                "module": concept["module"],
                "mastery": mastery,
                "mastery_percentage": round(mastery * 100, 2),
                "questions_attempted": sum(
                    1 for question in self.questions
                    if question["concept_id"] == concept_id
                    and question["question_id"] in self.attempted_questions
                ),
            })

        return report


    def get_prerequisite_report(self):
        """Assess prerequisite readiness and recommend learning actions."""

        minimum_attempts = 2
        mastery_threshold = 0.70
        report = []

        for concept_id, concept in self.concepts.items():
            prerequisite_results = []

            for prerequisite_id in concept["prerequisite_concept_ids"]:
                prerequisite = self.concepts[prerequisite_id]

                attempts = sum(
                    1 for question in self.questions
                    if question["concept_id"] == prerequisite_id
                    and question["question_id"] in self.attempted_questions
                )

                mastery = self.tracker.get_mastery(prerequisite_id)

                if attempts < minimum_attempts:
                    status = "Insufficient evidence"
                elif mastery < mastery_threshold:
                    status = "Needs attention"
                else:
                    status = "Ready"

                prerequisite_results.append({
                    "concept_id": prerequisite_id,
                    "concept_name": prerequisite["concept_name"],
                    "mastery_percentage": round(mastery * 100, 2),
                    "questions_attempted": attempts,
                    "status": status,
                })

            if not prerequisite_results:
                readiness = "No prerequisites"
                recommendation = "Start learning this concept."
            elif any(
                item["status"] == "Insufficient evidence"
                for item in prerequisite_results
            ):
                readiness = "Insufficient evidence"
                recommendation = (
                    "Complete more diagnostic questions for the "
                    "prerequisite concepts before assessing readiness."
                )
            elif any(
                item["status"] == "Needs attention"
                for item in prerequisite_results
            ):
                readiness = "Needs attention"
                names = [
                    item["concept_name"]
                    for item in prerequisite_results
                    if item["status"] == "Needs attention"
                ]
                recommendation = (
                    "Review these prerequisite concepts: "
                    + ", ".join(names)
                    + "."
                )
            else:
                readiness = "Ready"
                recommendation = (
                    "Prerequisites meet the demonstration readiness "
                    "criteria. Proceed to this concept."
                )

            report.append({
                "concept_id": concept_id,
                "concept_name": concept["concept_name"],
                "readiness": readiness,
                "prerequisites": prerequisite_results,
                "recommendation": recommendation,
            })

        return report
