import pytest

from src.cn_diagnostic_service import CNDiagnosticService


def test_service_loads_questions_and_concepts():
    service = CNDiagnosticService()

    assert len(service.get_questions()) == 53
    assert len(service.get_mastery_report()) == 17
    assert len(service.get_questions(module=1)) == 14


def test_questions_do_not_reveal_correct_answers():
    service = CNDiagnosticService()

    for question in service.get_questions():
        assert "correct_option" not in question


def test_correct_answer_increases_mastery():
    service = CNDiagnosticService()

    result = service.submit_answer("CN_Q001", "A")

    assert result["correct"] is True
    assert result["mastery_percentage"] == 60.0


def test_incorrect_answer_decreases_mastery():
    service = CNDiagnosticService()

    result = service.submit_answer("CN_Q001", "B")

    assert result["correct"] is False
    assert result["mastery_percentage"] < 20.0


def test_mastery_report_records_attempt():
    service = CNDiagnosticService()

    service.submit_answer("CN_Q001", "A")

    report = service.get_mastery_report()

    assert report[0]["questions_attempted"] == 1


def test_unknown_question_is_rejected():
    service = CNDiagnosticService()

    with pytest.raises(ValueError):
        service.submit_answer("INVALID_ID", "A")


def test_invalid_option_is_rejected():
    service = CNDiagnosticService()

    with pytest.raises(ValueError):
        service.submit_answer("CN_Q001", "Z")

def test_duplicate_question_attempt_is_rejected():
    service = CNDiagnosticService()

    service.submit_answer("CN_Q001", "A")

    with pytest.raises(ValueError, match="already been attempted"):
        service.submit_answer("CN_Q001", "B")

def test_module_2_has_32_questions_and_all_coverage_questions():
    from src.cn_diagnostic_service import CNDiagnosticService

    service = CNDiagnosticService()
    questions = service.get_questions(module=2)
    question_ids = [question["question_id"] for question in questions]

    assert len(questions) == 32
    assert question_ids[0] == "CN_Q022"
    assert question_ids[-1] == "CN_Q053"
    assert {"CN_Q050", "CN_Q051", "CN_Q052", "CN_Q053"}.issubset(set(question_ids))

