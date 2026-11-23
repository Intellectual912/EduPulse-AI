import pytest

from src.cn_knowledge_tracing import ConceptMasteryTracker


def test_initial_mastery():
    tracker = ConceptMasteryTracker()
    assert tracker.get_mastery("CN_NETWORKS") == 0.2


def test_correct_answer_increases_mastery():
    tracker = ConceptMasteryTracker()
    initial = tracker.get_mastery("CN_NETWORKS")

    updated = tracker.update("CN_NETWORKS", True)

    assert updated > initial


def test_incorrect_answer_decreases_mastery():
    tracker = ConceptMasteryTracker()
    tracker.update("CN_NETWORKS", True)

    before = tracker.get_mastery("CN_NETWORKS")
    after = tracker.update("CN_NETWORKS", False)

    assert after < before


def test_concepts_have_independent_mastery():
    tracker = ConceptMasteryTracker()

    tracker.update("CN_NETWORKS", True)

    assert tracker.get_mastery("CN_NETWORKS") > 0.2
    assert tracker.get_mastery("CN_TCP") == 0.2


def test_empty_concept_id_is_rejected():
    tracker = ConceptMasteryTracker()

    with pytest.raises(ValueError):
        tracker.update("", True)


def test_non_boolean_response_is_rejected():
    tracker = ConceptMasteryTracker()

    with pytest.raises(TypeError):
        tracker.update("CN_NETWORKS", 1)
