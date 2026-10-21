import pytest

from src.knowledge_tracing import (
    BKTParameters,
    trace_skill,
    update_mastery,
)


def test_mastery_stays_between_zero_and_one():
    history = trace_skill([False, True, True, False, True])
    assert all(0 <= value <= 1 for value in history)


def test_history_has_one_value_per_response():
    responses = [False, True, True, False, True]
    assert len(trace_skill(responses)) == len(responses)


def test_correct_response_increases_mastery_more_than_incorrect():
    params = BKTParameters()
    prior = 0.5

    after_correct = update_mastery(prior, True, params)
    after_incorrect = update_mastery(prior, False, params)

    assert after_correct > after_incorrect


def test_empty_response_history_is_empty():
    assert trace_skill([]) == []


def test_invalid_prior_mastery_is_rejected():
    with pytest.raises(ValueError):
        update_mastery(1.5, True)


def test_non_boolean_response_is_rejected():
    with pytest.raises(TypeError):
        update_mastery(0.5, 1)


def test_invalid_probability_is_rejected():
    params = BKTParameters(initial_mastery=1.5)
    with pytest.raises(ValueError):
        update_mastery(0.5, True, params)
