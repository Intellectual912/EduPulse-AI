"""Bayesian Knowledge Tracing (BKT) for a single mapped skill.

This module implements the update logic only. It does not infer which
real-world concept an assessment measures; that mapping must be supplied
and verified separately.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class BKTParameters:
    """Probabilities governing learning and answering a skill question."""

    initial_mastery: float = 0.20
    learn_probability: float = 0.15
    guess_probability: float = 0.20
    slip_probability: float = 0.10

    def validate(self):
        values = (
            self.initial_mastery,
            self.learn_probability,
            self.guess_probability,
            self.slip_probability,
        )
        if any(not 0 <= value <= 1 for value in values):
            raise ValueError("All BKT probabilities must be between 0 and 1.")
        if self.guess_probability + self.slip_probability >= 1:
            raise ValueError(
                "Guess probability plus slip probability must be less than 1."
            )


def update_mastery(
    prior_mastery: float,
    correct: bool,
    parameters: BKTParameters | None = None,
) -> float:
    """Update mastery after one observed correct/incorrect response."""

    params = parameters or BKTParameters()
    params.validate()

    if not 0 <= prior_mastery <= 1:
        raise ValueError("prior_mastery must be between 0 and 1.")
    if not isinstance(correct, bool):
        raise TypeError("correct must be True or False.")

    if correct:
        likelihood_mastered = 1 - params.slip_probability
        likelihood_unmastered = params.guess_probability
    else:
        likelihood_mastered = params.slip_probability
        likelihood_unmastered = 1 - params.guess_probability

    numerator = prior_mastery * likelihood_mastered
    denominator = numerator + (
        (1 - prior_mastery) * likelihood_unmastered
    )

    if denominator == 0:
        posterior = prior_mastery
    else:
        posterior = numerator / denominator

    # Learning can occur after the observed response.
    updated = posterior + (
        (1 - posterior) * params.learn_probability
    )
    return min(1.0, max(0.0, updated))


def trace_skill(
    responses: list[bool],
    parameters: BKTParameters | None = None,
) -> list[float]:
    """Return the estimated mastery after each response."""

    params = parameters or BKTParameters()
    params.validate()
    mastery = params.initial_mastery
    history = []

    for response in responses:
        mastery = update_mastery(mastery, response, params)
        history.append(round(mastery, 4))

    return history
