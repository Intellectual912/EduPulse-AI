"""Multi-concept Bayesian Knowledge Tracing for VTU BCS502.

This module tracks each concept independently. Concept mappings and
diagnostic questions must be curated and reviewed separately.
"""

from dataclasses import dataclass, field

from src.knowledge_tracing import BKTParameters, update_mastery


@dataclass
class ConceptMasteryTracker:
    """Maintain separate BKT mastery estimates for multiple concepts."""

    parameters: BKTParameters = field(default_factory=BKTParameters)
    mastery: dict[str, float] = field(default_factory=dict)

    def update(self, concept_id: str, correct: bool) -> float:
        """Update mastery for one concept after one response."""

        if not isinstance(concept_id, str) or not concept_id.strip():
            raise ValueError("concept_id must be a non-empty string.")

        if not isinstance(correct, bool):
            raise TypeError("correct must be True or False.")

        self.parameters.validate()

        prior = self.mastery.get(
            concept_id,
            self.parameters.initial_mastery
        )

        updated = update_mastery(
            prior,
            correct,
            self.parameters
        )

        self.mastery[concept_id] = round(updated, 4)
        return self.mastery[concept_id]

    def get_mastery(self, concept_id: str) -> float:
        """Return mastery for a concept, using initial mastery if unseen."""

        if not isinstance(concept_id, str) or not concept_id.strip():
            raise ValueError("concept_id must be a non-empty string.")

        return self.mastery.get(
            concept_id,
            self.parameters.initial_mastery
        )
