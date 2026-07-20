"""Parent selection strategies."""

from __future__ import annotations

import random
from typing import Protocol

from ..schemas.node import NodeRecord
from .archive import NodeArchive


class ParentSelector(Protocol):
    """Protocol for parent selection strategies."""

    def select(
        self,
        archive: NodeArchive,
        rng: random.Random,
        min_solved: int = 3,
    ) -> NodeRecord:
        ...


class EpsilonGreedySelector:
    """V1: epsilon-greedy over node_score."""

    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon

    def select(
        self,
        archive: NodeArchive,
        rng: random.Random,
        min_solved: int = 3,
    ) -> NodeRecord:
        eligible = archive.eligible_parents(min_solved)
        if not eligible:
            raise ValueError("No eligible parents in archive")

        if rng.random() < self.epsilon:
            return rng.choice(eligible)

        return max(eligible, key=lambda n: n.node_score or 0.0)


class ScoreProportionalSelector:
    """V2: score-proportional sampling."""

    def select(
        self,
        archive: NodeArchive,
        rng: random.Random,
        min_solved: int = 3,
    ) -> NodeRecord:
        eligible = archive.eligible_parents(min_solved)
        if not eligible:
            raise ValueError("No eligible parents in archive")

        scores = [max(n.node_score or 0.0, 0.001) for n in eligible]
        total = sum(scores)
        r = rng.random() * total
        cumulative = 0.0
        for node, score in zip(eligible, scores):
            cumulative += score
            if r <= cumulative:
                return node
        return eligible[-1]
