"""Parent selection strategies."""

from __future__ import annotations

import random
from typing import List, Optional, Protocol

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
    """V1: epsilon-greedy over node_score.

    Ablation-only. The main experiment must use ThompsonSamplingSelector so
    parent selection follows the HGM-style Beta posterior over utility
    measurements instead of an epsilon-greedy policy.
    """

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


def _pseudo_descendant_evals(
    node: NodeRecord,
    num_pseudo: int,
) -> List[float]:
    """HGM-style pseudo-descendant evaluations for a single node.

    P0-1: always summarise the node's trusted Level2 utilities by their mean
    and replicate ``num_pseudo`` times. That yields fractional observations
    such as ``[0.6] * 10`` which the Beta posterior must sum (not threshold).
    """
    own = list(getattr(node, "utility_measures", []) or [])
    if not own:
        return []
    mean = float(sum(own) / len(own))
    return [mean] * max(1, int(num_pseudo))


def descendant_evals(
    node: NodeRecord,
    archive: NodeArchive,
    num_pseudo: int,
) -> List[float]:
    """Aggregate a node's own + descendant utility measurements.

    The HGM-style descendant aggregation expands each node's Beta posterior
    with its descendants' outcomes so selection favors nodes whose subtree has
    demonstrated high utility.
    """
    evals = _pseudo_descendant_evals(node, num_pseudo)
    for desc in archive.descendants_of(node.node_id):
        evals.extend(list(getattr(desc, "utility_measures", []) or []))
    return evals


class ThompsonSamplingSelector:
    """HGM-style Thompson Sampling over descendant utility measurements.

    Each candidate node's Beta(alpha=1+successes, beta=1+failures) posterior is
    sampled and the node with the highest sampled theta is selected. Utility
    measurements come only from trusted Level2 outcomes (1.0 solved / 0.0
    unresolved). This is the default parent selector for the main experiment;
    EpsilonGreedySelector is kept for ablations only.
    """

    def __init__(self, num_pseudo_descendant_evals: int = 10):
        self.num_pseudo = max(1, int(num_pseudo_descendant_evals))

    def select(
        self,
        archive: NodeArchive,
        rng: random.Random,
        min_solved: int = 3,
    ) -> NodeRecord:
        eligible = archive.eligible_parents(min_solved)
        if not eligible:
            raise ValueError("No eligible parents in archive")

        best_node: Optional[NodeRecord] = None
        best_theta = -1.0
        for node in eligible:
            evals = descendant_evals(node, archive, self.num_pseudo)
            if not evals:
                alpha = 1.0
                beta = 1.0
            else:
                # P0-1: fractional posterior. Pseudo-descendant evals are
                # mean-replicated floats (e.g. [0.6]*10). Counting only
                # ``value >= 1.0`` treats every 0.6 as a full failure and
                # collapses Beta(7,5) -> Beta(1,11). Sum the fractional
                # observations directly so 0.6 x 10 => success=6, failure=4.
                successes = float(sum(evals))
                failures = float(len(evals)) - successes
                alpha = 1.0 + successes
                beta = 1.0 + max(0.0, failures)
            theta = rng.betavariate(alpha, beta)
            if theta > best_theta:
                best_theta = theta
                best_node = node
        assert best_node is not None
        return best_node
