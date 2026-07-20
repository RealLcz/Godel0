"""Scorer: computes a, b, and node_score from retention and frontier accuracy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreBundle:
    retention_rate: float
    frontier_accuracy: float
    solver_score: float
    proposer_score: float
    node_score: float


def compute_scores(
    *,
    retention_rate: float,
    frontier_accuracy: float,
    regression_weight: float,
    target_accuracy: float = 0.5,
) -> ScoreBundle:
    """Compute the node score.

    a = lambda * r + (1 - lambda) * p
    b = max(0, 1 - 2 * |p - target|)
    node_score = a * b

    Where:
        r = retention_rate (Level 1)
        p = frontier_accuracy (Level 2)
        lambda = regression_weight
    """
    r = max(0.0, min(1.0, retention_rate))
    p = max(0.0, min(1.0, frontier_accuracy))
    lam = max(0.0, min(1.0, regression_weight))

    a = lam * r + (1.0 - lam) * p
    b = max(0.0, 1.0 - 2.0 * abs(p - target_accuracy))
    node_score = a * b

    return ScoreBundle(
        retention_rate=r,
        frontier_accuracy=p,
        solver_score=a,
        proposer_score=b,
        node_score=node_score,
    )
