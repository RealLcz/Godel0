"""Scorer: computes a, b, and node_score from retention and frontier accuracy.

Phase 8: Scoring Ablation supports two modes:
  - "joint" (default, Godel0 original): node_score = a * b
  - "hgm" (HGM-style): solver_score = a, node_score = a; b is an eligibility
    gate only (not multiplied into the score).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ScoreBundle:
    retention_rate: float
    frontier_accuracy: float
    solver_score: float
    proposer_score: float
    node_score: float
    # HGM-mode eligibility gate (True iff the proposer quality gate passed).
    # In "joint" mode this is always True (no gate).
    eligible: bool = True


def compute_scores(
    *,
    retention_rate: float,
    frontier_accuracy: float,
    regression_weight: float,
    target_accuracy: float = 0.5,
    mode: str = "joint",
    valid_yield: Optional[float] = None,
    causal_ablation_pass: Optional[float] = None,
    batch_complete: bool = True,
    hgm_valid_yield_threshold: float = 0.20,
    hgm_causal_ablation_pass_threshold: float = 0.50,
    hgm_difficulty_min: float = 0.30,
) -> ScoreBundle:
    """Compute the node score.

    a = lambda * r + (1 - lambda) * p
    b = max(0, 1 - 2 * |p - target|)

    Joint mode:  node_score = a * b
    HGM mode:    node_score = a; b is an eligibility gate only.

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

    if mode == "hgm":
        # HGM gate: b is not multiplied; instead the proposer must pass
        # quality gates (valid_yield, causal_ablation, difficulty, batch).
        eligible = True
        if valid_yield is not None and valid_yield < hgm_valid_yield_threshold:
            eligible = False
        if causal_ablation_pass is not None and causal_ablation_pass < hgm_causal_ablation_pass_threshold:
            eligible = False
        # Difficulty gate: p must be in a reasonable range. Too high (too easy)
        # or too low (too hard) fails the gate.
        if p > 1.0 - hgm_difficulty_min:
            eligible = False
        if not batch_complete:
            eligible = False
        node_score = a
        return ScoreBundle(
            retention_rate=r,
            frontier_accuracy=p,
            solver_score=a,
            proposer_score=b,
            node_score=node_score,
            eligible=eligible,
        )

    # Joint mode (default): node_score = a * b
    node_score = a * b
    return ScoreBundle(
        retention_rate=r,
        frontier_accuracy=p,
        solver_score=a,
        proposer_score=b,
        node_score=node_score,
        eligible=True,
    )
