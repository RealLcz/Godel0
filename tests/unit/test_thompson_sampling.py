"""Unit tests for the Thompson Sampling selector (BUG-10/11/12)."""
from __future__ import annotations

import random

import pytest

from godel0.schemas.node import NodeRecord, NodeStatus
from godel0.tree.archive import NodeArchive
from godel0.tree.selection import (
    EpsilonGreedySelector,
    ThompsonSamplingSelector,
    descendant_evals,
)


def _make_node(
    node_id: str,
    utilities,
    parent: str = None,
    eligible: bool = True,
    score: float = 0.5,
    solved_count: int = 5,
) -> NodeRecord:
    return NodeRecord(
        node_id=node_id,
        code_commit="abc123",
        code_ref=f"refs/godel0/nodes/{node_id}",
        parent_node_id=parent,
        status=NodeStatus.COMPLETE,
        solver_score=score,
        node_score=score,
        retention_rate=0.9,
        frontier_accuracy=0.5,
        utility_measures=list(utilities),
        selection_eligible=eligible,
        solved_task_count=solved_count,
        generated_task_batch_id=f"batch_{node_id}",
    )


class TestThompsonSamplingSelector:
    def test_raises_when_no_eligible_parents(self, node_archive):
        node_archive.add(_make_node("n1", [1.0], eligible=False))
        selector = ThompsonSamplingSelector()
        rng = random.Random(42)
        with pytest.raises(ValueError):
            selector.select(node_archive, rng, min_solved=0)

    def test_returns_a_parent_when_eligible(self, node_archive):
        node_archive.add(_make_node("n1", [1.0, 1.0], eligible=True))
        selector = ThompsonSamplingSelector()
        rng = random.Random(42)
        result = selector.select(node_archive, rng, min_solved=0)
        assert result is not None
        assert result.node_id == "n1"

    def test_prefers_higher_utility_parent_on_average(self, node_archive):
        strong = _make_node("strong", [1.0] * 10, eligible=True)
        weak = _make_node("weak", [0.0] * 10, eligible=True)
        node_archive.add(strong)
        node_archive.add(weak)
        selector = ThompsonSamplingSelector()
        rng = random.Random(0)
        counts = {"strong": 0, "weak": 0}
        for _ in range(200):
            picked = selector.select(node_archive, rng, min_solved=0)
            if picked is not None:
                counts[picked.node_id] += 1
        assert counts["strong"] > counts["weak"]

    def test_descendant_evals_includes_descendant_utilities(self, node_archive):
        parent = _make_node("parent", [1.0, 0.0])
        node_archive.add(parent)
        child = _make_node("child", [1.0, 1.0], parent="parent")
        node_archive.add(child)
        evals = descendant_evals(parent, node_archive, num_pseudo=10)
        # parent's own (2 evals) + child's (2 evals) = 4
        assert len(evals) == 4
        assert sum(evals) == 3.0  # 1+0+1+1


class TestScorerHGMGate:
    def test_hgm_mode_ineligible_when_valid_yield_below_threshold(self):
        from godel0.controller.scorer import compute_scores

        result = compute_scores(
            retention_rate=1.0,
            frontier_accuracy=0.5,
            regression_weight=0.5,
            mode="hgm",
            valid_yield=0.1,  # below default 0.20
            causal_ablation_pass=1.0,
            batch_complete=True,
        )
        assert not result.eligible

    def test_hgm_mode_eligible_when_all_gates_pass(self):
        from godel0.controller.scorer import compute_scores

        result = compute_scores(
            retention_rate=1.0,
            frontier_accuracy=0.5,
            regression_weight=0.5,
            mode="hgm",
            valid_yield=0.5,
            causal_ablation_pass=0.8,
            batch_complete=True,
        )
        assert result.eligible
        # In HGM mode node_score = a (not a*b)
        assert result.node_score == result.solver_score

    def test_hgm_mode_ineligible_when_batch_incomplete(self):
        from godel0.controller.scorer import compute_scores

        result = compute_scores(
            retention_rate=1.0,
            frontier_accuracy=0.5,
            regression_weight=0.5,
            mode="hgm",
            valid_yield=1.0,
            causal_ablation_pass=1.0,
            batch_complete=False,
        )
        assert not result.eligible
