"""Unit tests for the scorer."""

from __future__ import annotations

import pytest

from godel0.controller.scorer import compute_scores


class TestScorer:
    def test_perfect_scores(self):
        """r=1.0, p=0.5 -> a=0.75, b=1.0, ab=0.75"""
        result = compute_scores(
            retention_rate=1.0,
            frontier_accuracy=0.5,
            regression_weight=0.5,
        )
        assert result.retention_rate == 1.0
        assert result.frontier_accuracy == 0.5
        assert result.solver_score == 0.75
        assert result.proposer_score == 1.0
        assert result.node_score == 0.75

    def test_balanced_scores(self):
        """r=0.8, p=0.5 -> a=0.65, b=1.0, ab=0.65"""
        result = compute_scores(
            retention_rate=0.8,
            frontier_accuracy=0.5,
            regression_weight=0.5,
        )
        assert result.solver_score == 0.65
        assert result.proposer_score == 1.0
        assert result.node_score == 0.65

    def test_zero_retention(self):
        """r=0, p=0.5 -> a=0.25, b=1.0, ab=0.25"""
        result = compute_scores(
            retention_rate=0.0,
            frontier_accuracy=0.5,
            regression_weight=0.5,
        )
        assert result.solver_score == 0.25
        assert result.node_score == 0.25

    def test_extreme_accuracy(self):
        """p=1.0 -> b=0 (too easy)"""
        result = compute_scores(
            retention_rate=1.0,
            frontier_accuracy=1.0,
            regression_weight=0.5,
        )
        assert result.proposer_score == 0.0
        assert result.node_score == 0.0

    def test_extreme_difficulty(self):
        """p=0.0 -> b=0 (too hard)"""
        result = compute_scores(
            retention_rate=1.0,
            frontier_accuracy=0.0,
            regression_weight=0.5,
        )
        assert result.proposer_score == 0.0
        assert result.node_score == 0.0

    def test_clamping(self):
        """Values outside [0,1] should be clamped."""
        result = compute_scores(
            retention_rate=1.5,
            frontier_accuracy=-0.5,
            regression_weight=0.5,
        )
        assert result.retention_rate == 1.0
        assert result.frontier_accuracy == 0.0

    def test_weight_extremes(self):
        """weight=0 -> a=p, weight=1 -> a=r"""
        r = 0.8
        p = 0.6

        result0 = compute_scores(retention_rate=r, frontier_accuracy=p, regression_weight=0.0)
        assert abs(result0.solver_score - p) < 1e-10

        result1 = compute_scores(retention_rate=r, frontier_accuracy=p, regression_weight=1.0)
        assert abs(result1.solver_score - r) < 1e-10


class TestHGMMode:
    def test_hgm_mode_node_score_equals_solver_score(self):
        """In HGM mode, node_score = a (not a*b)."""
        result = compute_scores(
            retention_rate=1.0,
            frontier_accuracy=0.5,
            regression_weight=0.5,
            mode="hgm",
        )
        assert result.solver_score == 0.75
        assert result.node_score == 0.75  # not multiplied by b
        assert result.eligible is True

    def test_hgm_mode_b_zero_still_scores(self):
        """In HGM mode, even if b=0 (too easy), node_score = a."""
        result = compute_scores(
            retention_rate=1.0,
            frontier_accuracy=1.0,
            regression_weight=0.5,
            mode="hgm",
        )
        assert result.proposer_score == 0.0  # b = 0
        assert result.node_score == 1.0  # a, not a*b=0
        # But eligible is False because difficulty gate fails (p > threshold)
        assert result.eligible is False

    def test_hgm_mode_valid_yield_gate(self):
        """HGM gate fails when valid_yield is below threshold."""
        result = compute_scores(
            retention_rate=1.0,
            frontier_accuracy=0.5,
            regression_weight=0.5,
            mode="hgm",
            valid_yield=0.05,
            hgm_valid_yield_threshold=0.20,
        )
        assert result.eligible is False

    def test_hgm_mode_causal_ablation_gate(self):
        """HGM gate fails when causal_ablation_pass is below threshold."""
        result = compute_scores(
            retention_rate=1.0,
            frontier_accuracy=0.5,
            regression_weight=0.5,
            mode="hgm",
            causal_ablation_pass=0.10,
            hgm_causal_ablation_pass_threshold=0.50,
        )
        assert result.eligible is False

    def test_joint_mode_no_gate(self):
        """Joint mode never gates (eligible always True)."""
        result = compute_scores(
            retention_rate=1.0,
            frontier_accuracy=1.0,
            regression_weight=0.5,
            mode="joint",
        )
        assert result.eligible is True
        assert result.node_score == 0.0  # a*b = 1.0 * 0.0
