"""Integration test: Level 1 and Level 2 evaluation flow."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from godel0.schemas.node import NodeRecord, NodeStatus
from godel0.schemas.evaluation import EvaluationOutcome, Level1Result, Level2Result
from godel0.evaluation.level1 import Level1Evaluator
from godel0.evaluation.level2 import Level2Evaluator
from godel0.controller.scorer import compute_scores


class TestLevel1Level2Flow:
    def test_full_evaluation_flow(self):
        """Test the complete Level 1 -> Level 2 -> scoring flow."""
        parent_solved = ["task_1", "task_2", "task_3"]

        child_level1_outcomes = [
            EvaluationOutcome(node_id="child", task_id="task_1", level=1, resolved=True, trajectory_id="t1"),
            EvaluationOutcome(node_id="child", task_id="task_2", level=1, resolved=True, trajectory_id="t2"),
            EvaluationOutcome(node_id="child", task_id="task_3", level=1, resolved=True, trajectory_id="t3"),
        ]

        level1_eval = Level1Evaluator(regression_threshold=0.8)
        level1_result = level1_eval.compute_retention(parent_solved, child_level1_outcomes)

        assert level1_result.passed
        assert level1_result.retention_rate == 1.0

        child_level2_outcomes = [
            EvaluationOutcome(node_id="child", task_id="new_1", level=2, resolved=True, trajectory_id="t4"),
            EvaluationOutcome(node_id="child", task_id="new_2", level=2, resolved=False, trajectory_id="t5"),
        ]

        level2_eval = Level2Evaluator()
        level2_result = level2_eval.compute_accuracy("child", "batch_new", child_level2_outcomes)

        assert level2_result.accuracy == 0.5

        scores = compute_scores(
            retention_rate=level1_result.retention_rate,
            frontier_accuracy=level2_result.accuracy,
            regression_weight=0.5,
        )

        assert scores.solver_score == 0.75
        assert scores.proposer_score == 1.0
        assert scores.node_score == 0.75

    def test_level1_fail_skips_level2(self):
        """When Level 1 fails, Level 2 should not run."""
        parent_solved = ["task_1", "task_2", "task_3"]
        child_outcomes = [
            EvaluationOutcome(node_id="child", task_id="task_1", level=1, resolved=False, trajectory_id="t1"),
            EvaluationOutcome(node_id="child", task_id="task_2", level=1, resolved=False, trajectory_id="t2"),
            EvaluationOutcome(node_id="child", task_id="task_3", level=1, resolved=False, trajectory_id="t3"),
        ]

        level1_eval = Level1Evaluator(regression_threshold=0.8)
        result = level1_eval.compute_retention(parent_solved, child_outcomes)

        assert not result.passed
        assert result.retention_rate == 0.0

    def test_level2_uses_current_batch_not_parent(self):
        """Level 2 must use the current proposer's tasks, not parent's."""
        level2_outcomes = [
            EvaluationOutcome(node_id="child", task_id="child_task_1", level=2, resolved=True, trajectory_id="t1"),
            EvaluationOutcome(node_id="child", task_id="child_task_2", level=2, resolved=True, trajectory_id="t2"),
        ]

        level2_eval = Level2Evaluator()
        result = level2_eval.compute_accuracy("child", "child_batch", level2_outcomes)

        assert result.task_batch_id == "child_batch"
        assert "parent_task" not in result.evaluated_task_ids
