"""Unit tests for Level 1 evaluator."""

from __future__ import annotations

import pytest

from godel0.evaluation.level1 import Level1Evaluator
from godel0.schemas.evaluation import EvaluationOutcome


class TestLevel1:
    def test_full_retention(self):
        evaluator = Level1Evaluator(regression_threshold=0.8)
        parent_solved = ["task_1", "task_2", "task_3"]
        outcomes = [
            EvaluationOutcome(node_id="child", task_id="task_1", level=1, resolved=True, trajectory_id="t1"),
            EvaluationOutcome(node_id="child", task_id="task_2", level=1, resolved=True, trajectory_id="t2"),
            EvaluationOutcome(node_id="child", task_id="task_3", level=1, resolved=True, trajectory_id="t3"),
        ]
        result = evaluator.compute_retention(parent_solved, outcomes)
        assert result.retention_rate == 1.0
        assert result.passed
        assert len(result.child_retained_task_ids) == 3
        assert len(result.child_forgotten_task_ids) == 0

    def test_partial_retention_pass(self):
        evaluator = Level1Evaluator(regression_threshold=0.8)
        parent_solved = ["task_1", "task_2", "task_3", "task_4", "task_5"]
        outcomes = [
            EvaluationOutcome(node_id="child", task_id="task_1", level=1, resolved=True, trajectory_id="t1"),
            EvaluationOutcome(node_id="child", task_id="task_2", level=1, resolved=True, trajectory_id="t2"),
            EvaluationOutcome(node_id="child", task_id="task_3", level=1, resolved=True, trajectory_id="t3"),
            EvaluationOutcome(node_id="child", task_id="task_4", level=1, resolved=True, trajectory_id="t4"),
            EvaluationOutcome(node_id="child", task_id="task_5", level=1, resolved=False, trajectory_id="t5"),
        ]
        result = evaluator.compute_retention(parent_solved, outcomes)
        assert result.retention_rate == 0.8
        assert result.passed
        assert len(result.child_forgotten_task_ids) == 1

    def test_retention_fail(self):
        evaluator = Level1Evaluator(regression_threshold=0.8)
        parent_solved = ["task_1", "task_2", "task_3", "task_4", "task_5"]
        outcomes = [
            EvaluationOutcome(node_id="child", task_id="task_1", level=1, resolved=True, trajectory_id="t1"),
            EvaluationOutcome(node_id="child", task_id="task_2", level=1, resolved=True, trajectory_id="t2"),
            EvaluationOutcome(node_id="child", task_id="task_3", level=1, resolved=False, trajectory_id="t3"),
            EvaluationOutcome(node_id="child", task_id="task_4", level=1, resolved=False, trajectory_id="t4"),
            EvaluationOutcome(node_id="child", task_id="task_5", level=1, resolved=False, trajectory_id="t5"),
        ]
        result = evaluator.compute_retention(parent_solved, outcomes)
        assert result.retention_rate == 0.4
        assert not result.passed
        assert len(result.child_forgotten_task_ids) == 3

    def test_empty_parent_solved(self):
        evaluator = Level1Evaluator()
        result = evaluator.compute_retention([], [])
        assert result.retention_rate == 1.0
        assert result.passed
