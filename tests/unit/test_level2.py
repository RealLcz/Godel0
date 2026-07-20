"""Unit tests for Level 2 evaluator."""

from __future__ import annotations

import pytest

from godel0.evaluation.level2 import Level2Evaluator
from godel0.schemas.evaluation import EvaluationOutcome


class TestLevel2:
    def test_all_solved(self):
        evaluator = Level2Evaluator()
        outcomes = [
            EvaluationOutcome(node_id="node", task_id="t1", level=2, resolved=True, trajectory_id="tr1"),
            EvaluationOutcome(node_id="node", task_id="t2", level=2, resolved=True, trajectory_id="tr2"),
            EvaluationOutcome(node_id="node", task_id="t3", level=2, resolved=True, trajectory_id="tr3"),
        ]
        result = evaluator.compute_accuracy("node", "batch_1", outcomes)
        assert result.accuracy == 1.0
        assert len(result.solved_task_ids) == 3
        assert len(result.failed_task_ids) == 0

    def test_half_solved(self):
        evaluator = Level2Evaluator()
        outcomes = [
            EvaluationOutcome(node_id="node", task_id="t1", level=2, resolved=True, trajectory_id="tr1"),
            EvaluationOutcome(node_id="node", task_id="t2", level=2, resolved=False, trajectory_id="tr2"),
        ]
        result = evaluator.compute_accuracy("node", "batch_1", outcomes)
        assert result.accuracy == 0.5
        assert len(result.solved_task_ids) == 1
        assert len(result.failed_task_ids) == 1

    def test_none_solved(self):
        evaluator = Level2Evaluator()
        outcomes = [
            EvaluationOutcome(node_id="node", task_id="t1", level=2, resolved=False, trajectory_id="tr1"),
            EvaluationOutcome(node_id="node", task_id="t2", level=2, resolved=False, trajectory_id="tr2"),
        ]
        result = evaluator.compute_accuracy("node", "batch_1", outcomes)
        assert result.accuracy == 0.0
        assert len(result.failed_task_ids) == 2

    def test_empty_outcomes(self):
        evaluator = Level2Evaluator()
        result = evaluator.compute_accuracy("node", "batch_1", [])
        assert result.accuracy == 0.0
