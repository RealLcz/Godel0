"""Level 1 evaluator: retention gate on parent-solved tasks."""

from __future__ import annotations

from typing import List, Optional

from ..schemas.evaluation import EvaluationOutcome, Level1Result
from ..schemas.node import NodeRecord


class Level1Evaluator:
    """Evaluates whether a child node retains the parent's solved capabilities.

    Level 1 runs the Child Solver on the Parent's solved tasks (T_parent).
    The child must retain at least `regression_threshold` fraction of
    the parent's solved tasks.
    """

    def __init__(self, regression_threshold: float = 0.8):
        self.regression_threshold = regression_threshold

    def compute_retention(
        self,
        parent_solved_task_ids: List[str],
        child_outcomes: List[EvaluationOutcome],
    ) -> Level1Result:
        """Compute retention from child outcomes on parent tasks.

        Args:
            parent_solved_task_ids: Tasks the parent solver successfully solved.
            child_outcomes: Child solver's outcomes on those same tasks.

        Returns:
            Level1Result with retention_rate and passed flag.
        """
        if not parent_solved_task_ids:
            return Level1Result(
                parent_node_id="",
                child_node_id="",
                evaluated_task_ids=[],
                parent_solved_task_ids=[],
                child_retained_task_ids=[],
                child_forgotten_task_ids=[],
                child_newly_solved_task_ids=[],
                retention_rate=1.0,
                threshold=self.regression_threshold,
                passed=True,
            )

        outcome_map = {o.task_id: o for o in child_outcomes}
        evaluated = [o.task_id for o in child_outcomes]

        retained = []
        forgotten = []
        for tid in parent_solved_task_ids:
            outcome = outcome_map.get(tid)
            if outcome and outcome.resolved:
                retained.append(tid)
            else:
                forgotten.append(tid)

        retention_rate = len(retained) / len(parent_solved_task_ids)
        passed = retention_rate >= self.regression_threshold

        parent_node_id = ""
        child_node_id = ""
        if child_outcomes:
            child_node_id = child_outcomes[0].node_id

        return Level1Result(
            parent_node_id=parent_node_id,
            child_node_id=child_node_id,
            evaluated_task_ids=evaluated,
            parent_solved_task_ids=parent_solved_task_ids,
            child_retained_task_ids=retained,
            child_forgotten_task_ids=forgotten,
            child_newly_solved_task_ids=[],
            retention_rate=retention_rate,
            threshold=self.regression_threshold,
            passed=passed,
        )
