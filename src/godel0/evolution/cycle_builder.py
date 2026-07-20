"""Node cycle summary builder."""

from __future__ import annotations

from typing import List, Optional

from ..schemas.cycle import CycleStage, NodeCycleSummary
from ..schemas.evaluation import EvaluationOutcome, Level1Result, Level2Result
from ..schemas.node import NodeRecord


class NodeCycleBuilder:
    """Builds complete or partial cycle summaries.

    The cycle builder does NOT call any LLM. It only performs deterministic
    statistical aggregation.
    """

    def build(
        self,
        node: NodeRecord,
        level1: Optional[Level1Result] = None,
        proposer_stats: Optional[dict] = None,
        level2: Optional[Level2Result] = None,
        is_root: bool = False,
    ) -> NodeCycleSummary:
        """Build a cycle summary from available results."""
        if is_root:
            stage = CycleStage.ROOT_BOOTSTRAP
        elif level1 and not level1.passed:
            stage = CycleStage.LEVEL1_FAILED
        elif level2 is None:
            stage = CycleStage.PROPOSER_FAILED
        else:
            stage = CycleStage.LEVEL2_COMPLETE

        summary = NodeCycleSummary(
            node_id=node.node_id,
            parent_node_id=node.parent_node_id,
            stage_reached=stage,
        )

        if level1:
            summary.level1_retention = level1.retention_rate
            summary.forgotten_task_ids = level1.child_forgotten_task_ids
            summary.newly_solved_parent_task_ids = level1.child_newly_solved_task_ids

        if proposer_stats:
            summary.proposer_requested_tasks = proposer_stats.get("requested", 0)
            summary.proposer_generated_candidates = proposer_stats.get("generated", 0)
            summary.proposer_accepted_tasks = proposer_stats.get("accepted", 0)
            gen = summary.proposer_generated_candidates
            if gen > 0:
                summary.proposer_valid_yield = summary.proposer_accepted_tasks / gen
            summary.proposer_rejection_distribution = proposer_stats.get("rejections", {})
            summary.proposer_operator_distribution = proposer_stats.get("operators", {})

        if level2:
            summary.level2_accuracy = level2.accuracy
            summary.level2_solved_task_ids = level2.solved_task_ids
            summary.level2_failed_task_ids = level2.failed_task_ids

        summary.solver_score = node.solver_score
        summary.proposer_score = node.proposer_score
        summary.node_score = node.node_score

        return summary
