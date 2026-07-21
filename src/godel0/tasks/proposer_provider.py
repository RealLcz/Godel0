"""ProposerTaskProvider: default task source invoking the node's RepoChain proposer.

This is the Godel0 counterpart to HGM's BenchmarkTaskProvider. It hides the
RepoPool / CandidateValidator / TaskCommitter / NodeProposerRunner plumbing
behind the single ``get_tasks(node, context)`` call, so the orchestrator no
longer threads five internal collaborators into a 13-parameter builder method.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..schemas.node import NodeRecord
from .batch import TaskBatchBuilder, TaskBatchResult
from .node_proposer import NodeProposerRunner
from .provider import TaskBatch, TaskGenerationContext, TaskProvider


def _result_to_batch(result: TaskBatchResult) -> TaskBatch:
    return TaskBatch(
        batch_id=result.batch_id,
        node_id=result.node_id,
        tasks=list(result.tasks),
        complete=bool(result.complete),
        rejected_candidates=result.rejected_candidates,
        rejection_reasons=dict(result.rejection_reasons),
        candidates_generated=result.candidates_generated,
        candidates_validated=result.candidates_validated,
        validation_reports=list(result.validation_reports),
        proposer_error=result.proposer_error,
        engine_rejections=list(result.engine_rejections),
    )


class ProposerTaskProvider:
    """Default Godel0 task provider: invokes the current node's proposer.

    The provider owns the RepoPool, CandidateValidator, TaskCommitter and
    NodeProposerRunner. The orchestrator only calls ``get_tasks`` with a
    ``TaskGenerationContext``; it never knows about RepoChain internals.
    """

    def __init__(
        self,
        batch_builder: TaskBatchBuilder,
        repo_pool,
        validator,
        task_committer,
        proposer_runner: NodeProposerRunner,
        task_store_dir: str = "./task_store",
    ):
        self.batch_builder = batch_builder
        self.repo_pool = repo_pool
        self.validator = validator
        self.task_committer = task_committer
        self.proposer_runner = proposer_runner
        self.task_store_dir = task_store_dir

    def get_tasks(
        self,
        node: NodeRecord,
        context: TaskGenerationContext,
    ) -> TaskBatch:
        bound_runner = (
            self.proposer_runner.for_node(node)
            if hasattr(self.proposer_runner, "for_node")
            else self.proposer_runner
        )
        output_dir = Path(context.output_dir) if context.output_dir else None
        result = self.batch_builder.build_for_node(
            node_id=node.node_id,
            repo_pool=self.repo_pool,
            validator=self.validator,
            task_committer=self.task_committer,
            proposer_runner=bound_runner,
            solver_trajectories=list(context.solver_trajectories),
            parent_task_ids=list(context.parent_task_ids),
            output_dir=output_dir,
            agent_code_dir="",
            model=context.model,
            run_id=context.run_id,
            task_store_dir=self.task_store_dir,
            bootstrap=context.bootstrap,
            # BUG-08/09: forward the split trajectory buckets so the builder
            # can enforce the 5+5 quota and record provenance.
            parent_failure_trajectories=list(context.parent_failure_trajectories),
            current_child_level1_trajectories=list(context.current_child_level1_trajectories),
        )
        return _result_to_batch(result)


__all__ = ["ProposerTaskProvider"]
