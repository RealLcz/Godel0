"""TaskProvider abstraction: decouples task source from the evolution loop.

The HGM-style outer loop calls ``task_provider.get_tasks(node, context)`` and
does not know whether tasks come from a static benchmark or from the node's own
evolvable RepoChain proposer. This mirrors the HGM design where the only
difference is ``BenchmarkTaskProvider`` vs ``ProposerTaskProvider``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Protocol, runtime_checkable

from ..schemas.node import NodeRecord
from ..schemas.task import TaskRecord


@dataclass
class TaskGenerationContext:
    """Everything the task provider needs to produce one batch.

    The orchestrator assembles this context; the provider never reaches back
    into orchestrator internals (no filesystem globs, no env vars). This is the
    seam that lets BenchmarkTaskProvider and ProposerTaskProvider be
    interchangeable.
    """

    node: NodeRecord
    parent: Optional[NodeRecord] = None
    level1_result: Optional[Any] = None

    # Phase 6 / BUG-08/09: trajectory sources are split so the 5+5 quota can be
    # enforced. parent_failure_trajectories holds the parent's Level2 unresolved
    # trajectories; current_child_level1_trajectories holds the current child's
    # Level1 unresolved/forgotten trajectories.
    parent_failure_trajectories: List[str] = field(default_factory=list)
    current_child_level1_trajectories: List[str] = field(default_factory=list)

    # Backwards-compatible flat trajectory list. Kept so legacy callers that
    # only have a single bucket still work; the provider prefers the split
    # buckets when they are populated.
    solver_trajectories: List[str] = field(default_factory=list)
    parent_task_ids: List[str] = field(default_factory=list)
    parent_solved_task_ids: List[str] = field(default_factory=list)

    run_id: str = "run"
    output_dir: Optional[Path] = None
    model: str = "deepseek/deepseek-chat"
    task_store_dir: str = "./task_store"
    bootstrap: bool = False


@dataclass
class TaskBatch:
    """Provider-agnostic result of one ``get_tasks`` call.

    Mirrors the fields the orchestrator previously read off ``TaskBatchResult``
    so the rest of the loop (Level 2, scoring, generation_summary.json) does not
    change.
    """

    batch_id: str
    node_id: str
    tasks: List[TaskRecord] = field(default_factory=list)
    complete: bool = False
    rejected_candidates: int = 0
    rejection_reasons: dict = field(default_factory=dict)
    candidates_generated: int = 0
    candidates_validated: int = 0
    validation_reports: List[dict] = field(default_factory=list)
    proposer_error: str = ""
    engine_rejections: List[dict] = field(default_factory=list)
    # P1-3: structured stage counters emitted by RepoChain / validator.
    repo_chain_stats: dict = field(default_factory=dict)


@runtime_checkable
class TaskProvider(Protocol):
    """Unified task source for the HGM-style evolution loop.

    HGM:  ``BenchmarkTaskProvider`` returns a fixed benchmark batch.
    Godel0: ``ProposerTaskProvider`` invokes the current node's RepoChain
    proposer subprocess and returns RepoChain-generated Coding Tasks.

    Both implementations return a ``TaskBatch`` whose ``tasks`` are
    ``TaskRecord`` objects with the same SWE-bench-style interface the solver
    already consumes.
    """

    def get_tasks(
        self,
        node: NodeRecord,
        context: TaskGenerationContext,
    ) -> TaskBatch:
        ...


__all__ = ["TaskGenerationContext", "TaskBatch", "TaskProvider"]
