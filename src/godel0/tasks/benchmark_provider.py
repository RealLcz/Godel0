"""BenchmarkTaskProvider: replays a fixed batch of tasks from the TaskStore.

This is the HGM-baseline seam. HGM uses a static benchmark; here the "benchmark"
is a previously committed, trusted-valid batch in the TaskStore. The provider
replays that batch against any node, so the orchestrator loop can run in
HGM-benchmark mode without ever invoking the proposer.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

from ..schemas.node import NodeRecord
from ..schemas.task import TaskRecord
from .provider import TaskBatch, TaskGenerationContext, TaskProvider


class BenchmarkTaskProvider:
    """Replay a fixed TaskStore batch as the task source.

    Args:
        task_store: TaskStore with committed tasks.
        benchmark_batch_id: The batch to replay. If None, the provider replays
            the most recent batch in the store (useful for smoke tests).
    """

    def __init__(self, task_store, benchmark_batch_id: Optional[str] = None):
        self.task_store = task_store
        self.benchmark_batch_id = benchmark_batch_id

    def get_tasks(
        self,
        node: NodeRecord,
        context: TaskGenerationContext,
    ) -> TaskBatch:
        if self.benchmark_batch_id:
            tasks = self.task_store.tasks_for_batch(self.benchmark_batch_id)
        else:
            tasks = []
            for task_id in self.task_store.all_task_ids():
                record = self.task_store.get(task_id)
                if record is not None:
                    tasks.append(record)

        replay_batch_id = f"benchmark_{uuid.uuid4().hex[:8]}"
        return TaskBatch(
            batch_id=replay_batch_id,
            node_id=node.node_id,
            tasks=list(tasks),
            complete=bool(tasks),
            rejected_candidates=0,
            rejection_reasons={},
            candidates_generated=len(tasks),
            candidates_validated=len(tasks),
            validation_reports=[],
            proposer_error="",
            engine_rejections=[],
        )


__all__ = ["BenchmarkTaskProvider"]
