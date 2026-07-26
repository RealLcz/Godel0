"""TaskStore: persistent storage for validated tasks."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import List, Optional

from ..schemas.task import TaskRecord
from ..storage.atomic import atomic_write_json, read_json


class TaskArtifacts:
    """Container for task file artifacts."""

    def __init__(
        self,
        problem_statement: str = "",
        bug_patch: str = "",
        oracle_patch: str = "",
        setup_patch: str = "",
        validation_report: Optional[dict] = None,
        failing_test_output: str = "",
        f2p_tests: Optional[list] = None,
        generation_context: Optional[dict] = None,
    ):
        self.problem_statement = problem_statement
        self.bug_patch = bug_patch
        self.oracle_patch = oracle_patch
        self.setup_patch = setup_patch
        self.validation_report = validation_report or {}
        self.failing_test_output = failing_test_output
        self.f2p_tests = f2p_tests or []
        self.generation_context = generation_context or {}


class TaskStore:
    """Persistent, immutable task store.

    Tasks are stored as directories:
        task_store/<task_id>/
            task.json
            problem_statement.md
            bug.patch
            oracle_reverse.patch
            validation.json
            failing_test_output.txt
            private/
                f2p_tests.json
                generation_context.json
            hashes.json
    """

    def __init__(self, store_dir: Path):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def put(self, record: TaskRecord, artifacts: TaskArtifacts) -> TaskRecord:
        """Commit a new task to the store. Task becomes immutable."""
        task_dir = self.store_dir / record.task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        private_dir = task_dir / "private"
        private_dir.mkdir(exist_ok=True)

        atomic_write_json(task_dir / "task.json", record.model_dump())

        (task_dir / "problem_statement.md").write_text(artifacts.problem_statement)
        (task_dir / "bug.patch").write_text(artifacts.bug_patch)

        if artifacts.oracle_patch:
            (task_dir / "oracle_reverse.patch").write_text(artifacts.oracle_patch)
        else:
            (task_dir / "oracle_reverse.patch").write_text("")

        (private_dir / "setup.patch").write_text(artifacts.setup_patch)

        atomic_write_json(task_dir / "validation.json", artifacts.validation_report)
        (task_dir / "failing_test_output.txt").write_text(artifacts.failing_test_output)

        atomic_write_json(private_dir / "f2p_tests.json", artifacts.f2p_tests)
        atomic_write_json(private_dir / "generation_context.json", artifacts.generation_context)

        hashes = {
            "bug_patch_sha256": hashlib.sha256(artifacts.bug_patch.encode()).hexdigest(),
            "problem_statement_sha256": hashlib.sha256(artifacts.problem_statement.encode()).hexdigest(),
        }
        atomic_write_json(task_dir / "hashes.json", hashes)

        record.content_hash = hashes["bug_patch_sha256"]
        return record

    def get(self, task_id: str) -> Optional[TaskRecord]:
        """Retrieve a task record by ID."""
        task_json = self.store_dir / task_id / "task.json"
        if not task_json.exists():
            return None
        data = read_json(task_json)
        return TaskRecord(**data)

    def materialize_public(self, task_id: str, destination: Path) -> None:
        """Copy public task artifacts to a destination directory."""
        task_dir = self.store_dir / task_id
        if not task_dir.exists():
            raise FileNotFoundError(f"Task not found: {task_id}")

        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        for f in ["task.json", "problem_statement.md", "bug.patch"]:
            src = task_dir / f
            if src.exists():
                shutil.copy2(src, destination / f)

    def materialize_private(self, task_id: str, destination: Path) -> None:
        """Copy private task artifacts (only for trusted evaluator)."""
        task_dir = self.store_dir / task_id / "private"
        if not task_dir.exists():
            raise FileNotFoundError(f"Task private dir not found: {task_id}")

        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)

        for f in ["f2p_tests.json", "generation_context.json", "setup.patch"]:
            src = task_dir / f
            if src.exists():
                shutil.copy2(src, destination / f)

    def get_f2p_tests(self, task_id: str) -> List[str]:
        """Get the F2P test list for a task (trusted only)."""
        f2p_path = self.store_dir / task_id / "private" / "f2p_tests.json"
        if not f2p_path.exists():
            return []
        return read_json(f2p_path)

    def get_bug_patch(self, task_id: str) -> str:
        """Get the public bug patch for a task."""
        path = self.store_dir / task_id / "bug.patch"
        if not path.exists():
            return ""
        return path.read_text()

    def get_problem_statement(self, task_id: str) -> str:
        """Get the public problem statement for a task."""
        path = self.store_dir / task_id / "problem_statement.md"
        if not path.exists():
            return ""
        return path.read_text()

    def get_setup_patch(self, task_id: str) -> str:
        """Get trusted generated-test/setup material for evaluator-only use."""
        path = self.store_dir / task_id / "private" / "setup.patch"
        return path.read_text() if path.exists() else ""

    def all_task_ids(self) -> List[str]:
        """List all task IDs in the store."""
        if not self.store_dir.exists():
            return []
        return sorted([
            d.name for d in self.store_dir.iterdir()
            if d.is_dir() and (d / "task.json").exists()
        ])

    def tasks_for_batch(self, batch_id: str) -> List[TaskRecord]:
        """List task records that belong to a batch."""
        tasks = []
        for task_id in self.all_task_ids():
            task = self.get(task_id)
            if task is not None and task.batch_id == batch_id:
                tasks.append(task)
        return tasks

    def tasks_for_proposer_node(self, proposer_node_id: str) -> List[TaskRecord]:
        """List every committed task emitted by a proposer node."""
        tasks: List[TaskRecord] = []
        for task_id in self.all_task_ids():
            task = self.get(task_id)
            if task is not None and task.proposer_node_id == proposer_node_id:
                tasks.append(task)
        return tasks

    def latest_batch_for_node(
        self,
        proposer_node_id: str,
        *,
        prefer_incomplete_below: Optional[int] = None,
    ) -> Optional[str]:
        """Recover the batch_id a crashed proposer was filling.

        Tasks are committed incrementally before the archive learns the
        batch id (job 216679 left ``generated_task_batch_id=null`` while
        five ``batch_097216cb`` tasks already sat in the store). Prefer an
        incomplete batch when ``prefer_incomplete_below`` is set (typically
        the configured ``batch_size``); otherwise return the newest batch.
        """
        by_batch: dict[str, List[TaskRecord]] = {}
        for task in self.tasks_for_proposer_node(proposer_node_id):
            if not task.batch_id:
                continue
            by_batch.setdefault(task.batch_id, []).append(task)
        if not by_batch:
            return None

        def _newest_key(batch_id: str) -> tuple:
            tasks = by_batch[batch_id]
            stamps = [getattr(t, "created_at", None) for t in tasks]
            stamps = [s for s in stamps if s is not None]
            newest = max(stamps) if stamps else None
            return (newest is not None, newest, len(tasks), batch_id)

        if prefer_incomplete_below is not None:
            incomplete = [
                batch_id
                for batch_id, tasks in by_batch.items()
                if 0 < len(tasks) < int(prefer_incomplete_below)
            ]
            if incomplete:
                return max(incomplete, key=_newest_key)

        return max(by_batch.keys(), key=_newest_key)

    def exists(self, task_id: str) -> bool:
        """Check if a task exists."""
        return (self.store_dir / task_id / "task.json").exists()
