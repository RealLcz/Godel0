"""Task committer: commits validated tasks to the TaskStore."""

from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path
from typing import Optional

from ..schemas.task import TaskRecord
from ..storage.atomic import atomic_write_json
from ..tasks.store import TaskArtifacts, TaskStore


class TaskCommitter:
    """Commits validated tasks to the TaskStore."""

    def __init__(self, task_store: TaskStore):
        self.task_store = task_store

    def commit_task(
        self,
        batch_id: str,
        proposer_node_id: str,
        repo_id: str,
        base_commit: str,
        bug_strategy: str,
        bug_patch: str,
        problem_statement: str,
        f2p_tests: list[str],
        baseline_test_command: str,
        failing_test_output: str = "",
        modified_files: list[str] = None,
        modified_entities: list[str] = None,
        validation_report: Optional[dict] = None,
        oracle_patch: str = "",
        setup_patch: str = "",
        solver_test_command: str = "",
        source_node: str = "",
        source_trajectory: str = "",
        source_type: str = "",
        source_node_id: str = "",
        source_trajectory_id: str = "",
        source_task_id: str = "",
        source_failure_stage: str = "",
    ) -> TaskRecord:
        """Create and commit a new task."""
        task_id = f"task_{uuid.uuid4().hex[:12]}"

        if not oracle_patch:
            oracle_patch = self._generate_reverse_patch(bug_patch)

        added_lines = bug_patch.count("\n+")
        deleted_lines = bug_patch.count("\n-")

        # P0-12: prefer explicit provenance fields; fall back to aliases.
        resolved_node_id = source_node_id or source_node
        resolved_traj_id = source_trajectory_id or source_trajectory

        record = TaskRecord(
            task_id=task_id,
            batch_id=batch_id,
            proposer_node_id=proposer_node_id,
            repo_id=repo_id,
            base_commit=base_commit,
            bug_strategy=bug_strategy,
            bug_patch_path=f"task_store/{task_id}/bug.patch",
            oracle_patch_path=f"task_store/{task_id}/oracle_reverse.patch",
            setup_patch_path=(f"task_store/{task_id}/private/setup.patch" if setup_patch else None),
            problem_statement_path=f"task_store/{task_id}/problem_statement.md",
            f2p_tests=f2p_tests,
            baseline_test_command=baseline_test_command,
            solver_test_command=solver_test_command or baseline_test_command,
            failing_test_output_path=f"task_store/{task_id}/failing_test_output.txt",
            modified_files=modified_files or [],
            modified_entities=modified_entities or [],
            patch_lines_added=added_lines,
            patch_lines_deleted=deleted_lines,
            execution_valid=True,
            trajectory_relevant=True,
            safety_valid=True,
            duplicate_valid=True,
            content_hash=hashlib.sha256(bug_patch.encode()).hexdigest(),
            source_node=resolved_node_id,
            source_trajectory=resolved_traj_id,
            source_type=source_type,
            source_node_id=resolved_node_id,
            source_trajectory_id=resolved_traj_id,
            source_task_id=source_task_id,
            source_failure_stage=source_failure_stage,
        )

        artifacts = TaskArtifacts(
            problem_statement=problem_statement,
            bug_patch=bug_patch,
            oracle_patch=oracle_patch,
            setup_patch=setup_patch,
            validation_report=validation_report or {},
            failing_test_output=failing_test_output,
            f2p_tests=f2p_tests,
        )

        return self.task_store.put(record, artifacts)

    def _generate_reverse_patch(self, patch: str) -> str:
        """Generate a syntactically valid reverse unified diff."""
        lines = patch.splitlines()
        reversed_lines = []
        index = 0
        hunk = re.compile(
            r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$"
        )
        while index < len(lines):
            line = lines[index]
            if line.startswith("--- ") and index + 1 < len(lines) and lines[index + 1].startswith("+++ "):
                old_path = line[4:]
                new_path = lines[index + 1][4:]
                reversed_lines.extend([f"--- {new_path}", f"+++ {old_path}"])
                index += 2
                continue
            if line.startswith("index ") and ".." in line:
                prefix, hashes = line.split(" ", 1)
                pair, *mode = hashes.split(" ", 1)
                old_hash, new_hash = pair.split("..", 1)
                suffix = f" {mode[0]}" if mode else ""
                reversed_lines.append(f"{prefix} {new_hash}..{old_hash}{suffix}")
            elif (match := hunk.match(line)) is not None:
                old_start, old_count, new_start, new_count, suffix = match.groups()
                old_range = new_start + (f",{new_count}" if new_count is not None else "")
                new_range = old_start + (f",{old_count}" if old_count is not None else "")
                reversed_lines.append(f"@@ -{old_range} +{new_range} @@{suffix}")
            elif line.startswith("+") and not line.startswith("+++"):
                reversed_lines.append("-" + line[1:])
            elif line.startswith("-") and not line.startswith("---"):
                reversed_lines.append("+" + line[1:])
            else:
                reversed_lines.append(line)
            index += 1
        return "\n".join(reversed_lines) + ("\n" if patch.endswith("\n") else "")
