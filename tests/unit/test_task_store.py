"""Unit tests for TaskStore."""

from __future__ import annotations

import pytest
from pathlib import Path

from godel0.tasks.store import TaskStore, TaskArtifacts
from godel0.schemas.task import TaskRecord


class TestTaskStore:
    def test_put_and_get(self, task_store):
        """Put a task and retrieve it."""
        record = TaskRecord(
            task_id="test_task_001",
            batch_id="batch_001",
            proposer_node_id="node_001",
            repo_id="toy/repo",
            base_commit="abc123",
            bug_strategy="procedural",
            bug_patch_path="task_store/test_task_001/bug.patch",
            problem_statement_path="task_store/test_task_001/problem_statement.md",
            baseline_test_command="pytest -q",
        )
        artifacts = TaskArtifacts(
            problem_statement="Test problem",
            bug_patch="diff --git a/file.py b/file.py\n+new",
            f2p_tests=["test_one"],
        )
        task_store.put(record, artifacts)

        retrieved = task_store.get("test_task_001")
        assert retrieved is not None
        assert retrieved.task_id == "test_task_001"
        assert retrieved.repo_id == "toy/repo"

    def test_materialize_public(self, task_store, tmp_path):
        """Public materialization should not include private data."""
        record = TaskRecord(
            task_id="test_task_002",
            batch_id="batch_001",
            proposer_node_id="node_001",
            repo_id="toy/repo",
            base_commit="abc123",
            bug_strategy="procedural",
            bug_patch_path="",
            problem_statement_path="",
            baseline_test_command="pytest -q",
        )
        artifacts = TaskArtifacts(
            problem_statement="Public problem",
            bug_patch="patch content",
            f2p_tests=["secret_test"],
            generation_context={"secret": "data"},
        )
        task_store.put(record, artifacts)

        dest = tmp_path / "public"
        task_store.materialize_public("test_task_002", dest)

        assert (dest / "problem_statement.md").exists()
        assert (dest / "bug.patch").exists()
        assert not (dest / "f2p_tests.json").exists()

    def test_materialize_private(self, task_store, tmp_path):
        """Private materialization should include F2P tests."""
        record = TaskRecord(
            task_id="test_task_003",
            batch_id="batch_001",
            proposer_node_id="node_001",
            repo_id="toy/repo",
            base_commit="abc123",
            bug_strategy="procedural",
            bug_patch_path="",
            problem_statement_path="",
            baseline_test_command="pytest -q",
        )
        artifacts = TaskArtifacts(
            problem_statement="Problem",
            bug_patch="patch",
            f2p_tests=["test_f2p_1", "test_f2p_2"],
        )
        task_store.put(record, artifacts)

        dest = tmp_path / "private"
        task_store.materialize_private("test_task_003", dest)

        assert (dest / "f2p_tests.json").exists()
        f2p = task_store.get_f2p_tests("test_task_003")
        assert "test_f2p_1" in f2p
        assert "test_f2p_2" in f2p

    def test_nonexistent_task(self, task_store):
        """Getting a nonexistent task should return None."""
        assert task_store.get("nonexistent") is None

    def test_all_task_ids(self, task_store):
        """List all task IDs."""
        for i in range(3):
            record = TaskRecord(
                task_id=f"task_{i}",
                batch_id="batch",
                proposer_node_id="node",
                repo_id="repo",
                base_commit="abc",
                bug_strategy="procedural",
                bug_patch_path="",
                problem_statement_path="",
                baseline_test_command="pytest",
            )
            task_store.put(record, TaskArtifacts(problem_statement="p", bug_patch="b"))

        ids = task_store.all_task_ids()
        assert len(ids) == 3
