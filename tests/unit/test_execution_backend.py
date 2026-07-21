"""Tests for the execution backend unification (Phase 9)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from godel0.execution.subprocess_runner import ExecutionBackend, ProcessResult, SubprocessRunner


class TestExecutionBackend:
    def test_subprocess_runner_is_execution_backend(self):
        runner = SubprocessRunner()
        assert isinstance(runner, ExecutionBackend)

    def test_subprocess_runner_run(self, tmp_path):
        runner = SubprocessRunner()
        result = runner.run(
            command=["python", "-c", "print('hello')"],
            cwd=tmp_path,
            env={},
            timeout_sec=10,
        )
        assert result.returncode == 0
        assert "hello" in result.stdout

    def test_subprocess_runner_timeout(self, tmp_path):
        runner = SubprocessRunner()
        result = runner.run(
            command=["python", "-c", "import time; time.sleep(10)"],
            cwd=tmp_path,
            env={},
            timeout_sec=1,
        )
        assert result.timed_out is True
        assert result.returncode == -15


class TestNodeProposerRunnerBackend:
    def test_runner_accepts_execution_backend(self, tmp_path):
        from godel0.tasks.node_proposer import NodeProposerRunner

        backend = SubprocessRunner()
        runner = NodeProposerRunner(
            agent_repo=tmp_path / "agent",
            scratch_root=tmp_path / "scratch",
            timeout_sec=60,
            execution_backend=backend,
        )
        assert runner.execution_backend is backend

    def test_runner_defaults_to_no_backend(self, tmp_path):
        from godel0.tasks.node_proposer import NodeProposerRunner

        runner = NodeProposerRunner(
            agent_repo=tmp_path / "agent",
            scratch_root=tmp_path / "scratch",
            timeout_sec=60,
        )
        assert runner.execution_backend is None

    def test_for_node_preserves_backend(self, tmp_path):
        from godel0.tasks.node_proposer import NodeProposerRunner

        backend = SubprocessRunner()
        runner = NodeProposerRunner(
            agent_repo=tmp_path / "agent",
            scratch_root=tmp_path / "scratch",
            timeout_sec=60,
            execution_backend=backend,
        )
        node = MagicMock()
        node.node_id = "test_node"
        bound = runner.for_node(node)
        assert bound.execution_backend is backend
