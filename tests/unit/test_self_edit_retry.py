"""A self-edit that leaves no usable diff must be retried, not discarded.

Job 213825: 30 of 33 self-edit trajectories died on the model context limit.
The agent had written nothing (or half a file), so the child build failed with
``Patch guard: Empty patch`` or ``Syntax guard: ... unexpected indent`` and the
whole expansion was wasted.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from godel0.evolution.self_edit import SelfEditRunner


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture()
def worktree(tmp_path: Path) -> Path:
    repo = tmp_path / "agent"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "proposer.py").write_text("def plan():\n    return 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


class _ScriptedAdapter:
    """Applies a scripted edit to the worktree on each call."""

    def __init__(self, actions):
        self.actions = list(actions)
        self.problem_statements: list[str] = []

    def run(self, agent_src, request):
        self.problem_statements.append(request.problem_statement)
        action = self.actions.pop(0) if self.actions else "noop"
        target = Path(request.git_dir) / "proposer.py"
        if action == "valid":
            target.write_text("def plan():\n    return 2\n", encoding="utf-8")
        elif action == "broken":
            target.write_text("    return 2\n", encoding="utf-8")
        Path(request.chat_history_file).write_text("trajectory", encoding="utf-8")
        return SimpleNamespace(success=True, patch_path=None, error=None)


def test_empty_attempt_is_retried_until_a_diff_appears(worktree: Path, tmp_path: Path):
    adapter = _ScriptedAdapter(["noop", "valid"])
    runner = SelfEditRunner(agent_adapter=adapter, max_attempts=3)

    result = runner.run(
        diagnosis=SimpleNamespace(problem_statement="Fix the planner"),
        worktree=worktree,
        output_dir=tmp_path / "self_evolve",
    )

    assert result.success is True
    assert result.attempts == 2
    assert "empty patch" in result.attempt_errors[0]
    assert "return 2" in (worktree / "proposer.py").read_text()


def test_broken_python_is_retried_and_worktree_is_reset(worktree: Path, tmp_path: Path):
    adapter = _ScriptedAdapter(["broken", "valid"])
    runner = SelfEditRunner(agent_adapter=adapter, max_attempts=3)

    result = runner.run(
        diagnosis=SimpleNamespace(problem_statement="Fix the planner"),
        worktree=worktree,
        output_dir=tmp_path / "self_evolve",
    )

    assert result.success is True
    assert result.attempts == 2
    assert "Invalid Python syntax" in result.attempt_errors[0]
    assert (worktree / "proposer.py").read_text() == "def plan():\n    return 2\n"


def test_all_attempts_unusable_reports_every_reason(worktree: Path, tmp_path: Path):
    adapter = _ScriptedAdapter(["noop", "broken", "noop"])
    runner = SelfEditRunner(agent_adapter=adapter, max_attempts=3)

    result = runner.run(
        diagnosis=SimpleNamespace(problem_statement="Fix the planner"),
        worktree=worktree,
        output_dir=tmp_path / "self_evolve",
    )

    assert result.success is False
    assert result.attempts == 3
    assert len(result.attempt_errors) == 3
    # The failed attempt was rolled back rather than left in the worktree.
    assert (worktree / "proposer.py").read_text() == "def plan():\n    return 1\n"


def test_retry_prompt_carries_the_editing_protocol_and_prior_failure(
    worktree: Path, tmp_path: Path
):
    adapter = _ScriptedAdapter(["noop", "valid"])
    runner = SelfEditRunner(agent_adapter=adapter, max_attempts=3)

    runner.run(
        diagnosis=SimpleNamespace(problem_statement="Fix the planner"),
        worktree=worktree,
        output_dir=tmp_path / "self_evolve",
    )

    first, second = adapter.problem_statements
    assert "Fix the planner" in first
    assert "Implement the improvement task" in first
    assert "simplest coherent" in first.lower()
    assert "previous attempt was discarded" in second.lower()
    assert "empty patch" in second


def test_single_attempt_runner_keeps_legacy_behaviour(worktree: Path, tmp_path: Path):
    adapter = _ScriptedAdapter(["noop"])
    runner = SelfEditRunner(agent_adapter=adapter, max_attempts=1)

    result = runner.run(
        diagnosis=SimpleNamespace(problem_statement="Fix the planner"),
        worktree=worktree,
        output_dir=tmp_path / "self_evolve",
    )

    assert result.success is False
    assert result.attempts == 1
    assert len(adapter.problem_statements) == 1


def test_missing_adapter_fails_fast(worktree: Path, tmp_path: Path):
    runner = SelfEditRunner(agent_adapter=None)

    result = runner.run(
        diagnosis=SimpleNamespace(problem_statement="Fix the planner"),
        worktree=worktree,
        output_dir=tmp_path / "self_evolve",
    )

    assert result.success is False
    assert "No agent adapter" in (result.error or "")
