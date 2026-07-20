"""Regression tests for solver patch evaluation."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from godel0.evaluation.runner import SolverEvaluationRunner
from godel0.execution.workspace_manager import WorkspaceManager
from godel0.git.repository import apply_patch, commit, diff_vs_commit, reset_to_commit, run_git
from godel0.proposer_trusted.task_committer import TaskCommitter
from godel0.schemas.node import NodeRecord, NodeStatus
from godel0.tasks.repo_pool import RepoPool, RepoSpec
from godel0.tasks.store import TaskStore
from godel0.tasks.workspace import TaskWorkspace
from initial_agent.src.proposer.trajectory_analyzer import TrajectoryView


def _buggy_clamp_patch(repo_path: Path) -> str:
    source_file = repo_path / "toy_module.py"
    original = source_file.read_text()
    source_file.write_text(original.replace("min(x, high)", "min(x, low)"))
    result = subprocess.run(
        ["git", "-C", str(repo_path), "diff"],
        capture_output=True,
        text=True,
        check=False,
    )
    source_file.write_text(original)
    reset_to_commit(repo_path, "HEAD")
    return result.stdout


def _solver_fix_patch(repo_path: Path, base_commit: str, bug_patch: str, tmp_path: Path) -> str:
    work = tmp_path / "bugged_for_solver_patch"
    shutil.copytree(repo_path, work)
    reset_to_commit(work, base_commit)
    assert apply_patch(work, bug_patch)
    bugged_commit = commit(work, "bugged base")

    source_file = work / "toy_module.py"
    source_file.write_text(source_file.read_text().replace("min(x, low)", "min(x, high)"))
    return diff_vs_commit(work, bugged_commit)


def test_solver_patch_must_pass_f2p_tests(toy_repo, tmp_path):
    bug_patch = _buggy_clamp_patch(toy_repo["path"])
    solver_patch = _solver_fix_patch(toy_repo["path"], toy_repo["commit"], bug_patch, tmp_path)

    task_store = TaskStore(tmp_path / "task_store")
    task = TaskCommitter(task_store).commit_task(
        batch_id="batch",
        proposer_node_id="root",
        repo_id="toy_repo",
        base_commit=toy_repo["commit"],
        bug_strategy="procedural",
        bug_patch=bug_patch,
        problem_statement="The clamp function mishandles upper bounds.",
        f2p_tests=["test_toy.py::test_clamp_upper_boundary"],
        baseline_test_command="python -m pytest test_toy.py -q",
    )

    repo_pool = RepoPool(tmp_path / "repo_pool")
    repo_pool.add(
        RepoSpec(
            repo_id="toy_repo",
            base_commit=toy_repo["commit"],
            path=str(toy_repo["path"]),
            test_command="python -m pytest test_toy.py -q",
        )
    )

    runner = SolverEvaluationRunner(
        task_store=task_store,
        workspace_manager=WorkspaceManager(tmp_path / "scratch"),
        repo_pool=repo_pool,
    )
    node = NodeRecord(
        node_id="node",
        code_commit="abc123",
        code_ref="refs/godel0/nodes/node",
        status=NodeStatus.COMPLETE,
    )

    outcome = runner.run_task(node, task, level=2, seed=1, solver_result_patch=solver_patch)

    assert outcome.resolved


def test_solver_patch_rejects_empty_patch(toy_repo, tmp_path):
    task_store = TaskStore(tmp_path / "task_store")
    task = TaskCommitter(task_store).commit_task(
        batch_id="batch",
        proposer_node_id="root",
        repo_id="toy_repo",
        base_commit=toy_repo["commit"],
        bug_strategy="procedural",
        bug_patch="diff --git a/toy_module.py b/toy_module.py\n",
        problem_statement="Problem.",
        f2p_tests=["test_toy.py::test_clamp_upper_boundary"],
        baseline_test_command="python -m pytest test_toy.py -q",
    )
    runner = SolverEvaluationRunner(
        task_store=task_store,
        workspace_manager=WorkspaceManager(tmp_path / "scratch"),
    )
    node = NodeRecord(node_id="node", code_commit="abc123", code_ref="ref")

    outcome = runner.run_task(node, task, level=2, seed=1, solver_result_patch="")

    assert not outcome.resolved
    eval_path = (
        tmp_path
        / "scratch"
        / "run"
        / "solver"
        / "node"
        / "trajectories"
        / "level_2"
        / task.task_id
        / "trajectory_eval.json"
    )
    assert eval_path.is_file()

    trajectory_path = eval_path.with_name("trajectory.jsonl")
    trajectory_path.write_text('{"role": "assistant", "content": "failed"}\n')
    trajectory = TrajectoryView.from_jsonl(str(trajectory_path))
    assert trajectory.trajectory_id == outcome.trajectory_id
    assert trajectory.task_id == task.task_id
    assert trajectory.node_id == "node"


def test_solver_artifact_dir_is_absolute_for_relative_scratch(tmp_path, monkeypatch):
    """A solver cwd change must not relocate trajectories into the task repo."""
    monkeypatch.chdir(tmp_path)
    runner = SolverEvaluationRunner(
        task_store=TaskStore(tmp_path / "task_store"),
        workspace_manager=WorkspaceManager(Path("relative_scratch")),
    )

    artifact_dir = runner._artifact_dir("run", "node", "task", 2)

    assert artifact_dir.is_absolute()
    assert artifact_dir == (
        tmp_path
        / "relative_scratch"
        / "run"
        / "solver"
        / "node"
        / "trajectories"
        / "level_2"
        / "task"
    )


def test_bugged_solver_snapshot_does_not_expose_clean_parent(toy_repo, tmp_path):
    bug_patch = _buggy_clamp_patch(toy_repo["path"])
    workspace = TaskWorkspace(tmp_path / "solver")
    repo = workspace.setup_bugged_repo(
        source_repo=toy_repo["path"],
        base_commit=toy_repo["commit"],
        bug_patch=bug_patch,
        task_id="task",
    )

    snapshot = workspace.seal_bugged_snapshot(repo)

    assert snapshot == run_git(repo, "rev-parse", "HEAD").stdout.strip()
    assert run_git(repo, "rev-list", "--count", "HEAD").stdout.strip() == "1"
    assert run_git(repo, "rev-parse", "HEAD^", check=False).returncode != 0
    assert "min(x, low)" in (repo / "toy_module.py").read_text()


def test_committed_oracle_is_a_valid_reverse_patch(toy_repo, tmp_path):
    bug_patch = _buggy_clamp_patch(toy_repo["path"])
    committer = TaskCommitter(TaskStore(tmp_path / "task_store"))
    oracle = committer._generate_reverse_patch(bug_patch)
    work = tmp_path / "oracle_check"
    shutil.copytree(toy_repo["path"], work)
    reset_to_commit(work, toy_repo["commit"])

    assert apply_patch(work, bug_patch)
    assert apply_patch(work, oracle)
    assert not diff_vs_commit(work, toy_repo["commit"])
