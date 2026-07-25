"""Concurrent evaluation workers share one agent repo and one scratch root.

Level1/Level2 now solve tasks in parallel, so several NodeWorktree instances
are live at once on the same repository. They must not serialise into each
other's directories or race on git's worktree bookkeeping.
"""

from __future__ import annotations

import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from godel0.git.worktree import NodeWorktree


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture()
def agent_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "agent"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "agent.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


def test_concurrent_worktrees_on_one_repo_stay_isolated(agent_repo: Path, tmp_path: Path):
    scratch = tmp_path / "scratch"
    head = subprocess.run(
        ["git", "-C", str(agent_repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()

    barrier = threading.Barrier(4, timeout=30)
    seen: dict[str, Path] = {}
    lock = threading.Lock()

    def worker(index: int) -> None:
        name = f"eval_node_task_{index}"
        with NodeWorktree(agent_repo, scratch, name, head) as worktree:
            # Every worker must hold a live, distinct checkout at the same time.
            barrier.wait()
            assert (worktree / "agent.py").read_text() == "VALUE = 1\n"
            with lock:
                seen[name] = worktree

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, range(4)))

    assert len(set(seen.values())) == 4


def test_worktrees_are_deregistered_after_use(agent_repo: Path, tmp_path: Path):
    scratch = tmp_path / "scratch"
    head = subprocess.run(
        ["git", "-C", str(agent_repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()

    for index in range(3):
        with NodeWorktree(agent_repo, scratch, f"eval_{index}", head):
            pass

    listing = subprocess.run(
        ["git", "-C", str(agent_repo), "worktree", "list"],
        check=True, capture_output=True, text=True,
    ).stdout
    assert "eval_0" not in listing
    assert "eval_2" not in listing
