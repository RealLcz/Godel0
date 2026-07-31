"""Tests for runtime artifact cleanup and least-attempted entry selection."""

from __future__ import annotations

import random
from pathlib import Path

from godel0.evolution.entry_selector import (
    ProposerFailureEntry,
    choose_least_attempted_failure,
)
from godel0.git.repository import (
    commit,
    diff_vs_commit,
    init_repo,
    is_runtime_artifact,
    restore_runtime_artifacts,
)


def test_is_runtime_artifact_patterns():
    assert is_runtime_artifact("tools/__pycache__/x.pyc")
    assert is_runtime_artifact("foo.backup")
    assert is_runtime_artifact("bar.bak")
    assert is_runtime_artifact("pkg/.pytest_cache/v")
    assert not is_runtime_artifact("proposer/schemas.py")


def test_restore_runtime_artifacts_keeps_source_changes(tmp_path: Path):
    repo = tmp_path / "agent"
    init_repo(repo)
    (repo / "coding_agent.py").write_text("VALUE = 1\n", encoding="utf-8")
    cache = repo / "__pycache__"
    cache.mkdir()
    (cache / "coding_agent.cpython-311.pyc").write_bytes(b"old")
    sha = commit(repo, "base")

    (repo / "coding_agent.py").write_text("VALUE = 2\n", encoding="utf-8")
    (cache / "coding_agent.cpython-311.pyc").write_bytes(b"new")
    tools = repo / "tools"
    tools.mkdir(exist_ok=True)
    (tools / "new_tool.py").write_text("X=1\n", encoding="utf-8")
    untracked = tools / "__pycache__"
    untracked.mkdir(parents=True)
    (untracked / "x.pyc").write_bytes(b"cache")

    restore_runtime_artifacts(repo, sha)
    patch = diff_vs_commit(repo, sha)

    assert "VALUE = 2" in patch
    assert "new_tool.py" in patch
    assert "__pycache__" not in patch
    assert ".pyc" not in patch
    assert not untracked.exists()


def test_choose_least_attempted_prefers_untried():
    failures = [
        ProposerFailureEntry(candidate_id="a"),
        ProposerFailureEntry(candidate_id="b"),
        ProposerFailureEntry(candidate_id="c"),
    ]
    chosen = choose_least_attempted_failure(
        failures,
        attempt_counts={"a": 2, "b": 0, "c": 1},
        rng=random.Random(0),
    )
    assert chosen is not None
    assert chosen.candidate_id == "b"


def test_choose_least_attempted_ties_use_rng():
    failures = [
        ProposerFailureEntry(candidate_id="a"),
        ProposerFailureEntry(candidate_id="b"),
    ]
    first = choose_least_attempted_failure(
        failures, attempt_counts={"a": 1, "b": 1}, rng=random.Random(1)
    )
    second = choose_least_attempted_failure(
        failures, attempt_counts={"a": 1, "b": 1}, rng=random.Random(1)
    )
    assert first is not None and second is not None
    assert first.candidate_id == second.candidate_id
