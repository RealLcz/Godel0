"""Generation chunks must overlap instead of queueing behind each other.

Job 217937 served exactly one vLLM request at a time across eight GPUs (KV
cache peaked at 0.5%), so ~38 min per accepted task was mostly the builder
waiting for one chunk before starting the next. Chunk size still obeys
plans_per_call, so a slow chunk still cannot take the batch down with it.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

from godel0.tasks.batch import TaskBatchBuilder


class _FakeRepoPool:
    def __init__(self, root: Path):
        self.pool_dir = root
        self._spec = SimpleNamespace(
            repo_id="ansible",
            base_commit="HEAD",
            path=str(root),
            test_command="pytest -q",
            install_command="pip install -e .",
            timeout_sec=120,
        )

    def all_repos(self):
        return [self._spec]


class _ConcurrencyProbe:
    """Records how many chunks were ever generating at the same moment."""

    def __init__(self, hold_sec: float = 0.15):
        self.hold_sec = hold_sec
        self.max_in_flight = 0
        self.attempts: list[int] = []
        self.output_dirs: list[str] = []
        self._in_flight = 0
        self._lock = threading.Lock()

    def generate_batch(self, request):
        with self._lock:
            self._in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self._in_flight)
            self.attempts.append(int(request.generation_attempt))
            self.output_dirs.append(str(request.output_dir))
        try:
            time.sleep(self.hold_sec)
        finally:
            with self._lock:
                self._in_flight -= 1
        return SimpleNamespace(
            accepted_candidates=[],
            rejected_candidates=[],
            pending_candidates=[],
            plans=[{"plan_id": f"plan-{request.generation_attempt}"}],
            completed=True,
            error="",
        )


def _run(tmp_path: Path, workers: int, probe: _ConcurrencyProbe):
    builder = TaskBatchBuilder(
        batch_size=10,
        max_candidates=8,
        workflow_config={"plans_per_call": 1},
        max_concurrent_attempts=workers,
    )
    return builder.build_for_node(
        node_id="root",
        repo_pool=_FakeRepoPool(tmp_path),
        validator=SimpleNamespace(),
        task_committer=None,
        proposer_runner=probe,
        output_dir=tmp_path / "proposer",
        bootstrap=True,
    )


def test_chunks_run_concurrently_when_workers_allow(tmp_path: Path):
    probe = _ConcurrencyProbe()
    _run(tmp_path, workers=4, probe=probe)

    assert probe.max_in_flight > 1
    assert probe.max_in_flight <= 4


def test_default_stays_serial(tmp_path: Path):
    probe = _ConcurrencyProbe()
    _run(tmp_path, workers=1, probe=probe)

    assert probe.max_in_flight == 1


def test_concurrent_chunks_get_distinct_attempts_and_output_dirs(tmp_path: Path):
    probe = _ConcurrencyProbe()
    _run(tmp_path, workers=4, probe=probe)

    # generation_attempt drives the bootstrap plan slice, so a repeat would
    # make two chunks build the very same plan.
    assert len(probe.attempts) == len(set(probe.attempts))
    assert len(probe.output_dirs) == len(set(probe.output_dirs))


def test_chunk_size_still_respects_plans_per_call(tmp_path: Path):
    class _TargetProbe(_ConcurrencyProbe):
        def __init__(self):
            super().__init__()
            self.targets: list[int] = []

        def generate_batch(self, request):
            self.targets.append(int(request.target_batch_size))
            return super().generate_batch(request)

    probe = _TargetProbe()
    _run(tmp_path, workers=4, probe=probe)

    assert probe.targets
    assert set(probe.targets) == {1}


def test_wave_never_over_requests_the_remaining_batch(tmp_path: Path):
    probe = _ConcurrencyProbe(hold_sec=0.0)
    builder = TaskBatchBuilder(
        batch_size=2,
        max_candidates=8,
        workflow_config={"plans_per_call": 1},
        max_concurrent_attempts=8,
    )
    builder.build_for_node(
        node_id="root",
        repo_pool=_FakeRepoPool(tmp_path),
        validator=SimpleNamespace(),
        task_committer=None,
        proposer_runner=probe,
        output_dir=tmp_path / "proposer",
        bootstrap=True,
    )

    # batch_size=2 with one plan per chunk means at most two chunks in flight,
    # even though eight workers were configured.
    assert probe.max_in_flight <= 2


def test_wave_spreads_source_quota_across_its_chunks(tmp_path: Path):
    """Chunks are built before any of them commits, so quota must be reserved."""

    class _QuotaProbe(_ConcurrencyProbe):
        def __init__(self):
            super().__init__(hold_sec=0.0)
            self.quotas: list[dict] = []

        def generate_batch(self, request):
            self.quotas.append(dict(request.generation_quotas or {}))
            return super().generate_batch(request)

    probe = _QuotaProbe()
    builder = TaskBatchBuilder(
        batch_size=10,
        max_candidates=4,
        source_quotas={"parent_failure": 5, "current_child_level1": 5},
        workflow_config={"plans_per_call": 1},
        max_concurrent_attempts=4,
    )
    builder.build_for_node(
        node_id="node_child",
        repo_pool=_FakeRepoPool(tmp_path),
        validator=SimpleNamespace(),
        task_committer=None,
        proposer_runner=probe,
        output_dir=tmp_path / "proposer",
        # Both sides need enough trajectories for the nominal 5/5 to survive
        # compute_effective_quotas; otherwise the lopsided split is correct.
        parent_failure_trajectories=[str(tmp_path / f"parent{i}.jsonl") for i in range(5)],
        current_child_level1_trajectories=[str(tmp_path / f"child{i}.jsonl") for i in range(5)],
    )

    first_wave = probe.quotas[:4]
    assert len(first_wave) == 4
    # A 5/5 split must not send all four chunks at the same source.
    assert sum(q.get("parent_failure", 0) for q in first_wave) < 4
    assert sum(q.get("current_child_level1", 0) for q in first_wave) < 4
    for quota in first_wave:
        assert sum(quota.values()) <= 1


class _AlwaysCrashRunner:
    def __init__(self):
        self.calls = 0
        self._lock = threading.Lock()

    def generate_batch(self, request):
        with self._lock:
            self.calls += 1
        raise RuntimeError("Node proposer produced no result: exit=-15; stderr=")


def test_all_chunks_crashing_does_not_look_like_a_dead_proposer(tmp_path: Path):
    """A crashed wave must keep spending budget, not abort like a zero-yield one."""
    runner = _AlwaysCrashRunner()
    builder = TaskBatchBuilder(
        batch_size=4,
        max_candidates=4,
        workflow_config={"plans_per_call": 1},
        max_concurrent_attempts=2,
    )
    result = builder.build_for_node(
        node_id="root",
        repo_pool=_FakeRepoPool(tmp_path),
        validator=SimpleNamespace(),
        task_committer=None,
        proposer_runner=runner,
        output_dir=tmp_path / "proposer",
        bootstrap=True,
    )

    assert "RuntimeError" in (result.proposer_error or "")
    assert runner.calls >= 4
    assert result.complete is False
