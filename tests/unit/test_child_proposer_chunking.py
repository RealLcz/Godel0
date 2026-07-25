"""Child batches must be chunked like bootstrap, not requested all at once.

Job 213825: ``bootstrap_plans_per_call`` only capped the bootstrap path. On
the child path ``TaskBatchBuilder`` asked for the whole batch in a single
proposer subprocess call, every call was SIGTERM'd at ``batch_timeout_sec``
with nothing persisted, and both children ended as ``proposer_failed``.
"""

from __future__ import annotations

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


class _RecordingRunner:
    """Records the per-call target and always returns an empty chunk."""

    def __init__(self):
        self.targets: list[int] = []
        self.quotas: list[dict] = []

    def generate_batch(self, request):
        self.targets.append(int(request.target_batch_size))
        self.quotas.append(dict(request.generation_quotas or {}))
        return SimpleNamespace(
            accepted_candidates=[],
            rejected_candidates=[],
            pending_candidates=[],
            plans=[{"plan_id": f"plan-{len(self.targets)}"}],
            completed=True,
            error="",
        )


def _build(tmp_path: Path, workflow_config: dict, bootstrap: bool = False):
    builder = TaskBatchBuilder(
        batch_size=10,
        max_candidates=4,
        workflow_config=workflow_config,
    )
    runner = _RecordingRunner()
    builder.build_for_node(
        node_id="node_child",
        repo_pool=_FakeRepoPool(tmp_path),
        validator=SimpleNamespace(),
        task_committer=None,
        proposer_runner=runner,
        output_dir=tmp_path / "proposer",
        bootstrap=bootstrap,
        parent_failure_trajectories=[str(tmp_path / "traj.jsonl")],
    )
    return runner


def test_child_path_requests_one_plan_per_call(tmp_path: Path):
    runner = _build(tmp_path, {"plans_per_call": 1})

    assert runner.targets, "proposer was never called"
    assert set(runner.targets) == {1}


def test_plans_per_call_caps_but_does_not_inflate(tmp_path: Path):
    runner = _build(tmp_path, {"plans_per_call": 3})

    assert runner.targets
    assert max(runner.targets) <= 3


def test_bootstrap_key_still_applies_when_plans_per_call_unset(tmp_path: Path):
    runner = _build(tmp_path, {"bootstrap_plans_per_call": 2})

    assert runner.targets
    assert max(runner.targets) <= 2


def test_uncapped_when_neither_key_is_set(tmp_path: Path):
    runner = _build(tmp_path, {})

    # Without a chunk size the builder keeps the legacy whole-batch request,
    # bounded only by the remaining candidate budget.
    assert runner.targets
    assert max(runner.targets) > 1


def test_source_quotas_are_scaled_down_to_the_chunk(tmp_path: Path):
    # Quotas describe the whole batch; the proposer builds one plan per quota
    # slot, so an unscaled quota rebuilds the full batch inside one call.
    runner = _build(tmp_path, {"plans_per_call": 1})

    assert runner.quotas
    for quota in runner.quotas:
        assert sum(quota.values()) <= 1


def test_chunk_quotas_keeps_the_split_when_it_fits():
    assert TaskBatchBuilder._chunk_quotas(
        {"parent_failure": 2, "current_child_level1": 1}, 5
    ) == {"parent_failure": 2, "current_child_level1": 1}


def test_chunk_quotas_never_drops_below_the_limit():
    for parent, child in ((10, 0), (0, 10), (5, 5), (7, 3)):
        for limit in (1, 2, 3):
            scaled = TaskBatchBuilder._chunk_quotas(
                {"parent_failure": parent, "current_child_level1": child}, limit
            )
            assert sum(scaled.values()) == limit
            assert scaled["parent_failure"] <= parent
            assert scaled["current_child_level1"] <= child
