"""Crash-resume: continue from committed tasks and completed nodes."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from godel0.controller.budget import Budget
from godel0.schemas.node import NodeRecord, NodeStatus
from godel0.schemas.task import TaskRecord
from godel0.tasks.batch import TaskBatchBuilder
from godel0.tasks.store import TaskStore
from godel0.tree.archive import NodeArchive


def _task(store: TaskStore, batch_id: str, node_id: str, idx: int) -> TaskRecord:
    record = TaskRecord(
        task_id=f"task_seed_{idx:02d}",
        batch_id=batch_id,
        proposer_node_id=node_id,
        repo_id="ansible",
        base_commit="abc",
        bug_strategy="repo_chain",
        bug_patch_path=f"task_store/task_seed_{idx:02d}/bug.patch",
        oracle_patch_path=f"task_store/task_seed_{idx:02d}/oracle_reverse.patch",
        problem_statement_path=f"task_store/task_seed_{idx:02d}/problem_statement.md",
        f2p_tests=["test_a"],
        baseline_test_command="pytest",
        source_type="bootstrap",
        created_at=datetime.now(timezone.utc),
    )
    from godel0.tasks.store import TaskArtifacts

    store.put(
        record,
        TaskArtifacts(
            problem_statement=f"problem {idx}",
            bug_patch=f"diff --git a/f{idx}.py b/f{idx}.py\n+x{idx}\n",
            oracle_patch="",
            failing_test_output="fail",
            validation_report={"passed": True},
        ),
    )
    return store.get(record.task_id)


def test_task_store_recovers_incomplete_batch(tmp_path: Path):
    store = TaskStore(tmp_path / "store")
    for i in range(5):
        _task(store, "batch_097216cb", "root", i)
    # A newer complete-looking sibling must not steal the incomplete one when
    # prefer_incomplete_below is set.
    for i in range(10):
        _task(store, "batch_newer_full", "root", 100 + i)

    recovered = store.latest_batch_for_node("root", prefer_incomplete_below=10)
    assert recovered == "batch_097216cb"
    assert len(store.tasks_for_batch(recovered)) == 5


def test_budget_from_archive_counts_complete_children(tmp_path: Path):
    archive = NodeArchive(tmp_path / "archive.jsonl")
    archive.add(
        NodeRecord(
            node_id="root",
            parent_node_id=None,
            code_commit="aaa",
            code_ref="refs/godel0/nodes/root",
            status=NodeStatus.COMPLETE,
        )
    )
    archive.add(
        NodeRecord(
            node_id="child_ok",
            parent_node_id="root",
            code_commit="bbb",
            code_ref="refs/godel0/nodes/child_ok",
            status=NodeStatus.COMPLETE,
        )
    )
    archive.add(
        NodeRecord(
            node_id="child_fail",
            parent_node_id="root",
            code_commit="ccc",
            code_ref="refs/godel0/nodes/child_fail",
            status=NodeStatus.LEVEL1_FAILED,
        )
    )
    budget = Budget.from_archive(archive, max_nodes=3, max_expansions=40)
    assert budget.nodes_created == 1
    assert budget.expansions_attempted == 2
    assert budget.remaining() == 2


def test_batch_builder_seeds_incomplete_batch(tmp_path: Path):
    store = TaskStore(tmp_path / "store")
    seeds = [_task(store, "batch_resume", "root", i) for i in range(3)]
    builder = TaskBatchBuilder(batch_size=5, max_candidates=10)
    # No repo pool / proposer: with a full seed it would complete; with a
    # partial seed and no generator it returns incomplete but keeps seeds.
    result = builder.build_for_node(
        node_id="root",
        task_store_dir=str(store.store_dir),
        resume_batch_id="batch_resume",
        seed_tasks=seeds,
        bootstrap=True,
        output_dir=tmp_path / "proposer",
    )
    assert result.batch_id == "batch_resume"
    assert len(result.tasks) == 3
    assert result.complete is False


def test_batch_builder_completes_when_seed_already_full(tmp_path: Path):
    store = TaskStore(tmp_path / "store")
    seeds = [_task(store, "batch_full", "root", i) for i in range(4)]
    builder = TaskBatchBuilder(batch_size=4, max_candidates=10)
    result = builder.build_for_node(
        node_id="root",
        task_store_dir=str(store.store_dir),
        resume_batch_id="batch_full",
        seed_tasks=seeds,
        bootstrap=True,
        output_dir=tmp_path / "proposer",
    )
    assert result.complete is True
    assert len(result.tasks) == 4


def test_resume_manager_sets_resume_from(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "runs" / "run_abc"
    run_dir.mkdir(parents=True)
    (run_dir / "nodes").mkdir()
    (run_dir / "config.resolved.yaml").write_text(
        "\n".join(
            [
                "run:",
                "  seed: 1",
                "  max_nodes: 3",
                "paths:",
                "  runs: ./runs",
                "  agent_repo: ./agent",
                "  repo_pool: ./repo_pool",
                "  task_store: ./task_store",
            ]
        )
    )

    captured = {}

    class FakeOrch:
        @classmethod
        def from_config(cls, config):
            captured["resume_from"] = config.run.resume_from
            captured["run_name"] = config.run.run_name
            return cls()

        def run(self):
            captured["ran"] = True

    monkeypatch.setattr(
        "godel0.controller.orchestrator.EvolutionOrchestrator",
        FakeOrch,
    )

    from godel0.controller.resume import ResumeManager

    ResumeManager(run_dir).resume()
    assert captured["ran"] is True
    assert Path(captured["resume_from"]).resolve() == run_dir.resolve()
    assert captured["run_name"] == "run_abc"
