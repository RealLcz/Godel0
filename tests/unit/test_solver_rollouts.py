"""P1-4: evaluation.solver_rollouts must actually run each Level2 task N times."""

from __future__ import annotations

from types import SimpleNamespace

from godel0.controller.orchestrator import EvolutionOrchestrator
from godel0.evaluation.level2 import Level2Evaluator
from godel0.schemas.evaluation import EvaluationOutcome, Level2Result


def test_level2_accuracy_aggregates_rollouts_per_task():
    evaluator = Level2Evaluator()
    outcomes = [
        EvaluationOutcome(
            node_id="n", task_id="t1", level=2, resolved=True, trajectory_id="a", rollout_index=0
        ),
        EvaluationOutcome(
            node_id="n", task_id="t1", level=2, resolved=False, trajectory_id="b", rollout_index=1
        ),
        EvaluationOutcome(
            node_id="n", task_id="t1", level=2, resolved=True, trajectory_id="c", rollout_index=2
        ),
        EvaluationOutcome(
            node_id="n", task_id="t2", level=2, resolved=False, trajectory_id="d", rollout_index=0
        ),
        EvaluationOutcome(
            node_id="n", task_id="t2", level=2, resolved=False, trajectory_id="e", rollout_index=1
        ),
        EvaluationOutcome(
            node_id="n", task_id="t2", level=2, resolved=False, trajectory_id="f", rollout_index=2
        ),
    ]
    result = evaluator.compute_accuracy("n", "batch", outcomes)
    # Mean over 6 rollouts: 2/6
    assert result.accuracy == 2 / 6
    assert set(result.evaluated_task_ids) == {"t1", "t2"}
    # t1 majority True (2/3); t2 all False
    assert result.solved_task_ids == ["t1"]
    assert result.failed_task_ids == ["t2"]
    assert len(result.outcomes) == 6


def test_evaluate_level2_runs_configured_rollouts(tmp_path):
    calls: list[tuple[str, int, int]] = []

    class FakeRunner:
        def run_task(self, *, node, task, level, seed, run_id, rollout_index=0):
            calls.append((task.task_id, int(seed), int(rollout_index)))
            return EvaluationOutcome(
                node_id=node.node_id,
                task_id=task.task_id,
                level=level,
                resolved=(rollout_index % 2 == 0),
                trajectory_id=f"{task.task_id}_r{rollout_index}",
                rollout_index=rollout_index,
            )

    orch = EvolutionOrchestrator.__new__(EvolutionOrchestrator)
    orch.level2_evaluator = Level2Evaluator()
    orch.solver_runner = FakeRunner()
    orch.config = SimpleNamespace(
        evaluation=SimpleNamespace(solver_rollouts=3),
        run=SimpleNamespace(seed=10),
    )
    result_path = tmp_path / "level2.json"
    orch.run_context = SimpleNamespace(
        run_id="run",
        paths=SimpleNamespace(level2_result=lambda node_id: result_path),
    )

    child = SimpleNamespace(node_id="child", level2_result_path="", solved_task_count=0)
    batch = SimpleNamespace(
        batch_id="b1",
        tasks=[
            SimpleNamespace(task_id="t1"),
            SimpleNamespace(task_id="t2"),
        ],
    )
    result = orch._evaluate_level2(child, batch)

    assert len(calls) == 6  # 2 tasks × 3 rollouts
    assert sorted(calls) == [
        ("t1", 10, 0),
        ("t1", 11, 1),
        ("t1", 12, 2),
        ("t2", 10, 0),
        ("t2", 11, 1),
        ("t2", 12, 2),
    ]
    assert isinstance(result, Level2Result)
    assert len(result.outcomes) == 6
    assert result_path.is_file()
    assert child.level2_result_path == str(result_path)


def test_solver_stats_uses_level2_outcomes_for_rollouts():
    orch = EvolutionOrchestrator.__new__(EvolutionOrchestrator)
    orch.config = SimpleNamespace(execution=SimpleNamespace(scratch_root="/tmp"))
    orch.run_context = SimpleNamespace(run_id="run")
    orch._trajectory_excerpts = lambda node_id: []  # type: ignore[method-assign]

    level2 = Level2Result(
        node_id="n",
        task_batch_id="b",
        evaluated_task_ids=["t1"],
        solved_task_ids=[],
        failed_task_ids=["t1"],
        accuracy=0.5,
        outcomes=[
            EvaluationOutcome(
                node_id="n", task_id="t1", level=2, resolved=True, trajectory_id="r0", rollout_index=0
            ),
            EvaluationOutcome(
                node_id="n", task_id="t1", level=2, resolved=False, trajectory_id="r1", rollout_index=1
            ),
            EvaluationOutcome(
                node_id="n", task_id="t1", level=2, resolved=True, trajectory_id="r2", rollout_index=2
            ),
        ],
    )
    stats = orch._solver_stats(SimpleNamespace(node_id="n"), level2=level2)
    assert stats["solver_rollouts"] == 3
    assert stats["tasks_with_multiple_rollouts"] == 1
    assert stats["stochastic_task_count"] == 1
    assert stats["evaluated_count"] == 3
