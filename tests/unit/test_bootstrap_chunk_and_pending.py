"""P0: bootstrap pending-only + chunk offset + zero-yield budget (guide §14.3–14.5)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from proposer.request import ProposerRequest, ProposerResult, RepoSpecInfo
from proposer.runner import ProposerRunner
from proposer.workflows.repo_chain.workflow import RepoChainWorkflow


def _request(**overrides) -> ProposerRequest:
    base = dict(
        node_id="root",
        run_id="run",
        agent_code_dir="/tmp/agent",
        repo_pool_dir="/tmp/pool",
        task_store_dir="/tmp/store",
        output_dir="/tmp/godel0_boot",
        bootstrap=True,
        target_batch_size=2,
        max_candidates=2,
        generation_attempt=0,
        workflow_config={"bootstrap_plans_per_call": 2},
        repo_specs=[
            RepoSpecInfo(
                repo_id="ansible",
                path="/tmp/repo",
                base_commit="HEAD",
                test_command="pytest",
            )
        ],
    )
    base.update(overrides)
    return ProposerRequest(**base)


def test_bootstrap_candidate_only_enters_pending():
    cand = SimpleNamespace(
        candidate_id="boot-1",
        plan_id="p1",
        generation_metadata={},
    )
    plan = SimpleNamespace(
        plan_id="bootstrap-plan-0",
        model_dump=lambda: {"plan_id": "bootstrap-plan-0"},
    )
    result = ProposerResult(run_id="run", node_id="root")
    candidates, plans = [cand], [plan]
    for c in candidates:
        result.add_pending_candidate(c)
    result.plans = [p.model_dump() for p in plans]
    result.completed = True

    assert result.accepted_candidates == []
    assert len(result.pending_candidates) == 1
    assert result.pending_candidates[0].candidate_id == "boot-1"


def test_bootstrap_chunk_offset_uses_generation_attempt():
    calls = []

    class _WF:
        bootstrap_plans_per_call = 2

        def bootstrap(self, **kwargs):
            calls.append(dict(kwargs))
            off = int(kwargs["plan_offset"])
            plan0 = SimpleNamespace(plan_id=f"plan-{off}")
            plan1 = SimpleNamespace(plan_id=f"plan-{off + 1}")
            return [], [plan0, plan1]

    runner = ProposerRunner(
        agent_adapter=SimpleNamespace(),
        allow_workflow_fallback=True,
        workflow_config={"bootstrap_plans_per_call": 2},
    )
    runner._workflow = _WF()
    base = _request()
    _, plans0 = runner._bootstrap_candidates(base)
    _, plans1 = runner._bootstrap_candidates(
        replace(base, generation_attempt=1, output_dir="/tmp/godel0_boot_offset1")
    )
    assert calls[0]["plan_offset"] == 0
    assert calls[0]["plan_limit"] == 2
    assert calls[1]["plan_offset"] == 2
    assert [p.plan_id for p in plans0] == ["plan-0", "plan-1"]
    assert [p.plan_id for p in plans1] == ["plan-2", "plan-3"]


def test_zero_yield_chunk_still_reports_plans_for_budget():
    """TaskBatchBuilder advances when generated_this_attempt uses len(plans)."""
    runner = ProposerRunner(
        agent_adapter=SimpleNamespace(),
        allow_workflow_fallback=True,
        workflow_config={"bootstrap_plans_per_call": 2},
    )

    class _WF:
        def bootstrap(self, **kwargs):
            plans = [
                SimpleNamespace(plan_id="a", model_dump=lambda: {"plan_id": "a"}),
                SimpleNamespace(plan_id="b", model_dump=lambda: {"plan_id": "b"}),
            ]
            return [], plans

    runner._workflow = _WF()
    request = _request(output_dir="/tmp/godel0_boot_zero")
    candidates, plans = runner._bootstrap_candidates(request)
    result = ProposerResult(run_id="run", node_id="root")
    for cand in candidates:
        result.add_pending_candidate(cand)
    result.plans = [
        p.model_dump() if hasattr(p, "model_dump") else {"plan_id": p.plan_id}
        for p in plans
    ]
    result.completed = True

    generated_this_attempt = (
        len(result.accepted_candidates)
        + len(result.rejected_candidates)
        + len(result.pending_candidates)
    )
    generated_this_attempt = max(generated_this_attempt, len(result.plans))
    assert candidates == []
    assert generated_this_attempt == 2


def test_workflow_bootstrap_slices_by_offset(monkeypatch):
    wf = RepoChainWorkflow(
        config=SimpleNamespace(
            bootstrap_plans_per_call=2,
            local_causal_ablation_mode="diagnostic",
            require_causal_ablation=False,
            require_generated_contracts=False,
            mutation_operator="trajectory_conditioned_chain_mutation",
        )
    )
    wf._backing_generator = SimpleNamespace(generate=lambda *a, **k: [])

    def fake_build(prior, repo_spec, target_count=10, max_plans=None, code_locator=None):
        return [
            SimpleNamespace(plan_id=f"p{i}", constraints=SimpleNamespace())
            for i in range(max_plans or 4)
        ]

    monkeypatch.setattr(
        "proposer.workflows.repo_chain.bootstrap.build_bootstrap_plans",
        fake_build,
    )
    cands0, plans0 = wf.bootstrap(
        repo_spec=SimpleNamespace(repo_id="ansible"),
        output_dir="/tmp/out0",
        target_count=2,
        max_candidates=2,
        plan_offset=0,
        plan_limit=2,
    )
    cands1, plans1 = wf.bootstrap(
        repo_spec=SimpleNamespace(repo_id="ansible"),
        output_dir="/tmp/out1",
        target_count=2,
        max_candidates=2,
        plan_offset=2,
        plan_limit=2,
    )
    assert [p.plan_id for p in plans0] == ["p0", "p1"]
    assert [p.plan_id for p in plans1] == ["p2", "p3"]
    assert cands0 == [] and cands1 == []
