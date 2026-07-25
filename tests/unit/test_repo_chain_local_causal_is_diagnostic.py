"""P0: local causal ablation is diagnostic by default (guide §14.1–14.2)."""

from __future__ import annotations

from types import SimpleNamespace

from proposer.workflows.repo_chain.workflow import RepoChainWorkflow
from proposer.workflows.repo_chain.causal_ablation import AblationResult


class _FakeBacking:
    def __init__(self, candidates):
        self._candidates = candidates

    def generate(self, plan, node_code_dir, repo_spec, output_dir):
        return list(self._candidates)


class _FailingAblation:
    def run(self, plan, repo_spec, candidates, contracts=None):
        return AblationResult(
            passed=False,
            details={"reason": "independently_active_file_count=1<2"},
        )


def _candidate():
    return SimpleNamespace(
        candidate_id="c1",
        generation_metadata={},
    )


def test_local_causal_failure_keeps_candidate_in_diagnostic_mode():
    wf = RepoChainWorkflow(
        config=SimpleNamespace(
            require_causal_ablation=True,
            local_causal_ablation_mode="diagnostic",
            require_generated_contracts=False,
            mutation_operator="trajectory_conditioned_chain_mutation",
        )
    )
    cand = _candidate()
    wf._backing_generator = _FakeBacking([cand])
    wf.ablation_stage = _FailingAblation()

    out = wf.generate(SimpleNamespace(constraints=SimpleNamespace()), "", None, "/tmp")
    assert len(out) == 1
    meta = out[0].generation_metadata
    assert meta["local_causal_ablation"]["passed"] is False
    assert meta["causal_analysis"]["passed_under_old_rule"] is False


def test_local_causal_hard_gate_still_drops_candidate():
    wf = RepoChainWorkflow(
        config=SimpleNamespace(
            require_causal_ablation=True,
            local_causal_ablation_mode="hard_gate",
            require_generated_contracts=False,
            mutation_operator="trajectory_conditioned_chain_mutation",
        )
    )
    cand = _candidate()
    wf._backing_generator = _FakeBacking([cand])
    wf.ablation_stage = _FailingAblation()

    out = wf.generate(SimpleNamespace(constraints=SimpleNamespace()), "", None, "/tmp")
    assert out == []
