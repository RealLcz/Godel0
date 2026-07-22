"""P1-3: RepoChain stats must be stage-emitted, not substring-inferred."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from godel0.schemas.evaluation import CandidateValidationReport
from godel0.schemas.repo_chain_stats import (
    CLEAN_CONTRACT_FAILURE,
    CONTRACT_GENERATION_FAILURE,
    MUTATION_FAILURE,
    NO_F2P,
    STATEMENT_LEAKAGE,
    TRUSTED_CAUSAL_FAILURE,
    accumulate_stages,
    empty_repo_chain_stats,
    increment_stage,
    merge_repo_chain_stats,
    stage_for_engine_rejection,
)
from godel0.storage.atomic import atomic_write_json, read_json


def test_stage_for_engine_rejection_classifies_emit_tokens():
    assert stage_for_engine_rejection(
        "clean_contract:generated tests failed on the unmodified repository"
    ) == CLEAN_CONTRACT_FAILURE
    assert (
        stage_for_engine_rejection("invalid_chain_plan:bad assertions")
        == CONTRACT_GENERATION_FAILURE
    )
    assert stage_for_engine_rejection("contract_not_restored") == CONTRACT_GENERATION_FAILURE
    assert stage_for_engine_rejection("mutation_patch_apply_failed:x") == MUTATION_FAILURE
    assert stage_for_engine_rejection("") == MUTATION_FAILURE


def test_accumulate_stages_counts_each_stage_once():
    stats = empty_repo_chain_stats()
    accumulate_stages(stats, [NO_F2P, NO_F2P, TRUSTED_CAUSAL_FAILURE])
    assert stats["no_f2p_count"] == 1
    assert stats["causal_ablation_failure_count"] == 1
    assert stats["mutation_materialization_failure_count"] == 0


def test_validator_report_failure_stages_drive_stats_not_reasons():
    report = CandidateValidationReport(
        candidate_id="c1",
        passed=False,
        rejection_reasons=["no_f2p", "trusted_causal_ablation_failed"],
        failure_stages=[NO_F2P, TRUSTED_CAUSAL_FAILURE],
    )
    stats = empty_repo_chain_stats()
    accumulate_stages(stats, report.failure_stages)
    # Substring noise in reasons must not be re-counted by callers.
    assert stats["no_f2p_count"] == 1
    assert stats["causal_ablation_failure_count"] == 1


def test_batch_style_aggregation_uses_emitted_stages_only():
    stats = empty_repo_chain_stats()
    # Engine emit
    increment_stage(stats, CLEAN_CONTRACT_FAILURE)
    # Validator emit
    accumulate_stages(stats, [NO_F2P, STATEMENT_LEAKAGE])
    # Rejection reason strings that would falsely inflate substring counters.
    rejection_reasons = {
        "no_f2p": 1,
        "statement_audit:leak": 1,
        "something_with_f2p_in_name": 99,
        "causal_ablation_note": 50,
    }
    # Structured path ignores rejection_reasons entirely.
    assert stats == {
        "contract_generation_failure_count": 0,
        "clean_contract_failure_count": 1,
        "mutation_materialization_failure_count": 0,
        "causal_ablation_failure_count": 0,
        "no_f2p_count": 1,
        "no_p2p_count": 0,
        "duplicate_count": 0,
        "statement_leakage_count": 1,
    }
    assert sum(rejection_reasons.values()) > stats["no_f2p_count"]


def test_extended_proposer_stats_does_not_double_count(tmp_path, monkeypatch):
    from godel0.controller.orchestrator import EvolutionOrchestrator

    proposer_dir = tmp_path / "proposer" / "node_a"
    proposer_dir.mkdir(parents=True)
    structured = empty_repo_chain_stats()
    structured["no_f2p_count"] = 2
    structured["causal_ablation_failure_count"] = 1
    atomic_write_json(
        proposer_dir / "generation_summary.json",
        {
            "repo_chain_stats": structured,
            **structured,
            # Poison pill: substring path would inflate these if still used.
            "rejection_reasons": {
                "no_f2p": 10,
                "trusted_causal_ablation_failed": 10,
                "leakage_in_statement": 10,
            },
        },
    )

    orch = EvolutionOrchestrator.__new__(EvolutionOrchestrator)
    orch.run_context = SimpleNamespace(
        paths=SimpleNamespace(proposer_dir=lambda node_id: tmp_path / "proposer" / node_id)
    )
    node = SimpleNamespace(node_id="node_a")
    extended = orch._extended_proposer_stats(node, {})
    assert extended["no_f2p_count"] == 2
    assert extended["causal_ablation_failure_count"] == 1
    assert extended["statement_leakage_count"] == 0


def test_repo_chain_reject_sets_last_rejection_stage():
    from swesmith.repo_chain import RepoChainGenerator

    gen = RepoChainGenerator(agent_adapter=object())
    gen._reject(CLEAN_CONTRACT_FAILURE, "clean_contract:unmodified repository failed")
    assert gen.last_rejection_stage == CLEAN_CONTRACT_FAILURE
    assert "unmodified repository" in gen.last_rejection

    gen._reject_detail("mutation_patch_apply_failed:boom")
    assert gen.last_rejection_stage == MUTATION_FAILURE
    assert gen.last_rejection.startswith("mutation_patch_apply_failed")


def test_merge_repo_chain_stats_sums_layers():
    a = empty_repo_chain_stats()
    b = empty_repo_chain_stats()
    a["no_f2p_count"] = 1
    b["no_f2p_count"] = 2
    b["duplicate_count"] = 4
    merged = merge_repo_chain_stats(a, b)
    assert merged["no_f2p_count"] == 3
    assert merged["duplicate_count"] == 4
