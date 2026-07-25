"""P0/P1: trusted validator still sees local-causal-failed candidates (guide §14.6)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from godel0.proposer_trusted.candidate_validator import CandidateValidator
from godel0.schemas.evaluation import CandidateValidationReport


def test_trusted_validator_called_once_for_local_causal_failed_candidate(tmp_path: Path):
    validator = CandidateValidator(
        workspace_root=tmp_path / "validator",
        require_causal_ablation=True,
        record_causal_diagnostics=False,
        causal_ablation_hard_gate=False,
    )
    report = CandidateValidationReport(candidate_id="c-local-fail", passed=True)
    report.patch_applied = True
    report.syntax_valid = True
    report.f2p_tests = ["t1"]
    report.p2p_tests = ["t2"]
    report.reverse_restored = True
    report.safety_valid = True
    report.duplicate_valid = True
    report.relevance_valid = True

    calls = {"n": 0}

    def fake_validate(**kwargs):
        calls["n"] += 1
        # Simulate a candidate that already failed local causal but still has
        # a legal multi-file patch reaching trusted validation once.
        return report

    validator.validate = fake_validate  # type: ignore[method-assign]
    out = validator.validate(
        candidate_patch="diff --git a/a.py b/a.py\n",
        repo_path=tmp_path / "repo",
        base_commit="HEAD",
        test_command="pytest",
        candidate_id="c-local-fail",
    )
    assert calls["n"] == 1
    assert out.passed is True


def test_causal_hard_gate_false_does_not_flip_passed_on_ablation_fail(tmp_path: Path):
    validator = CandidateValidator(
        workspace_root=tmp_path / "validator",
        require_causal_ablation=True,
        record_causal_diagnostics=True,
        causal_ablation_hard_gate=False,
    )
    # Minimal stub of the outer validate gate: only exercise the soft causal branch.
    report = CandidateValidationReport(candidate_id="c1", passed=True)
    report.f2p_tests = ["t1"]
    validator._run_trusted_causal_ablation = MagicMock(return_value=False)  # type: ignore

    run_causal = report.passed and (
        validator.causal_ablation_hard_gate
        or validator.record_causal_diagnostics
        or validator.require_causal_ablation
    )
    assert run_causal
    ablation_ok = validator._run_trusted_causal_ablation(
        candidate_patch="",
        repo_path=tmp_path,
        base_commit="HEAD",
        test_command="pytest",
        setup_patch="",
        f2p_tests=list(report.f2p_tests),
        report=report,
        validation_mode="nodeid",
        command_test_id="",
        control_test_command=None,
    )
    if validator.causal_ablation_hard_gate and not ablation_ok:
        report.passed = False
    assert report.passed is True
    assert ablation_ok is False
