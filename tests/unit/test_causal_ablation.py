"""Unit tests for the CausalAblationStage (BUG-06)."""
from __future__ import annotations

from types import SimpleNamespace

from proposer.workflows.repo_chain.causal_ablation import (
    AblationResult,
    CausalAblationResult,
    CausalAblationStage,
)


def _candidate(candidate_id="c1", causal=None):
    """Build a fake candidate with a generation_metadata.causal_ablation block."""
    metadata = {}
    if causal is not None:
        metadata["causal_ablation"] = causal
    return SimpleNamespace(
        candidate_id=candidate_id,
        generation_metadata=metadata,
    )


class TestCausalAblationStage:
    def test_passes_when_no_candidates(self):
        stage = CausalAblationStage()
        result = stage.run(plan=None, repo_spec=None, candidates=[], contracts=None)
        assert result.passed is True

    def test_rejects_candidate_missing_causal_metadata(self):
        stage = CausalAblationStage()
        cand = _candidate("c1", causal=None)  # no causal_ablation block
        result = stage.run(plan=None, repo_spec=None, candidates=[cand], contracts=None)
        assert result.passed is False

    def test_rejects_single_file_repair_restores_contract(self):
        stage = CausalAblationStage()
        causal = {
            "repair_only_one_file_passed": {"file_a.py": True},
            "repair_only_one_file_all_fail": False,
            "isolated_file_triggers_contract": {"file_a.py": True, "file_b.py": True},
            "independently_active_file_count": 2,
        }
        cand = _candidate("c1", causal=causal)
        result = stage.run(plan=None, repo_spec=None, candidates=[cand], contracts=None)
        assert result.passed is False

    def test_rejects_when_only_one_independently_active_file(self):
        stage = CausalAblationStage(min_independently_active=2)
        causal = {
            "repair_only_one_file_passed": {"file_a.py": False, "file_b.py": False},
            "repair_only_one_file_all_fail": True,
            "isolated_file_triggers_contract": {"file_a.py": True, "file_b.py": False},
            "independently_active_file_count": 1,  # below min of 2
        }
        cand = _candidate("c1", causal=causal)
        result = stage.run(plan=None, repo_spec=None, candidates=[cand], contracts=None)
        assert result.passed is False

    def test_passes_when_all_gates_satisfied(self):
        stage = CausalAblationStage(min_independently_active=2)
        causal = {
            "repair_only_one_file_passed": {"file_a.py": False, "file_b.py": False, "file_c.py": False},
            "repair_only_one_file_all_fail": True,
            "isolated_file_triggers_contract": {"file_a.py": True, "file_b.py": True, "file_c.py": True},
            "independently_active_file_count": 3,
        }
        cand = _candidate("c1", causal=causal)
        result = stage.run(plan=None, repo_spec=None, candidates=[cand], contracts=None)
        assert result.passed is True
        assert result.repair_one_file_still_fails is True
