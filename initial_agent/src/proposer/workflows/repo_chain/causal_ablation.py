"""Stage 6: Causal Ablation.

Verify that the task has genuine chain-level causal structure:
  - Repairing only one file -> complete contracts still fail.
  - A single isolated mutation -> does it independently trigger the contract?

This excludes tasks that are merely several unrelated single-file bugs
concatenated into a multi-file task. ``causal_ablation_pass`` is the core
quality signal for RepoChain tasks.

BUG-06: previously this stage was a stub that always returned ``passed=True``,
so loosely-coupled single-file bugs could slip through as RepoChain tasks even
when ``require_causal_ablation: true``. The stage now inspects the
``generation_metadata.causal_ablation`` block that the backing
``RepoChainGenerator`` emits for every candidate and rejects candidates that
fail either of the two causal checks below.

The authoritative ablation still runs in the trusted controller-side
``CandidateValidator``; this stage is the RepoChain-local pre-filter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AblationResult:
    passed: bool = True
    repair_one_file_still_fails: bool = True
    single_mutation_triggers_contract: bool = True
    details: Optional[dict] = None


@dataclass
class CausalAblationResult:
    """Structured ablation outcome (BUG-06 acceptance contract).

    Fields mirror the guidance in godel0_bugfix_guide_thompson_repochain_apptainer_cn.md
    so the controller-side scorer / special detector can consume them directly.
    """

    passed: bool
    repair_one_file_results: Dict[str, bool] = field(default_factory=dict)
    all_single_file_repairs_still_fail: bool = False
    independently_active_file_count: int = 0
    reason: str = ""


def _extract_causal_metadata(candidate: Any) -> Optional[dict]:
    """Return the ``causal_ablation`` dict from a candidate artifact, if any."""
    # CandidateArtifact dataclass
    metadata = getattr(candidate, "generation_metadata", None)
    if isinstance(metadata, dict):
        causal = metadata.get("causal_ablation")
        if isinstance(causal, dict):
            return causal
    # dict-shaped candidate
    if isinstance(candidate, dict):
        metadata = candidate.get("generation_metadata") or {}
        causal = metadata.get("causal_ablation") if isinstance(metadata, dict) else None
        if isinstance(causal, dict):
            return causal
    return None


class CausalAblationStage:
    """Stage 6: causal ablation check.

    Enforces two rules when ``require_causal_ablation`` is active:

    1. ``repair_only_one_file_all_fail`` -- repairing any single required file
       must NOT restore all target contracts.
    2. ``independently_active_file_count >= min_independently_active`` -- at
       least ``min_independently_active`` files must independently trigger the
       contract when mutated in isolation.

    Candidates whose backing generator did not emit a ``causal_ablation`` block
    are treated as failing, so misconfigured generators cannot silently bypass
    the gate.
    """

    def __init__(self, min_independently_active: int = 2) -> None:
        self.min_independently_active = max(1, int(min_independently_active))

    def run(
        self,
        plan: Any,
        repo_spec: Any,
        candidates: List[Any],
        contracts: Any = None,
    ) -> AblationResult:
        if not candidates:
            # Nothing to ablate; defer to upstream validation.
            return AblationResult(passed=True, details={"reason": "no_candidates"})

        repair_results: Dict[str, bool] = {}
        all_fail = True
        independently_active = 0
        rejected: List[str] = []

        for candidate in candidates:
            causal = _extract_causal_metadata(candidate)
            if causal is None:
                # Missing ablation metadata -> cannot prove causal structure.
                rejected.append(_candidate_id(candidate))
                continue

            # repair_only_one_file_passed: dict[file, bool]; True = single-file
            # repair succeeded (bad for a RepoChain task).
            repair_only = causal.get("repair_only_one_file_passed") or {}
            if isinstance(repair_only, dict):
                repair_results.update({str(k): bool(v) for k, v in repair_only.items()})
                if any(bool(v) for v in repair_only.values()):
                    all_fail = False

            independent = int(causal.get("independently_active_file_count", 0) or 0)
            if independent > independently_active:
                independently_active = independent

            passes_gate = bool(
                causal.get("repair_only_one_file_all_fail")
            ) and independent >= self.min_independently_active

            if not passes_gate:
                rejected.append(_candidate_id(candidate))

        passed = not rejected and all_fail and independently_active >= self.min_independently_active
        reason = ""
        if not passed:
            parts = []
            if not all_fail:
                parts.append("single_file_repair_restored_contract")
            if independently_active < self.min_independently_active:
                parts.append(
                    f"independently_active_file_count={independently_active}<{self.min_independently_active}"
                )
            if rejected:
                parts.append(f"rejected_candidates={len(rejected)}")
            reason = ",".join(parts)

        structured = CausalAblationResult(
            passed=passed,
            repair_one_file_results=repair_results,
            all_single_file_repairs_still_fail=all_fail,
            independently_active_file_count=independently_active,
            reason=reason,
        )
        return AblationResult(
            passed=passed,
            repair_one_file_still_fails=all_fail,
            single_mutation_triggers_contract=(
                independently_active >= self.min_independently_active
            ),
            details={
                "structured": asdict_safe(structured),
                "rejected_candidate_ids": rejected,
            },
        )


def _candidate_id(candidate: Any) -> str:
    for attr in ("candidate_id", "instance_id", "id"):
        value = getattr(candidate, attr, None)
        if value:
            return str(value)
    if isinstance(candidate, dict):
        for key in ("candidate_id", "instance_id", "id"):
            if candidate.get(key):
                return str(candidate[key])
    return "<unknown>"


def asdict_safe(obj: Any) -> dict:
    """Best-effort dataclass -> dict conversion (avoids importing dataclasses.asdict)."""
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return dict(obj) if isinstance(obj, dict) else {}


__all__ = [
    "AblationResult",
    "CausalAblationResult",
    "CausalAblationStage",
]
