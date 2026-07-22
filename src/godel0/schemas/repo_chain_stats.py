"""Structured RepoChain / trusted-validation failure stages (P1-3).

Stages emit these codes directly. Orchestrator aggregates counts — it must
NOT re-infer stats by substring-matching rejection_reasons (double-count risk).
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, MutableMapping, Optional

# Canonical stage codes (emitted by generators / validator).
CONTRACT_GENERATION_FAILURE = "contract_generation_failure"
CLEAN_CONTRACT_FAILURE = "clean_contract_failure"
MUTATION_FAILURE = "mutation_failure"
TRUSTED_CAUSAL_FAILURE = "trusted_causal_failure"
NO_F2P = "no_f2p"
NO_P2P = "no_p2p"
DUPLICATE = "duplicate"
STATEMENT_LEAKAGE = "statement_leakage"

# Stats dict keys consumed by special detectors / HGM gates.
STAT_KEY_BY_STAGE = {
    CONTRACT_GENERATION_FAILURE: "contract_generation_failure_count",
    CLEAN_CONTRACT_FAILURE: "clean_contract_failure_count",
    MUTATION_FAILURE: "mutation_materialization_failure_count",
    TRUSTED_CAUSAL_FAILURE: "causal_ablation_failure_count",
    NO_F2P: "no_f2p_count",
    NO_P2P: "no_p2p_count",
    DUPLICATE: "duplicate_count",
    STATEMENT_LEAKAGE: "statement_leakage_count",
}

ALL_STAT_KEYS = tuple(STAT_KEY_BY_STAGE.values())


def empty_repo_chain_stats() -> Dict[str, int]:
    return {key: 0 for key in ALL_STAT_KEYS}


def increment_stage(
    stats: MutableMapping[str, int],
    stage: Optional[str],
    *,
    amount: int = 1,
) -> None:
    """Increment the counter for one emitted stage code (no string guessing)."""
    if not stage:
        return
    key = STAT_KEY_BY_STAGE.get(str(stage).strip())
    if not key:
        return
    stats[key] = int(stats.get(key, 0) or 0) + int(amount)


def merge_repo_chain_stats(
    *layers: Optional[Mapping[str, int]],
) -> Dict[str, int]:
    out = empty_repo_chain_stats()
    for layer in layers:
        if not isinstance(layer, Mapping):
            continue
        for key in ALL_STAT_KEYS:
            if key in layer:
                out[key] = int(out.get(key, 0) or 0) + int(layer.get(key, 0) or 0)
    return out


def accumulate_stages(
    stats: MutableMapping[str, int],
    stages: Iterable[str],
) -> None:
    """Count each unique stage once (per candidate / plan)."""
    seen = set()
    for stage in stages:
        code = str(stage or "").strip()
        if not code or code in seen:
            continue
        seen.add(code)
        increment_stage(stats, code)


# Prefix / exact maps used only at the *emit* boundary inside RepoChainGenerator
# when assigning last_rejection_stage alongside last_rejection.
_ENGINE_PREFIX_TO_STAGE = (
    ("clean_contract", CLEAN_CONTRACT_FAILURE),
    ("invalid_chain_plan", CONTRACT_GENERATION_FAILURE),
    ("contract_not_restored", CONTRACT_GENERATION_FAILURE),
    ("mutation_", MUTATION_FAILURE),
    ("generated_contract_did_not_fail", MUTATION_FAILURE),
    ("target_contract_did_not_fail", MUTATION_FAILURE),
    ("compatibility_control_failed", MUTATION_FAILURE),
    ("bug_patch_", MUTATION_FAILURE),
    ("oracle_patch_", MUTATION_FAILURE),
    ("insufficient_context", MUTATION_FAILURE),
    ("missing_", MUTATION_FAILURE),
    ("generation_error", MUTATION_FAILURE),
)


def stage_for_engine_rejection(reason: str) -> str:
    """Assign a stage code when RepoChain sets last_rejection.

    This is emit-time classification of *RepoChain's own* rejection tokens,
    not a second pass over trusted rejection_reasons in the orchestrator.
    """
    text = str(reason or "").strip().lower()
    if not text:
        return MUTATION_FAILURE
    # Prefer clean-contract signals even when wrapped as invalid_chain_plan:...
    if text.startswith("clean_contract") or "unmodified repository" in text:
        return CLEAN_CONTRACT_FAILURE
    for prefix, stage in _ENGINE_PREFIX_TO_STAGE:
        if text.startswith(prefix) or prefix in text:
            return stage
    return MUTATION_FAILURE
