"""Root bootstrap capability prior.

When the root node has no solver trajectories yet, RepoChain generates T_0
from a capability prior instead of trajectory-conditioned weakness analysis.
These are capability *categories*, not fixed benchmark tasks.

P0-4: each capability can produce MULTIPLE plans targeting different
subsystems / semantic chains / anchors so a single bootstrap round can fill
K=10 trusted-valid tasks without collapsing to 7 deterministic duplicates.
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional, Set

BOOTSTRAP_CAPABILITY_PRIOR: List[str] = [
    "cross_file_localization",
    "multi_module_state_propagation",
    "configuration_precedence",
    "error_handling",
    "compatibility_preservation",
    "api_contract_reasoning",
    "multi_step_repository_reasoning",
]

# Per-capability semantic-chain variants used to diversify bootstrap plans.
# Each entry is (subsystem_hint, semantic_chain, anchor_hint).
_CAPABILITY_VARIANTS: Dict[str, List[tuple[str, str, str]]] = {
    "cross_file_localization": [
        ("parser", "parser→executor", "parse_entry"),
        ("inventory", "inventory→variable_manager", "inventory_load"),
        ("loader", "loader→task_state", "load_state"),
    ],
    "multi_module_state_propagation": [
        ("state", "state→propagator→consumer", "state_update"),
        ("cache", "cache→invalidator→reader", "cache_key"),
        ("session", "session→auth→handler", "session_id"),
    ],
    "configuration_precedence": [
        ("cli", "cli→config→defaults", "cli_flag"),
        ("env", "env→config→file", "env_var"),
        ("profile", "profile→override→base", "profile_name"),
    ],
    "error_handling": [
        ("io", "reader→error_handler→reporter", "read_error"),
        ("network", "client→retry→fallback", "request_error"),
        ("parse", "parser→validator→error", "parse_error"),
    ],
    "compatibility_preservation": [
        ("api", "legacy_api→adapter→modern", "legacy_call"),
        ("schema", "old_schema→migrator→new", "schema_version"),
        ("format", "v1_format→converter→v2", "format_tag"),
    ],
    "api_contract_reasoning": [
        ("public", "public_api→impl→contract", "public_method"),
        ("plugin", "plugin→host→contract", "plugin_hook"),
        ("rpc", "rpc→serializer→handler", "rpc_method"),
    ],
    "multi_step_repository_reasoning": [
        ("pipeline", "stage1→stage2→stage3", "pipeline_entry"),
        ("build", "discover→compile→link", "build_target"),
        ("deploy", "plan→apply→verify", "deploy_step"),
    ],
}


def build_bootstrap_plans(
    capability_prior: List[str],
    repo_spec,
    *,
    target_count: int = 10,
    max_plans: Optional[int] = None,
) -> List:
    """Build synthetic BugGenerationPlans from the capability prior.

    P0-4: expand each capability across multiple subsystem / semantic-chain
    / anchor variants and track diversity so we can fill K tasks.

    BUG-04: empty ``capability_prior`` yields zero plans (visible error).
    BUG-05: strategy is ``repo_chain`` (not ``lm_modify``).
    """
    try:
        from proposer.schemas import BugGenerationPlan, FailureSignature
    except Exception:
        return []

    limit = max_plans if max_plans is not None else max(target_count * 3, 21)
    used_subsystems: Set[str] = set()
    used_anchors: Set[str] = set()
    used_symbols: Set[str] = set()
    used_semantic_chains: Set[str] = set()

    plans: List = []
    # Round-robin across capabilities x variants until we hit the limit.
    round_idx = 0
    while len(plans) < limit:
        made_progress = False
        for capability in capability_prior:
            variants = _CAPABILITY_VARIANTS.get(capability) or [
                ("generic", f"{capability}→core", capability),
            ]
            if round_idx >= len(variants):
                # Exhausted variants for this capability; synthesize a fresh
                # variant keyed by round so we can still grow toward K.
                subsystem = f"{capability}_sub{round_idx}"
                chain = f"{capability}→variant{round_idx}"
                anchor = f"{capability}_anchor{round_idx}"
            else:
                subsystem, chain, anchor = variants[round_idx]

            # Prefer unused diversity dimensions; skip exact duplicates.
            if (
                subsystem in used_subsystems
                and chain in used_semantic_chains
                and anchor in used_anchors
            ):
                continue

            signature = FailureSignature(
                signature_id=f"bootstrap-{capability}-{round_idx}",
                failure_stage="localization",
                root_cause=f"bootstrap capability prior: {capability}",
                target_capability=capability,
            )
            plan = BugGenerationPlan(
                plan_id=f"bootstrap-{capability}-{round_idx}-{uuid.uuid4().hex[:8]}",
                source_trajectory_ids=[],
                failure_signature=signature,
                target_repo_id=getattr(repo_spec, "repo_id", ""),
                target_base_commit=getattr(repo_spec, "base_commit", ""),
                strategy="repo_chain",
                operator="",
                rationale=(
                    f"bootstrap plan for capability {capability} "
                    f"via {chain} ({subsystem}/{anchor})"
                ),
                task_blueprint={
                    "capability_gap": capability,
                    "failure_stage": "bootstrap",
                    "root_cause": f"bootstrap capability prior: {capability}",
                    "required_topology": "connected_cross_file_contract",
                    "source_trajectory_id": "",
                    "bootstrap": True,
                    "subsystem": subsystem,
                    "semantic_chain": chain,
                    "anchor_hint": anchor,
                    "diversity": {
                        "used_subsystems": sorted(used_subsystems | {subsystem}),
                        "used_anchor_files": sorted(used_anchors | {anchor}),
                        "used_symbols": sorted(used_symbols | {anchor}),
                        "used_semantic_chains": sorted(
                            used_semantic_chains | {chain}
                        ),
                    },
                },
            )
            plans.append(plan)
            used_subsystems.add(subsystem)
            used_anchors.add(anchor)
            used_symbols.add(anchor)
            used_semantic_chains.add(chain)
            made_progress = True
            if len(plans) >= limit:
                break
        if not made_progress:
            break
        round_idx += 1

    return plans


__all__ = ["BOOTSTRAP_CAPABILITY_PRIOR", "build_bootstrap_plans"]
