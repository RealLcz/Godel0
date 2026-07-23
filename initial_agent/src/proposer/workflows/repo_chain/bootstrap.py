"""Root bootstrap capability prior with real repository anchor selection.

When the root node has no solver trajectories yet, RepoChain generates T_0
from a capability prior instead of trajectory-conditioned weakness analysis.

P0-1 (bootstrap anchors): capability / subsystem / chain hints alone are NOT
enough. ``RepoChainGenerator`` expands context via
``declared_target_files(plan)`` → ``_related_files(requested=...)``, which
only reads ``plan.target_file`` / ``plan.target_files``. Plans that only
stuff ``task_blueprint["anchor_hint"]`` therefore produce
``insufficient_context_files`` and zero candidates.

Correct flow:

    Capability Prior
        → Synthetic FailureSignature (with code_patterns)
        → RepoIndex + CodeLocator.locate()
        → real target_file / target_symbol / target_files
        → RepoChain
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional, Set

BOOTSTRAP_CAPABILITY_PRIOR: List[str] = [
    "cross_file_localization",
    "multi_module_state_propagation",
    "configuration_precedence",
    "error_handling",
    "compatibility_preservation",
    "api_contract_reasoning",
    "multi_step_repository_reasoning",
]

# Per-capability search variants. Tokens become FailureSignature.code_patterns
# so CodeLocator can resolve them to real repository files/symbols.
# Each entry is (subsystem_hint, semantic_chain, search_tokens).
_CAPABILITY_VARIANTS: Dict[str, List[tuple[str, str, List[str]]]] = {
    "cross_file_localization": [
        ("parser", "parser→executor", ["parser", "parse", "executor", "tokenize"]),
        ("inventory", "inventory→variable_manager", ["inventory", "host", "variable"]),
        ("loader", "loader→task_state", ["loader", "load", "task", "play"]),
    ],
    "multi_module_state_propagation": [
        ("state", "state→propagator→consumer", ["state", "context", "variable", "manager"]),
        ("cache", "cache→invalidator→reader", ["cache", "fact", "gather"]),
        ("session", "session→auth→handler", ["session", "auth", "connection"]),
    ],
    "configuration_precedence": [
        ("cli", "cli→config→defaults", ["cli", "config", "option", "argument"]),
        ("env", "env→config→file", ["environ", "config", "setting"]),
        ("profile", "profile→override→base", ["profile", "override", "default"]),
    ],
    "error_handling": [
        ("io", "reader→error_handler→reporter", ["error", "exception", "fail", "display"]),
        ("network", "client→retry→fallback", ["retry", "timeout", "connection"]),
        ("parse", "parser→validator→error", ["validate", "parse", "error"]),
    ],
    "compatibility_preservation": [
        ("api", "legacy_api→adapter→modern", ["compat", "legacy", "deprecate", "adapter"]),
        ("schema", "old_schema→migrator→new", ["schema", "migrate", "version"]),
        ("format", "v1_format→converter→v2", ["format", "convert", "serialize"]),
    ],
    "api_contract_reasoning": [
        ("public", "public_api→impl→contract", ["public", "api", "interface", "entrypoint"]),
        ("plugin", "plugin→host→contract", ["plugin", "loader", "module_utils"]),
        ("rpc", "rpc→serializer→handler", ["rpc", "serialize", "handler"]),
    ],
    "multi_step_repository_reasoning": [
        ("pipeline", "stage1→stage2→stage3", ["pipeline", "executor", "strategy", "play"]),
        ("build", "discover→compile→link", ["build", "discover", "collection"]),
        ("deploy", "plan→apply→verify", ["deploy", "apply", "run", "playbook"]),
    ],
}


def _capability_tokens(capability: str) -> List[str]:
    return [tok for tok in capability.lower().replace("-", "_").split("_") if len(tok) >= 3]


def bootstrap_signature(
    capability: str,
    *,
    round_idx: int = 0,
    search_tokens: Optional[List[str]] = None,
) -> Any:
    """Build a synthetic FailureSignature for one capability prior entry."""
    from proposer.schemas import FailureSignature

    tokens = list(search_tokens or [])
    tokens.extend(_capability_tokens(capability))
    # Deduplicate while preserving order.
    seen: Set[str] = set()
    patterns: List[str] = []
    for tok in tokens:
        key = str(tok).lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        patterns.append(key)

    return FailureSignature(
        signature_id=f"bootstrap-{capability}-{round_idx}",
        failure_stage="localization",
        root_cause=f"bootstrap capability prior: {capability}",
        target_capability=capability,
        code_patterns=patterns,
        transfer_mode="same_repo_nearby",
    )


def _subsystem_from_path(file_path: str) -> str:
    """Derive a coarse subsystem key from a real repository-relative path."""
    parts = str(file_path).replace("\\", "/").split("/")
    # Prefer the package directory under common roots (lib/ansible/X/...).
    for idx, part in enumerate(parts):
        if part in {"lib", "src", "pkg"} and idx + 2 < len(parts):
            return "/".join(parts[idx + 1 : idx + 3])
        if part in {"ansible", "godel0"} and idx + 1 < len(parts):
            return "/".join(parts[idx : idx + 2])
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return parts[0] if parts else ""


def build_bootstrap_plans(
    capability_prior: List[str],
    repo_spec,
    *,
    target_count: int = 10,
    max_plans: Optional[int] = None,
    repo_index: Any = None,
    code_locator: Any = None,
) -> List:
    """Build BugGenerationPlans with *real* repository anchors.

    Without ``repo_index`` + ``code_locator``, this cannot resolve
    ``target_file`` / ``target_symbol`` and returns an empty list (fail loud)
    rather than emitting unusable plans that will all die at
    ``insufficient_context_files``.
    """
    try:
        from proposer.schemas import BugConstraints, BugGenerationPlan
        from proposer.code_locator import CodeLocator, RepoIndex
    except Exception:
        return []

    if repo_index is None:
        repo_dir = (
            getattr(repo_spec, "repo_dir", None)
            or getattr(repo_spec, "repo_path", None)
            or getattr(repo_spec, "path", None)
            or ""
        )
        if not repo_dir or not os.path.isdir(str(repo_dir)):
            return []
        source_dirs = ["."]
        try:
            from proposer.repo_profiles import get_profile

            profile = get_profile(getattr(repo_spec, "repo_id", "") or "")
            source_dirs = list(getattr(profile, "source_roots", None) or ["."])
        except Exception:
            pass
        repo_index = RepoIndex.build(
            repo_id=getattr(repo_spec, "repo_id", "") or "repo",
            repo_dir=str(repo_dir),
            base_commit=getattr(repo_spec, "base_commit", "") or "",
            source_dirs=source_dirs,
        )

    if code_locator is None:
        code_locator = CodeLocator()

    if not getattr(repo_index, "symbols", None):
        return []

    limit = max_plans if max_plans is not None else max(target_count * 3, 21)
    used_files: Set[str] = set()
    used_symbols: Set[str] = set()
    used_subsystems: Set[str] = set()
    used_semantic_chains: Set[str] = set()

    plans: List = []
    round_idx = 0
    stalled_rounds = 0
    while len(plans) < limit and stalled_rounds < 3:
        made_progress = False
        for capability in capability_prior:
            if len(plans) >= limit:
                break
            variants = _CAPABILITY_VARIANTS.get(capability) or [
                ("generic", f"{capability}→core", _capability_tokens(capability)),
            ]
            if round_idx < len(variants):
                subsystem_hint, chain, search_tokens = variants[round_idx]
            else:
                subsystem_hint = f"{capability}_sub{round_idx}"
                chain = f"{capability}→variant{round_idx}"
                search_tokens = _capability_tokens(capability)

            signature = bootstrap_signature(
                capability,
                round_idx=round_idx,
                search_tokens=search_tokens,
            )
            # Ask for more candidates than we need so we can skip already-used
            # real files / symbols / subsystems.
            candidates = code_locator.locate(
                signature,
                repo_index,
                max_results=12,
                used_symbols=sorted(used_symbols),
            )
            target = None
            for cand in candidates:
                file_path = str(getattr(cand, "file_path", "") or "")
                symbol = str(getattr(cand, "symbol_name", "") or "")
                if not file_path or not symbol:
                    continue
                subsystem = _subsystem_from_path(file_path)
                if file_path in used_files:
                    continue
                if symbol in used_symbols:
                    continue
                # Soft preference: avoid repeating the same subsystem when
                # alternatives exist, but allow it after we stall.
                if subsystem and subsystem in used_subsystems and stalled_rounds == 0:
                    continue
                target = cand
                break

            if target is None and candidates and stalled_rounds > 0:
                # Last resort: reuse a symbol-free slot but still require a
                # concrete file we have not used yet.
                for cand in candidates:
                    file_path = str(getattr(cand, "file_path", "") or "")
                    if file_path and file_path not in used_files:
                        target = cand
                        break

            if target is None:
                continue

            file_path = str(target.file_path)
            symbol = str(target.symbol_name)
            subsystem = _subsystem_from_path(file_path)

            # Seed multiple real files into target_files so _related_files can
            # expand even when import-following is sparse. Primary remains
            # targets[0]; extras are other positive locate hits not yet used.
            seed_files = [file_path]
            seed_symbols = [symbol] if symbol else []
            for cand in candidates:
                if len(seed_files) >= 3:
                    break
                extra = str(getattr(cand, "file_path", "") or "")
                extra_sym = str(getattr(cand, "symbol_name", "") or "")
                if not extra or extra in seed_files or extra in used_files:
                    continue
                seed_files.append(extra)
                if extra_sym and extra_sym not in seed_symbols:
                    seed_symbols.append(extra_sym)

            plan = BugGenerationPlan(
                plan_id=f"bootstrap-{capability}-{round_idx}-{uuid.uuid4().hex[:8]}",
                source_trajectory_ids=[],
                failure_signature=signature,
                target_repo_id=getattr(repo_spec, "repo_id", "") or repo_index.repo_id,
                target_base_commit=(
                    getattr(repo_spec, "base_commit", "") or repo_index.base_commit
                ),
                # Critical: real anchors consumed by declared_target_files().
                target_file=file_path,
                target_symbol=symbol,
                target_files=seed_files,
                target_symbols=seed_symbols,
                strategy="repo_chain",
                operator="",
                constraints=BugConstraints(
                    min_modified_files=2,
                    max_modified_files=6,
                    min_mutation_sites=3,
                    max_mutation_sites=8,
                    context_file_budget=10,
                    require_generated_tests=False,
                ),
                rationale=(
                    f"bootstrap plan for capability {capability} "
                    f"anchored at {file_path}::{symbol} via {chain}"
                ),
                task_blueprint={
                    "capability_gap": capability,
                    "failure_stage": "bootstrap",
                    "root_cause": f"bootstrap capability prior: {capability}",
                    "required_topology": "connected_cross_file_contract",
                    # P0-5: bootstrap has no failure trajectory source — leave
                    # identity fields empty rather than inventing node/task ids.
                    "source_type": "bootstrap",
                    "source_node_id": "",
                    "source_task_id": "",
                    "source_trajectory_id": "",
                    "source_failure_stage": "bootstrap",
                    "bootstrap": True,
                    "subsystem": subsystem or subsystem_hint,
                    "subsystem_hint": subsystem_hint,
                    "semantic_chain": chain,
                    "anchor_hint": search_tokens[0] if search_tokens else capability,
                    "anchor_file": file_path,
                    "anchor_symbol": symbol,
                    "diversity": {
                        "used_subsystems": sorted(used_subsystems | {subsystem}),
                        "used_anchor_files": sorted(used_files | {file_path}),
                        "used_symbols": sorted(used_symbols | {symbol}),
                        "used_semantic_chains": sorted(
                            used_semantic_chains | {chain}
                        ),
                    },
                },
            )
            plans.append(plan)
            used_files.add(file_path)
            if symbol:
                used_symbols.add(symbol)
            if subsystem:
                used_subsystems.add(subsystem)
            used_semantic_chains.add(chain)
            made_progress = True

        if not made_progress:
            stalled_rounds += 1
        else:
            stalled_rounds = 0
        round_idx += 1

    return plans


__all__ = [
    "BOOTSTRAP_CAPABILITY_PRIOR",
    "bootstrap_signature",
    "build_bootstrap_plans",
]
