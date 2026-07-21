from __future__ import annotations

import os
import uuid
from typing import List, Literal, Optional

from .code_locator import CodeLocator, RepoIndex
from .schemas import BugConstraints, BugGenerationPlan, FailureSignature

PlanStrategy = Literal[
    "lm_modify",
    "lm_rewrite",
    "procedural",
    "combine",
    "pr_mirror",
    "pr_replay",
]


class ProposerPlanner:
    """Creates BugGenerationPlan objects from FailureSignature + CodeTarget.

    The planner decides WHAT to test (capability), which repo/entity to
    target, which strategy/operator to use, and what constraints to apply.
    The SWE-smith engine later decides HOW to materialize the candidate.
    """

    def __init__(
        self,
        code_locator: Optional[CodeLocator] = None,
        default_constraints: Optional[BugConstraints] = None,
    ) -> None:
        self.code_locator = code_locator or CodeLocator()
        self.default_constraints = default_constraints or BugConstraints()
        self._strategy_weights: dict[str, float] = {}
        self._strategy_cursor = 0

    def configure_strategy_policy(
        self,
        weights: dict[str, float],
        *,
        offset: int = 0,
    ) -> None:
        """Apply the trusted run's allowed strategy mix deterministically."""
        supported = set(PlanStrategy.__args__)
        cleaned = {
            str(name): float(weight)
            for name, weight in weights.items()
            if name in supported and float(weight) > 0
        }
        total = sum(cleaned.values())
        self._strategy_weights = (
            {name: weight / total for name, weight in cleaned.items()}
            if total > 0
            else {}
        )
        self._strategy_cursor = max(0, int(offset))

    def create_plan(
        self,
        signature: FailureSignature,
        repo_index: RepoIndex,
        base_commit: str = "",
        used_symbols: Optional[List[str]] = None,
    ) -> Optional[BugGenerationPlan]:
        targets = self.code_locator.locate(
            signature,
            repo_index,
            max_results=5,
            used_symbols=used_symbols,
        )
        if not targets:
            return None
        target = targets[0]
        strategy = self._choose_strategy(signature)
        operator = self._choose_operator(signature, strategy)
        constraints = self._choose_constraints(signature, strategy)
        behavior = signature.behavior_pattern or {}
        reference_files = behavior.get("reference_files") or []
        if isinstance(reference_files, str):
            reference_files = [reference_files]
        target_files = list(reference_files) if strategy == "pr_replay" else [target.file_path]

        return BugGenerationPlan(
            plan_id=f"plan-{uuid.uuid4().hex[:12]}",
            source_trajectory_ids=[signature.source_trajectory_id] if signature.source_trajectory_id else [],
            failure_signature=signature,
            target_repo_id=target.repo_id,
            target_base_commit=base_commit or repo_index.base_commit,
            target_file=target.file_path,
            target_symbol=target.symbol_name,
            target_files=target_files,
            target_symbols=[target.symbol_name] if target.symbol_name else [],
            strategy=strategy,
            operator=operator,
            constraints=constraints,
            rationale=self._build_rationale(signature, target, strategy, operator),
            reference_commit=str(behavior.get("reference_commit") or ""),
            reference_parent=str(behavior.get("reference_parent") or ""),
            reference_patch=str(behavior.get("reference_patch") or ""),
            reference_patch_path=str(behavior.get("reference_patch_path") or ""),
            task_blueprint=self._build_task_blueprint(signature, target),
        )

    def create_plans(
        self,
        signatures: List[FailureSignature],
        repo_index: RepoIndex,
        base_commit: str = "",
        max_plans: int = 10,
    ) -> List[BugGenerationPlan]:
        plans: List[BugGenerationPlan] = []
        used_symbols: List[str] = []
        # Round-robin across failure signatures and code targets.  A batch of
        # K=10 valid tasks usually needs substantially more than ten raw
        # candidates, so stopping after one plan per trajectory is incorrect.
        stalled_rounds = 0
        while signatures and len(plans) < max_plans and stalled_rounds < 2:
            added = 0
            for sig in signatures:
                if len(plans) >= max_plans:
                    break
                plan = self.create_plan(
                    sig,
                    repo_index,
                    base_commit=base_commit,
                    used_symbols=used_symbols,
                )
                if plan is None:
                    continue
                plan.seed = len(plans) + 1
                plans.append(plan)
                used_symbols.append(plan.target_symbol)
                added += 1
            if added == 0:
                # Permit another candidate at an already-used anchor; its
                # seed and generated contract still differ, while trusted
                # duplicate detection remains authoritative.
                used_symbols.clear()
                stalled_rounds += 1
            else:
                stalled_rounds = 0
        return plans

    def _choose_strategy(self, signature: FailureSignature) -> PlanStrategy:
        if self._strategy_weights:
            # Golden-ratio stepping avoids restarting every subprocess on the
            # same strategy while retaining deterministic replay.
            position = (self._strategy_cursor * 0.6180339887498949) % 1.0
            self._strategy_cursor += 1
            cumulative = 0.0
            for name, weight in self._strategy_weights.items():
                cumulative += weight
                if position < cumulative:
                    return name  # type: ignore[return-value]
            return next(reversed(self._strategy_weights))  # type: ignore[return-value]
        behavior = signature.behavior_pattern or {}
        preferred = str(behavior.get("preferred_strategy") or "")
        supported = {
            "lm_modify",
            "lm_rewrite",
            "procedural",
            "combine",
            "pr_mirror",
            "pr_replay",
        }
        if preferred in supported:
            return preferred  # type: ignore[return-value]
        if any(
            behavior.get(key)
            for key in ("reference_commit", "reference_patch", "reference_patch_path")
        ):
            return "pr_replay"
        if signature.failure_stage in ("localization", "tool_use"):
            return "lm_modify"
        if signature.failure_stage == "patch_generation":
            return "lm_modify"
        if signature.failure_stage == "validation":
            return "procedural"
        if signature.failure_stage == "context_management":
            return "lm_rewrite"
        return "lm_modify"

    def _choose_operator(
        self,
        signature: FailureSignature,
        strategy: str,
    ) -> Optional[str]:
        if strategy == "procedural":
            if signature.preferred_operators:
                return signature.preferred_operators[0]
            return "subtle_logic_error"
        if strategy == "lm_modify":
            return "lm_introduce_bug"
        if strategy == "lm_rewrite":
            return "lm_rewrite_with_bug"
        if strategy == "combine":
            return "combine_operators"
        if strategy == "pr_mirror":
            return "mirror_pr"
        if strategy == "pr_replay":
            return "reverse_real_fix"
        return None

    def _choose_constraints(
        self,
        signature: FailureSignature,
        strategy: str,
    ) -> BugConstraints:
        constraints = self.default_constraints.model_copy()
        if strategy in ("pr_replay",):
            constraints.min_modified_files = 2
            constraints.max_modified_files = 6
            constraints.max_modified_lines = 160
            constraints.context_file_budget = 10
            constraints.min_mutation_sites = 3
            constraints.max_mutation_sites = 8
            constraints.require_generated_tests = True
        elif strategy in ("lm_modify", "lm_rewrite"):
            constraints.max_modified_lines = 30
        else:
            constraints.max_modified_lines = 20
        constraints.desired_behavior = (
            f"Induce a failure matching capability '{signature.target_capability}' "
            f"without breaking syntax validity."
        )
        return constraints

    def _build_task_blueprint(self, signature: FailureSignature, target) -> dict:
        blueprint = {
            "capability_gap": signature.target_capability,
            "failure_stage": signature.failure_stage,
            "root_cause": signature.root_cause,
            "anchor": {
                "file": target.file_path,
                "symbol": target.symbol_name,
            },
            "required_topology": "connected_cross_file_contract",
            "source_trajectory_id": signature.source_trajectory_id,
        }
        # Use RepoProfileRegistry to get repo-specific contract settings
        # (no hardcoded repo_id checks).
        from proposer.repo_profiles import get_profile

        profile = get_profile(str(target.repo_id))
        if profile.name != "default":
            blueprint.update(
                {
                    "contract_scenario": profile.contract_scenario,
                    "contract_test_style": profile.contract_test_style,
                    "contract_test_renderer": profile.contract_renderer,
                    "require_expected_counts": profile.require_expected_counts,
                }
            )
        return blueprint

    def _build_rationale(
        self,
        signature: FailureSignature,
        target,
        strategy: str,
        operator: Optional[str],
    ) -> str:
        return (
            f"Signature '{signature.signature_id}' targets capability "
            f"'{signature.target_capability}' (stage={signature.failure_stage}). "
            f"Selected target {target.symbol_name} in {target.file_path} "
            f"with novelty={target.novelty_score:.2f}. "
            f"Strategy={strategy}, operator={operator}."
        )


__all__ = ["ProposerPlanner"]
