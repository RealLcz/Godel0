"""RepoChainWorkflow: the default Proposer task-generation workflow.

RepoChain is a workflow, not a tool and not a mutation-backend mixer. It runs 8
stages: Weakness Identification -> Repository Transfer -> Semantic Chain
Discovery -> Contract Generation -> Chain Mutation -> Causal Ablation ->
Trusted Validation -> Task Packaging.

Stages 1-2 are delegated to the existing TrajectoryAnalyzer and CodeLocator.
Stages 3-6 are currently embedded inside ``RepoChainGenerator`` (in
swesmith/repo_chain.py); this workflow wraps that generator so the existing
multi-file chain logic keeps working while the stages are progressively
extracted into explicit modules. Stage 7 (Trusted Validation) and Stage 8
(Task Packaging) run in the trusted controller, not here.

P1-2 (v1): Stage 5 always uses ``trajectory_conditioned_chain_mutation``.
Weighted mutation_backends are not a Stage-5 selector.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

from .weakness_analysis import WeaknessAnalysisStage
from .repository_transfer import RepositoryTransferStage
from .chain_discovery import ChainDiscoveryStage
from .contract_generation import ContractGenerationStage
from .mutation_planning import MutationPlanningStage
from .causal_ablation import CausalAblationStage

# Fixed v1 Stage-5 operator (matches RepoChainGenerator metadata).
DEFAULT_MUTATION_OPERATOR = "trajectory_conditioned_chain_mutation"


class RepoChainWorkflow:
    """The default RepoChain task-generation workflow.

    Args:
        agent_adapter: LLM adapter for LM-driven steps.
        engine: optional SWESmithEngine (legacy / fallback registry only).
        trajectory_analyzer: existing analyzer for Stage 1.
        code_locator: existing locator for Stage 2.
        require_causal_ablation: if True, candidates failing Stage 6 are dropped.
        config: optional RepoChainWorkflowConfig (or compatible object / dict)
            carrying min/max files, mutation sites, context budget, operator.
        mutation_operator: fixed Stage-5 operator name (v1 default above).
    """

    def __init__(
        self,
        agent_adapter: Any = None,
        engine: Any = None,
        trajectory_analyzer: Any = None,
        code_locator: Any = None,
        require_causal_ablation: bool = True,
        config: Any = None,
        min_files: Optional[int] = None,
        max_files: Optional[int] = None,
        min_mutation_sites: Optional[int] = None,
        max_mutation_sites: Optional[int] = None,
        context_file_budget: Optional[int] = None,
        mutation_operator: Optional[str] = None,
        # Deprecated P1-2: accepted for call-site compatibility, ignored.
        mutation_backend_weights: Optional[dict] = None,
        mutation_backends: Optional[dict] = None,
    ) -> None:
        self.agent_adapter = agent_adapter
        self.engine = engine
        self.trajectory_analyzer = trajectory_analyzer
        self.code_locator = code_locator

        cfg = config
        if isinstance(cfg, dict):
            cfg = type("RCConfig", (), cfg)()
        self.config = cfg

        def _cfg_get(name: str, default: Any) -> Any:
            if cfg is not None and hasattr(cfg, name):
                value = getattr(cfg, name)
                if value is not None:
                    return value
            return default

        self.min_files = int(
            min_files if min_files is not None else _cfg_get("min_files", 2)
        )
        self.max_files = int(
            max_files if max_files is not None else _cfg_get("max_files", 6)
        )
        self.min_mutation_sites = int(
            min_mutation_sites
            if min_mutation_sites is not None
            else _cfg_get("min_mutation_sites", 3)
        )
        self.max_mutation_sites = int(
            max_mutation_sites
            if max_mutation_sites is not None
            else _cfg_get("max_mutation_sites", 8)
        )
        self.context_file_budget = int(
            context_file_budget
            if context_file_budget is not None
            else _cfg_get("context_file_budget", 10)
        )
        self.mutation_operator = str(
            mutation_operator
            if mutation_operator is not None
            else _cfg_get("mutation_operator", DEFAULT_MUTATION_OPERATOR)
            or DEFAULT_MUTATION_OPERATOR
        )
        # Keep empty for diagnostics only; weights no longer select backends.
        self.mutation_backend_weights: dict = {}
        _ = mutation_backend_weights, mutation_backends  # deprecated, ignored

        self.require_causal_ablation = bool(
            require_causal_ablation
            if config is None and not hasattr(cfg, "require_causal_ablation")
            else _cfg_get("require_causal_ablation", require_causal_ablation)
        )
        # Existing passing tests are the default oracle; generated contracts
        # remain an optional enhancement when this flag is True.
        self.require_generated_contracts = bool(
            _cfg_get("require_generated_contracts", False)
        )
        # Local causal ablation: diagnostic (default) or hard_gate (ablation only).
        self.local_causal_ablation_mode = str(
            _cfg_get("local_causal_ablation_mode", "diagnostic")
        ).strip().lower() or "diagnostic"
        self.bootstrap_plans_per_call = max(
            1, int(_cfg_get("bootstrap_plans_per_call", 2) or 2)
        )

        self.weakness_stage = WeaknessAnalysisStage(trajectory_analyzer)
        self.transfer_stage = RepositoryTransferStage(code_locator)
        self.chain_stage = ChainDiscoveryStage()
        self.contract_stage = ContractGenerationStage()
        self.mutation_stage = MutationPlanningStage(
            engine, mutation_operator=self.mutation_operator
        )
        self.ablation_stage = CausalAblationStage()
        self._backing_generator = None

    def _load_backing_generator(self):
        if self._backing_generator is not None:
            return self._backing_generator
        try:
            from swesmith.repo_chain import RepoChainGenerator

            self._backing_generator = RepoChainGenerator(self.agent_adapter)
        except Exception:
            self._backing_generator = None
        return self._backing_generator

    def _apply_constraints_to_plan(self, plan) -> None:
        """Stamp RepoChain difficulty constraints onto plan.constraints.

        P0-2: ``RepoChainGenerator`` reads ``plan.constraints`` (BugConstraints),
        etc., NOT ``plan.task_blueprint["constraints"]``. Writing only the
        blueprint left runtime on BugConstraints defaults (min/max files = 1).

        P1-2: also force strategy/operator to the fixed v1 chain mutation.
        """
        updates = {
            "min_modified_files": self.min_files,
            "max_modified_files": self.max_files,
            "min_mutation_sites": self.min_mutation_sites,
            "max_mutation_sites": self.max_mutation_sites,
            "context_file_budget": self.context_file_budget,
            "require_generated_tests": bool(self.require_generated_contracts),
        }
        constraints = getattr(plan, "constraints", None)
        if constraints is not None and hasattr(constraints, "model_copy"):
            plan.constraints = constraints.model_copy(update=updates)
        elif constraints is not None and hasattr(constraints, "copy"):
            # pydantic v1
            plan.constraints = constraints.copy(update=updates)
        else:
            try:
                from proposer.schemas import BugConstraints

                plan.constraints = BugConstraints(**updates)
            except Exception:
                pass

        # P1-2: Stage 5 is not a weighted backend mixer.
        try:
            plan.strategy = "repo_chain"
        except Exception:
            pass
        try:
            plan.operator = self.mutation_operator
        except Exception:
            pass

        blueprint = getattr(plan, "task_blueprint", None)
        if isinstance(blueprint, dict):
            meta = blueprint.setdefault("constraints", {})
            if not isinstance(meta, dict):
                meta = {}
                blueprint["constraints"] = meta
            meta.update(updates)
            blueprint["mutation_operator"] = self.mutation_operator

    def generate(
        self,
        plan,
        node_code_dir: str,
        repo_spec,
        output_dir: str,
    ) -> List:
        """Run stages 3-6 for one plan and return candidate artifacts.

        Stages 1-2 (weakness analysis, repository transfer) run in the
        ``ProposerRunner`` before plans are created. This method takes an
        already-formed ``BugGenerationPlan`` and delegates to the backing
        ``RepoChainGenerator`` for the chain discovery, contract generation,
        and mutation materialization (fixed operator, not backend weights).

        Local causal ablation is diagnostic by default: candidates are still
        returned so the trusted controller can judge admission.
        """
        self._apply_constraints_to_plan(plan)
        backing = self._load_backing_generator()
        if backing is None:
            return []
        candidates = backing.generate(plan, node_code_dir, repo_spec, output_dir)

        if self.require_causal_ablation and candidates:
            ablation = self.ablation_stage.run(
                plan, repo_spec, candidates, contracts=None
            )
            for candidate in candidates:
                metadata = getattr(candidate, "generation_metadata", None)
                if not isinstance(metadata, dict):
                    continue
                details = getattr(ablation, "details", None)
                metadata["local_causal_ablation"] = {
                    "passed": bool(ablation.passed),
                    "details": details,
                }
                metadata["causal_analysis"] = {
                    "passed_under_old_rule": bool(ablation.passed),
                    "details": details,
                }
            if (
                self.local_causal_ablation_mode == "hard_gate"
                and not ablation.passed
            ):
                return []
        return candidates

    def bootstrap(
        self,
        repo_spec,
        output_dir: str,
        capability_prior: Optional[List[str]] = None,
        target_count: int = 10,
        max_candidates: Optional[int] = None,
        plan_offset: int = 0,
        plan_limit: Optional[int] = None,
    ) -> tuple:
        """Root bootstrap mode: generate T_0 without solver trajectory conditioning.

        Builds capability-prior plans then runs a *chunk* through generate().
        Returns ``(candidates, plans_attempted)`` so zero-yield chunks still
        consume generation budget via ``result.plans``.
        """
        from .bootstrap import BOOTSTRAP_CAPABILITY_PRIOR, build_bootstrap_plans

        prior = capability_prior or BOOTSTRAP_CAPABILITY_PRIOR
        backing = self._load_backing_generator()
        if backing is None:
            return [], []
        chunk_size = max(
            1,
            int(
                plan_limit
                if plan_limit is not None
                else self.bootstrap_plans_per_call
            ),
        )
        offset = max(0, int(plan_offset or 0))
        needed = offset + chunk_size
        # Keep a generous upper bound so later chunks can still be built.
        build_limit = max(needed, int(max_candidates or chunk_size), int(target_count or 1))
        all_plans = build_bootstrap_plans(
            prior,
            repo_spec,
            target_count=max(int(target_count or 1), needed),
            max_plans=build_limit,
            code_locator=self.code_locator,
        )
        plans = all_plans[offset : offset + chunk_size]
        candidates: List = []
        emit_limit = (
            int(max_candidates)
            if max_candidates is not None
            else max(int(target_count or 1), chunk_size)
        )
        for plan in plans:
            if len(candidates) >= emit_limit:
                break
            produced = self.generate(
                plan, "", repo_spec, os.path.join(output_dir, plan.plan_id)
            )
            if not produced:
                # Bootstrap does not route through ProposerRunner's
                # _generate_candidates, so nothing else stamps the engine
                # rejection onto the plan. Without this the controller sees
                # zero-yield chunks with no reason attached and its
                # repo_chain_stats stay empty.
                self._stamp_plan_rejection(plan, backing)
            candidates.extend(produced)
        return candidates, plans

    @staticmethod
    def _stamp_plan_rejection(plan, backing) -> None:
        """Record why one plan produced no candidate onto its blueprint."""
        blueprint = getattr(plan, "task_blueprint", None)
        if not isinstance(blueprint, dict):
            return
        reason = str(getattr(backing, "last_rejection", "") or "")
        if not reason:
            reason = "engine_returned_no_candidates"
        blueprint["last_rejection"] = reason
        stage = str(getattr(backing, "last_rejection_stage", "") or "")
        if not stage:
            try:
                from godel0.schemas.repo_chain_stats import (
                    stage_for_engine_rejection,
                )

                stage = stage_for_engine_rejection(reason)
            except Exception:
                stage = "mutation_failure"
        blueprint["last_rejection_stage"] = stage


__all__ = ["RepoChainWorkflow", "DEFAULT_MUTATION_OPERATOR"]
