"""RepoChainWorkflow: the default Proposer task-generation workflow.

RepoChain is a workflow, not a tool and not a mutation backend. It runs 8
stages: Weakness Identification -> Repository Transfer -> Semantic Chain
Discovery -> Contract Generation -> Chain Mutation -> Causal Ablation ->
Trusted Validation -> Task Packaging.

Stages 1-2 are delegated to the existing TrajectoryAnalyzer and CodeLocator.
Stages 3-6 are currently embedded inside ``RepoChainGenerator`` (in
swesmith/repo_chain.py); this workflow wraps that generator so the existing
multi-file chain logic keeps working while the stages are progressively
extracted into explicit modules. Stage 7 (Trusted Validation) and Stage 8
(Task Packaging) run in the trusted controller, not here.
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


class RepoChainWorkflow:
    """The default RepoChain task-generation workflow.

    Args:
        agent_adapter: LLM adapter for LM-driven steps.
        engine: SWESmithEngine (mutation backend registry). The workflow uses
            it as the mutation backend in Stage 5.
        trajectory_analyzer: existing analyzer for Stage 1.
        code_locator: existing locator for Stage 2.
        mutation_backend_weights: weights for selecting mutation backends.
        require_causal_ablation: if True, candidates failing Stage 6 are dropped.
        config: optional RepoChainWorkflowConfig (or compatible object / dict)
            carrying min/max files, mutation sites, context budget, backends.
    """

    def __init__(
        self,
        agent_adapter: Any = None,
        engine: Any = None,
        trajectory_analyzer: Any = None,
        code_locator: Any = None,
        mutation_backend_weights: Optional[dict] = None,
        require_causal_ablation: bool = True,
        config: Any = None,
        min_files: Optional[int] = None,
        max_files: Optional[int] = None,
        min_mutation_sites: Optional[int] = None,
        max_mutation_sites: Optional[int] = None,
        context_file_budget: Optional[int] = None,
        mutation_backends: Optional[dict] = None,
    ) -> None:
        self.agent_adapter = agent_adapter
        self.engine = engine
        self.trajectory_analyzer = trajectory_analyzer
        self.code_locator = code_locator

        # P0-5: resolve constraints from config object / dict / explicit kwargs.
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
        backends = (
            mutation_backends
            if mutation_backends is not None
            else mutation_backend_weights
            if mutation_backend_weights is not None
            else _cfg_get("mutation_backends", {})
        )
        self.mutation_backend_weights = dict(backends or {})
        self.require_causal_ablation = bool(
            require_causal_ablation
            if config is None and not hasattr(cfg, "require_causal_ablation")
            else _cfg_get("require_causal_ablation", require_causal_ablation)
        )

        # Stage objects (delegating to existing components for now).
        self.weakness_stage = WeaknessAnalysisStage(trajectory_analyzer) if trajectory_analyzer else None
        self.transfer_stage = RepositoryTransferStage(code_locator) if code_locator else None
        self.chain_discovery_stage = ChainDiscoveryStage()
        self.contract_stage = ContractGenerationStage()
        self.mutation_stage = MutationPlanningStage(
            engine, mutation_backend_weights=self.mutation_backend_weights
        )
        self.ablation_stage = CausalAblationStage()

        # Lazily-loaded backing generator (the existing RepoChainGenerator).
        self._backing_generator = None

    def _load_backing_generator(self):
        """Lazily import and instantiate the existing RepoChainGenerator.

        The full chain-discovery + contract-generation + mutation logic
        currently lives in swesmith/repo_chain.py. Until each stage is fully
        extracted into this package, we delegate ``generate`` to it.
        """
        if self._backing_generator is not None:
            return self._backing_generator
        try:
            from swesmith.repo_chain import RepoChainGenerator  # type: ignore

            self._backing_generator = RepoChainGenerator(self.agent_adapter)
        except Exception:
            self._backing_generator = None
        return self._backing_generator

    def _apply_constraints_to_plan(self, plan) -> None:
        """P0-5: stamp workflow difficulty constraints onto the plan."""
        blueprint = getattr(plan, "task_blueprint", None)
        if not isinstance(blueprint, dict):
            return
        constraints = blueprint.setdefault("constraints", {})
        if not isinstance(constraints, dict):
            constraints = {}
            blueprint["constraints"] = constraints
        constraints["min_modified_files"] = self.min_files
        constraints["max_modified_files"] = self.max_files
        constraints["min_mutation_sites"] = self.min_mutation_sites
        constraints["max_mutation_sites"] = self.max_mutation_sites
        constraints["context_file_budget"] = self.context_file_budget
        if self.mutation_backend_weights:
            constraints["mutation_backends"] = dict(self.mutation_backend_weights)

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
        and mutation materialization.
        """
        self._apply_constraints_to_plan(plan)
        backing = self._load_backing_generator()
        if backing is None:
            return []
        candidates = backing.generate(plan, node_code_dir, repo_spec, output_dir)

        if self.require_causal_ablation:
            ablation = self.ablation_stage.run(plan, repo_spec, candidates, contracts=None)
            if not ablation.passed:
                return []
        return candidates

    def bootstrap(
        self,
        repo_spec,
        output_dir: str,
        capability_prior: Optional[List[str]] = None,
        target_count: int = 10,
        max_candidates: Optional[int] = None,
    ) -> List:
        """Root bootstrap mode: generate T_0 without solver trajectory conditioning.

        P0-4: keep generating diverse capability-prior plans until we have
        enough candidates (``target_count``) or hit ``max_candidates``.
        """
        from .bootstrap import BOOTSTRAP_CAPABILITY_PRIOR, build_bootstrap_plans

        prior = capability_prior or BOOTSTRAP_CAPABILITY_PRIOR
        backing = self._load_backing_generator()
        if backing is None:
            return []
        limit = max_candidates if max_candidates is not None else max(target_count * 3, 21)
        plans = build_bootstrap_plans(
            prior, repo_spec, target_count=target_count, max_plans=limit
        )
        candidates: List = []
        for plan in plans:
            if len(candidates) >= target_count:
                break
            if max_candidates is not None and len(candidates) >= max_candidates:
                break
            produced = self.generate(
                plan, "", repo_spec, os.path.join(output_dir, plan.plan_id)
            )
            candidates.extend(produced)
        return candidates


__all__ = ["RepoChainWorkflow"]
