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
    """

    def __init__(
        self,
        agent_adapter: Any = None,
        engine: Any = None,
        trajectory_analyzer: Any = None,
        code_locator: Any = None,
        mutation_backend_weights: Optional[dict] = None,
        require_causal_ablation: bool = True,
    ) -> None:
        self.agent_adapter = agent_adapter
        self.engine = engine
        self.trajectory_analyzer = trajectory_analyzer
        self.code_locator = code_locator
        self.mutation_backend_weights = dict(mutation_backend_weights or {})
        self.require_causal_ablation = require_causal_ablation

        # Stage objects (delegating to existing components for now).
        self.weakness_stage = WeaknessAnalysisStage(trajectory_analyzer) if trajectory_analyzer else None
        self.transfer_stage = RepositoryTransferStage(code_locator) if code_locator else None
        self.chain_discovery_stage = ChainDiscoveryStage()
        self.contract_stage = ContractGenerationStage()
        self.mutation_stage = MutationPlanningStage(
            engine, mutation_backend_weights=mutation_backend_weights
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
    ) -> List:
        """Root bootstrap mode: generate T_0 without solver trajectory conditioning.

        Uses a capability prior (cross_file_localization,
        multi_module_state_propagation, ...) instead of solver failures.
        Phase 3 will implement this fully; for now it delegates to the backing
        generator with a synthetic plan derived from the prior.
        """
        from .bootstrap import BOOTSTRAP_CAPABILITY_PRIOR, build_bootstrap_plans

        prior = capability_prior or BOOTSTRAP_CAPABILITY_PRIOR
        backing = self._load_backing_generator()
        if backing is None:
            return []
        plans = build_bootstrap_plans(prior, repo_spec)
        candidates: List = []
        for plan in plans:
            produced = self.generate(plan, "", repo_spec, os.path.join(output_dir, plan.plan_id))
            candidates.extend(produced)
        return candidates


__all__ = ["RepoChainWorkflow"]
