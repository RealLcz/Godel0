"""Stage 5: Chain Mutation.

Produce multiple mutation manifestations around a unified root invariant on the
semantic chain (2-6 production files, 3-8 mutation sites). All mutation sites
must serve the same behavioral regression; no random combinations of
independent bugs.

P1-2 (v1): Stage 5 uses the fixed operator
``trajectory_conditioned_chain_mutation`` materialized by
``RepoChainGenerator``. Weighted ``mutation_backends`` (lm_modify /
procedural / pr_replay) are **not** a Stage-5 selector in v1. A future
``ChainMutationBackend.generate_chain(...)`` registry can restore pluggable
backends without reviving fake weight tables on RepoChainWorkflowConfig.
"""

from __future__ import annotations

from typing import Optional

DEFAULT_MUTATION_OPERATOR = "trajectory_conditioned_chain_mutation"


class MutationPlanningStage:
    """Stage 5: plan / materialize chain-level mutations.

    In v1 this stage records the fixed operator and, when an engine is
    present, may delegate materialization. The live RepoChain path goes
    through ``RepoChainGenerator`` directly from ``RepoChainWorkflow.generate``.
    """

    def __init__(
        self,
        engine=None,
        mutation_operator: str = DEFAULT_MUTATION_OPERATOR,
        # Deprecated: ignored (kept for call-site compatibility).
        mutation_backend_weights=None,
    ):
        self.engine = engine
        self.mutation_operator = str(
            mutation_operator or DEFAULT_MUTATION_OPERATOR
        ).strip() or DEFAULT_MUTATION_OPERATOR
        self.mutation_backend_weights = {}

    def run(self, plan, node_code_dir, repo_spec, output_dir):
        try:
            plan.operator = self.mutation_operator
            plan.strategy = "repo_chain"
        except Exception:
            pass
        if self.engine is None:
            return []
        return self.engine.generate(plan, node_code_dir, repo_spec, output_dir)


__all__ = ["MutationPlanningStage", "DEFAULT_MUTATION_OPERATOR"]
