"""Stage 5: Chain Mutation.

Produce multiple mutation manifestations around a unified root invariant on the
semantic chain (2-6 production files, 3-8 mutation sites). All mutation sites
must serve the same behavioral regression; no random combinations of
independent bugs.

Mutation backends (lm_modify, procedural, pr_replay) are dispatched via
``SWESmithEngine``. This stage plans the mutations; the actual patch
materialization is delegated to the mutation backends.
"""

from __future__ import annotations

from typing import Dict, List, Optional


class MutationPlanningStage:
    """Stage 5: plan chain-level mutations around one root invariant.

    Delegates patch materialization to the configured mutation backends via
    the SWESmithEngine.
    """

    def __init__(self, engine, mutation_backend_weights: Optional[Dict[str, float]] = None):
        self.engine = engine
        self.mutation_backend_weights = dict(mutation_backend_weights or {})

    def run(self, plan, node_code_dir, repo_spec, output_dir):
        if self.engine is None:
            return []
        return self.engine.generate(plan, node_code_dir, repo_spec, output_dir)


__all__ = ["MutationPlanningStage"]
