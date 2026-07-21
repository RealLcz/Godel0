"""RepoChain workflow package.

The 8-stage Proposer default workflow. RepoChain is a workflow, not a tool
and not a mutation backend.
"""

from .workflow import RepoChainWorkflow
from .bootstrap import BOOTSTRAP_CAPABILITY_PRIOR, build_bootstrap_plans
from .causal_ablation import AblationResult, CausalAblationStage

__all__ = [
    "RepoChainWorkflow",
    "BOOTSTRAP_CAPABILITY_PRIOR",
    "build_bootstrap_plans",
    "AblationResult",
    "CausalAblationStage",
]
