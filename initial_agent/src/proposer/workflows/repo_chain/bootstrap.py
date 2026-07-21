"""Root bootstrap capability prior.

When the root node has no solver trajectories yet, RepoChain generates T_0
from a capability prior instead of trajectory-conditioned weakness analysis.
These are capability *categories*, not fixed benchmark tasks.
"""

from __future__ import annotations

import uuid
from typing import List

BOOTSTRAP_CAPABILITY_PRIOR: List[str] = [
    "cross_file_localization",
    "multi_module_state_propagation",
    "configuration_precedence",
    "error_handling",
    "compatibility_preservation",
    "api_contract_reasoning",
    "multi_step_repository_reasoning",
]


def build_bootstrap_plans(capability_prior: List[str], repo_spec) -> List:
    """Build synthetic BugGenerationPlans from the capability prior.

    Each capability becomes a plan with a FailureSignature stub so the
    RepoChainGenerator can run stages 2-8 without real solver trajectories.

    BUG-04: when ``capability_prior`` is empty the root produced zero plans.
    Callers that want the default prior should pass ``BOOTSTRAP_CAPABILITY_PRIOR``
    explicitly; this function no longer silently substitutes it so that an
    empty prior is a visible programming error rather than a 0-candidate batch.
    BUG-05: the plan strategy is now ``repo_chain`` (not ``lm_modify``) so the
    root T_0 actually enters the RepoChain workflow instead of degrading to a
    single-file LM modify.
    """
    try:
        from proposer.schemas import BugGenerationPlan, FailureSignature
    except Exception:
        return []

    plans: List = []
    for capability in capability_prior:
        signature = FailureSignature(
            signature_id=f"bootstrap-{capability}",
            failure_stage="localization",
            root_cause=f"bootstrap capability prior: {capability}",
            target_capability=capability,
        )
        plan = BugGenerationPlan(
            plan_id=f"bootstrap-{capability}-{uuid.uuid4().hex[:8]}",
            source_trajectory_ids=[],
            failure_signature=signature,
            target_repo_id=getattr(repo_spec, "repo_id", ""),
            target_base_commit=getattr(repo_spec, "base_commit", ""),
            strategy="repo_chain",
            operator="",
            rationale=f"bootstrap plan for capability {capability}",
            task_blueprint={
                "capability_gap": capability,
                "failure_stage": "bootstrap",
                "root_cause": f"bootstrap capability prior: {capability}",
                "required_topology": "connected_cross_file_contract",
                "source_trajectory_id": "",
                "bootstrap": True,
            },
        )
        plans.append(plan)
    return plans


__all__ = ["BOOTSTRAP_CAPABILITY_PRIOR", "build_bootstrap_plans"]
