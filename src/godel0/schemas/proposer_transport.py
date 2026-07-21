"""Trusted proposer transport schemas (BUG-25).

The trust-boundary transport between the trusted controller and the evolvable
node proposer subprocess must NOT live inside ``initial_agent/`` -- otherwise
a child node could self-edit ``proposer/request.py`` and break the contract.

This module defines the fixed V1 transport schema:

    ProposerRequestV1  -- what the controller sends to the proposer subprocess.
    ProposerResultV1   -- what the proposer subprocess returns.
    CandidateTransportV1 -- one candidate artifact on the wire.

The node workflow (``initial_agent/src/proposer/...``) is free to evolve, but
these schemas are trusted and are not self-editable (they live under
``src/godel0/schemas/`` which PatchGuard protects).

The schemas are intentionally permissive (``dict`` / ``Any`` for the evolvable
fields like ``generation_metadata`` and ``plans``) so a child node can extend
the *contents* of what it emits, but the *shape* the controller relies on
(candidate_id, plan_id, patch, strategy, ...) is fixed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CandidateTransportV1(BaseModel):
    """Fixed wire format for one candidate artifact (BUG-25).

    The controller only reads the fields below; any additional evolvable
    metadata travels inside ``generation_metadata`` (a free-form dict).
    """

    candidate_id: str
    plan_id: str = ""
    repo_id: str = ""
    base_commit: str = ""
    strategy: str = "unknown"
    operator: str = ""
    # The candidate bug patch the controller will validate.
    patch: str = ""
    # Issue draft / problem statement for the candidate.
    issue_draft: str = ""
    file_path: str = ""
    symbol_name: str = ""
    modified_files: List[str] = Field(default_factory=list)
    modified_entities: List[str] = Field(default_factory=list)
    # Free-form evolvable metadata (chain plan, causal ablation, source
    # provenance, ...). The controller does not assume any specific keys here.
    generation_metadata: Dict[str, Any] = Field(default_factory=dict)
    status: str = "pending_validation"

    model_config = {"extra": "allow"}


class ProposerRequestV1(BaseModel):
    """Fixed wire format for a proposer batch request (BUG-25).

    The controller constructs this and writes it to ``proposer_request.json``;
    the proposer subprocess reads it. The subprocess may evolve how it
    *consumes* the request, but the request shape is fixed.
    """

    node_id: str
    run_id: str
    agent_code_dir: str
    repo_pool_dir: str
    task_store_dir: str
    output_dir: str
    target_batch_size: int = 10
    max_candidates: int = 50
    solver_trajectories: List[str] = Field(default_factory=list)
    # BUG-08/09: split trajectory buckets.
    parent_failure_trajectories: List[str] = Field(default_factory=list)
    current_child_level1_trajectories: List[str] = Field(default_factory=list)
    parent_task_ids: List[str] = Field(default_factory=list)
    model: str = "deepseek/deepseek-chat"
    generation_attempt: int = 0
    strategy_weights: Dict[str, float] = Field(default_factory=dict)
    feedback_dir: Optional[str] = None
    repo_specs: List[Dict[str, Any]] = Field(default_factory=list)
    contract_test_renderer: str = ""
    bootstrap: bool = False

    model_config = {"extra": "allow"}


class ProposerResultV1(BaseModel):
    """Fixed wire format for a proposer batch result (BUG-25).

    The proposer subprocess writes this to ``proposer_result.json``; the
    trusted controller parses it with this schema. The controller never
    imports the evolvable ``initial_agent.src.proposer.request.ProposerResult``
    for parsing -- only this trusted type.
    """

    run_id: str = ""
    node_id: str = ""
    completed: bool = False
    accepted_candidates: List[CandidateTransportV1] = Field(default_factory=list)
    rejected_candidates: List[CandidateTransportV1] = Field(default_factory=list)
    pending_candidates: List[CandidateTransportV1] = Field(default_factory=list)
    failure_signatures: List[Dict[str, Any]] = Field(default_factory=list)
    plans: List[Dict[str, Any]] = Field(default_factory=list)
    error: str = ""
    timestamp: str = ""

    model_config = {"extra": "allow"}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProposerResultV1":
        """Parse a proposer result dict from the subprocess.

        Permissive: unknown keys are kept (``extra=allow``) and candidate
        fields are coerced into ``CandidateTransportV1``. This is the single
        trust-boundary parser the controller uses.
        """
        if not isinstance(data, dict):
            raise ValueError("ProposerResultV1.from_dict requires a dict")

        def _coerce_candidates(key: str) -> List[CandidateTransportV1]:
            items = data.get(key) or []
            if not isinstance(items, list):
                return []
            result: List[CandidateTransportV1] = []
            for item in items:
                if isinstance(item, CandidateTransportV1):
                    result.append(item)
                elif isinstance(item, dict):
                    result.append(CandidateTransportV1(**item))
            return result

        return cls(
            run_id=str(data.get("run_id", "")),
            node_id=str(data.get("node_id", "")),
            completed=bool(data.get("completed", False)),
            accepted_candidates=_coerce_candidates("accepted_candidates"),
            rejected_candidates=_coerce_candidates("rejected_candidates"),
            pending_candidates=_coerce_candidates("pending_candidates"),
            failure_signatures=list(data.get("failure_signatures") or []),
            plans=list(data.get("plans") or []),
            error=str(data.get("error") or ""),
            timestamp=str(data.get("timestamp") or ""),
        )


__all__ = [
    "CandidateTransportV1",
    "ProposerRequestV1",
    "ProposerResultV1",
]
