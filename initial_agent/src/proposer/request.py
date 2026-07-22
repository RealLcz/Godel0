from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RepoSpecInfo:
    """Lightweight repo spec carried inside ProposerRequest.

    This mirrors the control layer's RepoSpec but is a plain dataclass
    so it serializes cleanly to JSON without pydantic on the agent side.
    """

    repo_id: str
    base_commit: str
    path: str
    test_command: str = "pytest -q"
    install_command: str = "pip install -e ."
    timeout_sec: int = 120

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RepoSpecInfo":
        known = {fld for fld in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class ProposerRequest:
    """Input contract for a proposer batch run.

    The proposer reads this request (and only this request) as its input.
    It must NOT read trusted private inputs or directly write to TaskStore.

    The ``repo_specs`` field carries the full list of base repositories
    available for bug generation. Each spec points to a checked-out git
    repository at a specific ``base_commit`` with a known ``test_command``.
    """

    node_id: str
    run_id: str
    agent_code_dir: str
    repo_pool_dir: str
    task_store_dir: str
    output_dir: str
    target_batch_size: int = 10
    max_candidates: int = 50
    solver_trajectories: List[str] = field(default_factory=list)
    # BUG-08/09: split trajectory buckets so the proposer can tag each plan
    # with its source type (parent_failure vs current_child_level1) for
    # provenance. When empty, the proposer falls back to ``solver_trajectories``.
    parent_failure_trajectories: List[str] = field(default_factory=list)
    current_child_level1_trajectories: List[str] = field(default_factory=list)
    parent_task_ids: List[str] = field(default_factory=list)
    model: str = "deepseek/deepseek-chat"
    generation_attempt: int = 0
    strategy_weights: Dict[str, float] = field(default_factory=dict)
    feedback_dir: Optional[str] = None
    repo_specs: List[RepoSpecInfo] = field(default_factory=list)
    contract_test_renderer: str = ""
    bootstrap: bool = False
    # P0-5: optional RepoChainWorkflowConfig payload (dict form for JSON).
    workflow_config: Dict[str, Any] = field(default_factory=dict)
    # P0-10/11: effective per-source generation targets.
    generation_quotas: Dict[str, int] = field(default_factory=dict)
    # P0-6 / P0-23: production defaults mirror ProposerConfig.
    allow_workflow_fallback: bool = False
    allow_human_curated_data: bool = False

    @classmethod
    def load(cls, path: str) -> "ProposerRequest":
        """Load a ProposerRequest from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        known_fields = {fld for fld in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        if "repo_specs" in filtered and filtered["repo_specs"]:
            filtered["repo_specs"] = [
                RepoSpecInfo.from_dict(r) if isinstance(r, dict) else r
                for r in filtered["repo_specs"]
            ]
        return cls(**filtered)

    def save(self, path: str) -> None:
        """Serialize this request to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_repo(self, repo_id: str) -> Optional[RepoSpecInfo]:
        """Find a repo spec by ID."""
        for spec in self.repo_specs:
            if spec.repo_id == repo_id:
                return spec
        return None

    def first_repo(self) -> Optional[RepoSpecInfo]:
        """Get the first available repo spec, or None."""
        if self.repo_specs:
            return self.repo_specs[0]
        return None


@dataclass
class CandidateArtifact:
    """A single generated bug candidate produced by the engine."""

    candidate_id: str
    plan_id: str
    repo_id: str
    base_commit: str
    file_path: str
    symbol_name: str
    strategy: str
    operator: str = ""
    patch: str = ""
    issue_draft: str = ""
    local_test_notes: Dict[str, Any] = field(default_factory=dict)
    generation_trajectory: List[Dict[str, Any]] = field(default_factory=list)
    modified_files: List[str] = field(default_factory=list)
    modified_entities: List[str] = field(default_factory=list)
    generation_metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending_validation"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidateArtifact":
        known = {fld for fld in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


@dataclass
class ProposerResult:
    """Output contract for a proposer batch run."""

    run_id: str
    node_id: str
    completed: bool = False
    accepted_candidates: List[CandidateArtifact] = field(default_factory=list)
    rejected_candidates: List[CandidateArtifact] = field(default_factory=list)
    pending_candidates: List[CandidateArtifact] = field(default_factory=list)
    failure_signatures: List[Dict[str, Any]] = field(default_factory=list)
    plans: List[Dict[str, Any]] = field(default_factory=list)
    error: str = ""
    # P0-6: persist which workflow ran so silent degradation is auditable.
    workflow: str = "repo_chain"
    workflow_fallback: bool = False
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def add_candidate(self, candidate: CandidateArtifact, accepted: bool) -> None:
        if accepted:
            self.accepted_candidates.append(candidate)
        else:
            self.rejected_candidates.append(candidate)

    def add_pending_candidate(self, candidate: CandidateArtifact) -> None:
        self.pending_candidates.append(candidate)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "node_id": self.node_id,
            "completed": self.completed,
            "accepted_candidates": [c.to_dict() for c in self.accepted_candidates],
            "rejected_candidates": [c.to_dict() for c in self.rejected_candidates],
            "pending_candidates": [c.to_dict() for c in self.pending_candidates],
            "failure_signatures": self.failure_signatures,
            "plans": self.plans,
            "error": self.error,
            "workflow": self.workflow,
            "workflow_fallback": self.workflow_fallback,
            "timestamp": self.timestamp,
        }

    def save(self, output_dir: str) -> str:
        """Persist this result as ``proposer_result.json`` under output_dir."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "proposer_result.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def new_for(cls, request: ProposerRequest) -> "ProposerResult":
        return cls(run_id=request.run_id, node_id=request.node_id)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProposerResult":
        """Rebuild a result returned by an isolated node proposer process."""
        result = cls(
            run_id=str(data.get("run_id", "")),
            node_id=str(data.get("node_id", "")),
            completed=bool(data.get("completed", False)),
            failure_signatures=list(data.get("failure_signatures") or []),
            plans=list(data.get("plans") or []),
            error=str(data.get("error") or ""),
            workflow=str(data.get("workflow") or "repo_chain"),
            workflow_fallback=bool(data.get("workflow_fallback", False)),
            timestamp=str(data.get("timestamp") or ""),
        )
        result.accepted_candidates = [
            CandidateArtifact.from_dict(value)
            for value in data.get("accepted_candidates") or []
        ]
        result.rejected_candidates = [
            CandidateArtifact.from_dict(value)
            for value in data.get("rejected_candidates") or []
        ]
        result.pending_candidates = [
            CandidateArtifact.from_dict(value)
            for value in data.get("pending_candidates") or []
        ]
        return result


def new_candidate_id() -> str:
    return f"cand-{uuid.uuid4().hex[:12]}"


__all__ = [
    "ProposerRequest",
    "RepoSpecInfo",
    "CandidateArtifact",
    "ProposerResult",
    "new_candidate_id",
]
