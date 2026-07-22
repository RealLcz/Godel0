"""Strict task provenance helpers (P0-5).

Provenance must flow FailureSignature → BugGenerationPlan → Candidate → Task
without inventing missing fields (especially never substituting the current
proposer node for a parent failure source).
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

PROVENANCE_FIELDS = (
    "source_type",
    "source_node_id",
    "source_task_id",
    "source_trajectory_id",
    "source_failure_stage",
)


def provenance_from_signature(signature: Any) -> Dict[str, str]:
    """Extract identity fields already present on a FailureSignature."""
    if signature is None:
        return {key: "" for key in PROVENANCE_FIELDS}
    return {
        "source_type": str(getattr(signature, "source_type", "") or ""),
        "source_node_id": str(getattr(signature, "source_solver_node_id", "") or ""),
        "source_task_id": str(getattr(signature, "source_task_id", "") or ""),
        "source_trajectory_id": str(getattr(signature, "source_trajectory_id", "") or ""),
        "source_failure_stage": str(getattr(signature, "failure_stage", "") or ""),
    }


def provenance_from_blueprint(blueprint: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    """Read stamped provenance from a plan task_blueprint (no invention)."""
    bp = dict(blueprint or {})
    stage = str(
        bp.get("source_failure_stage") or bp.get("failure_stage") or ""
    )
    return {
        "source_type": str(bp.get("source_type") or ""),
        "source_node_id": str(bp.get("source_node_id") or ""),
        "source_task_id": str(bp.get("source_task_id") or ""),
        "source_trajectory_id": str(bp.get("source_trajectory_id") or ""),
        "source_failure_stage": stage,
    }


def merge_provenance(*layers: Mapping[str, Any]) -> Dict[str, str]:
    """Left-to-right: first non-empty value wins. Never invents values."""
    out = {key: "" for key in PROVENANCE_FIELDS}
    for layer in layers:
        if not layer:
            continue
        for key in PROVENANCE_FIELDS:
            if out[key]:
                continue
            value = layer.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                out[key] = text
    return out


def apply_provenance_to_mapping(target: Dict[str, Any], provenance: Mapping[str, str]) -> Dict[str, Any]:
    """Write non-empty provenance fields onto a metadata/blueprint dict."""
    for key in PROVENANCE_FIELDS:
        value = str(provenance.get(key) or "")
        if value:
            target[key] = value
    traj = str(provenance.get("source_trajectory_id") or "")
    if traj:
        ids = list(target.get("source_trajectory_ids") or [])
        if traj not in ids:
            ids = [traj] + ids
        target["source_trajectory_ids"] = ids
    return target
