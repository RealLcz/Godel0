"""Evidence selector: selects representative evidence for diagnosis."""

from __future__ import annotations

from typing import List, Optional

from ..schemas.cycle import (
    CycleEvidenceBundle,
    EvidenceItem,
    NodeCycleSummary,
    SpecialAlert,
    AlertPriority,
)


class CycleEvidenceSelector:
    """Selects a limited, representative set of evidence for the diagnoser.

    Ensures the total evidence fits within token budgets.
    """

    def __init__(
        self,
        max_solver_trajectories: int = 4,
        max_proposer_candidates: int = 4,
        max_tool_incidents: int = 2,
        max_raw_chars_per_item: int = 20000,
        max_total_evidence_chars: int = 120000,
        include_success_contrast: bool = True,
    ):
        self.max_solver = max_solver_trajectories
        self.max_proposer = max_proposer_candidates
        self.max_tool = max_tool_incidents
        self.max_chars_per_item = max_raw_chars_per_item
        self.max_total = max_total_evidence_chars
        self.include_contrast = include_success_contrast

    def select(
        self,
        summary: NodeCycleSummary,
        alerts: List[SpecialAlert],
        artifacts: dict = None,
    ) -> CycleEvidenceBundle:
        """Select evidence items from the cycle."""
        artifacts = artifacts or {}
        items: List[EvidenceItem] = []
        total_chars = 0

        for alert in alerts:
            if alert.priority in (AlertPriority.CRITICAL, AlertPriority.HIGH):
                item = EvidenceItem(
                    evidence_id=f"alert_{alert.alert_id}",
                    evidence_type="tool_incident" if alert.source.value == "shared" else "solver_trajectory",
                    source_stage=alert.source.value,
                    summary=alert.recommended_attention,
                    token_estimate=len(alert.recommended_attention) // 4,
                    importance=alert.severity,
                )
                items.append(item)
                total_chars += item.token_estimate * 4

        solver_trajs = artifacts.get("solver_trajectories", [])
        for i, traj in enumerate(solver_trajs[:self.max_solver]):
            excerpt = str(traj)[:self.max_chars_per_item]
            item = EvidenceItem(
                evidence_id=f"solver_traj_{i}",
                evidence_type="solver_trajectory",
                source_stage="solver",
                summary=excerpt[:500],
                raw_excerpt_path=None,
                token_estimate=len(excerpt) // 4,
                importance=0.8 if i == 0 else 0.5,
            )
            if total_chars + len(excerpt) > self.max_total:
                break
            items.append(item)
            total_chars += len(excerpt)

        proposer_candidates = artifacts.get("proposer_candidates", [])
        for i, cand in enumerate(proposer_candidates[:self.max_proposer]):
            excerpt = str(cand)[:self.max_chars_per_item]
            item = EvidenceItem(
                evidence_id=f"proposer_cand_{i}",
                evidence_type="proposer_candidate",
                source_stage="proposer",
                summary=excerpt[:500],
                raw_excerpt_path=None,
                token_estimate=len(excerpt) // 4,
                importance=0.8 if i == 0 else 0.5,
            )
            if total_chars + len(excerpt) > self.max_total:
                break
            items.append(item)
            total_chars += len(excerpt)

        if self.include_contrast and artifacts.get("success_contrast"):
            contrast = str(artifacts["success_contrast"])[:self.max_chars_per_item]
            item = EvidenceItem(
                evidence_id="success_contrast",
                evidence_type="success_contrast",
                source_stage="solver",
                summary=contrast[:500],
                raw_excerpt_path=None,
                token_estimate=len(contrast) // 4,
                importance=0.7,
            )
            items.append(item)
            total_chars += len(contrast)

        return CycleEvidenceBundle(
            node_id=summary.node_id,
            special_alerts=alerts,
            items=items,
            total_token_estimate=total_chars // 4,
            truncated=total_chars > self.max_total,
        )
