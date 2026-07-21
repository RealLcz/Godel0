"""Evidence selector: alert-conditioned representative evidence for diagnosis.

HGM-style: the evidence selector picks the most relevant evidence for the
diagnoser based on the primary special alert. Different alert types surface
different evidence (no_f2p candidates, empty-patch trajectories, failed chain
plans, ablation results, ...). A success contrast is always included so the
diagnoser can compare failure vs success.
"""

from __future__ import annotations

from typing import List, Optional

from ..schemas.cycle import (
    AlertPriority,
    CycleEvidenceBundle,
    EvidenceItem,
    NodeCycleSummary,
    SpecialAlert,
)


class CycleEvidenceSelector:
    """Selects alert-conditioned evidence for the diagnoser."""

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
        """Select evidence items, conditioned on the primary alert."""
        artifacts = artifacts or {}
        items: List[EvidenceItem] = []
        total_chars = 0

        # Always include the primary alert summary as an evidence item.
        primary = self._primary_alert(alerts)
        if primary is not None:
            item = EvidenceItem(
                evidence_id=f"alert_{primary.alert_id}",
                evidence_type="tool_incident"
                if primary.source.value == "shared"
                else "solver_trajectory",
                source_stage=primary.source.value,
                summary=primary.recommended_attention,
                token_estimate=len(primary.recommended_attention) // 4,
                importance=primary.severity,
            )
            items.append(item)
            total_chars += item.token_estimate * 4

        # Alert-conditioned retrieval.
        if primary is not None:
            branch_items, branch_chars = self._branch_for_alert(
                primary, artifacts, summary
            )
            for item in branch_items:
                if total_chars + item.token_estimate * 4 > self.max_total:
                    break
                items.append(item)
                total_chars += item.token_estimate * 4

        # Always include 1 success contrast (if available).
        if self.include_contrast and artifacts.get("success_contrast"):
            contrast = str(artifacts["success_contrast"])[: self.max_chars_per_item]
            item = EvidenceItem(
                evidence_id="success_contrast",
                evidence_type="success_contrast",
                source_stage="solver",
                summary=contrast[:500],
                token_estimate=len(contrast) // 4,
                importance=0.7,
            )
            if total_chars + len(contrast) <= self.max_total:
                items.append(item)
                total_chars += len(contrast)

        return CycleEvidenceBundle(
            node_id=summary.node_id,
            special_alerts=alerts,
            items=items,
            total_token_estimate=total_chars // 4,
            truncated=total_chars > self.max_total,
        )

    def _primary_alert(self, alerts: List[SpecialAlert]) -> Optional[SpecialAlert]:
        """Pick the highest-priority alert as the conditioning signal."""
        if not alerts:
            return None
        priority_order = {
            AlertPriority.CRITICAL: 3,
            AlertPriority.HIGH: 2,
            AlertPriority.NORMAL: 1,
        }
        return max(alerts, key=lambda a: (priority_order.get(a.priority, 0), a.severity))

    def _branch_for_alert(
        self,
        alert: SpecialAlert,
        artifacts: dict,
        summary: NodeCycleSummary,
    ) -> tuple[List[EvidenceItem], int]:
        """Return evidence items specific to the alert type."""
        atype = alert.alert_type
        items: List[EvidenceItem] = []
        total_chars = 0

        if atype == "no_f2p_dominant":
            # 2 no_f2p candidates + 1 success F2P + 1 chain plan.
            candidates = artifacts.get("proposer_candidates", [])
            no_f2p = [
                c for c in candidates
                if "no_f2p" in str(c).lower() or "f2p" in str(c).lower()
            ]
            for i, cand in enumerate(no_f2p[:2]):
                excerpt = str(cand)[: self.max_chars_per_item]
                item = EvidenceItem(
                    evidence_id=f"no_f2p_cand_{i}",
                    evidence_type="proposer_candidate",
                    source_stage="proposer",
                    summary=excerpt[:500],
                    token_estimate=len(excerpt) // 4,
                    importance=0.9,
                )
                items.append(item)
                total_chars += len(excerpt)
            chain_plans = artifacts.get("chain_plans", [])
            if chain_plans:
                plan = str(chain_plans[0])[: self.max_chars_per_item]
                items.append(EvidenceItem(
                    evidence_id="chain_plan_0",
                    evidence_type="proposer_candidate",
                    source_stage="proposer",
                    summary=plan[:500],
                    token_estimate=len(plan) // 4,
                    importance=0.8,
                ))
                total_chars += len(plan)

        elif atype == "solver_empty_patch":
            # 3 empty-patch trajectories + 1 success patch + task quality summary.
            trajs = artifacts.get("solver_trajectories", [])
            empty_trajs = [t for t in trajs if "empty" in str(t).lower() or len(str(t)) < 200]
            for i, traj in enumerate((empty_trajs or trajs)[:3]):
                excerpt = str(traj)[: self.max_chars_per_item]
                item = EvidenceItem(
                    evidence_id=f"empty_traj_{i}",
                    evidence_type="solver_trajectory",
                    source_stage="solver",
                    summary=excerpt[:500],
                    token_estimate=len(excerpt) // 4,
                    importance=0.9,
                )
                items.append(item)
                total_chars += len(excerpt)
            quality = artifacts.get("task_quality_summary", "")
            if quality:
                q = str(quality)[: self.max_chars_per_item]
                items.append(EvidenceItem(
                    evidence_id="task_quality_summary",
                    evidence_type="proposer_batch_summary",
                    source_stage="proposer",
                    summary=q[:500],
                    token_estimate=len(q) // 4,
                    importance=0.7,
                ))
                total_chars += len(q)

        elif atype == "causal_ablation_failure":
            # failed chain plan + mutation sites + ablation results + 1 success contrast.
            chain_plans = artifacts.get("chain_plans", [])
            if chain_plans:
                plan = str(chain_plans[0])[: self.max_chars_per_item]
                items.append(EvidenceItem(
                    evidence_id="failed_chain_plan",
                    evidence_type="proposer_candidate",
                    source_stage="proposer",
                    summary=plan[:500],
                    token_estimate=len(plan) // 4,
                    importance=0.9,
                ))
                total_chars += len(plan)
            ablation = artifacts.get("ablation_results", "")
            if ablation:
                a = str(ablation)[: self.max_chars_per_item]
                items.append(EvidenceItem(
                    evidence_id="ablation_results",
                    evidence_type="proposer_batch_summary",
                    source_stage="proposer",
                    summary=a[:500],
                    token_estimate=len(a) // 4,
                    importance=0.9,
                ))
                total_chars += len(a)

        else:
            # Default: generic first-N retrieval (backward compat).
            trajs = artifacts.get("solver_trajectories", [])
            for i, traj in enumerate(trajs[: self.max_solver]):
                excerpt = str(traj)[: self.max_chars_per_item]
                item = EvidenceItem(
                    evidence_id=f"solver_traj_{i}",
                    evidence_type="solver_trajectory",
                    source_stage="solver",
                    summary=excerpt[:500],
                    token_estimate=len(excerpt) // 4,
                    importance=0.8 if i == 0 else 0.5,
                )
                items.append(item)
                total_chars += len(excerpt)
            candidates = artifacts.get("proposer_candidates", [])
            for i, cand in enumerate(candidates[: self.max_proposer]):
                excerpt = str(cand)[: self.max_chars_per_item]
                item = EvidenceItem(
                    evidence_id=f"proposer_cand_{i}",
                    evidence_type="proposer_candidate",
                    source_stage="proposer",
                    summary=excerpt[:500],
                    token_estimate=len(excerpt) // 4,
                    importance=0.8 if i == 0 else 0.5,
                )
                items.append(item)
                total_chars += len(excerpt)

        return items, total_chars
