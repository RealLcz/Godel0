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
        # BUG-22: per-evidence-type raw text budgets (chars). Primary failure
        # evidence gets up to 8k; supporting/candidate/contrast get 2-4k.
        primary_raw_chars: int = 8000,
        supporting_raw_chars: int = 4000,
        contrast_raw_chars: int = 4000,
    ):
        self.max_solver = max_solver_trajectories
        self.max_proposer = max_proposer_candidates
        self.max_tool = max_tool_incidents
        self.max_chars_per_item = max_raw_chars_per_item
        self.max_total = max_total_evidence_chars
        self.include_contrast = include_success_contrast
        self.primary_raw_chars = primary_raw_chars
        self.supporting_raw_chars = supporting_raw_chars
        self.contrast_raw_chars = contrast_raw_chars

    def _make_item(
        self,
        evidence_id: str,
        evidence_type: str,
        source_stage: str,
        raw: str,
        importance: float,
        raw_budget: int,
        is_primary: bool = False,
    ) -> tuple[EvidenceItem, int]:
        """BUG-22: build an EvidenceItem carrying a real representative excerpt.

        ``raw`` is the full text we have (e.g. a 20k trajectory excerpt). We
        keep up to ``raw_budget`` chars of it in ``raw_text`` so the diagnoser
        sees real evidence, and derive a short ``summary`` from the first
        500 chars for backwards-compatible consumers. Returns (item, chars).
        """
        text = str(raw or "")
        budget = min(raw_budget, self.max_chars_per_item)
        raw_text = text[:budget] if text else None
        summary = text[:500] if text else ""
        item = EvidenceItem(
            evidence_id=evidence_id,
            evidence_type=evidence_type,
            source_stage=source_stage,
            summary=summary,
            raw_text=raw_text,
            token_estimate=(len(raw_text) if raw_text else len(summary)) // 4,
            importance=importance,
        )
        return item, (len(raw_text) if raw_text else len(summary))

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
            # BUG-23: success contrast must include real trajectory detail
            # (tool sequence, files inspected, patch, test behavior), not just
            # "SOLVED task X". We keep up to ``contrast_raw_chars`` of the
            # raw contrast text in ``raw_text`` and surface a richer summary.
            item, item_chars = self._make_item(
                evidence_id="success_contrast",
                evidence_type="success_contrast",
                source_stage="solver",
                raw=contrast,
                importance=0.7,
                raw_budget=self.contrast_raw_chars,
            )
            if total_chars + item_chars <= self.max_total:
                items.append(item)
                total_chars += item_chars

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

        def _add(item: EvidenceItem, chars: int) -> None:
            items.append(item)
            nonlocal total_chars
            total_chars += chars

        if atype == "no_f2p_dominant":
            # 2 no_f2p candidates + 1 success F2P + 1 chain plan.
            candidates = artifacts.get("proposer_candidates", [])
            no_f2p = [
                c for c in candidates
                if "no_f2p" in str(c).lower() or "f2p" in str(c).lower()
            ]
            for i, cand in enumerate(no_f2p[:2]):
                excerpt = str(cand)[: self.max_chars_per_item]
                # BUG-22: primary failure evidence gets up to 8k raw chars.
                item, chars = self._make_item(
                    evidence_id=f"no_f2p_cand_{i}",
                    evidence_type="proposer_candidate",
                    source_stage="proposer",
                    raw=excerpt,
                    importance=0.9,
                    raw_budget=self.primary_raw_chars if i == 0 else self.supporting_raw_chars,
                )
                _add(item, chars)
            chain_plans = artifacts.get("chain_plans", [])
            if chain_plans:
                plan = str(chain_plans[0])[: self.max_chars_per_item]
                item, chars = self._make_item(
                    evidence_id="chain_plan_0",
                    evidence_type="proposer_candidate",
                    source_stage="proposer",
                    raw=plan,
                    importance=0.8,
                    raw_budget=self.supporting_raw_chars,
                )
                _add(item, chars)

        elif atype == "solver_empty_patch":
            # 3 empty-patch trajectories + 1 success patch + task quality summary.
            trajs = artifacts.get("solver_trajectories", [])
            empty_trajs = [t for t in trajs if "empty" in str(t).lower() or len(str(t)) < 200]
            for i, traj in enumerate((empty_trajs or trajs)[:3]):
                excerpt = str(traj)[: self.max_chars_per_item]
                item, chars = self._make_item(
                    evidence_id=f"empty_traj_{i}",
                    evidence_type="solver_trajectory",
                    source_stage="solver",
                    raw=excerpt,
                    importance=0.9,
                    raw_budget=self.primary_raw_chars if i == 0 else self.supporting_raw_chars,
                )
                _add(item, chars)
            quality = artifacts.get("task_quality_summary", "")
            if quality:
                q = str(quality)[: self.max_chars_per_item]
                item, chars = self._make_item(
                    evidence_id="task_quality_summary",
                    evidence_type="proposer_batch_summary",
                    source_stage="proposer",
                    raw=q,
                    importance=0.7,
                    raw_budget=self.supporting_raw_chars,
                )
                _add(item, chars)

        elif atype == "causal_ablation_failure":
            # failed chain plan + mutation sites + ablation results + 1 success contrast.
            chain_plans = artifacts.get("chain_plans", [])
            if chain_plans:
                plan = str(chain_plans[0])[: self.max_chars_per_item]
                item, chars = self._make_item(
                    evidence_id="failed_chain_plan",
                    evidence_type="proposer_candidate",
                    source_stage="proposer",
                    raw=plan,
                    importance=0.9,
                    raw_budget=self.primary_raw_chars,
                )
                _add(item, chars)
            ablation = artifacts.get("ablation_results", "")
            if ablation:
                a = str(ablation)[: self.max_chars_per_item]
                item, chars = self._make_item(
                    evidence_id="ablation_results",
                    evidence_type="proposer_batch_summary",
                    source_stage="proposer",
                    raw=a,
                    importance=0.9,
                    raw_budget=self.supporting_raw_chars,
                )
                _add(item, chars)

        else:
            # Default: generic first-N retrieval (backward compat).
            trajs = artifacts.get("solver_trajectories", [])
            for i, traj in enumerate(trajs[: self.max_solver]):
                excerpt = str(traj)[: self.max_chars_per_item]
                item, chars = self._make_item(
                    evidence_id=f"solver_traj_{i}",
                    evidence_type="solver_trajectory",
                    source_stage="solver",
                    raw=excerpt,
                    importance=0.8 if i == 0 else 0.5,
                    raw_budget=self.primary_raw_chars if i == 0 else self.supporting_raw_chars,
                )
                _add(item, chars)
            candidates = artifacts.get("proposer_candidates", [])
            for i, cand in enumerate(candidates[: self.max_proposer]):
                excerpt = str(cand)[: self.max_chars_per_item]
                item, chars = self._make_item(
                    evidence_id=f"proposer_cand_{i}",
                    evidence_type="proposer_candidate",
                    source_stage="proposer",
                    raw=excerpt,
                    importance=0.8 if i == 0 else 0.5,
                    raw_budget=self.supporting_raw_chars,
                )
                _add(item, chars)

        return items, total_chars
