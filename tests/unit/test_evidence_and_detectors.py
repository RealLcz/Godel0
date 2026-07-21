"""Unit tests for evidence selector (BUG-22/23) and solver special detector (BUG-21)."""
from __future__ import annotations

from godel0.evolution.evidence_selector import CycleEvidenceSelector
from godel0.evolution.special_detectors import SolverSpecialDetector
from godel0.schemas.cycle import (
    AlertPriority,
    AlertSource,
    NodeCycleSummary,
    SpecialAlert,
)


def _alert(
    alert_type: str,
    source: AlertSource = AlertSource.SOLVER,
    priority: AlertPriority = AlertPriority.HIGH,
    severity: float = 0.8,
) -> SpecialAlert:
    return SpecialAlert(
        alert_id=f"a_{alert_type}",
        alert_type=alert_type,
        source=source,
        priority=priority,
        triggered=True,
        severity=severity,
        recommended_attention=f"investigate {alert_type}",
    )


class TestEvidenceSelectorRawText:
    """BUG-22: evidence items must carry raw_text, not just a 500-char summary."""

    def test_solver_empty_patch_branch_carries_raw_text(self):
        sel = CycleEvidenceSelector()
        summary = NodeCycleSummary(node_id="n1")
        alert = _alert("solver_empty_patch")
        # 10k-char trajectory excerpt; raw_text should keep up to 8k.
        long_traj = "FAIL empty patch\n" + "x" * 10000
        artifacts = {
            "solver_trajectories": [long_traj, "empty2", "empty3"],
            "task_quality_summary": "quality summary here",
        }
        bundle = sel.select(summary, [alert], artifacts)
        ids = [it.evidence_id for it in bundle.items]
        assert "empty_traj_0" in ids
        traj0 = next(it for it in bundle.items if it.evidence_id == "empty_traj_0")
        # Primary failure evidence gets up to primary_raw_chars (8000).
        assert traj0.raw_text is not None
        assert len(traj0.raw_text) <= 8000
        assert len(traj0.raw_text) > 500  # more than the old 500-char summary

    def test_no_f2p_branch_carries_raw_text(self):
        sel = CycleEvidenceSelector()
        summary = NodeCycleSummary(node_id="n1")
        alert = _alert("no_f2p_dominant", source=AlertSource.PROPOSER)
        artifacts = {
            "proposer_candidates": [
                "no_f2p candidate with detail " + "y" * 3000,
                "no_f2p second " + "z" * 2000,
            ],
            "chain_plans": ["chain plan detail"],
        }
        bundle = sel.select(summary, [alert], artifacts)
        cand0 = next(
            it for it in bundle.items if it.evidence_id == "no_f2p_cand_0"
        )
        assert cand0.raw_text is not None
        # Primary evidence budget is 8000, but the excerpt itself is ~3025 chars.
        assert len(cand0.raw_text) > 500

    def test_success_contrast_carries_raw_text(self):
        """BUG-23: success contrast must include real trajectory detail."""
        sel = CycleEvidenceSelector()
        summary = NodeCycleSummary(node_id="n1")
        alert = _alert("solver_empty_patch")
        contrast = (
            "SOLVED task T1\n"
            "patch: a.diff\n"
            "tools: view,edit,bash\n"
            "tests: 3 pass, 0 fail\n"
            + "detail " * 500
        )
        artifacts = {"success_contrast": contrast}
        bundle = sel.select(summary, [alert], artifacts)
        contrast_item = next(
            (it for it in bundle.items if it.evidence_id == "success_contrast"),
            None,
        )
        assert contrast_item is not None
        assert contrast_item.raw_text is not None
        # Contrast budget is 4000 chars.
        assert len(contrast_item.raw_text) <= 4000
        # Should contain real trajectory detail, not just "SOLVED task T1".
        assert "patch" in contrast_item.raw_text
        assert "tools" in contrast_item.raw_text

    def test_raw_text_budgets_are_configurable(self):
        sel = CycleEvidenceSelector(
            primary_raw_chars=1000,
            supporting_raw_chars=500,
            contrast_raw_chars=300,
        )
        summary = NodeCycleSummary(node_id="n1")
        alert = _alert("solver_empty_patch")
        long_traj = "x" * 5000
        artifacts = {
            "solver_trajectories": [long_traj, long_traj, long_traj],
        }
        bundle = sel.select(summary, [alert], artifacts)
        traj0 = next(it for it in bundle.items if it.evidence_id == "empty_traj_0")
        traj1 = next(it for it in bundle.items if it.evidence_id == "empty_traj_1")
        assert len(traj0.raw_text) <= 1000
        assert len(traj1.raw_text) <= 500

    def test_empty_raw_produces_no_raw_text(self):
        sel = CycleEvidenceSelector()
        item, _ = sel._make_item(
            evidence_id="x",
            evidence_type="solver_trajectory",
            source_stage="solver",
            raw="",
            importance=0.5,
            raw_budget=1000,
        )
        assert item.raw_text is None
        assert item.summary == ""

    def test_summary_is_first_500_chars(self):
        sel = CycleEvidenceSelector()
        text = "A" * 800
        item, _ = sel._make_item(
            evidence_id="x",
            evidence_type="solver_trajectory",
            source_stage="solver",
            raw=text,
            importance=0.5,
            raw_budget=800,
        )
        assert len(item.summary) == 500
        assert len(item.raw_text) == 800


class TestSolverSpecialDetectorStochasticity:
    """BUG-21: solver_stochasticity alert must be gated on solver_rollouts."""

    def test_alert_not_triggered_below_min_rollouts(self):
        det = SolverSpecialDetector()
        summary = NodeCycleSummary(node_id="n1")
        # 3 stochastic tasks out of 5 evaluated would normally trigger, but
        # solver_rollouts=2 is below the default min_rollouts=3.
        alerts = det.detect(
            summary,
            evaluated_count=5,
            stochastic_task_count=3,
            solver_rollouts=2,
            tasks_with_multiple_rollouts=1,
        )
        types = [a.alert_type for a in alerts]
        assert "solver_stochasticity" not in types

    def test_alert_triggered_at_or_above_min_rollouts(self):
        det = SolverSpecialDetector()
        summary = NodeCycleSummary(node_id="n1")
        alerts = det.detect(
            summary,
            evaluated_count=5,
            stochastic_task_count=3,
            solver_rollouts=3,
            tasks_with_multiple_rollouts=2,
        )
        types = [a.alert_type for a in alerts]
        assert "solver_stochasticity" in types

    def test_alert_not_triggered_when_stochastic_ratio_below_threshold(self):
        det = SolverSpecialDetector()
        summary = NodeCycleSummary(node_id="n1")
        # Only 1 stochastic task out of 5 evaluated -> ratio 0.2 < 0.3.
        alerts = det.detect(
            summary,
            evaluated_count=5,
            stochastic_task_count=1,
            solver_rollouts=5,
            tasks_with_multiple_rollouts=3,
        )
        types = [a.alert_type for a in alerts]
        assert "solver_stochasticity" not in types
