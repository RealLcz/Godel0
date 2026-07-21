"""Unit tests for cycle builder and diagnosis."""

from __future__ import annotations

import pytest

from godel0.evolution.cycle_builder import NodeCycleBuilder
from godel0.evolution.evidence_selector import CycleEvidenceSelector
from godel0.evolution.diagnose import CycleDiagnoser
from godel0.schemas.cycle import (
    NodeCycleSummary, SpecialAlert, AlertSource, AlertPriority,
    CycleEvidenceBundle, EvidenceItem, CycleStage,
)
from godel0.schemas.node import NodeRecord, NodeStatus
from godel0.schemas.evaluation import Level1Result, Level2Result
from datetime import datetime, timezone


def make_node(node_id: str = "test_node") -> NodeRecord:
    return NodeRecord(
        node_id=node_id,
        code_commit="abc",
        code_ref="ref",
        status=NodeStatus.COMPLETE,
    )


class TestCycleBuilder:
    def test_root_bootstrap(self):
        builder = NodeCycleBuilder()
        node = make_node("root")
        summary = builder.build(node, is_root=True)
        assert summary.stage_reached == CycleStage.ROOT_BOOTSTRAP

    def test_level1_failed(self):
        builder = NodeCycleBuilder()
        node = make_node("child")
        level1 = Level1Result(
            parent_node_id="parent",
            child_node_id="child",
            evaluated_task_ids=["t1"],
            parent_solved_task_ids=["t1"],
            child_retained_task_ids=[],
            child_forgotten_task_ids=["t1"],
            child_newly_solved_task_ids=[],
            retention_rate=0.0,
            threshold=0.8,
            passed=False,
        )
        summary = builder.build(node, level1=level1)
        assert summary.stage_reached == CycleStage.LEVEL1_FAILED
        assert summary.level1_retention == 0.0

    def test_complete_cycle(self):
        builder = NodeCycleBuilder()
        node = make_node("child")
        level1 = Level1Result(
            parent_node_id="parent",
            child_node_id="child",
            evaluated_task_ids=["t1"],
            parent_solved_task_ids=["t1"],
            child_retained_task_ids=["t1"],
            child_forgotten_task_ids=[],
            child_newly_solved_task_ids=[],
            retention_rate=1.0,
            threshold=0.8,
            passed=True,
        )
        level2 = Level2Result(
            node_id="child",
            task_batch_id="batch",
            evaluated_task_ids=["t1", "t2"],
            solved_task_ids=["t1"],
            failed_task_ids=["t2"],
            accuracy=0.5,
        )
        proposer_stats = {
            "requested": 10,
            "generated": 20,
            "accepted": 10,
            "rejections": {"no_f2p": 5, "syntax_error": 3},
            "operators": {"change_operator": 8, "invert_if": 4},
        }
        summary = builder.build(node, level1=level1, proposer_stats=proposer_stats, level2=level2)
        assert summary.stage_reached == CycleStage.LEVEL2_COMPLETE
        assert summary.level2_accuracy == 0.5
        assert summary.proposer_valid_yield == 0.5


class TestEvidenceSelector:
    def test_select_within_budget(self):
        selector = CycleEvidenceSelector(max_total_evidence_chars=10000)
        summary = NodeCycleSummary(node_id="test")
        alerts = [
            SpecialAlert(
                alert_id="a1",
                alert_type="solver_regression",
                source=AlertSource.SOLVER,
                priority=AlertPriority.CRITICAL,
                triggered=True,
                severity=0.9,
                recommended_attention="Regression detected",
            )
        ]
        artifacts = {
            "solver_trajectories": [{"event": "test"} for _ in range(10)],
            "proposer_candidates": [{"candidate": "test"} for _ in range(10)],
        }
        bundle = selector.select(summary, alerts, artifacts)
        assert len(bundle.items) <= 10
        assert any(item.evidence_type == "solver_trajectory" for item in bundle.items)

    def test_alerts_included(self):
        selector = CycleEvidenceSelector()
        summary = NodeCycleSummary(node_id="test")
        alerts = [
            SpecialAlert(
                alert_id="critical_1",
                alert_type="solver_regression",
                source=AlertSource.SOLVER,
                priority=AlertPriority.CRITICAL,
                triggered=True,
                severity=0.9,
                recommended_attention="Critical issue",
            )
        ]
        bundle = selector.select(summary, alerts)
        assert len(bundle.special_alerts) == 1
        assert bundle.special_alerts[0].alert_id == "critical_1"

    def test_no_f2p_dominant_branch(self):
        selector = CycleEvidenceSelector()
        summary = NodeCycleSummary(node_id="test")
        alerts = [
            SpecialAlert(
                alert_id="no_f2p",
                alert_type="no_f2p_dominant",
                source=AlertSource.PROPOSER,
                priority=AlertPriority.HIGH,
                triggered=True,
                severity=0.8,
                recommended_attention="No F2P",
            )
        ]
        artifacts = {
            "proposer_candidates": [{"no_f2p": True, "detail": "candidate failed f2p"}],
            "chain_plans": ["chain_plan_detail"],
            "success_contrast": "success case",
        }
        bundle = selector.select(summary, alerts, artifacts)
        no_f2p_items = [i for i in bundle.items if i.evidence_id.startswith("no_f2p_cand")]
        assert len(no_f2p_items) >= 1
        chain_items = [i for i in bundle.items if i.evidence_id == "chain_plan_0"]
        assert len(chain_items) == 1
        contrast_items = [i for i in bundle.items if i.evidence_type == "success_contrast"]
        assert len(contrast_items) == 1

    def test_causal_ablation_branch(self):
        selector = CycleEvidenceSelector()
        summary = NodeCycleSummary(node_id="test")
        alerts = [
            SpecialAlert(
                alert_id="causal_fail",
                alert_type="causal_ablation_failure",
                source=AlertSource.PROPOSER,
                priority=AlertPriority.CRITICAL,
                triggered=True,
                severity=0.9,
                recommended_attention="Causal ablation failed",
            )
        ]
        artifacts = {
            "chain_plans": ["failed_chain_plan_detail"],
            "ablation_results": "ablation: repair_one_file restored all",
            "success_contrast": "success case",
        }
        bundle = selector.select(summary, alerts, artifacts)
        failed_items = [i for i in bundle.items if i.evidence_id == "failed_chain_plan"]
        assert len(failed_items) == 1
        ablation_items = [i for i in bundle.items if i.evidence_id == "ablation_results"]
        assert len(ablation_items) == 1


class TestCycleDiagnoser:
    def test_deterministic_diagnosis_with_critical(self):
        diagnoser = CycleDiagnoser()
        summary = NodeCycleSummary(node_id="test")
        evidence = CycleEvidenceBundle(
            node_id="test",
            special_alerts=[
                SpecialAlert(
                    alert_id="crit_1",
                    alert_type="solver_regression",
                    source=AlertSource.SOLVER,
                    priority=AlertPriority.CRITICAL,
                    triggered=True,
                    severity=0.9,
                    recommended_attention="Severe regression",
                )
            ],
            items=[],
        )
        diagnosis = diagnoser.diagnose("test", summary, evidence)
        assert diagnosis.primary_root_cause
        assert diagnosis.selected_alert_id == "crit_1"
        assert "coding_agent" in diagnosis.recommended_edit_scopes

    def test_deterministic_diagnosis_no_alerts(self):
        diagnoser = CycleDiagnoser()
        summary = NodeCycleSummary(
            node_id="test",
            level2_accuracy=0.3,
        )
        evidence = CycleEvidenceBundle(node_id="test")
        diagnosis = diagnoser.diagnose("test", summary, evidence)
        assert diagnosis.primary_root_cause
        assert len(diagnosis.source_stages) >= 1

    def test_one_root_cause_only(self):
        diagnoser = CycleDiagnoser()
        summary = NodeCycleSummary(node_id="test")
        evidence = CycleEvidenceBundle(
            node_id="test",
            special_alerts=[
                SpecialAlert(
                    alert_id="a1",
                    alert_type="t1",
                    source=AlertSource.SOLVER,
                    priority=AlertPriority.CRITICAL,
                    triggered=True,
                    severity=0.9,
                    recommended_attention="issue 1",
                ),
                SpecialAlert(
                    alert_id="a2",
                    alert_type="t2",
                    source=AlertSource.PROPOSER,
                    priority=AlertPriority.HIGH,
                    triggered=True,
                    severity=0.7,
                    recommended_attention="issue 2",
                ),
            ],
        )
        diagnosis = diagnoser.diagnose("test", summary, evidence)
        assert diagnosis.selected_alert_id is not None
