"""Integration test: joint cycle diagnosis."""

from __future__ import annotations

import pytest

from godel0.evolution.cycle_builder import NodeCycleBuilder
from godel0.evolution.special_detectors import CompositeSpecialDetector
from godel0.evolution.evidence_selector import CycleEvidenceSelector
from godel0.evolution.diagnose import CycleDiagnoser
from godel0.schemas.cycle import (
    NodeCycleSummary, CycleStage, AlertPriority,
    CycleEvidenceBundle,
)
from godel0.schemas.node import NodeRecord, NodeStatus
from godel0.schemas.evaluation import Level1Result


class TestJointDiagnosis:
    def test_simultaneous_solver_and_proposer_issues(self):
        """When both Solver empty patches and Proposer no-F2P exist,
        two independent alerts are generated, but Diagnoser selects ONE root cause."""
        node = NodeRecord(
            node_id="test_node",
            code_commit="abc",
            code_ref="ref",
            status=NodeStatus.COMPLETE,
        )

        level1 = Level1Result(
            parent_node_id="parent",
            child_node_id="test_node",
            evaluated_task_ids=["t1", "t2"],
            parent_solved_task_ids=["t1", "t2"],
            child_retained_task_ids=["t1"],
            child_forgotten_task_ids=["t2"],
            child_newly_solved_task_ids=[],
            retention_rate=0.5,
            threshold=0.8,
            passed=False,
        )

        builder = NodeCycleBuilder()
        summary = builder.build(node, level1=level1)
        summary.proposer_accepted_tasks = 0
        summary.proposer_requested_tasks = 10
        summary.proposer_generated_candidates = 20

        detector = CompositeSpecialDetector()
        alerts = detector.detect(
            summary,
            solver_stats={"empty_patch_count": 2, "evaluated_count": 5},
        )

        solver_alerts = [a for a in alerts if a.source.value == "solver"]
        proposer_alerts = [a for a in alerts if a.source.value == "proposer"]
        assert len(solver_alerts) >= 1
        assert len(proposer_alerts) >= 1

        selector = CycleEvidenceSelector()
        evidence = selector.select(summary, alerts)

        diagnoser = CycleDiagnoser()
        diagnosis = diagnoser.diagnose(node.node_id, summary, evidence)

        assert diagnosis.primary_root_cause
        assert diagnosis.selected_alert_id is not None

    def test_recommended_scopes_can_span_roles(self):
        """recommended_edit_scopes can include tools + proposer_logic
        for the same root cause."""
        node = NodeRecord(
            node_id="test_node",
            code_commit="abc",
            code_ref="ref",
            status=NodeStatus.COMPLETE,
        )
        summary = NodeCycleSummary(
            node_id="test_node",
            proposer_valid_yield=0.1,
            proposer_generated_candidates=20,
            proposer_accepted_tasks=2,
            proposer_requested_tasks=10,
        )
        evidence = selector_bundle = CycleEvidenceBundle(
            node_id="test_node",
            special_alerts=[],
        )
        diagnoser = CycleDiagnoser()
        diagnosis = diagnoser.diagnose(node.node_id, summary, evidence)

        assert diagnosis.recommended_edit_scopes
        assert len(diagnosis.recommended_edit_scopes) >= 1

    def test_no_choose_solver_or_proposer_function(self):
        """The code must not contain a choose_solver_or_proposer() function."""
        import godel0.controller.orchestrator as orch
        assert not hasattr(orch, "choose_solver_or_proposer")

        import godel0.evolution.diagnose as diag
        assert not hasattr(diag, "choose_solver_or_proposer")

    def test_evidence_budget_enforced(self):
        """EvidenceSelector must respect token budget."""
        selector = CycleEvidenceSelector(
            max_solver_trajectories=4,
            max_proposer_candidates=4,
            max_total_evidence_chars=5000,
        )
        summary = NodeCycleSummary(node_id="test")
        artifacts = {
            "solver_trajectories": [{"data": "x" * 10000} for _ in range(20)],
            "proposer_candidates": [{"data": "y" * 10000} for _ in range(20)],
        }
        bundle = selector.select(summary, [], artifacts)

        solver_items = [i for i in bundle.items if i.evidence_type == "solver_trajectory"]
        proposer_items = [i for i in bundle.items if i.evidence_type == "proposer_candidate"]

        assert len(solver_items) <= 4
        assert len(proposer_items) <= 4
