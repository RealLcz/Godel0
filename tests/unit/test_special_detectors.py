"""Unit tests for special detectors."""

from __future__ import annotations

import pytest

from godel0.evolution.special_detectors import (
    SolverSpecialDetector,
    ProposerSpecialDetector,
    CompositeSpecialDetector,
)
from godel0.schemas.cycle import NodeCycleSummary, AlertPriority


class TestSolverSpecialDetector:
    def test_empty_patch_detection(self):
        detector = SolverSpecialDetector()
        summary = NodeCycleSummary(node_id="test")
        alerts = detector.detect(
            summary,
            empty_patch_count=3,
            evaluated_count=10,
        )
        empty_alerts = [a for a in alerts if a.alert_type == "solver_empty_patches"]
        assert len(empty_alerts) == 1
        assert empty_alerts[0].priority == AlertPriority.HIGH

    def test_regression_detection(self):
        detector = SolverSpecialDetector()
        summary = NodeCycleSummary(
            node_id="test",
            level1_retention=0.5,
            forgotten_task_ids=["t1", "t2"],
        )
        alerts = detector.detect(summary, config={"regression_threshold": 0.8})
        regression_alerts = [a for a in alerts if a.alert_type == "solver_regression"]
        assert len(regression_alerts) == 1
        assert regression_alerts[0].priority == AlertPriority.CRITICAL

    def test_no_alerts_when_healthy(self):
        detector = SolverSpecialDetector()
        summary = NodeCycleSummary(
            node_id="test",
            level1_retention=0.9,
        )
        alerts = detector.detect(summary, empty_patch_count=0, evaluated_count=10)
        assert len(alerts) == 0


class TestProposerSpecialDetector:
    def test_empty_batch_detection(self):
        detector = ProposerSpecialDetector()
        summary = NodeCycleSummary(
            node_id="test",
            proposer_requested_tasks=10,
            proposer_accepted_tasks=0,
        )
        alerts = detector.detect(summary)
        empty_alerts = [a for a in alerts if a.alert_type == "proposer_empty_task_batch"]
        assert len(empty_alerts) == 1
        assert empty_alerts[0].priority == AlertPriority.CRITICAL

    def test_low_yield_detection(self):
        detector = ProposerSpecialDetector()
        summary = NodeCycleSummary(
            node_id="test",
            proposer_generated_candidates=20,
            proposer_accepted_tasks=2,
            proposer_valid_yield=0.1,
            proposer_requested_tasks=10,
        )
        alerts = detector.detect(summary)
        yield_alerts = [a for a in alerts if a.alert_type == "proposer_low_valid_yield"]
        assert len(yield_alerts) == 1

    def test_difficulty_mismatch(self):
        detector = ProposerSpecialDetector()
        summary = NodeCycleSummary(
            node_id="test",
            level2_accuracy=0.8,
            proposer_requested_tasks=10,
            proposer_accepted_tasks=10,
        )
        alerts = detector.detect(summary)
        diff_alerts = [a for a in alerts if a.alert_type == "proposer_difficulty_mismatch"]
        assert len(diff_alerts) >= 1


class TestCompositeDetector:
    def test_combined_detection(self):
        detector = CompositeSpecialDetector()
        summary = NodeCycleSummary(
            node_id="test",
            level1_retention=0.3,
            proposer_accepted_tasks=0,
            proposer_requested_tasks=10,
            forgotten_task_ids=["t1", "t2", "t3"],
        )
        alerts = detector.detect(summary)
        assert len(alerts) >= 2
        critical = [a for a in alerts if a.priority == AlertPriority.CRITICAL]
        assert len(critical) >= 1
