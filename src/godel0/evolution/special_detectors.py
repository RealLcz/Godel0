"""Special case detectors for systematic failure patterns."""

from __future__ import annotations

from typing import List

from ..schemas.cycle import AlertPriority, AlertSource, NodeCycleSummary, SpecialAlert
from ..schemas.evaluation import CandidateValidationReport
from ..schemas.trajectory import TrajectoryRecord


class SolverSpecialDetector:
    """Detects systematic solver failure patterns (HGM-style)."""

    def detect(
        self,
        summary: NodeCycleSummary,
        trajectories: List[TrajectoryRecord] = None,
        empty_patch_count: int = 0,
        evaluated_count: int = 0,
        config: dict = None,
    ) -> List[SpecialAlert]:
        alerts: List[SpecialAlert] = []
        config = config or {}
        empty_ratio_threshold = config.get("solver_empty_patch_ratio", 0.10)

        if evaluated_count > 0 and empty_patch_count / evaluated_count >= empty_ratio_threshold:
            alerts.append(SpecialAlert(
                alert_id="solver_empty_patches",
                alert_type="solver_empty_patches",
                source=AlertSource.SOLVER,
                priority=AlertPriority.HIGH,
                triggered=True,
                severity=min(1.0, empty_patch_count / max(evaluated_count, 1)),
                confidence=0.8,
                metric_name="empty_patch_ratio",
                observed_value=empty_patch_count / evaluated_count,
                threshold=empty_ratio_threshold,
                recommended_attention="Solver produces too many empty or test-only patches",
            ))

        if summary.level1_retention is not None:
            threshold = config.get("regression_threshold", 0.8)
            if summary.level1_retention < threshold:
                alerts.append(SpecialAlert(
                    alert_id="solver_regression",
                    alert_type="solver_regression",
                    source=AlertSource.SOLVER,
                    priority=AlertPriority.CRITICAL,
                    triggered=True,
                    severity=1.0 - summary.level1_retention,
                    confidence=0.9,
                    metric_name="retention_rate",
                    observed_value=summary.level1_retention,
                    threshold=threshold,
                    recommended_attention="Level 1 regression detected",
                ))

        if summary.forgotten_task_ids:
            alerts.append(SpecialAlert(
                alert_id="solver_forgotten_tasks",
                alert_type="solver_forgotten_tasks",
                source=AlertSource.SOLVER,
                priority=AlertPriority.NORMAL,
                triggered=True,
                severity=len(summary.forgotten_task_ids) / 10.0,
                confidence=0.7,
                metric_name="forgotten_count",
                observed_value=len(summary.forgotten_task_ids),
                recommended_attention="Solver forgot some parent-solved tasks",
            ))

        return alerts


class ProposerSpecialDetector:
    """Detects systematic proposer failure patterns."""

    def detect(
        self,
        summary: NodeCycleSummary,
        candidates: List[CandidateValidationReport] = None,
        config: dict = None,
    ) -> List[SpecialAlert]:
        alerts: List[SpecialAlert] = []
        config = config or {}
        min_yield = config.get("proposer_min_valid_yield", 0.20)

        if summary.proposer_accepted_tasks == 0 and summary.proposer_requested_tasks > 0:
            alerts.append(SpecialAlert(
                alert_id="proposer_empty_task_batch",
                alert_type="proposer_empty_task_batch",
                source=AlertSource.PROPOSER,
                priority=AlertPriority.CRITICAL,
                triggered=True,
                severity=1.0,
                confidence=0.9,
                metric_name="accepted_tasks",
                observed_value=0,
                recommended_attention="Proposer generated zero valid tasks",
            ))

        if summary.proposer_valid_yield is not None and summary.proposer_valid_yield < min_yield:
            alerts.append(SpecialAlert(
                alert_id="proposer_low_valid_yield",
                alert_type="proposer_low_valid_yield",
                source=AlertSource.PROPOSER,
                priority=AlertPriority.HIGH,
                triggered=True,
                severity=1.0 - summary.proposer_valid_yield,
                confidence=0.8,
                metric_name="valid_yield",
                observed_value=summary.proposer_valid_yield,
                threshold=min_yield,
                recommended_attention="Proposer valid candidate yield too low",
            ))

        if summary.proposer_rejection_distribution:
            dominant = max(summary.proposer_rejection_distribution.values())
            total = sum(summary.proposer_rejection_distribution.values())
            if total > 0 and dominant / total > 0.5:
                dominant_reason = max(summary.proposer_rejection_distribution, key=summary.proposer_rejection_distribution.get)
                alerts.append(SpecialAlert(
                    alert_id="proposer_dominant_rejection",
                    alert_type="proposer_dominant_rejection",
                    source=AlertSource.PROPOSER,
                    priority=AlertPriority.NORMAL,
                    triggered=True,
                    severity=dominant / total,
                    confidence=0.6,
                    metric_name="dominant_rejection_ratio",
                    observed_value=dominant / total,
                    recommended_attention=f"Dominant rejection: {dominant_reason}",
                ))

        if summary.level2_accuracy is not None:
            if summary.level2_accuracy > config.get("difficulty_high", 0.60):
                alerts.append(SpecialAlert(
                    alert_id="proposer_difficulty_too_easy",
                    alert_type="proposer_difficulty_mismatch",
                    source=AlertSource.PROPOSER,
                    priority=AlertPriority.NORMAL,
                    triggered=True,
                    severity=summary.level2_accuracy,
                    confidence=0.5,
                    metric_name="level2_accuracy",
                    observed_value=summary.level2_accuracy,
                    recommended_attention="Tasks may be too easy",
                ))
            elif summary.level2_accuracy < config.get("difficulty_low", 0.40):
                alerts.append(SpecialAlert(
                    alert_id="proposer_difficulty_too_hard",
                    alert_type="proposer_difficulty_mismatch",
                    source=AlertSource.PROPOSER,
                    priority=AlertPriority.NORMAL,
                    triggered=True,
                    severity=1.0 - summary.level2_accuracy,
                    confidence=0.5,
                    metric_name="level2_accuracy",
                    observed_value=summary.level2_accuracy,
                    recommended_attention="Tasks may be too hard (ambiguous)",
                ))

        return alerts


class SharedSpecialDetector:
    """Detects shared tool and runtime failures."""

    def detect(
        self,
        summary: NodeCycleSummary,
        tool_events: List[dict] = None,
    ) -> List[SpecialAlert]:
        alerts: List[SpecialAlert] = []
        tool_events = tool_events or []

        tool_failures: dict[str, int] = {}
        for event in tool_events:
            tool_name = event.get("tool", "unknown")
            if event.get("error"):
                tool_failures[tool_name] = tool_failures.get(tool_name, 0) + 1

        for tool, count in tool_failures.items():
            if count >= 2:
                alerts.append(SpecialAlert(
                    alert_id=f"tool_failure_{tool}",
                    alert_type="tool_repeated_failure",
                    source=AlertSource.SHARED,
                    priority=AlertPriority.HIGH if count >= 3 else AlertPriority.NORMAL,
                    triggered=True,
                    severity=min(1.0, count / 5.0),
                    confidence=0.7,
                    metric_name=f"{tool}_failure_count",
                    observed_value=count,
                    recommended_attention=f"Tool '{tool}' failed {count} times",
                ))

        return alerts


class CompositeSpecialDetector:
    """Combines all special detectors."""

    def __init__(self):
        self.solver = SolverSpecialDetector()
        self.proposer = ProposerSpecialDetector()
        self.shared = SharedSpecialDetector()

    def detect(
        self,
        summary: NodeCycleSummary,
        trajectories: List[TrajectoryRecord] = None,
        candidates: List[CandidateValidationReport] = None,
        tool_events: List[dict] = None,
        solver_stats: dict = None,
        config: dict = None,
    ) -> List[SpecialAlert]:
        alerts: List[SpecialAlert] = []
        solver_stats = solver_stats or {}
        alerts.extend(self.solver.detect(summary, trajectories, **solver_stats, config=config))
        alerts.extend(self.proposer.detect(summary, candidates, config=config))
        alerts.extend(self.shared.detect(summary, tool_events))
        return alerts
