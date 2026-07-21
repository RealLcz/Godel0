"""Special case detectors for systematic failure patterns.

HGM-style: prioritize cross-task systematic failures over individual unresolved
tasks. Detectors only detect anomalies and provide severity/confidence; they do
not decide which component to modify.

Solver-side alerts (Stage 1 of the plan's Section 12.1):
  solver_empty_patch, solver_test_only_patch, solver_stochasticity,
  solver_context_overflow, solver_timeout, solver_repeated_tool_loop,
  solver_regression, solver_localization_collapse

Proposer / RepoChain-side alerts (Section 12.2):
  proposer_empty_task_batch, contract_generation_failure,
  clean_contract_failure, mutation_materialization_failure,
  no_f2p_dominant, no_p2p, causal_ablation_failure, low_valid_yield,
  duplicate_collapse, difficulty_too_easy, difficulty_too_hard,
  context_overflow, repo_subsystem_collapse, statement_leakage

Shared / runtime alerts:
  tool_repeated_failure
"""

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
        test_only_patch_count: int = 0,
        evaluated_count: int = 0,
        timeout_count: int = 0,
        context_overflow_count: int = 0,
        stochastic_task_count: int = 0,
        repeated_tool_loop_count: int = 0,
        localization_collapse_count: int = 0,
        config: dict = None,
    ) -> List[SpecialAlert]:
        alerts: List[SpecialAlert] = []
        config = config or {}
        empty_ratio_threshold = config.get("solver_empty_patch_ratio", 0.10)
        solver_stats = summary.solver_special_stats or {}

        if evaluated_count > 0 and empty_patch_count / evaluated_count >= empty_ratio_threshold:
            alerts.append(SpecialAlert(
                alert_id="solver_empty_patches",
                alert_type="solver_empty_patch",
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

        if evaluated_count > 0 and test_only_patch_count / evaluated_count >= empty_ratio_threshold:
            alerts.append(SpecialAlert(
                alert_id="solver_test_only_patches",
                alert_type="solver_test_only_patch",
                source=AlertSource.SOLVER,
                priority=AlertPriority.HIGH,
                triggered=True,
                severity=min(1.0, test_only_patch_count / max(evaluated_count, 1)),
                confidence=0.7,
                metric_name="test_only_patch_ratio",
                observed_value=test_only_patch_count / evaluated_count,
                threshold=empty_ratio_threshold,
                recommended_attention="Solver produces test-only patches without production fixes",
            ))

        if evaluated_count > 0 and timeout_count / evaluated_count >= 0.3:
            alerts.append(SpecialAlert(
                alert_id="solver_timeout",
                alert_type="solver_timeout",
                source=AlertSource.SOLVER,
                priority=AlertPriority.HIGH,
                triggered=True,
                severity=min(1.0, timeout_count / max(evaluated_count, 1)),
                confidence=0.8,
                metric_name="timeout_ratio",
                observed_value=timeout_count / evaluated_count,
                recommended_attention="Solver frequently times out",
            ))

        if evaluated_count > 0 and context_overflow_count / evaluated_count >= 0.2:
            alerts.append(SpecialAlert(
                alert_id="solver_context_overflow",
                alert_type="solver_context_overflow",
                source=AlertSource.SOLVER,
                priority=AlertPriority.HIGH,
                triggered=True,
                severity=min(1.0, context_overflow_count / max(evaluated_count, 1)),
                confidence=0.7,
                metric_name="context_overflow_ratio",
                observed_value=context_overflow_count / evaluated_count,
                recommended_attention="Solver hits context window limits",
            ))

        if evaluated_count > 0 and stochastic_task_count / evaluated_count >= 0.3:
            alerts.append(SpecialAlert(
                alert_id="solver_stochasticity",
                alert_type="solver_stochasticity",
                source=AlertSource.SOLVER,
                priority=AlertPriority.NORMAL,
                triggered=True,
                severity=min(1.0, stochastic_task_count / max(evaluated_count, 1)),
                confidence=0.6,
                metric_name="stochastic_task_ratio",
                observed_value=stochastic_task_count / evaluated_count,
                recommended_attention="Solver results are unstable across rollouts",
            ))

        if repeated_tool_loop_count >= 2:
            alerts.append(SpecialAlert(
                alert_id="solver_repeated_tool_loop",
                alert_type="solver_repeated_tool_loop",
                source=AlertSource.SOLVER,
                priority=AlertPriority.NORMAL,
                triggered=True,
                severity=min(1.0, repeated_tool_loop_count / 5.0),
                confidence=0.7,
                metric_name="repeated_tool_loop_count",
                observed_value=float(repeated_tool_loop_count),
                recommended_attention="Solver repeatedly calls the same tool without expanding the call chain",
            ))

        if localization_collapse_count >= 2:
            alerts.append(SpecialAlert(
                alert_id="solver_localization_collapse",
                alert_type="solver_localization_collapse",
                source=AlertSource.SOLVER,
                priority=AlertPriority.HIGH,
                triggered=True,
                severity=min(1.0, localization_collapse_count / 5.0),
                confidence=0.7,
                metric_name="localization_collapse_count",
                observed_value=float(localization_collapse_count),
                recommended_attention="Solver fails to localize the relevant files",
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
    """Detects systematic proposer / RepoChain failure patterns."""

    def detect(
        self,
        summary: NodeCycleSummary,
        candidates: List[CandidateValidationReport] = None,
        proposer_stats: dict = None,
        config: dict = None,
    ) -> List[SpecialAlert]:
        alerts: List[SpecialAlert] = []
        config = config or {}
        min_yield = config.get("proposer_min_valid_yield", 0.20)
        proposer_stats = proposer_stats or {}
        rejections = dict(summary.proposer_rejection_distribution)

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
                alert_type="low_valid_yield",
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

        # Dominant rejection alerts.
        total_rejections = sum(rejections.values())
        if total_rejections > 0:
            for reason, count in rejections.items():
                ratio = count / total_rejections
                if ratio > 0.5:
                    alert_type = self._rejection_alert_type(reason)
                    alerts.append(SpecialAlert(
                        alert_id=f"proposer_dominant_{alert_type}",
                        alert_type=alert_type,
                        source=AlertSource.PROPOSER,
                        priority=AlertPriority.HIGH if ratio > 0.7 else AlertPriority.NORMAL,
                        triggered=True,
                        severity=ratio,
                        confidence=0.7,
                        metric_name=f"{alert_type}_ratio",
                        observed_value=ratio,
                        recommended_attention=f"Dominant rejection: {reason} ({count}/{total_rejections})",
                    ))

        # Causal ablation failure: the most important RepoChain quality signal.
        causal_failures = proposer_stats.get("causal_ablation_failure_count", 0)
        if causal_failures >= 2:
            alerts.append(SpecialAlert(
                alert_id="causal_ablation_failure",
                alert_type="causal_ablation_failure",
                source=AlertSource.PROPOSER,
                priority=AlertPriority.CRITICAL,
                triggered=True,
                severity=min(1.0, causal_failures / 10.0),
                confidence=0.9,
                metric_name="causal_ablation_failure_count",
                observed_value=float(causal_failures),
                recommended_attention="Tasks lack chain-level causal structure: repairing one file restores all contracts",
            ))

        # Contract generation failures.
        contract_failures = proposer_stats.get("contract_generation_failure_count", 0)
        if contract_failures >= 2:
            alerts.append(SpecialAlert(
                alert_id="contract_generation_failure",
                alert_type="contract_generation_failure",
                source=AlertSource.PROPOSER,
                priority=AlertPriority.HIGH,
                triggered=True,
                severity=min(1.0, contract_failures / 10.0),
                confidence=0.8,
                metric_name="contract_generation_failure_count",
                observed_value=float(contract_failures),
                recommended_attention="Contract generation frequently fails",
            ))

        # Clean contract failures (contract fails on clean repo).
        clean_contract_failures = proposer_stats.get("clean_contract_failure_count", 0)
        if clean_contract_failures >= 1:
            alerts.append(SpecialAlert(
                alert_id="clean_contract_failure",
                alert_type="clean_contract_failure",
                source=AlertSource.PROPOSER,
                priority=AlertPriority.HIGH,
                triggered=True,
                severity=min(1.0, clean_contract_failures / 5.0),
                confidence=0.9,
                metric_name="clean_contract_failure_count",
                observed_value=float(clean_contract_failures),
                recommended_attention="Contracts fail on the clean repository (invalid contract)",
            ))

        # Statement leakage.
        leakage_count = proposer_stats.get("statement_leakage_count", 0)
        if leakage_count >= 1:
            alerts.append(SpecialAlert(
                alert_id="statement_leakage",
                alert_type="statement_leakage",
                source=AlertSource.PROPOSER,
                priority=AlertPriority.HIGH,
                triggered=True,
                severity=min(1.0, leakage_count / 5.0),
                confidence=0.9,
                metric_name="statement_leakage_count",
                observed_value=float(leakage_count),
                recommended_attention="Problem statements leak mutation-site details",
            ))

        # Difficulty mismatch.
        if summary.level2_accuracy is not None:
            if summary.level2_accuracy > config.get("difficulty_high", 0.60):
                alerts.append(SpecialAlert(
                    alert_id="proposer_difficulty_too_easy",
                    alert_type="difficulty_too_easy",
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
                    alert_type="difficulty_too_hard",
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

    def _rejection_alert_type(self, reason: str) -> str:
        """Map a rejection reason string to a canonical alert_type."""
        reason_lower = reason.lower()
        if "no_f2p" in reason_lower or "f2p" in reason_lower:
            return "no_f2p_dominant"
        if "no_p2p" in reason_lower or "p2p" in reason_lower:
            return "no_p2p"
        if "duplicate" in reason_lower:
            return "duplicate_collapse"
        if "mutation" in reason_lower or "materialization" in reason_lower:
            return "mutation_materialization_failure"
        if "context" in reason_lower or "overflow" in reason_lower:
            return "context_overflow"
        if "leakage" in reason_lower or "leak" in reason_lower:
            return "statement_leakage"
        if "subsystem" in reason_lower:
            return "repo_subsystem_collapse"
        return "proposer_dominant_rejection"


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
                    observed_value=float(count),
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
        proposer_stats: dict = None,
        config: dict = None,
    ) -> List[SpecialAlert]:
        alerts: List[SpecialAlert] = []
        solver_stats = solver_stats or {}
        proposer_stats = proposer_stats or {}
        alerts.extend(self.solver.detect(
            summary, trajectories, **solver_stats, config=config,
        ))
        alerts.extend(self.proposer.detect(
            summary, candidates, proposer_stats=proposer_stats, config=config,
        ))
        alerts.extend(self.shared.detect(summary, tool_events))
        return alerts
