"""Stage 1: Weakness Identification.

Input: Solver trajectories, Solver outcomes, Previous task metadata.
Output: CapabilityGap + FailureSignature.

This stage is currently backed by ``TrajectoryAnalyzer.extract_signatures``
in proposer/trajectory_analyzer.py. The workflow delegates to it until the
richer FailureSignature fields from the refactor plan (root_cause,
capability_gap, reasoning_pattern, code_topology, tool_behavior,
failed_fix_pattern, transfer_constraints, forbidden_copy_features) are
fully implemented.
"""

from __future__ import annotations

from typing import List


class WeaknessAnalysisStage:
    """Stage 1: identify abstract capability gaps from solver failures."""

    def __init__(self, trajectory_analyzer):
        self.trajectory_analyzer = trajectory_analyzer

    def run(self, traces, outcomes) -> List:
        return self.trajectory_analyzer.extract_signatures(traces, outcomes)


__all__ = ["WeaknessAnalysisStage"]
