"""Stage 6: Causal Ablation.

Verify that the task has genuine chain-level causal structure:
  - Repairing only one file -> complete contracts still fail.
  - A single isolated mutation -> does it independently trigger the contract?

This excludes tasks that are merely several unrelated single-file bugs
concatenated into a multi-file task. ``causal_ablation_pass`` is the core
quality signal for RepoChain tasks.

Phase 2 stub: returns ``True`` (pass) by default. Real enforcement lands in
Phase 4 where ``causal_ablation_failure`` special alerts are wired.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AblationResult:
    passed: bool = True
    repair_one_file_still_fails: bool = True
    single_mutation_triggers_contract: bool = True
    details: dict = None


class CausalAblationStage:
    """Stage 6: causal ablation check.

    Stub: returns a passing result. Real ablation logic lands in Phase 4.
    """

    def run(self, plan, repo_spec, candidates, contracts):
        return AblationResult(passed=True)


__all__ = ["AblationResult", "CausalAblationStage"]
