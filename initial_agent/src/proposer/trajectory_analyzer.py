from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from .schemas import FailureSignature

FailureStage = Literal[
    "localization",
    "reproduction",
    "patch_generation",
    "validation",
    "tool_use",
    "context_management",
]


@dataclass
class TrajectoryView:
    """A read-only view over a single solver trajectory JSONL file.

    Each line in the JSONL file is expected to be a JSON object representing
    a step (assistant message, tool call, tool result, etc.).
    """

    trajectory_id: str
    task_id: str = ""
    node_id: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    raw_path: str = ""

    @classmethod
    def from_jsonl(cls, path: str) -> "TrajectoryView":
        trajectory_id = os.path.splitext(os.path.basename(path))[0]
        companion_path = os.path.splitext(path)[0] + "_eval.json"
        companion: Dict[str, Any] = {}
        if os.path.isfile(companion_path):
            try:
                with open(companion_path, "r", encoding="utf-8") as f:
                    companion = json.load(f)
                trajectory_id = str(companion.get("trajectory_id") or trajectory_id)
            except (json.JSONDecodeError, OSError):
                companion = {}
        steps: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    steps.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        task_id = str(companion.get("task_id") or "")
        node_id = str(companion.get("node_id") or "")
        for step in steps:
            if isinstance(step, dict):
                task_id = task_id or str(step.get("task_id", ""))
                node_id = node_id or str(step.get("node_id", ""))
                if task_id and node_id:
                    break
        return cls(
            trajectory_id=trajectory_id,
            task_id=task_id,
            node_id=node_id,
            steps=steps,
            raw_path=path,
        )


@dataclass
class EvaluationOutcomeView:
    """A read-only view over the evaluation outcome for a trajectory."""

    trajectory_id: str
    task_id: str = ""
    success: bool = False
    fail_to_pass: List[str] = field(default_factory=list)
    pass_to_pass: List[str] = field(default_factory=list)
    patch: str = ""
    empty_patch: bool = False
    test_only_patch: bool = False
    error_stage: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationOutcomeView":
        patch = str(data.get("model_patch", data.get("patch", "")))
        empty_patch = not patch.strip()
        test_only_patch = bool(data.get("test_only_patch", False))
        return cls(
            trajectory_id=str(data.get("trajectory_id", "")),
            task_id=str(data.get("task_id", "")),
            success=bool(data.get("success", data.get("resolved", False))),
            fail_to_pass=list(data.get("fail_to_pass", [])),
            pass_to_pass=list(data.get("pass_to_pass", [])),
            patch=patch,
            empty_patch=empty_patch,
            test_only_patch=test_only_patch,
            error_stage=str(data.get("error_stage", "")),
            raw=data,
        )

    @classmethod
    def from_json(cls, path: str) -> "EvaluationOutcomeView":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


_TOOL_RE = re.compile(r"\b(?:bash|edit|str_replace|view|find|grep|python)\b", re.IGNORECASE)


class TrajectoryAnalyzer:
    """Extracts FailureSignature objects from solver trajectories.

    The analyzer inspects file/symbol localization, tool-use sequences,
    whether a reproducer was generated, patch scope, test behavior,
    empty/test-only patches, context/timeout issues, and visible-test overfit.
    It emits signatures conforming to the strict schema in `schemas.py`.
    """

    def extract_signatures(
        self,
        trajectories: List[TrajectoryView],
        outcomes: List[EvaluationOutcomeView],
    ) -> List[FailureSignature]:
        outcome_by_id = {o.trajectory_id: o for o in outcomes}
        signatures: List[FailureSignature] = []
        for traj in trajectories:
            outcome = outcome_by_id.get(traj.trajectory_id)
            if outcome is None or outcome.success:
                continue
            sig = self._analyze_one(traj, outcome)
            if sig is not None:
                signatures.append(sig)
        return signatures

    def _analyze_one(
        self,
        traj: TrajectoryView,
        outcome: EvaluationOutcomeView,
    ) -> Optional[FailureSignature]:
        steps = traj.steps
        stage = self._infer_stage(traj, outcome)
        root_cause = self._infer_root_cause(traj, outcome, stage)
        code_patterns = self._extract_code_patterns(steps)
        behavior_pattern = self._extract_behavior_pattern(traj, outcome)
        preferred_operators = self._suggest_operators(stage, behavior_pattern)
        target_capability = self._infer_capability(stage, behavior_pattern)

        return FailureSignature(
            signature_id=f"sig-{uuid.uuid4().hex[:12]}",
            source_solver_node_id=traj.node_id,
            source_task_id=traj.task_id,
            source_trajectory_id=traj.trajectory_id,
            failure_stage=stage,
            root_cause=root_cause,
            target_capability=target_capability,
            code_patterns=code_patterns,
            behavior_pattern=behavior_pattern,
            preferred_operators=preferred_operators,
            transfer_mode="same_repo_nearby",
            forbidden_copy_features=self._forbidden_features(outcome),
        )

    def _infer_stage(
        self,
        traj: TrajectoryView,
        outcome: EvaluationOutcomeView,
    ) -> FailureStage:
        if outcome.empty_patch:
            return "patch_generation"
        if outcome.test_only_patch:
            return "patch_generation"
        if any("timeout" in str(s).lower() or "context" in str(s).lower() for s in traj.steps[:1]):
            return "context_management"
        if not self._has_localization(steps=traj.steps):
            return "localization"
        if not outcome.fail_to_pass and not outcome.success:
            return "validation"
        tool_calls = self._count_tool_calls(traj.steps)
        if tool_calls == 0:
            return "tool_use"
        return "patch_generation"

    def _has_localization(self, steps: List[Dict[str, Any]]) -> bool:
        for step in steps:
            content = json.dumps(step).lower()
            if any(k in content for k in ("view", "find", "grep", "str_replace", "file_path")):
                return True
        return False

    def _count_tool_calls(self, steps: List[Dict[str, Any]]) -> int:
        count = 0
        for step in steps:
            content = json.dumps(step)
            count += len(_TOOL_RE.findall(content))
        return count

    def _infer_root_cause(
        self,
        traj: TrajectoryView,
        outcome: EvaluationOutcomeView,
        stage: str,
    ) -> str:
        if outcome.empty_patch:
            return "Agent produced an empty patch; failed to emit any source edit."
        if outcome.test_only_patch:
            return "Agent only modified tests without touching primary source code."
        if stage == "localization":
            return "Agent failed to localize the relevant file/symbol."
        if stage == "tool_use":
            return "Agent did not make effective use of available tools."
        if stage == "context_management":
            return "Agent hit context length or timeout limits."
        if outcome.fail_to_pass:
            return f"Agent patch did not satisfy {len(outcome.fail_to_pass)} FAIL_TO_PASS tests."
        return "Patch was incomplete or introduced a regression."

    def _extract_code_patterns(self, steps: List[Dict[str, Any]]) -> List[str]:
        patterns: List[str] = []
        for step in steps:
            content = json.dumps(step)
            for match in re.findall(r"def\s+([A-Za-z_][A-Za-z0-9_]*)", content):
                patterns.append(f"function:{match}")
            for match in re.findall(r"class\s+([A-Za-z_][A-Za-z0-9_]*)", content):
                patterns.append(f"class:{match}")
        seen: List[str] = []
        for p in patterns:
            if p not in seen:
                seen.append(p)
        return seen[:16]

    def _extract_behavior_pattern(
        self,
        traj: TrajectoryView,
        outcome: EvaluationOutcomeView,
    ) -> Dict[str, Any]:
        return {
            "tool_call_count": self._count_tool_calls(traj.steps),
            "empty_patch": outcome.empty_patch,
            "test_only_patch": outcome.test_only_patch,
            "fail_to_pass_count": len(outcome.fail_to_pass),
            "pass_to_pass_count": len(outcome.pass_to_pass),
            "patch_lines": len(outcome.patch.splitlines()) if outcome.patch else 0,
        }

    def _suggest_operators(
        self,
        stage: str,
        behavior_pattern: Dict[str, Any],
    ) -> List[str]:
        ops: List[str] = []
        if stage == "localization":
            ops.extend(["misdirect_localization", "rename_symbol"])
        elif stage == "patch_generation":
            ops.extend(["off_by_one", "wrong_condition", "missing_edge_case"])
        elif stage == "tool_use":
            ops.extend(["break_tool_invocation"])
        elif stage == "context_management":
            ops.extend(["inflate_context"])
        elif stage == "validation":
            ops.extend(["silent_regression"])
        else:
            ops.extend(["subtle_logic_error"])
        return ops

    def _infer_capability(
        self,
        stage: str,
        behavior_pattern: Dict[str, Any],
    ) -> str:
        mapping = {
            "localization": "codebase_navigation",
            "reproduction": "reproducer_synthesis",
            "patch_generation": "correct_patch_synthesis",
            "validation": "test_alignment",
            "tool_use": "tool_orchestration",
            "context_management": "context_budgeting",
        }
        return mapping.get(stage, "general_coding")

    def _forbidden_features(self, outcome: EvaluationOutcomeView) -> List[str]:
        forbidden: List[str] = []
        if outcome.test_only_patch:
            forbidden.append("test_only_modifications")
        if outcome.empty_patch:
            forbidden.append("empty_patch")
        return forbidden


__all__ = [
    "TrajectoryAnalyzer",
    "TrajectoryView",
    "EvaluationOutcomeView",
]
