"""Select one Proposer or Solver failure entry for HGM-style diagnose."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class ProposerFailureEntry:
    """One rejected / invalid proposer candidate."""

    candidate_id: str
    reason: str = ""
    validation_report: Dict[str, Any] = field(default_factory=dict)
    attempt_dir: Optional[Path] = None
    candidate_dir: Optional[Path] = None
    stdout_log: str = ""
    stderr_log: str = ""
    bug_patch: str = ""
    problem_statement: str = ""
    mutation_diff: str = ""
    failing_test_output: str = ""

    @property
    def id(self) -> str:
        return self.candidate_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "reason": self.reason,
            "attempt_dir": str(self.attempt_dir) if self.attempt_dir else None,
            "candidate_dir": str(self.candidate_dir) if self.candidate_dir else None,
            "validation_report": self.validation_report,
            "has_stdout": bool(self.stdout_log.strip()),
            "has_stderr": bool(self.stderr_log.strip()),
            "has_bug_patch": bool(self.bug_patch.strip()),
            "has_problem_statement": bool(self.problem_statement.strip()),
        }


@dataclass
class SolverFailureEntry:
    """One failed solver task with trajectory artifacts."""

    task_id: str
    level: int
    trajectory_path: Optional[Path] = None
    patch_path: Optional[Path] = None
    eval_path: Optional[Path] = None
    problem_statement: str = ""
    failing_test_output: str = ""
    trajectory_text: str = ""
    predicted_patch: str = ""
    eval_log: str = ""

    @property
    def id(self) -> str:
        return f"L{self.level}:{self.task_id}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "level": self.level,
            "trajectory_path": str(self.trajectory_path) if self.trajectory_path else None,
            "patch_path": str(self.patch_path) if self.patch_path else None,
            "eval_path": str(self.eval_path) if self.eval_path else None,
            "has_problem_statement": bool(self.problem_statement.strip()),
            "has_trajectory": bool(self.trajectory_text.strip()),
            "has_predicted_patch": bool(self.predicted_patch.strip()),
            "has_eval_log": bool(self.eval_log.strip()),
        }


def _read_text(path: Optional[Path], limit: int = 0) -> str:
    if path is None or not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if limit > 0 and len(text) > limit:
        return text[:limit]
    return text


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _find_candidate_dir(proposer_dir: Path, candidate_id: str) -> Optional[Path]:
    matches = list(proposer_dir.glob(f"**/proposer_candidates/**/{candidate_id}"))
    for match in matches:
        if match.is_dir():
            return match
    # Some layouts nest one extra level under the candidate id.
    matches = list(proposer_dir.glob(f"**/{candidate_id}"))
    for match in matches:
        if match.is_dir() and (
            (match / "bug.patch").exists()
            or (match / "mutation.diff").exists()
            or (match / "validation.json").exists()
        ):
            return match
    return None


def _attempt_dir_for_candidate(proposer_dir: Path, candidate_id: str) -> Optional[Path]:
    cand_dir = _find_candidate_dir(proposer_dir, candidate_id)
    if cand_dir is None:
        return None
    for parent in cand_dir.parents:
        if parent.name.startswith("attempt_"):
            return parent
        if parent == proposer_dir:
            break
    return None


def _hydrate_proposer_entry(
    proposer_dir: Path,
    candidate_id: str,
    reason: str,
    report: Optional[Dict[str, Any]] = None,
) -> ProposerFailureEntry:
    report = dict(report or {})
    attempt_dir = _attempt_dir_for_candidate(proposer_dir, candidate_id)
    candidate_dir = _find_candidate_dir(proposer_dir, candidate_id)
    stdout = _read_text(attempt_dir / "proposer.stdout.log" if attempt_dir else None)
    stderr = _read_text(attempt_dir / "proposer.stderr.log" if attempt_dir else None)
    bug_patch = ""
    problem_statement = ""
    mutation_diff = ""
    if candidate_dir is not None:
        bug_patch = _read_text(candidate_dir / "bug.patch")
        if not bug_patch:
            # Nested cand_chain_* under plan dir.
            nested = list(candidate_dir.glob("**/bug.patch"))
            if nested:
                bug_patch = _read_text(nested[0])
        problem_statement = _read_text(candidate_dir / "problem_statement.md")
        if not problem_statement:
            nested_ps = list(candidate_dir.glob("**/problem_statement.md"))
            if nested_ps:
                problem_statement = _read_text(nested_ps[0])
        mutation_diff = _read_text(candidate_dir / "mutation.diff")
        if not mutation_diff:
            nested_md = list(candidate_dir.glob("**/mutation.diff"))
            if nested_md:
                mutation_diff = _read_text(nested_md[0])
    failing = str(report.get("failing_test_output") or "")
    if not reason:
        reasons = report.get("rejection_reasons") or []
        if isinstance(reasons, list):
            reason = "; ".join(str(r) for r in reasons)
        else:
            reason = str(reasons or report.get("reason") or "")
    return ProposerFailureEntry(
        candidate_id=candidate_id,
        reason=reason,
        validation_report=report,
        attempt_dir=attempt_dir,
        candidate_dir=candidate_dir,
        stdout_log=stdout,
        stderr_log=stderr,
        bug_patch=bug_patch,
        problem_statement=problem_statement,
        mutation_diff=mutation_diff,
        failing_test_output=failing,
    )


def list_proposer_failures(proposer_dir: Path) -> List[ProposerFailureEntry]:
    """Collect rejected proposer candidates from trusted_feedback / validation reports."""
    proposer_dir = Path(proposer_dir)
    if not proposer_dir.is_dir():
        return []

    by_id: Dict[str, ProposerFailureEntry] = {}

    # Prefer trusted_feedback rejects.
    feedback_dir = proposer_dir / "trusted_feedback"
    if feedback_dir.is_dir():
        for path in sorted(feedback_dir.glob("*.json")):
            data = _load_json(path)
            if not isinstance(data, dict):
                continue
            if data.get("accepted") is True:
                continue
            candidate_id = str(data.get("candidate_id") or path.stem)
            notes = data.get("notes")
            report: Dict[str, Any] = {}
            if isinstance(notes, dict):
                report = notes
            elif isinstance(notes, str):
                try:
                    parsed = json.loads(notes.replace("'", '"'))
                    if isinstance(parsed, dict):
                        report = parsed
                except json.JSONDecodeError:
                    report = {"notes": notes}
            entry = _hydrate_proposer_entry(
                proposer_dir,
                candidate_id,
                reason=str(data.get("reason") or ""),
                report=report,
            )
            by_id[candidate_id] = entry

    # Merge generation_summary.validation_reports with passed=false.
    summary_path = proposer_dir / "generation_summary.json"
    summary = _load_json(summary_path) if summary_path.is_file() else None
    if isinstance(summary, dict):
        for report in summary.get("validation_reports") or []:
            if not isinstance(report, dict):
                continue
            if report.get("passed") is True:
                continue
            candidate_id = str(report.get("candidate_id") or "").strip()
            if not candidate_id:
                continue
            if candidate_id in by_id:
                # Enrich report / failing output if missing.
                existing = by_id[candidate_id]
                if not existing.validation_report:
                    existing.validation_report = report
                if not existing.failing_test_output:
                    existing.failing_test_output = str(
                        report.get("failing_test_output") or ""
                    )
                continue
            by_id[candidate_id] = _hydrate_proposer_entry(
                proposer_dir,
                candidate_id,
                reason="",
                report=report,
            )

    # Fallback: scan attempt dirs for validation.json with passed=false.
    if not by_id:
        for path in proposer_dir.glob("attempt_*/proposer_candidates/**/validation.json"):
            report = _load_json(path)
            if not isinstance(report, dict) or report.get("passed") is True:
                continue
            candidate_id = str(
                report.get("candidate_id") or path.parent.name
            ).strip()
            if not candidate_id or candidate_id in by_id:
                continue
            by_id[candidate_id] = _hydrate_proposer_entry(
                proposer_dir, candidate_id, reason="", report=report
            )

    return list(by_id.values())


def choose_least_attempted_failure(
    failures: Sequence[Any],
    attempt_counts: Optional[Dict[str, int]] = None,
    rng: Optional[random.Random] = None,
) -> Optional[Any]:
    """Prefer never-tried entries, then least-tried, then seeded random."""
    if not failures:
        return None
    counts = attempt_counts or {}
    chooser = rng.choice if rng is not None else random.choice
    minimum = min(counts.get(getattr(failure, "id", ""), 0) for failure in failures)
    candidates = [
        failure
        for failure in failures
        if counts.get(getattr(failure, "id", ""), 0) == minimum
    ]
    return chooser(candidates)


def choose_proposer_failure(
    proposer_dir: Path,
    rng: Optional[random.Random] = None,
    attempt_counts: Optional[Dict[str, int]] = None,
) -> Optional[ProposerFailureEntry]:
    """Pick one proposer failure case (HGM choose_entry analog)."""
    failures = list_proposer_failures(proposer_dir)
    return choose_least_attempted_failure(failures, attempt_counts, rng)


def _solver_rollout_dir(scratch_solver_root: Path, task_id: str, level: int) -> Optional[Path]:
    """Find the newest rollout dir for a task at the given level."""
    level_dir = scratch_solver_root / "trajectories" / f"level_{level}" / task_id
    if not level_dir.is_dir():
        return None
    rollouts = sorted(
        [p for p in level_dir.iterdir() if p.is_dir() and p.name.startswith("rollout_")],
        key=lambda p: p.name,
    )
    if not rollouts:
        return None
    return rollouts[-1]


def _hydrate_solver_entry(
    task_id: str,
    level: int,
    scratch_solver_root: Path,
    task_store_root: Optional[Path],
) -> Optional[SolverFailureEntry]:
    rollout = _solver_rollout_dir(scratch_solver_root, task_id, level)
    if rollout is None:
        return None
    traj_path = rollout / "trajectory.jsonl"
    if not traj_path.is_file():
        # Some runners name it trajectory.md / other — still require some log.
        alt = list(rollout.glob("trajectory*"))
        traj_path = alt[0] if alt else None
        if traj_path is None or not traj_path.is_file():
            return None
    patch_path = rollout / "model_patch.diff"
    if not patch_path.is_file():
        patch_path = None
    eval_path = rollout / "trajectory_eval.json"
    if not eval_path.is_file():
        eval_path = None

    problem_statement = ""
    failing_test_output = ""
    if task_store_root is not None:
        task_dir = Path(task_store_root) / task_id
        problem_statement = _read_text(task_dir / "problem_statement.md")
        failing_test_output = _read_text(task_dir / "failing_test_output.txt")

    eval_log = failing_test_output
    if not eval_log and eval_path is not None:
        eval_log = _read_text(eval_path)

    return SolverFailureEntry(
        task_id=task_id,
        level=level,
        trajectory_path=traj_path,
        patch_path=patch_path,
        eval_path=eval_path,
        problem_statement=problem_statement,
        failing_test_output=failing_test_output,
        trajectory_text=_read_text(traj_path),
        predicted_patch=_read_text(patch_path),
        eval_log=eval_log,
    )


def list_solver_failures(
    *,
    scratch_solver_root: Path,
    failed_task_ids: Sequence[str],
    level: int,
    task_store_root: Optional[Path] = None,
) -> List[SolverFailureEntry]:
    """Hydrate solver failure entries that have trajectories on disk."""
    out: List[SolverFailureEntry] = []
    for task_id in failed_task_ids:
        entry = _hydrate_solver_entry(
            str(task_id),
            level,
            Path(scratch_solver_root),
            Path(task_store_root) if task_store_root else None,
        )
        if entry is not None:
            out.append(entry)
    return out


def choose_solver_failure(
    *,
    level2_result_path: Optional[Path],
    level1_result_path: Optional[Path],
    scratch_solver_root: Path,
    task_store_root: Optional[Path] = None,
    rng: Optional[random.Random] = None,
    attempt_counts: Optional[Dict[str, int]] = None,
) -> Optional[SolverFailureEntry]:
    """Prefer Level2 failed tasks; fall back to Level1 forgotten tasks."""
    scratch_solver_root = Path(scratch_solver_root)

    failed_ids: List[str] = []
    if level2_result_path is not None and Path(level2_result_path).is_file():
        data = _load_json(Path(level2_result_path))
        if isinstance(data, dict):
            failed_ids = [str(x) for x in (data.get("failed_task_ids") or [])]

    entries = list_solver_failures(
        scratch_solver_root=scratch_solver_root,
        failed_task_ids=failed_ids,
        level=2,
        task_store_root=task_store_root,
    )
    if entries:
        return choose_least_attempted_failure(entries, attempt_counts, rng)

    forgotten: List[str] = []
    if level1_result_path is not None and Path(level1_result_path).is_file():
        data = _load_json(Path(level1_result_path))
        if isinstance(data, dict):
            forgotten = [str(x) for x in (data.get("child_forgotten_task_ids") or [])]

    entries = list_solver_failures(
        scratch_solver_root=scratch_solver_root,
        failed_task_ids=forgotten,
        level=1,
        task_store_root=task_store_root,
    )
    if entries:
        return choose_least_attempted_failure(entries, attempt_counts, rng)
    return None
