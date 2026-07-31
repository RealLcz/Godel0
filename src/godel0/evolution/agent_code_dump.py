"""Role-scoped agent code dump for HGM-style diagnosis (get_current_code analog)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

PROPOSER_ALWAYS_FILES: Tuple[str, ...] = (
    "proposer/proposer_main.py",
    "proposer/schemas.py",
)

SOLVER_ALWAYS_FILES: Tuple[str, ...] = (
    "coding_agent.py",
    "llm_withtools.py",
)

PROPOSER_CANDIDATE_PREFIXES: Tuple[str, ...] = (
    "proposer/",
    "swesmith/",
)

SOLVER_CANDIDATE_PREFIXES: Tuple[str, ...] = (
    "coding_agent.py",
    "llm.py",
    "llm_withtools.py",
    "tools/",
    "prompts/",
    "utils/",
)

# Legacy aliases used by older callers / tests.
PROPOSER_INCLUDE_PREFIXES = PROPOSER_CANDIDATE_PREFIXES
SOLVER_INCLUDE_PREFIXES = SOLVER_CANDIDATE_PREFIXES

EXCLUDE_NAME_SUBSTRINGS: Tuple[str, ...] = (
    "__pycache__",
    ".pyc",
    "self_improvement_prompt",
    "self_improve_step",
    ".git/",
    "proposer/request.py",
)


def _should_exclude(rel: str) -> bool:
    lower = rel.replace("\\", "/").lower()
    return any(token in lower for token in EXCLUDE_NAME_SUBSTRINGS)


def _matches_prefixes(rel: str, prefixes: Sequence[str]) -> bool:
    normalized = rel.replace("\\", "/")
    for prefix in prefixes:
        if prefix.endswith("/"):
            if normalized.startswith(prefix) or normalized == prefix.rstrip("/"):
                return True
        elif normalized == prefix:
            return True
    return False


def _score_file(rel: str, evidence_hints: Sequence[str]) -> int:
    """Higher score = more relevant to the failure evidence."""
    score = 0
    normalized = rel.replace("\\", "/")
    for hint in evidence_hints:
        hint_norm = hint.replace("\\", "/").strip()
        if not hint_norm:
            continue
        if normalized == hint_norm or normalized.endswith("/" + hint_norm):
            score += 100
        elif hint_norm in normalized or normalized in hint_norm:
            score += 40
        else:
            # Match basename mentions.
            base = Path(normalized).name
            if base and base in hint_norm:
                score += 20
    # Prefer live entrypoints and planners slightly.
    if normalized.endswith(("proposer_main.py", "coding_agent.py", "schemas.py")):
        score += 5
    if "/workflows/" in normalized or normalized.startswith("tools/"):
        score += 2
    return score


def list_code_files(
    repo_root: Path,
    role: str,
    *,
    max_files: int = 12,
    evidence_hints: Optional[Sequence[str]] = None,
) -> List[Path]:
    """List a focused set of files for a diagnose role."""
    repo_root = Path(repo_root)
    if role == "proposer":
        always = PROPOSER_ALWAYS_FILES
        prefixes = PROPOSER_CANDIDATE_PREFIXES
    elif role == "solver":
        always = SOLVER_ALWAYS_FILES
        prefixes = SOLVER_CANDIDATE_PREFIXES
    else:
        raise ValueError(f"Unknown role: {role}")

    hints = list(evidence_hints or [])
    selected: List[Path] = []
    seen = set()

    def _add(rel: str) -> None:
        if rel in seen:
            return
        path = repo_root / rel
        if path.is_file() and not _should_exclude(rel):
            selected.append(path)
            seen.add(rel)

    for rel in always:
        _add(rel)

    candidates: List[Tuple[int, str, Path]] = []
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(repo_root).as_posix()
        if rel in seen or _should_exclude(rel):
            continue
        if not _matches_prefixes(rel, prefixes):
            continue
        if path.suffix.lower() not in {
            ".py",
            ".md",
            ".txt",
            ".yaml",
            ".yml",
            ".jinja",
            ".j2",
            "",
        }:
            if path.suffix:
                continue
        candidates.append((_score_file(rel, hints), rel, path))

    # Prefer evidence-linked files, then stable path order.
    candidates.sort(key=lambda item: (-item[0], item[1]))
    for _, rel, path in candidates:
        if len(selected) >= max_files:
            break
        selected.append(path)
        seen.add(rel)

    return selected


def dump_agent_code(
    repo_root: Path,
    role: str,
    *,
    max_chars: int = 80_000,
    max_files: int = 12,
    extra_files: Iterable[Path] = (),
    evidence_hints: Optional[Sequence[str]] = None,
) -> str:
    """Concatenate role-scoped agent sources with path headers."""
    repo_root = Path(repo_root).resolve()
    parts: List[str] = []
    used = 0
    seen = set()

    def _append(path: Path) -> None:
        nonlocal used
        try:
            rel = path.resolve().relative_to(repo_root).as_posix()
        except ValueError:
            rel = path.name
        if rel in seen:
            return
        seen.add(rel)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        chunk = f"\n===== FILE: {rel} =====\n{text}"
        if used + len(chunk) > max_chars:
            remain = max_chars - used
            if remain <= 0:
                return
            chunk = chunk[:remain] + "\n... [code dump truncated]\n"
            parts.append(chunk)
            used = max_chars
            return
        parts.append(chunk)
        used += len(chunk)

    for path in list_code_files(
        repo_root, role, max_files=max_files, evidence_hints=evidence_hints
    ):
        if used >= max_chars:
            break
        _append(path)

    for path in extra_files:
        if used >= max_chars:
            break
        p = Path(path)
        if p.is_file():
            _append(p)

    return "".join(parts).strip()


PROPOSER_CODE_SUMMARY = """# Coding / Proposer Agent Summary

- **Proposer entry**: `proposer/` (planner, runner, workflows)
- **RepoChain / SWE-smith helpers**: `swesmith/`
- Self-edit should improve task *generation* quality (valid F2P tasks, robust
  mutation planning, better grounding in existing tests) without hard-coding
  repository- or task-specific constants, and without bypassing trusted validation.
"""

SOLVER_CODE_SUMMARY = """# Coding Agent Summary

- **Main File**: `coding_agent.py`
  - Primary Class: `AgenticSystem`
  - The `forward()` function is the central entry point.
  - Prompts are located either within `forward()` or in the `prompts/` directory.
- **Tools**: `tools/`
  - Each tool exposes `tool_info()` / `tool_function()`.
- Prefer focused changes to the live workflow over introducing a new subsystem.
"""


def role_code_summary(role: str) -> str:
    if role == "proposer":
        return PROPOSER_CODE_SUMMARY
    if role == "solver":
        return SOLVER_CODE_SUMMARY
    raise ValueError(f"Unknown role: {role}")
