"""Role-scoped agent code dump for HGM-style diagnosis (get_current_code analog)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

PROPOSER_INCLUDE_PREFIXES: Tuple[str, ...] = (
    "proposer/",
    "swesmith/",
    "utils/",
)

SOLVER_INCLUDE_PREFIXES: Tuple[str, ...] = (
    "coding_agent.py",
    "llm.py",
    "llm_withtools.py",
    "tools/",
    "prompts/",
    "utils/",
)

EXCLUDE_NAME_SUBSTRINGS: Tuple[str, ...] = (
    "__pycache__",
    ".pyc",
    "self_improvement_prompt",
    "self_improve_step",
    ".git/",
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


def list_code_files(repo_root: Path, role: str) -> List[Path]:
    """List files to include for a diagnose role."""
    repo_root = Path(repo_root)
    if role == "proposer":
        prefixes = PROPOSER_INCLUDE_PREFIXES
    elif role == "solver":
        prefixes = SOLVER_INCLUDE_PREFIXES
    else:
        raise ValueError(f"Unknown role: {role}")

    files: List[Path] = []
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(repo_root).as_posix()
        if _should_exclude(rel):
            continue
        if not _matches_prefixes(rel, prefixes):
            continue
        # Prefer source / text files.
        if path.suffix.lower() not in {".py", ".md", ".txt", ".yaml", ".yml", ".jinja", ".j2", ""}:
            if path.suffix:
                continue
        files.append(path)
    return files


def dump_agent_code(
    repo_root: Path,
    role: str,
    *,
    max_chars: int = 200_000,
    extra_files: Iterable[Path] = (),
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

    for path in list_code_files(repo_root, role):
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
- **Shared utilities**: `utils/`
- Self-edit should improve task *generation* quality (valid F2P tasks, robust
  mutation planning, better grounding in existing tests) without hard-coding
  repository- or task-specific constants.
"""

SOLVER_CODE_SUMMARY = """# Coding Agent Summary

- **Main File**: `coding_agent.py`
  - Primary Class: `AgenticSystem`
  - The `forward()` function is the central entry point.
  - Prompts are located either within `forward()` or in the `prompts/` directory.
- **Tools**: `tools/`
  - Each tool exposes `tool_info()` / `tool_function()`.
- **Utilities**: `utils/`
- Prefer general mechanisms wired into the live tool path over task-specific hardcoding.
"""


def role_code_summary(role: str) -> str:
    if role == "proposer":
        return PROPOSER_CODE_SUMMARY
    if role == "solver":
        return SOLVER_CODE_SUMMARY
    raise ValueError(f"Unknown role: {role}")
