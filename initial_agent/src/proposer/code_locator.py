from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .schemas import CodeTarget, FailureSignature


@dataclass
class RepoIndex:
    """A lightweight index over a checked-out repository.

    The index scans Python files for top-level and nested functions/classes
    so the locator can match failure signatures against concrete symbols.

    ``source_dirs`` controls which subdirectories to scan. For a typical
    repo this is ``["."]`` (scan everything). For Ansible it is
    ``["lib", "test/lib"]`` so only importable packages are indexed.
    """

    repo_id: str
    base_commit: str = ""
    repo_dir: str = ""
    source_dirs: List[str] = field(default_factory=lambda: ["."])
    symbols: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def build(
        cls,
        repo_id: str,
        repo_dir: str,
        base_commit: str = "",
        source_dirs: Optional[List[str]] = None,
    ) -> "RepoIndex":
        if source_dirs is None:
            source_dirs = ["."]

        symbols: List[Dict[str, Any]] = []
        for source_dir in source_dirs:
            scan_dir = os.path.join(repo_dir, source_dir) if source_dir != "." else repo_dir
            if not os.path.isdir(scan_dir):
                continue
            for root, _dirs, files in os.walk(scan_dir):
                rel_root = os.path.relpath(root, repo_dir)
                if rel_root == ".":
                    parts: List[str] = []
                else:
                    parts = rel_root.split(os.sep)
                # Skip hidden dirs, __pycache__, .git
                if any(p.startswith(".") or p == "__pycache__" for p in parts):
                    continue
                for fname in files:
                    if not fname.endswith(".py"):
                        continue
                    fpath = os.path.join(root, fname)
                    rel = os.path.relpath(fpath, repo_dir)
                    for sym in _extract_symbols(fpath):
                        sym["file_path"] = rel
                        symbols.append(sym)
        return cls(
            repo_id=repo_id,
            base_commit=base_commit,
            repo_dir=repo_dir,
            source_dirs=source_dirs,
            symbols=symbols,
        )


def _extract_symbols(fpath: str) -> List[Dict[str, Any]]:
    try:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
        tree = ast.parse(source, filename=fpath)
    except (SyntaxError, ValueError):
        return []
    out: List[Dict[str, Any]] = []
    lines = source.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "function"
        elif isinstance(node, ast.ClassDef):
            kind = "class"
        else:
            continue
        line_start = getattr(node, "lineno", 0)
        line_end = getattr(node, "end_lineno", line_start)
        snippet = ""
        if line_start and line_end and lines:
            lo = max(0, line_start - 1)
            hi = min(len(lines), line_end)
            snippet = "\n".join(lines[lo:hi])
        out.append(
            {
                "symbol_name": node.name,
                "symbol_type": kind,
                "line_start": line_start,
                "line_end": line_end,
                "source": snippet,
            }
        )
    return out


@dataclass
class RepoSpec:
    """Specification of a repository checkout available to the engine.

    ``repo_dir`` is the path to the checked-out repository.
    ``base_commit`` is the git commit SHA.
    """

    repo_id: str
    repo_dir: str
    base_commit: str = ""
    test_command: str = ""
    install_command: str = ""
    timeout_sec: int = 120

    @property
    def repo_path(self) -> str:
        """Compatibility alias used by SWE-smith engines."""
        return self.repo_dir

    @classmethod
    def from_index(cls, index: RepoIndex) -> "RepoSpec":
        return cls(
            repo_id=index.repo_id,
            repo_dir=index.repo_dir,
            base_commit=index.base_commit,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RepoSpec":
        known = {fld for fld in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known}
        if "repo_path" in data and "repo_dir" not in filtered:
            filtered["repo_dir"] = data["repo_path"]
        return cls(**filtered)


class CodeLocator:
    """Locates candidate code targets for a FailureSignature.

    Candidate selection considers: code pattern matching, repository-native
    test coverage, novelty relative to source tasks, entity size, operator
    support, and historical candidate usage.
    """

    def locate(
        self,
        signature: FailureSignature,
        repo_index: RepoIndex,
        max_results: int = 5,
        used_symbols: Optional[List[str]] = None,
    ) -> List[CodeTarget]:
        used = set(used_symbols or [])
        scored: List[tuple] = []
        patterns: List[tuple[str, str]] = []
        for raw_pattern in signature.code_patterns:
            raw = str(raw_pattern).lower()
            prefix, separator, value = raw.partition(":")
            if separator and prefix in {"function", "class"}:
                patterns.append((prefix, value))
            else:
                patterns.append(("", raw))
        for sym in repo_index.symbols:
            file_path = str(sym.get("file_path", ""))
            if _is_test_path(file_path):
                continue
            name = str(sym.get("symbol_name", ""))
            source = str(sym.get("source", ""))
            score = 0.0
            symbol_type = str(sym.get("symbol_type", "")).lower()
            for expected_type, pat in patterns:
                if expected_type and expected_type != symbol_type:
                    continue
                if pat and pat in source.lower():
                    score += 1.0
                if pat and pat in name.lower():
                    score += 1.5
                if pat and pat == name.lower():
                    score += 2.0
                # Path tokens matter for bootstrap capability priors
                # (e.g. "inventory" → lib/ansible/inventory/...).
                if pat and pat in file_path.lower():
                    score += 1.2
            if signature.target_capability and signature.target_capability in source.lower():
                score += 0.5
            # Also score capability tokens against the file path.
            if signature.target_capability:
                for tok in str(signature.target_capability).lower().replace("-", "_").split("_"):
                    if len(tok) >= 4 and tok in file_path.lower():
                        score += 0.4
                        break
            line_span = int(sym.get("line_end", 0)) - int(sym.get("line_start", 0))
            if 0 < line_span <= 60:
                score += 0.5
            elif line_span > 200:
                score -= 0.5
            if name in used:
                score -= 1.0
            novelty = max(0.0, 1.0 - (1.0 if name in used else 0.0))
            target = CodeTarget(
                repo_id=repo_index.repo_id,
                file_path=file_path,
                symbol_name=name,
                symbol_type=str(sym.get("symbol_type", "function")),
                line_start=int(sym.get("line_start", 0)),
                line_end=int(sym.get("line_end", 0)),
                source=source,
                has_test_coverage=self._has_test_coverage(repo_index, str(sym.get("file_path", ""))),
                novelty_score=novelty,
            )
            scored.append((score, target))
        scored.sort(key=lambda t: t[0], reverse=True)
        # Prefer positive-scoring targets; fall back to the top of the
        # ranking so bootstrap still gets a real file when patterns are weak.
        positive = [t for score, t in scored if score > 0 and t.symbol_name]
        if positive:
            return positive[:max_results]
        return [t for _score, t in scored[:max_results] if t.symbol_name]

    def _has_test_coverage(self, index: RepoIndex, file_path: str) -> bool:
        if not file_path:
            return False
        base = os.path.splitext(os.path.basename(file_path))[0]
        for sym in index.symbols:
            fp = str(sym.get("file_path", ""))
            if "test" in fp.lower() and base in os.path.basename(fp):
                return True
        return False


def _is_test_path(file_path: str) -> bool:
    normalized = file_path.replace("\\", "/").lower()
    parts = normalized.split("/")
    name = parts[-1] if parts else ""
    return (
        "test" in parts
        or "tests" in parts
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


__all__ = [
    "CodeLocator",
    "RepoIndex",
    "RepoSpec",
]
