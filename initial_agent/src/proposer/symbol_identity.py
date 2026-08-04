from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def canonical_symbol_id(file_path: str, qualified_name: str, kind: str) -> str:
    """Return a stable repository-local identity for one Python symbol."""
    identity = f"{file_path.replace(chr(92), '/')}::{kind}::{qualified_name}"
    return "sym_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def extract_canonical_symbols(
    source: str,
    *,
    file_path: str,
    include_source: bool = False,
) -> List[Dict[str, Any]]:
    try:
        tree = ast.parse(source, filename=file_path)
    except (SyntaxError, ValueError):
        return []
    lines = source.splitlines()
    entries: List[Dict[str, Any]] = []

    def visit(body: Sequence[ast.stmt], parents: List[str]) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                kind = "class"
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "function"
            else:
                continue
            qualified_name = ".".join(parents + [node.name])
            line_start = int(getattr(node, "lineno", 0) or 0)
            line_end = int(getattr(node, "end_lineno", line_start) or line_start)
            entry: Dict[str, Any] = {
                "symbol_id": canonical_symbol_id(file_path, qualified_name, kind),
                "file_path": file_path,
                "qualified_name": qualified_name,
                "symbol_name": node.name,
                "symbol_type": kind,
                "line_start": line_start,
                "line_end": line_end,
            }
            if include_source:
                entry["source"] = "\n".join(
                    lines[max(0, line_start - 1) : min(len(lines), line_end)]
                )
            entries.append(entry)
            visit(getattr(node, "body", []), parents + [node.name])

    visit(tree.body, [])
    return entries


def build_canonical_catalog(
    root: Path,
    files: Iterable[str],
    *,
    include_source: bool = False,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for relative in dict.fromkeys(str(path).replace("\\", "/") for path in files):
        path = root / relative
        if not path.is_file() or path.suffix != ".py":
            continue
        entries.extend(
            extract_canonical_symbols(
                path.read_text(encoding="utf-8", errors="replace"),
                file_path=relative,
                include_source=include_source,
            )
        )
    return entries


__all__ = [
    "build_canonical_catalog",
    "canonical_symbol_id",
    "extract_canonical_symbols",
]
