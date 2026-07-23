"""Retrieve nearby existing repository tests for RepoChain contracts.

Primary path (require_generated_contracts=False):
  production semantic chain
    → retrieve nearby existing tests
    → keep tests that pass on the clean repository
    → those tests become the FAIL_TO_PASS oracle
    → Proposer plans a capability-conditioned multi-file mutation

Optional enhancement (require_generated_contracts=True):
  same retrieval is used only as API / fixture grounding while the Proposer
  still emits a *new* generated contract validated via clean_contract.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple


_DEFAULT_TEST_ROOTS = (
    "test/units",
    "tests/units",
    "test",
    "tests",
)

_SKIP_DIR_PARTS = {
    "__pycache__",
    ".git",
    "inventory_test_data",
    "fixtures",
    "data",
}


def production_to_unit_subdir(production_path: str) -> str:
    """Map ``lib/ansible/inventory/host.py`` → ``inventory`` (best-effort)."""
    relative = str(production_path or "").replace("\\", "/").lstrip("/")
    parts = [p for p in relative.split("/") if p and p != "."]
    if not parts:
        return ""
    # Strip common source roots and trailing filename.
    if parts[0] in {"lib", "src", "packages"}:
        parts = parts[1:]
    if parts and parts[0] in {"ansible", "godel0"}:
        parts = parts[1:]
    if parts and parts[-1].endswith(".py"):
        parts = parts[:-1]
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
    return parts[0] if parts else ""


def _path_tokens(path: str) -> Set[str]:
    return {
        token.lower()
        for token in re.split(r"[^A-Za-z0-9_]+", str(path or ""))
        if len(token) >= 4
        and token.lower()
        not in {
            "test",
            "tests",
            "units",
            "unit",
            "python",
            "init",
            "lib",
            "main",
            "ansible",
        }
    }


def _is_usable_test_file(relative: str) -> bool:
    relative = relative.replace("\\", "/")
    if not relative.endswith(".py"):
        return False
    name = Path(relative).name
    if name in {"__init__.py", "conftest.py"}:
        return False
    if not (name.startswith("test_") or name.endswith("_test.py")):
        return False
    parts = set(relative.split("/"))
    if parts & _SKIP_DIR_PARTS:
        return False
    return True


def candidate_test_dirs(
    root: Path,
    production_files: Sequence[str],
    *,
    test_roots: Optional[Sequence[str]] = None,
) -> List[Path]:
    """Prefer ``test/units/<subsystem>/`` mirrors of production files."""
    roots = [str(r).rstrip("/") for r in (test_roots or _DEFAULT_TEST_ROOTS)]
    dirs: List[Path] = []
    seen: Set[str] = set()

    def _add(path: Path) -> None:
        key = path.as_posix()
        if key in seen or not path.is_dir():
            return
        seen.add(key)
        dirs.append(path)

    for production in production_files:
        sub = production_to_unit_subdir(production)
        if not sub:
            continue
        for test_root in roots:
            _add(root / test_root / sub)
            # Also one-level deeper mirrors: inventory/manager → inventory
            # already covered; keep parent of nested packages.
    for test_root in roots:
        _add(root / test_root)
    return dirs


def score_existing_test(
    test_relative: str,
    *,
    production_files: Sequence[str],
    anchor_tokens: Set[str],
) -> int:
    score = 0
    lower = test_relative.lower()
    for production in production_files:
        sub = production_to_unit_subdir(production)
        if sub and f"/{sub.lower()}/" in f"/{lower}/":
            score += 5
        for token in _path_tokens(production):
            if token in lower:
                score += 2
    for token in anchor_tokens:
        if token in lower:
            score += 1
    # Prefer shallow, focused unit tests over huge integration trees.
    depth = test_relative.count("/")
    score -= max(0, depth - 4)
    return score


def retrieve_nearby_existing_tests(
    root: Path,
    production_files: Sequence[str],
    *,
    budget: int = 4,
    test_roots: Optional[Sequence[str]] = None,
) -> List[str]:
    """Return relative paths of existing tests near the production chain."""
    if budget <= 0:
        return []
    root = Path(root)
    production = [
        str(p).replace("\\", "/").lstrip("/")
        for p in production_files
        if str(p or "").strip()
    ]
    if not production:
        return []

    anchor_tokens: Set[str] = set()
    for path in production:
        anchor_tokens |= _path_tokens(path)

    scored: List[Tuple[int, str]] = []
    seen: Set[str] = set()
    for directory in candidate_test_dirs(root, production, test_roots=test_roots):
        try:
            iterator: Iterable[Path] = directory.rglob("*.py")
        except OSError:
            continue
        for path in iterator:
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:
                continue
            if relative in seen or not _is_usable_test_file(relative):
                continue
            seen.add(relative)
            score = score_existing_test(
                relative,
                production_files=production,
                anchor_tokens=anchor_tokens,
            )
            if score > 0:
                scored.append((-score, relative))

    scored.sort()
    return [relative for _, relative in scored[:budget]]


def _strip_license_header(source: str) -> str:
    lines = source.splitlines()
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        if not stripped:
            idx += 1
            continue
        if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
            # Keep going through comment / module docstring preamble.
            if stripped.startswith('"""') or stripped.startswith("'''"):
                quote = stripped[:3]
                if stripped.count(quote) >= 2 and len(stripped) > 3:
                    idx += 1
                    continue
                idx += 1
                while idx < len(lines) and quote not in lines[idx]:
                    idx += 1
                if idx < len(lines):
                    idx += 1
                continue
            idx += 1
            continue
        break
    return "\n".join(lines[idx:])


def excerpt_existing_test(
    source: str,
    *,
    max_chars: int = 3500,
) -> str:
    """Keep imports, fixtures/setUp, and a few short test bodies."""
    text = _strip_license_header(source or "")
    if not text.strip():
        return ""
    lines = text.splitlines()
    keep: List[str] = []
    i = 0
    test_blocks = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if (
            stripped.startswith("import ")
            or stripped.startswith("from ")
            or stripped.startswith("@pytest")
            or stripped.startswith("@unittest")
            or stripped.startswith("class ")
        ):
            keep.append(line)
            i += 1
            continue
        if re.match(r"^\s*def (setUp|tearDown|setup_|teardown_|test_)", line):
            block = [line]
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if nxt.strip() and not nxt.startswith((" ", "\t")) and not nxt.startswith("#"):
                    break
                if re.match(r"^\s*def ", nxt) and not nxt.startswith(line[: len(line) - len(line.lstrip())]):
                    # nested def inside method — still part of block if indented more
                    if len(nxt) - len(nxt.lstrip()) <= len(line) - len(line.lstrip()):
                        break
                block.append(nxt)
                i += 1
                if sum(len(x) + 1 for x in block) > 1200:
                    block.append("        # ... clipped ...")
                    break
            keep.extend(block)
            if re.match(r"^\s*def test_", line):
                test_blocks += 1
                if test_blocks >= 2:
                    break
            continue
        if stripped.startswith("pytest.fixture") or "fixture" in stripped and stripped.startswith("@"):
            keep.append(line)
            i += 1
            continue
        i += 1

    excerpt = "\n".join(keep).strip() or text[:max_chars]
    if len(excerpt) > max_chars:
        half = max_chars // 2
        excerpt = excerpt[:half] + "\n# ... clipped ...\n" + excerpt[-half:]
    return excerpt


def build_existing_test_grounding(
    root: Path,
    production_files: Sequence[str],
    *,
    budget: int = 4,
    max_chars_per_file: int = 3500,
    test_roots: Optional[Sequence[str]] = None,
) -> Tuple[str, List[str]]:
    """Build prompt grounding text + list of retrieved test paths."""
    tests = retrieve_nearby_existing_tests(
        root,
        production_files,
        budget=budget,
        test_roots=test_roots,
    )
    if not tests:
        return (
            "(No nearby existing tests were found for this production chain. "
            "Infer public APIs carefully from the production source bundle only.)",
            [],
        )

    chunks: List[str] = [
        "Use the following EXISTING repository tests only as grounding for "
        "real API usage, fixtures, constructors, and expected public behavior. "
        "Do NOT copy them as the final contract. Emit a newly named generated "
        "test file that exercises the planned semantic chain.",
    ]
    for relative in tests:
        path = Path(root) / relative
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        excerpt = excerpt_existing_test(source, max_chars=max_chars_per_file)
        if not excerpt.strip():
            continue
        chunks.append(
            f"\n## EXISTING TEST (grounding only): {relative}\n"
            f"```python\n{excerpt}\n```"
        )
    if len(chunks) == 1:
        return (
            "(Nearby test paths were found but could not be read as grounding.)",
            tests,
        )
    return "\n".join(chunks), tests


__all__ = [
    "build_existing_test_grounding",
    "candidate_test_dirs",
    "excerpt_existing_test",
    "production_to_unit_subdir",
    "retrieve_nearby_existing_tests",
]
