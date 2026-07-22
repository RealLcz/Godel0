"""Patch utilities for diff manipulation."""

from __future__ import annotations

import re
from typing import List


def normalize_patch(patch: str) -> str:
    """Normalize a patch for deduplication."""
    lines = patch.splitlines()
    normalized = []
    for line in lines:
        line = re.sub(r"@@ -\d+,\d+ \+\d+,\d+ @@", "@@ ... @@", line)
        line = re.sub(r"index [0-9a-f]{7,}\.\.[0-9a-f]{7,}.*", "index ...", line)
        normalized.append(line)
    return "\n".join(normalized)


def patch_hash(patch: str) -> str:
    """Compute a hash of the normalized patch."""
    import hashlib
    normalized = normalize_patch(patch)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def extract_changed_files(patch: str) -> List[str]:
    """Extract list of changed file paths from a patch."""
    files: list[str] = []

    def add(path: str) -> None:
        if not path or path == "/dev/null":
            return
        if path.startswith("a/") or path.startswith("b/"):
            path = path[2:]
        if path not in files:
            files.append(path)

    for line in patch.splitlines():
        if line.startswith("diff --git"):
            match = re.match(r"diff --git a/(.*) b/(.*)", line)
            if match:
                add(match.group(2))
        elif line.startswith("+++ "):
            path = line[4:].strip().split("\t", 1)[0]
            add(path)
        elif line.startswith("--- ") and not files:
            path = line[4:].strip().split("\t", 1)[0]
            add(path)
    return files


def count_patch_lines(patch: str) -> tuple[int, int]:
    """Count added and deleted lines in a patch."""
    added = 0
    deleted = 0
    for line in patch.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            deleted += 1
    return added, deleted


def is_source_only(patch: str, test_patterns: list[str] | None = None) -> bool:
    """Check if a patch only modifies source files (not tests)."""
    if test_patterns is None:
        test_patterns = ["test_", "_test.py", "/tests/", "/test/", "conftest.py"]
    files = extract_changed_files(patch)
    if not files:
        return False
    for f in files:
        for pattern in test_patterns:
            if pattern in f:
                return False
    return True


def filter_patch_by_files(patch: str, target_files: list[str]) -> str:
    """Filter a patch to only include changes to target files."""
    lines = patch.splitlines()
    filtered = []
    include = False
    for line in lines:
        if line.startswith("diff --git"):
            include = any(f"a/{t}" in line and f"b/{t}" in line for t in target_files)
        if include:
            filtered.append(line)
    return "\n".join(filtered)


def split_patch_by_file(patch: str) -> dict[str, str]:
    """P0-7: split a unified diff into per-file patch fragments.

    Returns ``{relative_path: file_patch}``. Used by trusted causal ablation
    to restore one file at a time while keeping the remaining bug applied.
    """
    files = extract_changed_files(patch)
    return {path: filter_patch_by_files(patch, [path]) for path in files}
