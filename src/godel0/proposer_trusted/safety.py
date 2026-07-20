"""Safety checks for candidate patches."""

from __future__ import annotations

import re
from typing import List

from ..git.patch import extract_changed_files, count_patch_lines


UNSAFE_PATTERNS = [
    r"import\s+os\s*;\s*os\.system",
    r"subprocess\.(call|run|Popen)\s*\(",
    r"eval\s*\(",
    r"exec\s*\(",
    r"__import__\s*\(",
    r"os\.remove\s*\(",
    r"shutil\.rmtree\s*\(",
    r"open\s*\([^)]*['\"]w['\"]",
    r"requests\.(get|post|put|delete)\s*\(",
    r"urllib\.request",
    r"socket\.",
    r"http\.client",
]

TEST_FILE_PATTERNS = [
    "test_",
    "_test.py",
    "/tests/",
    "/test/",
    "conftest.py",
    "spec_",
]

DEPENDENCY_PATTERNS = [
    "requirements.txt",
    "setup.py",
    "pyproject.toml",
    "Pipfile",
    "poetry.lock",
]


def check_safety(patch: str, max_patch_lines: int = 80) -> tuple[bool, list[str]]:
    """Check if a patch is safe.

    Returns (is_safe, rejection_reasons).
    """
    reasons: List[str] = []

    added, deleted = count_patch_lines(patch)
    if added + deleted > max_patch_lines * 2:
        reasons.append("patch_too_large")

    changed = extract_changed_files(patch)

    for f in changed:
        for pattern in TEST_FILE_PATTERNS:
            if pattern in f:
                reasons.append(f"modifies_test_file: {f}")
                break

    for f in changed:
        for pattern in DEPENDENCY_PATTERNS:
            if f.endswith(pattern) or f == pattern:
                reasons.append(f"modifies_dependency: {f}")
                break

    for pattern in UNSAFE_PATTERNS:
        if re.search(pattern, patch):
            reasons.append(f"unsafe_pattern: {pattern}")

    if not changed:
        reasons.append("empty_patch")

    is_safe = len(reasons) == 0
    return is_safe, reasons
