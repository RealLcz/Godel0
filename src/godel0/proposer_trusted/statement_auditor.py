"""Statement auditor: checks problem statements for leakage."""

from __future__ import annotations

import re
from typing import List, Optional

from ..git.patch import extract_changed_files


LEAKAGE_PATTERNS = [
    r"def\s+\w+\s*\(",
    r"class\s+\w+\s*[\(:]",
    r"import\s+\w+",
    r"from\s+\w+\s+import",
    r"return\s+",
    r"if\s+.*:",
    r"for\s+.*:",
    r"while\s+.*:",
]

ANSWER_LEAKAGE_INDICATORS = [
    "the bug is",
    "the fix is",
    "the answer is",
    "you need to change",
    "modify the line",
    "replace",
    "the correct code",
    "inserted a bug",
    "intentionally introduced",
    "oracle patch",
    "reverse patch",
    "hidden test",
]


def audit_statement(
    problem_statement: str,
    bug_patch: str,
    f2p_tests: List[str],
) -> tuple[bool, List[str]]:
    """Audit a problem statement for answer leakage.

    Returns (is_valid, issues).
    """
    issues: List[str] = []

    changed_files = extract_changed_files(bug_patch)
    for f in changed_files:
        if f in problem_statement:
            issues.append(f"leaks_file_path: {f}")

    for pattern in LEAKAGE_PATTERNS:
        matches = re.findall(pattern, problem_statement)
        if len(matches) > 3:
            issues.append(f"leaks_code_pattern: {pattern}")

    statement_lower = problem_statement.lower()
    for indicator in ANSWER_LEAKAGE_INDICATORS:
        if indicator in statement_lower:
            issues.append(f"answer_leakage: '{indicator}'")
            break

    for test in f2p_tests:
        if test in problem_statement:
            issues.append(f"leaks_test_name: {test}")
            break

    is_valid = len(issues) == 0
    return is_valid, issues
