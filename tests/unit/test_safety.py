"""Unit tests for safety checks."""

from __future__ import annotations

import pytest

from godel0.proposer_trusted.safety import check_safety
from godel0.proposer_trusted.duplicate_detector import DuplicateDetector
from godel0.proposer_trusted.statement_auditor import audit_statement


class TestSafetyCheck:
    def test_safe_patch(self):
        patch = """diff --git a/module.py b/module.py
--- a/module.py
+++ b/module.py
@@ -1,3 +1,3 @@
-def foo():
+def foo(x):
     return x
"""
        is_safe, reasons = check_safety(patch)
        assert is_safe
        assert len(reasons) == 0

    def test_test_file_modification(self):
        patch = """diff --git a/test_module.py b/test_module.py
--- a/test_module.py
+++ b/test_module.py
@@ -1,3 +1,3 @@
-def test_foo():
+def test_foo(x):
     pass
"""
        is_safe, reasons = check_safety(patch)
        assert not is_safe
        assert any("test_file" in r for r in reasons)

    def test_unsafe_pattern(self):
        patch = """diff --git a/module.py b/module.py
+import os; os.system("rm -rf /")
"""
        is_safe, reasons = check_safety(patch)
        assert not is_safe

    def test_dependency_modification(self):
        patch = """diff --git a/requirements.txt b/requirements.txt
--- a/requirements.txt
+++ b/requirements.txt
+malicious-package
"""
        is_safe, reasons = check_safety(patch)
        assert not is_safe


class TestDuplicateDetector:
    def test_unique_patch(self):
        detector = DuplicateDetector()
        assert detector.check("patch1", "repo1", "file.py", "func", "change_operator")

    def test_duplicate_patch(self):
        detector = DuplicateDetector()
        detector.check("patch1", "repo1", "file.py", "func", "change_operator")
        assert not detector.check("patch1", "repo1", "file.py", "func", "change_operator")

    def test_reset(self):
        detector = DuplicateDetector()
        detector.check("patch1", "repo1", "file.py", "func", "op")
        detector.reset()
        assert detector.check("patch1", "repo1", "file.py", "func", "op")


class TestStatementAuditor:
    def test_clean_statement(self):
        statement = "The function does not handle edge cases properly."
        patch = "diff --git a/module.py b/module.py\n+pass"
        is_valid, issues = audit_statement(statement, patch, ["test_one"])
        assert is_valid

    def test_leaks_answer(self):
        statement = "The bug is in the clamp function. You need to change the comparison."
        patch = "diff --git a/module.py b/module.py\n+pass"
        is_valid, issues = audit_statement(statement, patch, [])
        assert not is_valid
        assert any("leakage" in i for i in issues)

    def test_leaks_file_path(self):
        statement = "Check src/deep/module/file.py for the issue."
        patch = "diff --git a/src/deep/module/file.py b/src/deep/module/file.py\n+pass"
        is_valid, issues = audit_statement(statement, patch, [])
        assert not is_valid
