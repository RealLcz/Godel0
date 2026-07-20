"""Unit tests for the patch guard."""

from __future__ import annotations

import pytest

from godel0.evolution.patch_guard import PatchGuard


class TestPatchGuard:
    def test_allowed_file(self):
        guard = PatchGuard()
        patch = """diff --git a/coding_agent.py b/coding_agent.py
index 1234567..abcdefg 100644
--- a/coding_agent.py
+++ b/coding_agent.py
@@ -1,3 +1,3 @@
-import old
+import new
"""
        report = guard.check(patch)
        assert report.passed
        assert "coding_agent.py" in report.allowed_files

    def test_forbidden_file(self):
        guard = PatchGuard()
        patch = """diff --git a/../../etc/passwd b/../../etc/passwd
--- a/../../etc/passwd
+++ b/../../etc/passwd
@@ -1,1 +1,1 @@
-root
+hacked
"""
        report = guard.check(patch)
        assert not report.passed

    def test_allowed_proposer(self):
        guard = PatchGuard()
        patch = """diff --git a/proposer/runner.py b/proposer/runner.py
--- a/proposer/runner.py
+++ b/proposer/runner.py
@@ -1,1 +1,1 @@
-old
+new
"""
        report = guard.check(patch)
        assert report.passed

    def test_allowed_tools(self):
        guard = PatchGuard()
        patch = """diff --git a/tools/new_tool.py b/tools/new_tool.py
--- a/tools/new_tool.py
+++ b/tools/new_tool.py
@@ -1,1 +1,1 @@
-old
+new
"""
        report = guard.check(patch)
        assert report.passed

    def test_empty_patch_rejected(self):
        guard = PatchGuard()
        report = guard.check("")
        assert not report.passed

    def test_git_directory_rejected(self):
        guard = PatchGuard()
        patch = """diff --git a/.git/config b/.git/config
--- a/.git/config
+++ b/.git/config
"""
        report = guard.check(patch)
        assert not report.passed
