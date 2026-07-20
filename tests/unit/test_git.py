"""Unit tests for git operations."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from godel0.git.repository import (
    init_repo, add_all, commit, get_head_sha, diff_vs_commit,
    apply_patch, reset_to_commit, list_changed_files,
)
from godel0.git.patch import (
    extract_changed_files, count_patch_lines, is_source_only,
    normalize_patch, patch_hash,
)
from godel0.git.node_refs import create_node_ref, get_node_sha, node_exists


class TestGitOperations:
    def test_init_and_commit(self, tmp_path):
        repo = tmp_path / "test_repo"
        init_repo(repo)
        (repo / "file.py").write_text("print('hello')")
        sha = commit(repo, "initial commit")
        assert len(sha) == 40

    def test_diff_vs_commit(self, tmp_path):
        repo = tmp_path / "test_repo"
        init_repo(repo)
        (repo / "file.py").write_text("print('hello')")
        sha = commit(repo, "initial commit")

        (repo / "file.py").write_text("print('world')")
        diff = diff_vs_commit(repo, sha)
        assert "print('world')" in diff
        assert "print('hello')" in diff

    def test_diff_vs_commit_includes_relative_untracked_and_ignores_caches(self, tmp_path):
        repo = tmp_path / "test_repo"
        init_repo(repo)
        (repo / "file.py").write_text("print('hello')")
        sha = commit(repo, "initial commit")

        (repo / "tools").mkdir()
        (repo / "tools" / "new_tool.py").write_text("VALUE = 1\n")
        (repo / "tools" / "__pycache__").mkdir()
        (repo / "tools" / "__pycache__" / "new_tool.cpython-310.pyc").write_bytes(b"cache")

        diff = diff_vs_commit(repo, sha)

        assert "diff --git a/tools/new_tool.py b/tools/new_tool.py" in diff
        assert str(repo) not in diff
        assert "__pycache__" not in diff

    def test_apply_and_reverse_patch(self, tmp_path):
        repo = tmp_path / "test_repo"
        init_repo(repo)
        (repo / "file.py").write_text("line1\nline2\nline3\n")
        sha = commit(repo, "initial")

        patch = """diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 line1
-line2
+line2_modified
 line3
"""
        assert apply_patch(repo, patch)
        content = (repo / "file.py").read_text()
        assert "line2_modified" in content

    def test_reset_to_commit(self, tmp_path):
        repo = tmp_path / "test_repo"
        init_repo(repo)
        (repo / "file.py").write_text("original")
        sha1 = commit(repo, "first")

        (repo / "file.py").write_text("modified")
        commit(repo, "second")

        reset_to_commit(repo, sha1)
        assert (repo / "file.py").read_text() == "original"

    def test_list_changed_files(self, tmp_path):
        repo = tmp_path / "test_repo"
        init_repo(repo)
        (repo / "a.py").write_text("a")
        (repo / "b.py").write_text("b")
        sha = commit(repo, "initial")

        (repo / "a.py").write_text("modified")
        (repo / "c.py").write_text("new")
        files = list_changed_files(repo, sha)
        assert "a.py" in files


class TestPatchUtils:
    def test_extract_changed_files(self):
        patch = """diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -1,1 +1,1 @@
-old
+new
diff --git a/bar.py b/bar.py
--- a/bar.py
+++ b/bar.py
@@ -1,1 +1,1 @@
-old
+new
"""
        files = extract_changed_files(patch)
        assert "foo.py" in files
        assert "bar.py" in files

    def test_count_lines(self):
        patch = """diff --git a/foo.py b/foo.py
+added line 1
+added line 2
-removed line
"""
        added, deleted = count_patch_lines(patch)
        assert added == 2
        assert deleted == 1

    def test_is_source_only(self):
        patch = """diff --git a/module.py b/module.py
+new code
"""
        assert is_source_only(patch)

    def test_not_source_only(self):
        patch = """diff --git a/test_module.py b/test_module.py
+new test
"""
        assert not is_source_only(patch)

    def test_patch_hash_deterministic(self):
        patch = "diff --git a/f.py b/f.py\n+new"
        h1 = patch_hash(patch)
        h2 = patch_hash(patch)
        assert h1 == h2


class TestNodeRefs:
    def test_create_and_get_ref(self, tmp_path):
        repo = tmp_path / "test_repo"
        init_repo(repo)
        (repo / "file.py").write_text("content")
        sha = commit(repo, "initial")

        create_node_ref(repo, "test_node", sha)
        assert node_exists(repo, "test_node")
        assert get_node_sha(repo, "test_node") == sha

    def test_nonexistent_ref(self, tmp_path):
        repo = tmp_path / "test_repo"
        init_repo(repo)
        assert not node_exists(repo, "nonexistent")
        assert get_node_sha(repo, "nonexistent") is None
