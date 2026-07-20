"""Regression tests for trusted candidate validation."""

from __future__ import annotations

import subprocess
from pathlib import Path

from godel0.proposer_trusted.candidate_validator import CandidateValidator
from godel0.proposer_trusted.duplicate_detector import DuplicateDetector
from godel0.git.repository import get_head_sha, reset_to_commit


def _buggy_clamp_patch(repo_path: Path) -> str:
    source_file = repo_path / "toy_module.py"
    original = source_file.read_text()
    source_file.write_text(original.replace("min(x, high)", "min(x, low)"))
    result = subprocess.run(
        ["git", "-C", str(repo_path), "diff"],
        capture_output=True,
        text=True,
        check=False,
    )
    source_file.write_text(original)
    reset_to_commit(repo_path, get_head_sha(repo_path))
    return result.stdout


def test_validator_forces_verbose_pytest_for_f2p_parsing(toy_repo, tmp_path):
    patch = _buggy_clamp_patch(toy_repo["path"])
    validator = CandidateValidator(tmp_path / "validator", test_timeout_sec=30)

    report = validator.validate(
        candidate_patch=patch,
        repo_path=toy_repo["path"],
        base_commit=toy_repo["commit"],
        test_command="python -m pytest test_toy.py -q",
        candidate_id="quiet_pytest",
    )

    assert report.passed
    assert report.f2p_tests
    assert len(report.f2p_tests) == len(set(report.f2p_tests))


def test_validator_accepts_patch_relevant_to_declared_symbol(toy_repo, tmp_path):
    patch = _buggy_clamp_patch(toy_repo["path"])
    validator = CandidateValidator(tmp_path / "validator", test_timeout_sec=30)

    report = validator.validate(
        candidate_patch=patch,
        repo_path=toy_repo["path"],
        base_commit=toy_repo["commit"],
        test_command="python -m pytest test_toy.py -q",
        candidate_id="relevant_symbol",
        target_file="toy_module.py",
        target_symbol="clamp",
    )

    assert report.passed
    assert report.relevance_valid


def test_validator_rejects_patch_irrelevant_to_declared_symbol(toy_repo, tmp_path):
    patch = _buggy_clamp_patch(toy_repo["path"])
    validator = CandidateValidator(tmp_path / "validator", test_timeout_sec=30)

    report = validator.validate(
        candidate_patch=patch,
        repo_path=toy_repo["path"],
        base_commit=toy_repo["commit"],
        test_command="python -m pytest test_toy.py -q",
        candidate_id="wrong_symbol",
        target_file="toy_module.py",
        target_symbol="divide",
    )

    assert not report.passed
    assert not report.relevance_valid
    assert any("irrelevant_target_symbol" in r for r in report.rejection_reasons)


def test_validator_rejects_patch_irrelevant_to_declared_file(toy_repo, tmp_path):
    patch = _buggy_clamp_patch(toy_repo["path"])
    validator = CandidateValidator(tmp_path / "validator", test_timeout_sec=30)

    report = validator.validate(
        candidate_patch=patch,
        repo_path=toy_repo["path"],
        base_commit=toy_repo["commit"],
        test_command="python -m pytest test_toy.py -q",
        candidate_id="wrong_file",
        target_file="other_module.py",
    )

    assert not report.passed
    assert not report.relevance_valid
    assert any("irrelevant_target_file" in r for r in report.rejection_reasons)


def test_validator_reports_unusable_clean_test_command(toy_repo, tmp_path, monkeypatch):
    patch = _buggy_clamp_patch(toy_repo["path"])
    validator = CandidateValidator(tmp_path / "validator", test_timeout_sec=30)
    monkeypatch.setattr(
        validator,
        "_run_tests",
        lambda *_args: {
            "returncode": 2,
            "stdout": "collected 0 items / 1 error",
            "stderr": "collection error",
            "timed_out": False,
        },
    )

    report = validator.validate(
        candidate_patch=patch,
        repo_path=toy_repo["path"],
        base_commit=toy_repo["commit"],
        test_command="python -m pytest test_toy.py -q",
        candidate_id="clean_collection_error",
    )

    assert not report.passed
    assert "clean_tests_unusable" in report.rejection_reasons
    assert "no_f2p" not in report.rejection_reasons


def test_validator_reports_clean_test_timeout(toy_repo, tmp_path, monkeypatch):
    patch = _buggy_clamp_patch(toy_repo["path"])
    validator = CandidateValidator(tmp_path / "validator", test_timeout_sec=30)
    monkeypatch.setattr(
        validator,
        "_run_tests",
        lambda *_args: {
            "returncode": -1,
            "stdout": "",
            "stderr": "Timeout",
            "timed_out": True,
        },
    )

    report = validator.validate(
        candidate_patch=patch,
        repo_path=toy_repo["path"],
        base_commit=toy_repo["commit"],
        test_command="python -m pytest test_toy.py -q",
        candidate_id="clean_timeout",
    )

    assert not report.passed
    assert "clean_test_timeout" in report.rejection_reasons
    assert not report.timeout_valid


def test_validator_supports_exit_code_regression_commands(toy_repo, tmp_path):
    patch = _buggy_clamp_patch(toy_repo["path"])
    validator = CandidateValidator(tmp_path / "validator", test_timeout_sec=30)

    report = validator.validate(
        candidate_patch=patch,
        repo_path=toy_repo["path"],
        base_commit=toy_repo["commit"],
        test_command=(
            "python -c \"import toy_module; "
            "assert toy_module.clamp(10, 0, 5) == 5\""
        ),
        candidate_id="exit_code_candidate",
        validation_mode="exit_code",
        command_test_id="command::clamp_regression",
        control_test_command=(
            "python -c \"import toy_module; "
            "assert toy_module.divide(4, 2) == 2\""
        ),
    )

    assert report.passed
    assert report.f2p_tests == ["command::clamp_regression"]
    assert report.p2p_tests == [
        "command::control::command::clamp_regression"
    ]
    assert report.reverse_restored


def test_validator_rejects_exit_code_command_without_regression(toy_repo, tmp_path):
    patch = _buggy_clamp_patch(toy_repo["path"])
    validator = CandidateValidator(tmp_path / "validator", test_timeout_sec=30)

    report = validator.validate(
        candidate_patch=patch,
        repo_path=toy_repo["path"],
        base_commit=toy_repo["commit"],
        test_command="python -c \"import toy_module\"",
        candidate_id="exit_code_no_regression",
        validation_mode="exit_code",
        command_test_id="command::no_regression",
        control_test_command="python -c \"import toy_module\"",
    )

    assert not report.passed
    assert not report.f2p_tests
    assert "no_f2p" in report.rejection_reasons


def test_duplicate_detector_rejects_reused_signature_with_different_patch():
    detector = DuplicateDetector()

    assert detector.check("patch1", "repo", "module.py", "func", "change_operator")
    assert not detector.check("patch2", "repo", "module.py", "func", "change_operator")


def test_failed_validation_does_not_poison_duplicate_detector(toy_repo, tmp_path):
    patch = _buggy_clamp_patch(toy_repo["path"])
    validator = CandidateValidator(tmp_path / "validator", test_timeout_sec=30)

    rejected = validator.validate(
        candidate_patch=patch,
        repo_path=toy_repo["path"],
        base_commit=toy_repo["commit"],
        test_command="python -m pytest missing_test.py -q",
        candidate_id="invalid_first_attempt",
        repo_id="toy",
        target_file="toy_module.py",
        target_symbol="clamp",
        operator="change_operator",
    )
    accepted = validator.validate(
        candidate_patch=patch,
        repo_path=toy_repo["path"],
        base_commit=toy_repo["commit"],
        test_command="python -m pytest test_toy.py -q",
        candidate_id="valid_retry",
        repo_id="toy",
        target_file="toy_module.py",
        target_symbol="clamp",
        operator="change_operator",
    )

    assert not rejected.passed
    assert accepted.passed


def test_multifile_duplicate_detection_uses_components_not_anchor_signature():
    detector = DuplicateDetector()
    first = """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1 +1 @@
-old_a
+new_a
diff --git a/b.py b/b.py
--- a/b.py
+++ b/b.py
@@ -1 +1 @@
-old_b
+new_b
"""
    different_sites = """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -5 +5 @@
-other_a
+changed_a
diff --git a/b.py b/b.py
--- a/b.py
+++ b/b.py
@@ -8 +8 @@
-other_b
+changed_b
"""
    reused_component = """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1 +1 @@
-old_a
+new_a
diff --git a/c.py b/c.py
--- a/c.py
+++ b/c.py
@@ -1 +1 @@
-old_c
+new_c
"""

    assert detector.check(first, "repo", "a.py", "", "repo_compose")
    assert detector.check(
        different_sites,
        "repo",
        "a.py",
        "",
        "repo_compose",
    )
    assert not detector.check(
        reused_component,
        "repo",
        "a.py",
        "",
        "repo_compose",
    )


def test_pytest_parser_preserves_parameter_spaces_and_normalizes_workspace(tmp_path):
    repo = tmp_path / "validate_candidate" / "repo"
    repo.mkdir(parents=True)
    validator = CandidateValidator(tmp_path / "validator")
    passed_node = (
        "tests/test_config.py::test_value[quoted value-str-"
        f"{repo}/fixtures/config.yml-yaml]"
    )
    failed_node = "tests/test_config.py::test_other[param with space]"
    result = {
        "stdout": (
            f"{passed_node} PASSED [ 50%]\n"
            f"FAILED {failed_node} - AssertionError\n"
        ),
        "stderr": "",
    }

    assert validator._parse_passed_tests(result, repo) == [
        passed_node.replace(str(repo), "<REPO>")
    ]
    assert validator._parse_failed_tests(result, repo) == [failed_node]
