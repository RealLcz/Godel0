"""P0-6: Trusted Causal Ablation necessity + isolation activity."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from godel0.proposer_trusted.candidate_validator import CandidateValidator
from godel0.schemas.evaluation import CandidateValidationReport


MULTI_FILE_PATCH = """\
diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1 +1 @@
-clean_a
+bug_a
diff --git a/b.py b/b.py
--- a/b.py
+++ b/b.py
@@ -1 +1 @@
-clean_b
+bug_b
diff --git a/c.py b/c.py
--- a/c.py
+++ b/c.py
@@ -1 +1 @@
-clean_c
+filler_c
"""


class TestTrustedCausalAblationIsolation:
    def test_filler_edit_not_counted_as_independently_active(self, tmp_path):
        """A,B real bugs + C filler: leave-one-out still fails for all,
        but only A and B trigger in isolation. Old code wrongly counted C.
        """
        validator = CandidateValidator(
            workspace_root=tmp_path,
            require_causal_ablation=True,
            min_independently_active=2,
        )
        report = CandidateValidationReport(candidate_id="c1", passed=True)

        # Track which workspace path / mode we are in via the target name.
        def fake_prepare_full(repo_copy, repo_path, base_commit, setup_patch, candidate_patch):
            repo_copy.mkdir(parents=True, exist_ok=True)
            (repo_copy / "mode.txt").write_text("full", encoding="utf-8")

        def fake_prepare_iso(repo_copy, repo_path, base_commit, setup_patch, file_patch):
            repo_copy.mkdir(parents=True, exist_ok=True)
            # Encode which file was applied from the patch header.
            if "a.py" in file_patch and "bug_a" in file_patch:
                tag = "iso_a"
            elif "b.py" in file_patch and "bug_b" in file_patch:
                tag = "iso_b"
            else:
                tag = "iso_c"
            (repo_copy / "mode.txt").write_text(tag, encoding="utf-8")

        def fake_reverse(repo_copy, file_patch):
            # After reverse, leave-one-out workspace still has other bugs.
            (repo_copy / "mode.txt").write_text("leave_one_out", encoding="utf-8")
            return True

        def fake_run_tests(repo_copy, test_command):
            mode = (repo_copy / "mode.txt").read_text(encoding="utf-8").strip()
            # Leave-one-out: always still failing (A+B real bugs remain).
            if mode == "leave_one_out":
                return {"returncode": 1, "timed_out": False, "stdout": "", "stderr": ""}
            # Isolation: A or B alone fails; C filler alone passes.
            if mode in {"iso_a", "iso_b"}:
                return {"returncode": 1, "timed_out": False, "stdout": "", "stderr": ""}
            if mode == "iso_c":
                return {"returncode": 0, "timed_out": False, "stdout": "", "stderr": ""}
            return {"returncode": 1, "timed_out": False, "stdout": "", "stderr": ""}

        with (
            patch.object(validator, "_prepare_ablation_workspace", side_effect=fake_prepare_full),
            patch.object(validator, "_prepare_isolation_workspace", side_effect=fake_prepare_iso),
            patch(
                "godel0.proposer_trusted.candidate_validator.reverse_patch",
                side_effect=fake_reverse,
            ),
            patch.object(validator, "_run_tests", side_effect=fake_run_tests),
        ):
            ok = validator._run_trusted_causal_ablation(
                candidate_patch=MULTI_FILE_PATCH,
                repo_path=tmp_path,
                base_commit="HEAD",
                test_command="true",
                setup_patch="",
                f2p_tests=["t1"],
                report=report,
                validation_mode="exit_code",
                command_test_id="",
                control_test_command=None,
            )

        assert ok is True
        assert report.trusted_causal_ablation_pass is True
        # Necessity: no single-file repair restores.
        assert report.repair_one_file_results == {
            "a.py": False,
            "b.py": False,
            "c.py": False,
        }
        # Isolation: only true bugs count.
        assert report.isolated_file_triggers == {
            "a.py": True,
            "b.py": True,
            "c.py": False,
        }
        assert report.independently_active_file_count == 2

    def test_rejects_when_only_one_file_triggers_isolation(self, tmp_path):
        validator = CandidateValidator(
            workspace_root=tmp_path,
            require_causal_ablation=True,
            min_independently_active=2,
        )
        report = CandidateValidationReport(candidate_id="c1", passed=True)

        def fake_prepare_full(repo_copy, repo_path, base_commit, setup_patch, candidate_patch):
            repo_copy.mkdir(parents=True, exist_ok=True)
            (repo_copy / "mode.txt").write_text("full", encoding="utf-8")

        def fake_prepare_iso(repo_copy, repo_path, base_commit, setup_patch, file_patch):
            repo_copy.mkdir(parents=True, exist_ok=True)
            tag = "iso_a" if "bug_a" in file_patch else "iso_other"
            (repo_copy / "mode.txt").write_text(tag, encoding="utf-8")

        def fake_reverse(repo_copy, file_patch):
            (repo_copy / "mode.txt").write_text("leave_one_out", encoding="utf-8")
            return True

        def fake_run_tests(repo_copy, test_command):
            mode = (repo_copy / "mode.txt").read_text(encoding="utf-8").strip()
            if mode == "leave_one_out":
                # Repairing A restores (single-file solvable) — also fail necessity.
                # Actually for this test we want necessity pass but isolation fail:
                # leave-one-out still fails for all.
                return {"returncode": 1, "timed_out": False}
            if mode == "iso_a":
                return {"returncode": 1, "timed_out": False}
            return {"returncode": 0, "timed_out": False}

        with (
            patch.object(validator, "_prepare_ablation_workspace", side_effect=fake_prepare_full),
            patch.object(validator, "_prepare_isolation_workspace", side_effect=fake_prepare_iso),
            patch(
                "godel0.proposer_trusted.candidate_validator.reverse_patch",
                side_effect=fake_reverse,
            ),
            patch.object(validator, "_run_tests", side_effect=fake_run_tests),
        ):
            ok = validator._run_trusted_causal_ablation(
                candidate_patch=MULTI_FILE_PATCH,
                repo_path=tmp_path,
                base_commit="HEAD",
                test_command="true",
                setup_patch="",
                f2p_tests=["t1"],
                report=report,
                validation_mode="exit_code",
                command_test_id="",
                control_test_command=None,
            )

        assert ok is False
        assert report.independently_active_file_count == 1
        assert report.isolated_file_triggers["a.py"] is True
        assert report.isolated_file_triggers["b.py"] is False
        assert report.isolated_file_triggers["c.py"] is False
        assert any(
            r.startswith("independently_active_file_count=")
            for r in report.rejection_reasons
        )

    def test_rejects_when_single_file_repair_restores(self, tmp_path):
        validator = CandidateValidator(
            workspace_root=tmp_path,
            require_causal_ablation=True,
            min_independently_active=2,
        )
        report = CandidateValidationReport(candidate_id="c1", passed=True)
        call_n = {"n": 0}

        def fake_prepare_full(repo_copy, repo_path, base_commit, setup_patch, candidate_patch):
            repo_copy.mkdir(parents=True, exist_ok=True)
            (repo_copy / "mode.txt").write_text("full", encoding="utf-8")

        def fake_prepare_iso(repo_copy, repo_path, base_commit, setup_patch, file_patch):
            repo_copy.mkdir(parents=True, exist_ok=True)
            (repo_copy / "mode.txt").write_text("iso", encoding="utf-8")

        def fake_reverse(repo_copy, file_patch):
            call_n["n"] += 1
            # First leave-one-out (a.py) restores — necessity fails.
            tag = "restored" if call_n["n"] == 1 else "leave_one_out"
            (repo_copy / "mode.txt").write_text(tag, encoding="utf-8")
            return True

        def fake_run_tests(repo_copy, test_command):
            mode = (repo_copy / "mode.txt").read_text(encoding="utf-8").strip()
            if mode == "restored":
                return {"returncode": 0, "timed_out": False}
            if mode == "iso":
                return {"returncode": 1, "timed_out": False}
            return {"returncode": 1, "timed_out": False}

        with (
            patch.object(validator, "_prepare_ablation_workspace", side_effect=fake_prepare_full),
            patch.object(validator, "_prepare_isolation_workspace", side_effect=fake_prepare_iso),
            patch(
                "godel0.proposer_trusted.candidate_validator.reverse_patch",
                side_effect=fake_reverse,
            ),
            patch.object(validator, "_run_tests", side_effect=fake_run_tests),
        ):
            ok = validator._run_trusted_causal_ablation(
                candidate_patch=MULTI_FILE_PATCH,
                repo_path=tmp_path,
                base_commit="HEAD",
                test_command="true",
                setup_patch="",
                f2p_tests=["t1"],
                report=report,
                validation_mode="exit_code",
                command_test_id="",
                control_test_command=None,
            )

        assert ok is False
        assert report.repair_one_file_results["a.py"] is True
        assert "single_file_repair_restored_contract" in report.rejection_reasons
