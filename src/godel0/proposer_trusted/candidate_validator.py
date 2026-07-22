"""Trusted Candidate Validator: validates generated bug candidates."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
import ast
from pathlib import Path
from typing import List, Optional

from ..errors import CandidateValidationError
from ..schemas.evaluation import CandidateValidationReport
from ..git.repository import run_git, reset_to_commit, apply_patch, reverse_patch
from ..git.patch import (
    extract_changed_files,
    count_patch_lines,
    is_source_only,
    filter_patch_by_files,
    split_patch_by_file,
)
from .duplicate_detector import DuplicateDetector
from .safety import check_safety


class CandidateValidator:
    """Validates bug candidates in a clean, isolated workspace.

    The validation flow:
    1. Create clean workspace.
    2. Checkout pinned base_commit.
    3. Run baseline test command, record T_pass.
    4. Check candidate patch only modifies allowed source files.
    5. Apply candidate patch.
    6. Check syntax, import, and install.
    7. Run same test command.
    8. Compute F2P = clean_pass ∩ bugged_fail.
    9. Require |F2P| >= 1.
    10. Reverse candidate patch.
    11. Re-run F2P, confirm restored to pass.
    12. Run relevance, duplicate, and safety checks.
    """

    def __init__(
        self,
        workspace_root: Path,
        test_timeout_sec: int = 120,
        max_patch_lines: int = 80,
        forbid_test_file_edits: bool = True,
        duplicate_detector: Optional[DuplicateDetector] = None,
        execution_backend=None,
        require_causal_ablation: bool = False,
        min_independently_active: int = 2,
    ):
        self.workspace_root = Path(workspace_root)
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.test_timeout_sec = test_timeout_sec
        self.max_patch_lines = max_patch_lines
        self.forbid_test_file_edits = forbid_test_file_edits
        self.duplicate_detector = duplicate_detector or DuplicateDetector()
        # BUG-15/10.8: trusted repository tests (clean, bugged, reverse, F2P,
        # P2P, causal ablation) run through the same ExecutionBackend so the
        # whole chain is end-to-end Apptainer. When None, fall back to direct
        # subprocess (backward compatible).
        self.execution_backend = execution_backend
        # P0-7: authoritative trusted causal ablation gate. Default False so
        # unit tests / ablations that inject a bare validator keep working;
        # EvolutionOrchestrator.from_config enables it from RepoChain config.
        self.require_causal_ablation = require_causal_ablation
        self.min_independently_active = max(1, int(min_independently_active))

    def validate(
        self,
        candidate_patch: str,
        repo_path: Path,
        base_commit: str,
        test_command: str,
        candidate_id: str = "",
        install_command: Optional[str] = None,
        repo_id: str = "",
        target_file: str = "",
        target_symbol: str = "",
        operator: str = "",
        validation_mode: str = "pytest",
        command_test_id: str = "",
        control_test_command: Optional[str] = None,
        setup_patch: str = "",
    ) -> CandidateValidationReport:
        """Validate a single bug candidate."""
        report = CandidateValidationReport(
            candidate_id=candidate_id or "unknown",
            passed=False,
        )

        if validation_mode not in {"pytest", "exit_code"}:
            report.rejection_reasons.append(
                f"unsupported_validation_mode: {validation_mode}"
            )
            return report
        if validation_mode == "exit_code" and not control_test_command:
            report.rejection_reasons.append("exit_code_requires_control_test")
            return report

        if not candidate_patch.strip():
            report.rejection_reasons.append("empty_patch")
            return report

        if setup_patch.strip():
            setup_files = extract_changed_files(setup_patch)
            if not setup_files or any(
                not self._is_test_path(path) for path in setup_files
            ):
                report.rejection_reasons.append("setup_patch_must_only_modify_tests")
                return report

        added, deleted = count_patch_lines(candidate_patch)
        if added + deleted > self.max_patch_lines * 2:
            report.rejection_reasons.append("patch_scope_too_large")
            return report

        changed_files = extract_changed_files(candidate_patch)
        report.source_only = is_source_only(candidate_patch)
        if self.forbid_test_file_edits and not report.source_only:
            report.rejection_reasons.append("modifies_test_files")
            return report

        is_safe, safety_reasons = check_safety(candidate_patch, max_patch_lines=self.max_patch_lines)
        if safety_reasons and not self.forbid_test_file_edits:
            safety_reasons = [r for r in safety_reasons if "test_file" not in r]
            is_safe = not safety_reasons
        report.safety_valid = is_safe
        if not is_safe:
            report.rejection_reasons.extend(safety_reasons)
            return report

        duplicate_target = target_file or (changed_files[0] if len(changed_files) == 1 else "")
        duplicate_args = {
            "patch": candidate_patch,
            "repo_id": repo_id,
            "target_file": duplicate_target,
            "target_symbol": target_symbol,
            "operator": operator,
        }
        report.duplicate_valid = self.duplicate_detector.is_unique(
            **duplicate_args,
        )
        if not report.duplicate_valid:
            report.rejection_reasons.append("duplicate")
            return report

        workspace = self.workspace_root / f"validate_{candidate_id}"
        workspace.mkdir(parents=True, exist_ok=True)

        try:
            report = self._run_validation(
                candidate_patch=candidate_patch,
                repo_path=repo_path,
                base_commit=base_commit,
                test_command=self._ensure_verbose_pytest(test_command),
                workspace=workspace,
                install_command=install_command,
                target_file=target_file,
                target_symbol=target_symbol,
                report=report,
                validation_mode=validation_mode,
                command_test_id=command_test_id,
                control_test_command=control_test_command,
                setup_patch=setup_patch,
            )
        except Exception as e:
            report.rejection_reasons.append(f"validation_error: {str(e)[:200]}")
        finally:
            import shutil
            shutil.rmtree(workspace, ignore_errors=True)

        report.passed = (
            report.patch_applied
            and report.syntax_valid
            and len(report.f2p_tests) >= 1
            and len(report.p2p_tests) >= 1
            and report.reverse_restored
            and report.safety_valid
            and report.duplicate_valid
            and report.relevance_valid
        )

        # P0-7: Trusted Causal Ablation (authoritative). Local RepoChain
        # metadata is advisory only; this re-executes per-file repairs.
        if report.passed and self.require_causal_ablation:
            ablation_ok = self._run_trusted_causal_ablation(
                candidate_patch=candidate_patch,
                repo_path=repo_path,
                base_commit=base_commit,
                test_command=self._ensure_verbose_pytest(test_command),
                setup_patch=setup_patch,
                f2p_tests=list(report.f2p_tests),
                report=report,
                validation_mode=validation_mode,
                command_test_id=command_test_id,
                control_test_command=control_test_command,
            )
            if not ablation_ok:
                report.passed = False
                if "trusted_causal_ablation_failed" not in report.rejection_reasons:
                    report.rejection_reasons.append("trusted_causal_ablation_failed")

        if report.passed:
            report.duplicate_valid = self.duplicate_detector.record(
                **duplicate_args,
            )
            if not report.duplicate_valid:
                report.passed = False
                report.rejection_reasons.append("duplicate")

        if not report.passed and not report.rejection_reasons:
            if not report.f2p_tests:
                report.rejection_reasons.append("no_f2p")
            if not report.reverse_restored:
                report.rejection_reasons.append("reverse_not_restored")
            if not report.syntax_valid:
                report.rejection_reasons.append("syntax_error")

        return report

    def _run_trusted_causal_ablation(
        self,
        *,
        candidate_patch: str,
        repo_path: Path,
        base_commit: str,
        test_command: str,
        setup_patch: str,
        f2p_tests: List[str],
        report: CandidateValidationReport,
        validation_mode: str,
        command_test_id: str,
        control_test_command: Optional[str],
    ) -> bool:
        """P0-7: re-execute causal ablation under Trusted Validator authority.

        For each modified source file F:
          full bug applied → restore only F → re-run F2P tests.
        If restoring F alone makes ALL F2P tests pass again, the task is
        single-file solvable via F (``repair_one_file_results[F]=True``).

        Gate:
          - at least 2 source files
          - no single-file repair restores all F2P
          - independently_active_file_count >= min_independently_active
            (files whose solo repair still leaves at least one F2P failing)
        """
        source_files = [
            f for f in extract_changed_files(candidate_patch)
            if not self._is_test_path(f)
        ]
        if len(source_files) < 2:
            report.trusted_causal_ablation_pass = False
            report.independently_active_file_count = len(source_files)
            report.repair_one_file_results = {f: True for f in source_files}
            report.rejection_reasons.append("not_multi_file")
            return False

        per_file = split_patch_by_file(candidate_patch)
        repair_results: dict = {}
        independently_active = 0

        with tempfile.TemporaryDirectory(
            prefix="causal_ablation_", dir=str(self.workspace_root)
        ) as tmp:
            workspace = Path(tmp)
            for file_path in source_files:
                file_patch = per_file.get(file_path) or filter_patch_by_files(
                    candidate_patch, [file_path]
                )
                if not file_patch.strip():
                    repair_results[file_path] = False
                    continue
                # Fresh workspace: apply full bug, then reverse only this file.
                repo_copy = workspace / f"ablate_{file_path.replace('/', '_')}"
                try:
                    self._prepare_ablation_workspace(
                        repo_copy, repo_path, base_commit, setup_patch, candidate_patch
                    )
                except Exception as exc:
                    report.rejection_reasons.append(
                        f"causal_ablation_setup_error:{file_path}:{str(exc)[:80]}"
                    )
                    repair_results[file_path] = False
                    continue

                # Restore only this file (reverse its hunk).
                reverse_ok = reverse_patch(repo_copy, file_patch)
                if not reverse_ok:
                    # Fallback: reset and re-apply full patch minus this file.
                    try:
                        reset_to_commit(repo_copy, base_commit)
                        if setup_patch.strip():
                            apply_patch(repo_copy, setup_patch)
                        others = filter_patch_by_files(
                            candidate_patch,
                            [f for f in source_files if f != file_path],
                        )
                        if others.strip():
                            apply_patch(repo_copy, others)
                        reverse_ok = True
                    except Exception:
                        reverse_ok = False
                if not reverse_ok:
                    repair_results[file_path] = False
                    continue

                result = self._run_tests(repo_copy, test_command)
                if result.get("timed_out"):
                    repair_results[file_path] = False
                    continue
                if validation_mode == "exit_code":
                    # Restored if primary test now passes.
                    restored = result.get("returncode") == 0
                else:
                    passed = self._parse_passed_tests(result, repo_copy)
                    restored = all(t in passed for t in f2p_tests) if f2p_tests else False

                # True = single-file repair restored contracts (BAD for RepoChain).
                repair_results[file_path] = bool(restored)
                if not restored:
                    independently_active += 1

        report.repair_one_file_results = repair_results
        report.independently_active_file_count = independently_active
        all_single_fail = not any(repair_results.values()) if repair_results else False
        passed = (
            all_single_fail
            and independently_active >= self.min_independently_active
        )
        report.trusted_causal_ablation_pass = bool(passed)
        if not passed:
            if any(repair_results.values()):
                report.rejection_reasons.append("single_file_repair_restored_contract")
            if independently_active < self.min_independently_active:
                report.rejection_reasons.append(
                    f"independently_active_file_count={independently_active}"
                    f"<{self.min_independently_active}"
                )
        return bool(passed)

    def _prepare_ablation_workspace(
        self,
        repo_copy: Path,
        repo_path: Path,
        base_commit: str,
        setup_patch: str,
        candidate_patch: str,
    ) -> None:
        """Clone/reset workspace and apply setup + full bug patch."""
        import shutil

        if repo_copy.exists():
            shutil.rmtree(repo_copy, ignore_errors=True)
        clone_result = subprocess.run(
            ["git", "clone", "--shared", "--quiet", str(repo_path), str(repo_copy)],
            capture_output=True,
            text=True,
            check=False,
        )
        if clone_result.returncode != 0:
            shutil.copytree(repo_path, repo_copy, dirs_exist_ok=False)
        if not (repo_copy / ".git").exists():
            subprocess.run(["git", "init"], cwd=str(repo_copy), capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "validator@godel0.ai"],
                cwd=str(repo_copy),
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Godel0 Validator"],
                cwd=str(repo_copy),
                capture_output=True,
            )
            subprocess.run(["git", "add", "-A"], cwd=str(repo_copy), capture_output=True)
            subprocess.run(
                ["git", "commit", "--allow-empty", "-m", "validator base"],
                cwd=str(repo_copy),
                capture_output=True,
            )
        reset_to_commit(repo_copy, base_commit)
        if setup_patch.strip():
            apply_patch(repo_copy, setup_patch)
        if not apply_patch(repo_copy, candidate_patch):
            raise CandidateValidationError("failed to apply full bug for ablation")

    def _run_validation(
        self,
        candidate_patch: str,
        repo_path: Path,
        base_commit: str,
        test_command: str,
        workspace: Path,
        install_command: Optional[str],
        target_file: str,
        target_symbol: str,
        report: CandidateValidationReport,
        validation_mode: str,
        command_test_id: str,
        control_test_command: Optional[str],
        setup_patch: str,
    ) -> CandidateValidationReport:
        """Run the full validation flow in a workspace."""
        import shutil
        repo_copy = workspace / "repo"
        # A shared local clone avoids copying the source repository's Git object
        # store for every candidate while preserving an isolated worktree.
        clone_result = subprocess.run(
            ["git", "clone", "--shared", "--quiet", str(repo_path), str(repo_copy)],
            capture_output=True,
            text=True,
            check=False,
        )
        if clone_result.returncode != 0:
            shutil.rmtree(repo_copy, ignore_errors=True)
            shutil.copytree(repo_path, repo_copy, dirs_exist_ok=False)

        # Initialize git if .git was not copied (e.g., from a non-git source)
        if not (repo_copy / ".git").exists():
            subprocess.run(["git", "init"], cwd=str(repo_copy), capture_output=True)
            subprocess.run(["git", "config", "user.email", "validator@godel0.ai"], cwd=str(repo_copy), capture_output=True)
            subprocess.run(["git", "config", "user.name", "Godel0 Validator"], cwd=str(repo_copy), capture_output=True)
            subprocess.run(["git", "add", "-A"], cwd=str(repo_copy), capture_output=True)
            subprocess.run(["git", "commit", "--allow-empty", "-m", "validator base"], cwd=str(repo_copy), capture_output=True)

        # Reset to base commit to ensure clean state
        try:
            reset_to_commit(repo_copy, base_commit)
        except Exception:
            # If base_commit doesn't exist in this repo (e.g., shallow clone),
            # just use the current state
            pass

        if setup_patch.strip() and not apply_patch(repo_copy, setup_patch):
            report.rejection_reasons.append("setup_patch_apply_failed")
            return report

        report.relevance_valid, relevance_reasons = self._check_relevance(
            repo_copy,
            candidate_patch,
            target_file=target_file,
            target_symbol=target_symbol,
        )
        if not report.relevance_valid:
            report.rejection_reasons.extend(relevance_reasons)
            return report

        clean_result = self._run_tests(repo_copy, test_command)
        if clean_result.get("timed_out"):
            report.rejection_reasons.append("clean_test_timeout")
            return report
        if validation_mode == "exit_code":
            if clean_result.get("returncode") != 0:
                report.rejection_reasons.append("clean_test_command_failed")
                return report
            primary_test_id = command_test_id or "command::primary"
            control_test_id = f"command::control::{primary_test_id}"
            clean_control = self._run_tests(repo_copy, control_test_command or "")
            if clean_control.get("timed_out"):
                report.rejection_reasons.append("clean_control_test_timeout")
                return report
            if clean_control.get("returncode") != 0:
                report.rejection_reasons.append("clean_control_test_failed")
                return report
            clean_passed = [primary_test_id, control_test_id]
        else:
            clean_passed = self._parse_passed_tests(clean_result, repo_copy)
        report.clean_passed_tests = clean_passed
        if not clean_passed:
            report.rejection_reasons.append("clean_tests_unusable")
            return report

        patch_ok = apply_patch(repo_copy, candidate_patch)
        report.patch_applied = patch_ok
        if not patch_ok:
            report.rejection_reasons.append("patch_apply_failed")
            return report

        report.syntax_valid = self._check_syntax(repo_copy, candidate_patch)
        if not report.syntax_valid:
            return report

        report.import_valid = True

        bugged_result = self._run_tests(repo_copy, test_command)
        if bugged_result.get("timed_out"):
            report.rejection_reasons.append("bugged_test_timeout")
            return report
        if validation_mode == "exit_code":
            if bugged_result.get("returncode") == 0:
                bugged_failed = []
                bugged_passed = [primary_test_id]
            else:
                bugged_failed = [primary_test_id]
                bugged_passed = []
            bugged_control = self._run_tests(repo_copy, control_test_command or "")
            if bugged_control.get("timed_out"):
                report.rejection_reasons.append("bugged_control_test_timeout")
                return report
            if bugged_control.get("returncode") != 0:
                report.rejection_reasons.append("bugged_control_test_failed")
                return report
            bugged_passed.append(control_test_id)
        else:
            bugged_failed = self._parse_failed_tests(bugged_result, repo_copy)
            bugged_passed = self._parse_passed_tests(bugged_result, repo_copy)
        report.bugged_failed_tests = bugged_failed
        report.bugged_passed_tests = bugged_passed

        # F2P / P2P (P0-8): ALWAYS overwrite from trusted execution results.
        # Never trust proposer-declared f2p_tests / FAIL_TO_PASS metadata —
        # candidate.generation_metadata may advertise F2P lists that did not
        # come from the clean ∩ bugged intersection computed here.
        report.f2p_tests = [t for t in bugged_failed if t in clean_passed]
        # P2P: tests that passed before and still pass after bug
        report.p2p_tests = [t for t in bugged_passed if t in clean_passed]

        if not report.f2p_tests:
            return report

        # Require at least 1 P2P (bug should not break everything)
        if not report.p2p_tests:
            report.rejection_reasons.append("no_p2p")
            return report

        reverse_ok = reverse_patch(repo_copy, candidate_patch)
        if not reverse_ok:
            try:
                reset_to_commit(repo_copy, base_commit)
                reverse_ok = bool(
                    not setup_patch.strip() or apply_patch(repo_copy, setup_patch)
                )
            except Exception:
                pass

        if reverse_ok:
            restore_result = self._run_tests(repo_copy, test_command)
            if restore_result.get("timed_out"):
                report.rejection_reasons.append("restore_test_timeout")
                return report
            if validation_mode == "exit_code":
                report.reverse_restored = restore_result.get("returncode") == 0
            else:
                restore_passed = self._parse_passed_tests(restore_result, repo_copy)
                report.reverse_restored = all(t in restore_passed for t in report.f2p_tests)
        else:
            report.reverse_restored = False

        report.timeout_valid = True

        return report

    @staticmethod
    def _is_test_path(path: str) -> bool:
        normalized = str(path).replace("\\", "/").lower()
        parts = normalized.split("/")
        name = parts[-1] if parts else ""
        return bool(
            "test" in parts
            or "tests" in parts
            or name.startswith("test_")
            or name.endswith("_test.py")
            or name == "conftest.py"
        )

    def _check_relevance(
        self,
        repo_path: Path,
        patch: str,
        *,
        target_file: str = "",
        target_symbol: str = "",
    ) -> tuple[bool, list[str]]:
        """Check that a candidate patch edits its declared target."""
        changed_files = extract_changed_files(patch)
        if not changed_files:
            return False, ["relevance_no_changed_files"]

        normalized_changed = {_normalize_repo_path(f) for f in changed_files}
        normalized_target = _normalize_repo_path(target_file)

        if normalized_target:
            if normalized_target not in normalized_changed:
                return False, [f"irrelevant_target_file: {target_file}"]
            file_for_symbol = normalized_target
        elif len(normalized_changed) == 1:
            file_for_symbol = next(iter(normalized_changed))
        else:
            file_for_symbol = ""

        if not target_symbol:
            return True, []
        if not file_for_symbol:
            return False, ["relevance_target_symbol_without_single_file"]
        if not file_for_symbol.endswith(".py"):
            return True, []

        source_path = repo_path / file_for_symbol
        if not source_path.exists():
            return False, [f"relevance_target_file_missing: {file_for_symbol}"]

        symbol_ranges = _python_symbol_ranges(source_path, target_symbol)
        if not symbol_ranges:
            return False, [f"relevance_target_symbol_missing: {target_symbol}"]

        changed_lines = _changed_old_lines_for_file(patch, file_for_symbol)
        if not changed_lines:
            return False, [f"relevance_no_changed_lines: {file_for_symbol}"]

        for line in changed_lines:
            for start, end in symbol_ranges:
                if start <= line <= end:
                    return True, []
        return False, [f"irrelevant_target_symbol: {target_symbol}"]

    def _ensure_verbose_pytest(self, test_command: str) -> str:
        """Ensure pytest prints individual test IDs needed for F2P/P2P."""
        parts = test_command.split()
        if not any("pytest" in p for p in parts):
            return test_command
        if "-v" in parts or "-vv" in parts or any(p.startswith("--verbose") for p in parts):
            return test_command
        filtered = [p for p in parts if p not in {"-q", "--quiet"}]
        return " ".join(filtered + ["-v"])

    def _run_tests(self, repo_path: Path, test_command: str) -> dict:
        """Run a test command and return results.

        BUG-15/10.8: route through the ExecutionBackend when one is configured
        so trusted repository tests also run inside Apptainer (end-to-end
        chain). Falls back to direct subprocess for backward compatibility.
        """
        if self.execution_backend is not None:
            try:
                import shlex as _shlex

                parts = _shlex.split(test_command) if isinstance(test_command, str) else list(test_command)
                binds = {Path(repo_path): "/workspace"}
                result = self.execution_backend.run(
                    command=parts,
                    cwd=Path(repo_path),
                    env={},
                    timeout_sec=self.test_timeout_sec,
                    binds=binds,
                )
                return {
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "timed_out": result.timed_out,
                }
            except Exception as e:
                return {
                    "returncode": -1,
                    "stdout": "",
                    "stderr": str(e),
                    "timed_out": False,
                }
        try:
            result = subprocess.run(
                test_command,
                shell=True,
                cwd=str(repo_path),
                capture_output=True,
                text=True,
                timeout=self.test_timeout_sec,
            )
            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": "Timeout",
                "timed_out": True,
            }
        except Exception as e:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "timed_out": False,
            }

    def _parse_passed_tests(
        self,
        result: dict,
        repo_path: Optional[Path] = None,
    ) -> List[str]:
        """Parse pytest output to find passing tests.

        Handles both verbose (-v) and non-verbose output:
        - Verbose:   test_file.py::TestClass::test_method PASSED
        - Non-verbose: .....F.s (summary line) + "X passed, Y failed"
        """
        return _parse_pytest_tests(result, {"PASSED"}, repo_path)

    def _parse_failed_tests(
        self,
        result: dict,
        repo_path: Optional[Path] = None,
    ) -> List[str]:
        """Parse pytest output to find failing tests.

        Handles both verbose (-v) and non-verbose output:
        - Verbose:   test_file.py::TestClass::test_method FAILED
        - Non-verbose: FAILED test_file.py::TestClass::test_method - error msg
        """
        return _parse_pytest_tests(result, {"FAILED", "ERROR"}, repo_path)

    def _check_syntax(self, repo_path: Path, patch: str) -> bool:
        """Check that patched Python files have valid syntax."""
        changed = extract_changed_files(patch)
        for f in changed:
            if not f.endswith(".py"):
                continue
            file_path = repo_path / f
            if not file_path.exists():
                continue
            try:
                with open(file_path) as fh:
                    compile(fh.read(), str(file_path), "exec")
            except SyntaxError:
                return False
        return True


def _dedupe(items: List[str]) -> List[str]:
    """Deduplicate test IDs while preserving parser order."""
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _parse_pytest_tests(
    result: dict,
    statuses: set[str],
    repo_path: Optional[Path] = None,
) -> List[str]:
    """Parse complete pytest node IDs, including parameter values with spaces."""
    combined = f"{result.get('stdout', '')}\n{result.get('stderr', '')}"
    tests: List[str] = []
    repo_prefix = str(repo_path.resolve()) if repo_path else ""

    for raw_line in combined.splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw_line).strip()
        verbose = re.match(
            r"^(?P<nodeid>.+?::.+?)\s+"
            r"(?P<status>PASSED|FAILED|ERROR)(?:\s+\[[^]]*\])?$",
            line,
        )
        summary = re.match(
            r"^(?P<status>FAILED|ERROR)\s+"
            r"(?P<nodeid>.+?::.+?)(?:\s+-\s+.*)?$",
            line,
        )
        match = verbose or summary
        if not match or match.group("status") not in statuses:
            continue
        nodeid = match.group("nodeid").rstrip(":-")
        if repo_prefix:
            nodeid = nodeid.replace(repo_prefix, "<REPO>")
        tests.append(nodeid)
    return _dedupe(tests)


def _normalize_repo_path(path: str) -> str:
    """Normalize a patch/repo path to repo-relative POSIX form."""
    path = str(path or "").strip()
    if not path:
        return ""
    path = path.replace("\\", "/")
    if path.startswith("a/") or path.startswith("b/"):
        path = path[2:]
    return path.lstrip("./")


def _python_symbol_ranges(source_path: Path, target_symbol: str) -> list[tuple[int, int]]:
    """Return line ranges for a Python function/class symbol."""
    try:
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
    except (OSError, SyntaxError):
        return []

    requested = target_symbol.split(".")[-1]
    ranges: list[tuple[int, int]] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name != requested:
            continue
        start = getattr(node, "lineno", 0)
        end = getattr(node, "end_lineno", start) or start
        if start:
            ranges.append((start, end))
    return ranges


def _changed_old_lines_for_file(patch: str, target_file: str) -> set[int]:
    """Return original-file line numbers touched by a unified diff."""
    target_file = _normalize_repo_path(target_file)
    lines: set[int] = set()
    current_file = ""
    old_line = 0
    in_target = False

    for line in patch.splitlines():
        if line.startswith("diff --git"):
            current_file = ""
            in_target = False
            continue
        if line.startswith("+++ "):
            current_file = _normalize_repo_path(line[4:].strip().split("\t", 1)[0])
            in_target = current_file == target_file
            continue
        if not in_target:
            continue
        if line.startswith("@@"):
            match = __import__("re").match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            if match:
                old_line = int(match.group(1))
            continue
        if not old_line:
            continue
        if line.startswith("-") and not line.startswith("---"):
            lines.add(old_line)
            old_line += 1
        elif line.startswith("+") and not line.startswith("+++"):
            # Pure insertions belong to the current original-file location.
            lines.add(old_line)
        else:
            old_line += 1

    return lines
