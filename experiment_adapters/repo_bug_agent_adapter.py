"""Bounded coding-agent loop for repository-level bug construction."""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Optional, Tuple


REPO_BUG_SYSTEM_PROMPT = """\
You are a software mutation engineer constructing a difficult repository-level
repair task. The repository is currently healthy. Introduce one coherent
behavioral regression; do not repair or merely explain the existing code.

Work under this strict call budget:
- Calls 1-8: inspect only narrow slices of the anchor production files and tests.
- Call 9: apply the first multi-file mutation with the edit_files tool.
- Remaining calls: run only the targeted tests, refine the mutation, and inspect
  the final diff.

The final diff must satisfy the blueprint's production-file bounds (up to six
related files). Each edit must be part of one producer/consumer protocol or
causal chain. Prefer an incomplete protocol migration, representation drift, or
incorrect value handoff that propagates through every edited layer. Do not use
unrelated boolean flips, arbitrary loop truncation, or several independent bugs
merely to satisfy the file count. Do not edit
tests, generated files, dependency locks, or git metadata. Do not run the full
test suite, trace an already-passing behavior exhaustively, or commit changes.
Break established behavior deliberately: remove or invert a branch, corrupt a
value handoff, or skip required propagation already asserted by baseline tests.
Do not add fallback handling, compatibility aliases, broader exception handling,
missing-value support, or comments-only code; those are repairs, not regressions.
At least two final file mutations must each independently fail a baseline test
for the same contract, so restoring one file cannot solve the task.
Use edit_files with exact old/new snippets instead of rewriting complete files.
Copy the shortest unique 1-5 line old_text verbatim from shell output; never
reconstruct an entire function or class from memory. Bash is read-only and any
worktree change made by a shell command is automatically reverted.
Every worktree change triggers an automatic F2P and per-file ablation probe. Do
not stop until that probe reports strict_ready=true. If it reports inert files
or a single-file repair, inspect those exact paths and revise the causal chain.
The shell already starts at the repository root. Use only repository-relative
paths. Never search for, reconstruct, or cd into the temporary repository path.
"""


class RepoBugAgentAdapter:
    """Run a short, mutation-specific tool loop without changing Solver code."""

    def __init__(
        self,
        *,
        client_factory: Optional[Callable[[str], Tuple[Any, str]]] = None,
        max_llm_calls: int = 32,
        max_output_tokens: int = 1536,
        max_tool_result_chars: int = 6000,
        edit_deadline_call: int = 9,
        shell_timeout_sec: int = 90,
        disable_qwen_thinking: bool = True,
    ) -> None:
        self.client_factory = client_factory
        self.max_llm_calls = max(1, max_llm_calls)
        self.max_output_tokens = max(256, max_output_tokens)
        self.max_tool_result_chars = max(1000, max_tool_result_chars)
        self.edit_deadline_call = max(1, edit_deadline_call)
        self.shell_timeout_sec = max(1, shell_timeout_sec)
        self.disable_qwen_thinking = disable_qwen_thinking

    def generate_repo_bug(
        self,
        workspace: str,
        request: str,
        plan: Any,
        *,
        output_dir: Optional[str] = None,
    ) -> str:
        root = Path(workspace).resolve()
        trace_dir = Path(output_dir).resolve() if output_dir else root.parent
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = trace_dir / "trajectory.jsonl"
        trace_path.write_text("", encoding="utf-8")

        model = str(getattr(plan, "model", "") or "deepseek/deepseek-chat")
        client, actual_model = self._create_client(model)
        timeout_sec = int(
            getattr(getattr(plan, "constraints", None), "generation_timeout_sec", 1200)
            or 1200
        )
        started = time.monotonic()
        messages: list[Any] = [
            {"role": "system", "content": REPO_BUG_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{request}\n\n"
                    "The bash working directory is already the repository root. "
                    "Use relative paths from the blueprint. Begin with focused "
                    "inspection and do not give a prose-only answer."
                ),
            },
        ]
        tools = self._tool_specs()
        no_edit_reminders = 0
        correction_reads_remaining = 0
        blueprint = dict(getattr(plan, "task_blueprint", None) or {})
        constraints = getattr(plan, "constraints", None)
        min_modified_files = max(
            2,
            int(getattr(constraints, "min_modified_files", 2) or 2),
        )
        max_modified_files = max(
            min_modified_files,
            int(getattr(constraints, "max_modified_files", 6) or 6),
        )
        validation_command = str(blueprint.get("validation_command") or "").strip()
        clean_passed: set[str] = set()
        last_probe: Optional[dict[str, Any]] = None
        probe_requires_revision = False

        self._write_event(
            trace_path,
            "start",
            model=actual_model,
            max_llm_calls=self.max_llm_calls,
            timeout_sec=timeout_sec,
        )

        if validation_command:
            clean_result = self._run_validation_command(
                validation_command,
                root,
                timeout=self.shell_timeout_sec,
            )
            clean_passed, clean_failed = self._pytest_statuses(clean_result)
            self._write_event(
                trace_path,
                "clean_baseline",
                returncode=clean_result["returncode"],
                timed_out=clean_result["timed_out"],
                passed_count=len(clean_passed),
                failed_count=len(clean_failed),
            )
            if clean_result["timed_out"] or not clean_passed:
                self._write_event(
                    trace_path,
                    "quality_gate_rejected",
                    reason=(
                        "clean_test_timeout"
                        if clean_result["timed_out"]
                        else "clean_tests_unusable"
                    ),
                )
                (trace_dir / "model_patch.diff").write_text("", encoding="utf-8")
                self._write_event(
                    trace_path,
                    "finish",
                    elapsed_sec=round(time.monotonic() - started, 3),
                    patch_chars=0,
                    changed_files=[],
                )
                return ""

        for call_index in range(1, self.max_llm_calls + 1):
            elapsed = time.monotonic() - started
            if elapsed >= timeout_sec * 0.9:
                self._write_event(trace_path, "timeout", elapsed_sec=round(elapsed, 3))
                break

            current_patch = self._repository_diff(root)
            has_edits = bool(current_patch)
            if validation_command:
                probe_requires_revision = not self._probe_matches_patch(
                    last_probe,
                    current_patch,
                )
            must_edit_now = (
                call_index >= self.edit_deadline_call
                and (not has_edits or probe_requires_revision)
                and correction_reads_remaining == 0
            )
            if must_edit_now:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "MANDATORY ACTION: the repository either has no mutation or "
                            "its latest automatic ablation probe is not strict. Your next "
                            "call must use edit_files to revise at least two related "
                            "production files. Stop further investigation and do not "
                            "finish until strict_ready=true."
                        ),
                    }
                )

            try:
                response = client.chat.completions.create(
                    **self._completion_kwargs(
                        actual_model=actual_model,
                        messages=messages,
                        tools=(
                            [tool for tool in tools if tool["function"]["name"] == "edit_files"]
                            if must_edit_now
                            else tools
                        ),
                        forced_tool_name="edit_files" if must_edit_now else None,
                    )
                )
            except Exception as exc:
                self._write_event(
                    trace_path,
                    "model_error",
                    call=call_index,
                    error=str(exc)[:2000],
                )
                break

            choice = response.choices[0]
            message = choice.message
            tool_calls = list(getattr(message, "tool_calls", None) or [])
            content = str(getattr(message, "content", "") or "")
            self._write_event(
                trace_path,
                "assistant",
                call=call_index,
                finish_reason=str(getattr(choice, "finish_reason", "") or ""),
                content=content,
                tool_names=[call.function.name for call in tool_calls],
                usage=self._usage(response),
                has_edits=has_edits,
            )
            messages.append(message)

            if not tool_calls:
                patch_now = self._repository_diff(root)
                quality_ready = (
                    not validation_command
                    or self._probe_matches_patch(last_probe, patch_now)
                )
                if patch_now and quality_ready:
                    break
                no_edit_reminders += 1
                if no_edit_reminders >= 2:
                    self._write_event(
                        trace_path,
                        "stopped_without_acceptable_edits",
                        call=call_index,
                        has_edits=bool(patch_now),
                        quality_ready=quality_ready,
                    )
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "A prose answer is not a result. Use edit_files now to "
                            "introduce or revise the multi-file regression until the "
                            "automatic probe reports strict_ready=true."
                        ),
                    }
                )
                continue

            for tool_call in tool_calls:
                tool_name = str(tool_call.function.name)
                patch_before_tool = self._repository_diff(root)
                try:
                    tool_input = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    tool_input = {}
                    tool_result = f"Invalid JSON tool arguments: {exc}"
                except Exception as exc:
                    tool_input = {}
                    tool_result = f"Invalid tool arguments: {exc}"
                else:
                    if must_edit_now and tool_name != "edit_files":
                        tool_result = (
                            "Tool denied: the exploration budget is exhausted and "
                            "the repository still has no edits. The next tool call "
                            "must be edit_files with exact snippets touching at "
                            "least two related production files."
                        )
                    else:
                        revision_error = self._revision_edit_restriction(
                            tool_name,
                            tool_input,
                            last_probe,
                        )
                        tool_result = revision_error or self._run_tool(
                            tool_name,
                            tool_input,
                            root,
                        )
                patch_after_tool = self._repository_diff(root)
                probe_feedback = ""
                if (
                    validation_command
                    and patch_after_tool
                    and patch_after_tool != patch_before_tool
                ):
                    last_probe = self._probe_candidate(
                        root,
                        validation_command,
                        clean_passed,
                        min_modified_files=min_modified_files,
                        max_modified_files=max_modified_files,
                        reject_mechanical_shortcuts=bool(
                            blueprint.get("contract_test")
                        ),
                    )
                    probe_requires_revision = not bool(last_probe.get("strict_ready"))
                    probe_feedback = self._format_probe_feedback(last_probe)
                    tool_result = f"{tool_result}\n\n{probe_feedback}"
                    self._write_event(
                        trace_path,
                        "quality_probe",
                        call=call_index,
                        **last_probe,
                    )
                reopen_correction_read = (
                    tool_name == "edit_files"
                    and tool_result.startswith("Error:")
                )
                if (
                    correction_reads_remaining > 0
                    and tool_name != "edit_files"
                ):
                    correction_reads_remaining -= 1
                self._write_event(
                    trace_path,
                    "tool",
                    call=call_index,
                    name=tool_name,
                    arguments=tool_input,
                    result=tool_result,
                    has_edits=bool(self._repository_diff(root)),
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": self._compact_tool_result(tool_result),
                    }
                )
                if reopen_correction_read:
                    correction_reads_remaining = 1
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The edit transaction failed. You have exactly one "
                                "bash call to read the exact relative files and lines "
                                "reported above. Then call edit_files again with "
                                "verbatim old_text from the repository."
                            ),
                        }
                    )
                elif probe_feedback and probe_requires_revision:
                    correction_reads_remaining = 1
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The automatic F2P ablation gate rejected this mutation. "
                                "You have exactly one focused bash read of the reported "
                                "files/tests, then you must call edit_files again. You are "
                                "still the bug generator, not the solver: preserve every "
                                "active bug file that already has only_file_f2p and add "
                                "failure-causing behavior to the inert files. Make every "
                                "touched file necessary and keep an unaffected test passing."
                            ),
                        }
                    )

        patch = self._repository_diff(root)
        if validation_command and not self._probe_matches_patch(last_probe, patch):
            reason = "missing_probe" if last_probe is None else "probe_not_strict_or_stale"
            self._write_event(
                trace_path,
                "quality_gate_rejected",
                reason=reason,
                patch_chars=len(patch),
                changed_files=self._changed_files(root),
            )
            self._discard_repository_diff(root)
            patch = self._repository_diff(root)
            if patch:
                self._write_event(
                    trace_path,
                    "quality_gate_cleanup_failed",
                    patch_chars=len(patch),
                    changed_files=self._changed_files(root),
                )
            else:
                patch = ""
        (trace_dir / "model_patch.diff").write_text(patch, encoding="utf-8")
        self._write_event(
            trace_path,
            "finish",
            elapsed_sec=round(time.monotonic() - started, 3),
            patch_chars=len(patch),
            changed_files=self._changed_files(root),
        )
        return patch

    def _create_client(self, model: str) -> Tuple[Any, str]:
        if self.client_factory is not None:
            return self.client_factory(model)
        from llm import create_client

        return create_client(model)

    def _completion_kwargs(
        self,
        *,
        actual_model: str,
        messages: list[Any],
        tools: list[dict[str, Any]],
        forced_tool_name: Optional[str],
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": actual_model,
            "messages": messages,
            "tools": tools,
            "tool_choice": (
                {
                    "type": "function",
                    "function": {"name": forced_tool_name},
                }
                if forced_tool_name
                else "auto"
            ),
            "parallel_tool_calls": False,
            "temperature": 0.4,
            "max_tokens": self.max_output_tokens,
        }
        if self.disable_qwen_thinking and "qwen" in actual_model.lower():
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False}
            }
        return kwargs

    def _run_tool(self, name: str, arguments: dict[str, Any], root: Path) -> str:
        if name == "bash":
            return self._run_bash(str(arguments.get("command") or ""), root)
        if name == "edit_files":
            return self._edit_files(arguments.get("edits"), root)
        if name == "apply_patch":
            return self._apply_patch(str(arguments.get("patch") or ""), root)
        return f"Unknown tool: {name}"

    def _run_bash(self, command: str, root: Path) -> str:
        if not command.strip():
            return "Error: command is empty"
        patch_before = self._repository_diff(root)
        try:
            result = subprocess.run(
                command,
                cwd=root,
                shell=True,
                executable="/bin/bash",
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                timeout=self.shell_timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {self.shell_timeout_sec}s"
        output = result.stdout
        if result.stderr:
            output += ("\n" if output else "") + "STDERR:\n" + result.stderr
        patch_after = self._repository_diff(root)
        if patch_after != patch_before:
            restored = self._set_repository_patch(root, patch_before)
            output += (
                "\n" if output else ""
            ) + (
                "BASH_WRITE_REVERTED: bash is read-only; use edit_files for "
                "worktree changes."
                if restored
                else "BASH_WRITE_RESTORE_FAILED: stop and inspect the worktree."
            )
        return f"exit_code={result.returncode}\n{output}".rstrip()

    def _run_validation_command(
        self,
        command: str,
        root: Path,
        *,
        timeout: int,
    ) -> dict[str, Any]:
        try:
            result = subprocess.run(
                command,
                cwd=root,
                shell=True,
                executable="/bin/bash",
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                timeout=max(1, timeout),
                check=False,
            )
            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired as exc:
            return {
                "returncode": -2,
                "stdout": self._decode_timeout_output(exc.stdout),
                "stderr": self._decode_timeout_output(exc.stderr),
                "timed_out": True,
            }
        except Exception as exc:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": str(exc),
                "timed_out": False,
            }

    def _pytest_statuses(
        self,
        result: dict[str, Any],
    ) -> tuple[set[str], set[str]]:
        passed: set[str] = set()
        failed: set[str] = set()
        combined = f"{result.get('stdout', '')}\n{result.get('stderr', '')}"
        ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
        statuses = {"PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL", "XPASS"}
        for raw_line in combined.splitlines():
            line = ansi_escape.sub("", raw_line).strip()
            if "::" not in line:
                continue
            parts = line.split()
            node_id = next((part.rstrip(":-") for part in parts if "::" in part), "")
            status = next((part for part in parts if part in statuses), "")
            if not node_id or not status:
                continue
            if status == "PASSED":
                passed.add(node_id)
            elif status in {"FAILED", "ERROR"}:
                failed.add(node_id)
        return passed, failed

    def _probe_candidate(
        self,
        root: Path,
        validation_command: str,
        clean_passed: set[str],
        *,
        min_modified_files: int = 2,
        max_modified_files: int = 6,
        reject_mechanical_shortcuts: bool = False,
    ) -> dict[str, Any]:
        full_patch = self._repository_diff(root)
        changed_files = self._changed_files(root)
        patch_sha256 = self._patch_sha256(full_patch)
        probe: dict[str, Any] = {
            "patch_sha256": patch_sha256,
            "changed_files": changed_files,
            "full_f2p": [],
            "full_p2p_count": 0,
            "only_file_f2p": {},
            "without_file_f2p": {},
            "standalone_inert_files": [],
            "single_repair_sufficient_files": [],
            "setup_errors": [],
            "timed_out": False,
            "syntax_valid": self._python_syntax_valid(root, changed_files),
            "causal_shortcut_reasons": (
                self._causal_shortcut_reasons(full_patch)
                if reject_mechanical_shortcuts
                else []
            ),
            "strict_ready": False,
        }
        if not full_patch or len(changed_files) < min_modified_files:
            probe["setup_errors"].append(
                f"candidate_must_modify_at_least_{min_modified_files}_files"
            )
            return probe
        if len(changed_files) > max_modified_files:
            probe["setup_errors"].append(
                f"candidate_must_modify_at_most_{max_modified_files}_files"
            )
            return probe

        full_result = self._run_validation_command(
            validation_command,
            root,
            timeout=self.shell_timeout_sec,
        )
        full_passed, full_failed = self._pytest_statuses(full_result)
        full_f2p = sorted(clean_passed & full_failed)
        probe["full_f2p"] = full_f2p
        probe["full_p2p_count"] = len(clean_passed & full_passed)
        probe["full_returncode"] = full_result["returncode"]
        probe["timed_out"] = bool(full_result["timed_out"])
        if full_result["timed_out"] or not full_f2p:
            return probe

        blocks = {
            path: self._repository_diff(root, path=path)
            for path in changed_files
        }
        empty_blocks = [path for path, block in blocks.items() if not block]
        if empty_blocks:
            probe["setup_errors"].extend(
                f"missing_file_patch:{path}" for path in empty_blocks
            )
            return probe

        variant_cache: dict[str, dict[str, Any]] = {}

        def run_variant(variant_patch: str) -> dict[str, Any]:
            digest = self._patch_sha256(variant_patch)
            if digest in variant_cache:
                return variant_cache[digest]
            if not self._set_repository_patch(root, variant_patch):
                result = {
                    "failed": set(),
                    "timed_out": False,
                    "setup_error": "variant_patch_apply_failed",
                }
            else:
                command_result = self._run_validation_command(
                    validation_command,
                    root,
                    timeout=self.shell_timeout_sec,
                )
                _, failed = self._pytest_statuses(command_result)
                result = {
                    "failed": failed,
                    "timed_out": bool(command_result["timed_out"]),
                    "setup_error": "",
                }
            variant_cache[digest] = result
            return result

        try:
            for path in changed_files:
                only_result = run_variant(blocks[path])
                without_patch = "".join(
                    blocks[other] for other in changed_files if other != path
                )
                without_result = run_variant(without_patch)
                only_f2p = sorted(set(full_f2p) & only_result["failed"])
                without_f2p = sorted(set(full_f2p) & without_result["failed"])
                probe["only_file_f2p"][path] = only_f2p
                probe["without_file_f2p"][path] = without_f2p
                for mode, result in (
                    ("only", only_result),
                    ("without", without_result),
                ):
                    if result["setup_error"]:
                        probe["setup_errors"].append(
                            f"{mode}:{path}:{result['setup_error']}"
                        )
                    if result["timed_out"]:
                        probe["timed_out"] = True
        finally:
            if not self._set_repository_patch(root, full_patch):
                probe["setup_errors"].append("full_patch_restore_failed")

        probe["standalone_inert_files"] = [
            path for path in changed_files if not probe["only_file_f2p"].get(path)
        ]
        probe["single_repair_sufficient_files"] = [
            path for path in changed_files if not probe["without_file_f2p"].get(path)
        ]
        probe["strict_ready"] = bool(
            probe["syntax_valid"]
            and probe["full_f2p"]
            and probe["full_p2p_count"]
            and not probe["standalone_inert_files"]
            and not probe["single_repair_sufficient_files"]
            and not probe["causal_shortcut_reasons"]
            and not probe["setup_errors"]
            and not probe["timed_out"]
            and self._patch_sha256(self._repository_diff(root)) == patch_sha256
        )
        return probe

    def _format_probe_feedback(self, probe: dict[str, Any]) -> str:
        only_file_f2p = probe.get("only_file_f2p") or {}
        active_bug_files = [
            path for path, tests in only_file_f2p.items() if tests
        ]
        feedback = {
            "strict_ready": bool(probe.get("strict_ready")),
            "syntax_valid": bool(probe.get("syntax_valid")),
            "full_f2p": probe.get("full_f2p") or [],
            "full_p2p_count": int(probe.get("full_p2p_count") or 0),
            "only_file_f2p": only_file_f2p,
            "active_bug_files_preserve_unchanged": active_bug_files,
            "without_file_f2p": probe.get("without_file_f2p") or {},
            "standalone_inert_files": probe.get("standalone_inert_files") or [],
            "single_repair_sufficient_files": (
                probe.get("single_repair_sufficient_files") or []
            ),
            "setup_errors": probe.get("setup_errors") or [],
            "timed_out": bool(probe.get("timed_out")),
            "causal_shortcut_reasons": (
                probe.get("causal_shortcut_reasons") or []
            ),
        }
        guidance: list[str] = []
        if not probe.get("full_f2p"):
            guidance.append(
                "The complete patch breaks no baseline test. Replace behavior, "
                "not comments or fallback handling; preserve at least two final "
                "file diffs and contradict concrete assertions in the listed tests."
            )
        if any(
            str(error).startswith("candidate_must_modify_at_least_")
            for error in probe.get("setup_errors") or []
        ):
            guidance.append(
                "Your last transaction restored one prior file to HEAD, leaving a "
                "single-file diff. Keep the existing mutation and add a causal "
                "mutation in another file; do not alternate which file is changed."
            )
        if probe.get("standalone_inert_files"):
            guidance.append(
                "Do not repair or restore active_bug_files_preserve_unchanged. "
                "Each standalone_inert_file must independently fail at least one "
                "test from full_f2p when its mutation is applied alone. Submit an "
                "active file as a no-op entry if the tool schema needs another item."
            )
        if probe.get("single_repair_sufficient_files"):
            guidance.append(
                "For each single_repair_sufficient_file, strengthen mutations in "
                "the remaining files so repairing that one file is insufficient."
            )
        if probe.get("causal_shortcut_reasons"):
            guidance.append(
                "Replace mechanical mutation shortcuts with one coherent protocol "
                "or value-handoff regression. Do not truncate iterations, erase "
                "values with None, hard-code fixtures, or directly invert a boolean."
            )
        feedback["revision_guidance"] = guidance
        return "AUTOMATIC F2P ABLATION PROBE\n" + json.dumps(
            feedback,
            indent=2,
            ensure_ascii=False,
        )

    def _revision_edit_restriction(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        probe: Optional[dict[str, Any]],
    ) -> str:
        if tool_name != "edit_files" or not probe:
            return ""
        inert_files = set(probe.get("standalone_inert_files") or [])
        if not inert_files:
            return ""
        active_files = {
            path
            for path, tests in (probe.get("only_file_f2p") or {}).items()
            if tests
        }
        changed_active = []
        for edit in tool_input.get("edits") or []:
            if not isinstance(edit, dict):
                continue
            path = str(edit.get("path") or "")
            if path in active_files and edit.get("old_text") != edit.get("new_text"):
                changed_active.append(path)
        if not changed_active:
            return ""
        return (
            "Error: no files changed; revision rejected\n"
            "You are attempting to repair active bug files already proven to cause "
            "F2P failures: "
            + ", ".join(sorted(set(changed_active)))
            + ". Preserve those mutations unchanged and modify only the reported "
            "standalone_inert_files (or add another related production file)."
        )

    def _probe_matches_patch(
        self,
        probe: Optional[dict[str, Any]],
        patch: str,
    ) -> bool:
        return bool(
            patch
            and probe
            and probe.get("strict_ready")
            and probe.get("patch_sha256") == self._patch_sha256(patch)
        )

    def _patch_sha256(self, patch: str) -> str:
        return hashlib.sha256(patch.encode("utf-8")).hexdigest()

    def _causal_shortcut_reasons(self, patch: str) -> list[str]:
        added = [
            line[1:].strip()
            for line in patch.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        ]
        removed = [
            line[1:].strip()
            for line in patch.splitlines()
            if line.startswith("-") and not line.startswith("---")
        ]
        reasons: set[str] = set()
        for line in added:
            if re.search(r"\blist\(.+\)\s*\[:\s*1\s*\]", line):
                reasons.add("arbitrary_iteration_truncation")
            if re.match(r"return\s+None(?:\s*#.*)?$", line):
                reasons.add("direct_return_none")
            if re.search(r"\b(?:test1|test2|host1|host2)\b", line, re.IGNORECASE):
                reasons.add("hard_coded_contract_fixture")
            if line.startswith("if not (") and line.endswith("):"):
                original = "if " + line[len("if not (") : -2] + ":"
                if original in removed:
                    reasons.add("mechanical_condition_inversion")
            assignment = re.match(
                r"(?P<lhs>[A-Za-z_][\w.]*)\s*=\s*(?P<value>True|False)$",
                line,
            )
            if assignment:
                opposite = "False" if assignment.group("value") == "True" else "True"
                if f"{assignment.group('lhs')} = {opposite}" in removed:
                    reasons.add("mechanical_boolean_flip")
            for keyword in re.findall(r"\b([A-Za-z_]\w*)=None\b", line):
                if any(
                    re.search(
                        rf"\b{re.escape(keyword)}=(?!None\b)[A-Za-z_]\w*",
                        old,
                    )
                    for old in removed
                ):
                    reasons.add("keyword_value_erasure")
        return sorted(reasons)

    def _python_syntax_valid(self, root: Path, paths: list[str]) -> bool:
        for path in paths:
            target = root / path
            if path.endswith(".py") and target.is_file():
                try:
                    ast.parse(target.read_text(encoding="utf-8"), filename=path)
                except (OSError, SyntaxError, UnicodeError):
                    return False
        return True

    def _set_repository_patch(self, root: Path, target_patch: str) -> bool:
        current_patch = self._repository_diff(root)
        if current_patch == target_patch:
            return True
        if current_patch and not self._git_apply(root, current_patch, reverse=True):
            return False
        if target_patch and not self._git_apply(root, target_patch):
            if current_patch:
                self._git_apply(root, current_patch)
            return False
        return True

    def _git_apply(self, root: Path, patch: str, *, reverse: bool = False) -> bool:
        args = ["git", "apply", "--whitespace=nowarn"]
        if reverse:
            args.append("--reverse")
        args.append("-")
        try:
            result = subprocess.run(
                args,
                cwd=root,
                input=patch,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0

    def _discard_repository_diff(self, root: Path) -> bool:
        return self._set_repository_patch(root, "")

    def _decode_timeout_output(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)

    def _apply_patch(self, patch: str, root: Path) -> str:
        if not patch.strip():
            return "Error: patch is empty"
        invalid_path = self._invalid_patch_path(patch)
        if invalid_path:
            return f"Error: patch path is not allowed: {invalid_path}"
        result = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            cwd=root,
            input=patch,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            return f"Error: git apply failed\n{result.stderr.strip()}"
        return "Patch applied.\n" + self._run_bash("git diff --stat", root)

    def _edit_files(self, edits: Any, root: Path) -> str:
        if not isinstance(edits, list) or not 2 <= len(edits) <= 6:
            return "Error: edits must contain between 2 and 6 file edits"

        staged: dict[str, tuple[Path, str, str]] = {}
        seen_paths: set[str] = set()
        ignored_noop_paths: list[str] = []
        errors: list[str] = []
        for index, edit in enumerate(edits, start=1):
            if not isinstance(edit, dict):
                errors.append(f"edit {index}: expected an object")
                continue
            relative_path = str(edit.get("path") or "")
            path_error = self._invalid_relative_path(relative_path)
            if path_error:
                errors.append(f"edit {index}: {path_error}")
                continue
            seen_paths.add(relative_path)

            target = (root / relative_path).resolve()
            try:
                target.relative_to(root)
            except ValueError:
                errors.append(f"edit {index}: path escapes repository: {relative_path}")
                continue
            if not target.is_file():
                errors.append(f"edit {index}: file does not exist: {relative_path}")
                continue

            old_text = str(edit.get("old_text") or "")
            new_text = str(edit.get("new_text") or "")
            if not old_text:
                errors.append(f"edit {index}: old_text must be non-empty")
                continue
            if old_text == new_text:
                ignored_noop_paths.append(relative_path)
                continue
            if relative_path in staged:
                _, original_content, content = staged[relative_path]
            else:
                content = target.read_text(encoding="utf-8")
                original_content = content
            occurrences = content.count(old_text)
            if occurrences != 1:
                closest = self._closest_context(content, old_text)
                errors.append(
                    f"edit {index}: old_text occurs {occurrences} times in "
                    f"{relative_path}. Copy the shortest unique 1-5 lines "
                    f"verbatim from this real context:\n{closest}"
                )
                continue
            staged[relative_path] = (
                target,
                original_content,
                content.replace(old_text, new_text, 1),
            )

        if len(seen_paths) < 2:
            errors.append("edits must target at least two distinct production files")
        if not staged:
            errors.append("at least one edit must materially change its file")
        if errors:
            return "Error: no files changed; transaction rejected\n" + "\n".join(errors)

        for target, _, content in staged.values():
            target.write_text(content, encoding="utf-8")
        final_changed_files = self._changed_files(root)
        if len(final_changed_files) < 2:
            for target, original, _ in staged.values():
                target.write_text(original, encoding="utf-8")
            return (
                "Error: no files changed; transaction rejected\n"
                "The proposed revision would leave the final candidate with fewer "
                "than two changed files. Do not restore one prior mutation while "
                "switching to another file; keep at least two production files "
                "materially different from HEAD."
            )
        result = "Files edited transactionally."
        if ignored_noop_paths:
            result += " Ignored unchanged entries: " + ", ".join(ignored_noop_paths) + "."
        return result + "\n" + self._run_bash("git diff --stat", root)

    def _invalid_relative_path(self, raw_path: str) -> str:
        path = PurePosixPath(raw_path)
        if not raw_path or path.is_absolute() or ".." in path.parts:
            return f"invalid repository-relative path: {raw_path}"
        if path.parts and path.parts[0] in {"test", "tests"}:
            return f"test edits are not allowed: {raw_path}"
        return ""

    def _closest_context(self, content: str, requested: str) -> str:
        source_lines = content.splitlines()
        requested_lines = requested.splitlines()
        if not source_lines:
            return "<empty file>"
        window_size = max(1, len(requested_lines))
        best_index = 0
        best_ratio = -1.0
        requested_normalized = "\n".join(line.strip() for line in requested_lines)
        for index in range(max(1, len(source_lines) - window_size + 1)):
            window = source_lines[index : index + window_size]
            ratio = difflib.SequenceMatcher(
                None,
                requested_normalized,
                "\n".join(line.strip() for line in window),
            ).ratio()
            if ratio > best_ratio:
                best_index = index
                best_ratio = ratio
        start = max(0, best_index - 2)
        end = min(len(source_lines), best_index + window_size + 2)
        return "\n".join(
            f"{line_number + 1}: {source_lines[line_number]}"
            for line_number in range(start, end)
        )

    def _invalid_patch_path(self, patch: str) -> str:
        for line in patch.splitlines():
            if not line.startswith(("--- ", "+++ ")):
                continue
            raw_path = line[4:].split("\t", 1)[0].strip()
            if raw_path == "/dev/null":
                continue
            if raw_path.startswith(("a/", "b/")):
                raw_path = raw_path[2:]
            path = PurePosixPath(raw_path)
            if path.is_absolute() or ".." in path.parts:
                return raw_path
            if path.parts and path.parts[0] in {"test", "tests"}:
                return raw_path
        return ""

    def _repository_diff(self, root: Path, *, path: Optional[str] = None) -> str:
        command = ["git", "diff", "--binary", "HEAD", "--"]
        if path:
            command.append(path)
        result = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout if result.returncode == 0 else ""

    def _changed_files(self, root: Path) -> list[str]:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD", "--"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        return [line for line in result.stdout.splitlines() if line]

    def _compact_tool_result(self, result: str) -> str:
        if len(result) <= self.max_tool_result_chars:
            return result
        head_chars = self.max_tool_result_chars * 2 // 3
        tail_chars = self.max_tool_result_chars - head_chars
        omitted = len(result) - self.max_tool_result_chars
        return (
            result[:head_chars]
            + f"\n... <{omitted} chars omitted from tool history> ...\n"
            + result[-tail_chars:]
        )

    def _write_event(self, path: Path, event: str, **payload: Any) -> None:
        record = {"event": event, **payload}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _usage(self, response: Any) -> dict[str, int]:
        usage = getattr(response, "usage", None)
        return {
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        }

    def _tool_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": (
                        "Run a focused shell command from the repository root. "
                        "Use sed/rg for narrow reads and targeted tests only."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Shell command to run.",
                            }
                        },
                        "required": ["command"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "edit_files",
                    "description": (
                        "Transactionally perform 2-6 exact snippet replacements "
                        "spanning at least two production files. Multiple sequential "
                        "replacements may target one file. No file changes if any "
                        "old_text is absent."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "edits": {
                                "type": "array",
                                "minItems": 2,
                                "maxItems": 6,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "path": {"type": "string"},
                                        "old_text": {"type": "string"},
                                        "new_text": {"type": "string"},
                                    },
                                    "required": ["path", "old_text", "new_text"],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["edits"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "apply_patch",
                    "description": (
                        "Apply a small unified diff to production files in the "
                        "repository. Paths must be relative a/... and b/... paths."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patch": {
                                "type": "string",
                                "description": "Complete git-style unified diff.",
                            }
                        },
                        "required": ["patch"],
                        "additionalProperties": False,
                    },
                },
            },
        ]


__all__ = ["RepoBugAgentAdapter"]
