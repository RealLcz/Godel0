#!/usr/bin/env python3
"""Probe a high-coverage LM Rewrite across one connected repository slice.

The probe deliberately stays outside the default proposer loop. It masks at
least a requested fraction of function implementation lines in six related
production files, asks one model call to reimplement every masked body, then
measures the resulting patch instead of assuming that a large rewrite is a
useful bug.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Iterable, Optional


STUB_MARKER = "LM_REWRITE_STUB"

REWRITE_DOMAINS: dict[str, dict[str, Any]] = {
    "yaml_pipeline": {
        "description": "YAML loading, construction, source metadata, and dumping",
        "files": [
            "lib/ansible/parsing/dataloader.py",
            "lib/ansible/parsing/utils/yaml.py",
            "lib/ansible/parsing/yaml/loader.py",
            "lib/ansible/parsing/yaml/constructor.py",
            "lib/ansible/parsing/yaml/objects.py",
            "lib/ansible/parsing/yaml/dumper.py",
        ],
        "tests": [
            "test/units/parsing/test_dataloader.py",
            "test/units/parsing/utils/test_yaml.py",
            "test/units/parsing/yaml/test_loader.py",
            "test/units/parsing/yaml/test_constructor.py",
            "test/units/parsing/yaml/test_objects.py",
            "test/units/parsing/yaml/test_dumper.py",
        ],
        "contract": (
            "Bytes and text flow from DataLoader through JSON/YAML selection, "
            "AnsibleLoader, AnsibleConstructor, position-aware YAML objects, and "
            "AnsibleDumper without losing value, vault, unsafe-value, or source "
            "metadata semantics."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--godel0-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--repo-pool", default="repo_pool")
    parser.add_argument("--repo-id", default="ansible")
    parser.add_argument("--domain", choices=sorted(REWRITE_DOMAINS), default="yaml_pipeline")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--vllm-host", required=True)
    parser.add_argument("--vllm-port", default="8000")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--rewrite-ratio", type=float, default=0.60)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=32768)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--agent-timeout", type=int, default=1200)
    parser.add_argument("--test-timeout", type=int, default=300)
    parser.add_argument(
        "--reuse-response",
        default="",
        help="Skip generation and evaluate an existing raw_response.txt",
    )
    parser.add_argument("--run-solver", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.rewrite_ratio <= 1:
        raise ValueError("--rewrite-ratio must be in (0, 1]")

    root = Path(args.godel0_root).resolve()
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "initial_agent" / "src"))
    sys.path.insert(0, str(root))

    os.environ["VLLM_HOST"] = args.vllm_host
    os.environ["VLLM_PORT"] = str(args.vllm_port)
    os.environ.pop("QWEN_API_BASE_URL", None)

    from experiment_adapters.common_agent_adapter import CommonAgentAdapter
    from godel0.proposer_trusted.candidate_validator import CandidateValidator
    from godel0.tasks.repo_pool import RepoPool
    from scripts import run_repo_level_closed_loop as closed_loop
    from swesmith.candidate import CandidateArtifact
    from swesmith.patch_utils import make_git_diff

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_tmp = output_dir / "runtime_tmp"
    (runtime_tmp / "ansible_local").mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(runtime_tmp)
    os.environ["ANSIBLE_LOCAL_TEMP"] = str(runtime_tmp / "ansible_local")

    pool = RepoPool((root / args.repo_pool).resolve())
    spec = pool.get(args.repo_id)
    if spec is None:
        raise RuntimeError(f"Repository not found: {args.repo_id}")
    source_repo = Path(spec.path)
    if not source_repo.is_absolute():
        source_repo = (root / source_repo).resolve()

    domain = REWRITE_DOMAINS[args.domain]
    original_sources = {
        path: _read_file_at_commit(source_repo, spec.base_commit, path)
        for path in domain["files"]
    }
    masked_sources: dict[str, str] = {}
    mask_manifest: dict[str, dict[str, Any]] = {}
    for path, source in original_sources.items():
        masked, manifest = mask_function_implementations(
            source,
            target_ratio=args.rewrite_ratio,
        )
        if manifest["masked_ratio"] + 1e-9 < args.rewrite_ratio:
            raise RuntimeError(
                f"Cannot mask {args.rewrite_ratio:.0%} of implementation lines in {path}"
            )
        masked_sources[path] = masked
        mask_manifest[path] = manifest

    masked_dir = output_dir / "masked_inputs"
    for path, source in masked_sources.items():
        target = masked_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
    (output_dir / "mask_manifest.json").write_text(
        json.dumps(mask_manifest, indent=2),
        encoding="utf-8",
    )

    prompt = build_rewrite_prompt(domain, masked_sources, args.rewrite_ratio)
    (output_dir / "rewrite_prompt.txt").write_text(prompt, encoding="utf-8")
    dependency_evidence = _dependency_evidence(
        domain["files"],
        original_sources,
        closed_loop,
    )
    report: dict[str, Any] = {
        "configuration": {
            "model": args.model,
            "repo_id": spec.repo_id,
            "base_commit": spec.base_commit,
            "domain": args.domain,
            "file_count": len(domain["files"]),
            "rewrite_ratio": args.rewrite_ratio,
            "attempts": args.attempts,
            "temperature": args.temperature,
            "max_output_tokens": args.max_output_tokens,
            "run_solver": args.run_solver,
            "reuse_response": args.reuse_response,
        },
        "files": list(domain["files"]),
        "tests": list(domain["tests"]),
        "mask_manifest": mask_manifest,
        "static_dependency_evidence": dependency_evidence,
        "prompt_chars": len(prompt),
        "attempt_results": [],
        "started_at": _utc_now(),
    }

    def checkpoint() -> None:
        (output_dir / "repo_lm_rewrite_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    checkpoint()
    validator = CandidateValidator(
        workspace_root=output_dir / "validator",
        test_timeout_sec=args.test_timeout,
        max_patch_lines=10000,
        forbid_test_file_edits=True,
    )
    solver_adapter = CommonAgentAdapter() if args.run_solver else None
    test_command = closed_loop._test_command(spec.test_command, domain["tests"])

    for attempt_index in range(max(0, args.attempts)):
        attempt_number = attempt_index + 1
        attempt_dir = output_dir / "attempts" / f"attempt_{attempt_number:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        row: dict[str, Any] = {
            "attempt": attempt_number,
            "started_at": _utc_now(),
        }
        report["attempt_results"].append(row)
        checkpoint()
        print(
            f"\n[{attempt_number}/{args.attempts}] rewriting {len(domain['files'])} "
            f"files with target ratio {args.rewrite_ratio:.0%}",
            flush=True,
        )

        generation_started = time.monotonic()
        try:
            if args.reuse_response:
                reused_path = Path(args.reuse_response).resolve()
                raw_response = reused_path.read_text(encoding="utf-8")
                response_metadata = {
                    "reused": True,
                    "source_path": str(reused_path),
                }
            else:
                raw_response, response_metadata = request_rewrite(
                    host=args.vllm_host,
                    port=str(args.vllm_port),
                    model=args.model,
                    prompt=prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_output_tokens,
                    timeout=args.request_timeout,
                    seed=7300 + attempt_index,
                )
        except Exception as exc:
            row.update(
                {
                    "generation_runtime_sec": round(time.monotonic() - generation_started, 3),
                    "result": "model_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            checkpoint()
            print(f"model error: {row['error']}", flush=True)
            continue

        (attempt_dir / "raw_response.txt").write_text(raw_response, encoding="utf-8")
        model_sources, parse_metadata = parse_file_blocks(raw_response)
        expected_files = list(domain["files"])
        missing_files = [path for path in expected_files if path not in model_sources]
        extra_files = sorted(set(model_sources) - set(expected_files))
        model_sources = {
            path: _normalize_source(model_sources[path])
            for path in expected_files
            if path in model_sources
        }
        for path, source in model_sources.items():
            target = attempt_dir / "model_rewritten" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source, encoding="utf-8")

        rewritten_sources: dict[str, str] = {}
        transplant_metadata: dict[str, Any] = {}
        transplant_errors: dict[str, str] = {}
        for path, model_source in model_sources.items():
            try:
                rewritten, metadata = transplant_selected_function_bodies(
                    original_sources[path],
                    model_source,
                    mask_manifest[path],
                )
            except (SyntaxError, ValueError) as exc:
                transplant_errors[path] = f"{type(exc).__name__}: {exc}"
                continue
            rewritten_sources[path] = rewritten
            transplant_metadata[path] = metadata
            target = attempt_dir / "rewritten" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(rewritten, encoding="utf-8")

        syntax_errors = _syntax_errors(rewritten_sources)
        remaining_stub_files = [
            path for path, source in rewritten_sources.items() if STUB_MARKER in source
        ]
        rewrite_metrics = {
            path: compute_rewrite_metrics(
                original_sources[path],
                rewritten_sources[path],
            )
            for path in expected_files
            if path in rewritten_sources
        }
        total_implementation_lines = sum(
            metrics["implementation_line_count"]
            for metrics in rewrite_metrics.values()
        )
        changed_implementation_lines = sum(
            metrics["changed_implementation_line_count"]
            for metrics in rewrite_metrics.values()
        )
        aggregate_actual_ratio = (
            changed_implementation_lines / total_implementation_lines
            if total_implementation_lines
            else 0.0
        )
        actual_ratio_gate = bool(
            not missing_files
            and aggregate_actual_ratio + 1e-9 >= args.rewrite_ratio
        )

        patch_parts = [
            make_git_diff(original_sources[path], rewritten_sources[path], filename=path)
            for path in expected_files
            if path in rewritten_sources and original_sources[path] != rewritten_sources[path]
        ]
        bug_patch = "".join(part for part in patch_parts if part)
        changed_files = [
            path
            for path in expected_files
            if path in rewritten_sources and original_sources[path] != rewritten_sources[path]
        ]
        patch_path = attempt_dir / "bug.patch"
        patch_path.write_text(bug_patch, encoding="utf-8")
        generation_complete = bool(
            not missing_files
            and not extra_files
            and 5 <= len(changed_files) <= 6
            and not syntax_errors
            and not transplant_errors
            and not remaining_stub_files
            and bug_patch
        )
        row.update(
            {
                "generation_runtime_sec": round(time.monotonic() - generation_started, 3),
                "response_chars": len(raw_response),
                "response_metadata": response_metadata,
                "parse_metadata": parse_metadata,
                "missing_files": missing_files,
                "extra_files": extra_files,
                "changed_files": changed_files,
                "syntax_errors": syntax_errors,
                "transplant_errors": transplant_errors,
                "transplant_metadata": transplant_metadata,
                "remaining_stub_files": remaining_stub_files,
                "rewrite_metrics": rewrite_metrics,
                "aggregate_changed_implementation_ratio": aggregate_actual_ratio,
                "actual_ratio_gate": actual_ratio_gate,
                "generation_complete": generation_complete,
                "patch_chars": len(bug_patch),
            }
        )
        checkpoint()
        print(
            f"generation complete={generation_complete} files={len(changed_files)} "
            f"actual_ratio_gate={actual_ratio_gate} patch_chars={len(bug_patch)}",
            flush=True,
        )
        if not generation_complete:
            row["result"] = "invalid_rewrite_output"
            row["quality_label"] = classify_quality(row)
            checkpoint()
            continue

        candidate_digest = hashlib.sha256(bug_patch.encode("utf-8")).hexdigest()[:12]
        candidate = CandidateArtifact(
            candidate_id=f"cand_repo_rewrite_{candidate_digest}",
            plan_id=f"repo_rewrite_{args.domain}_{attempt_number}",
            strategy="repo_lm_rewrite",
            operator="masked_repository_reimplementation",
            target_file=expected_files[0],
            target_symbol="",
            bug_patch=bug_patch,
            mutation_site={
                "masked_symbols": {
                    path: mask_manifest[path]["selected_symbols"]
                    for path in expected_files
                }
            },
            seed=7300 + attempt_index,
            before_snippet="",
            after_snippet="",
            generation_metadata={
                "rewrite_ratio_target": args.rewrite_ratio,
                "mask_manifest": mask_manifest,
                "rewrite_metrics": rewrite_metrics,
                "static_dependency_evidence": dependency_evidence,
            },
            modified_files=changed_files,
            modified_entities=[
                symbol
                for path in expected_files
                for symbol in mask_manifest[path]["selected_symbols"]
            ],
        )
        candidate.save(str(attempt_dir / "candidate"))
        problem_statement = (
            "A regression affects Ansible's YAML data pipeline. Loading, construction, "
            "source-aware values, and serialization no longer preserve one consistent "
            "behavioral contract. Diagnose the repository-level cause and restore the "
            "expected behavior without changing tests."
        )

        validation_started = time.monotonic()
        task = closed_loop._validate_and_package(
            candidate=candidate,
            phase="rewrite_probe",
            domain_id=args.domain,
            problem_statement=problem_statement,
            test_files=domain["tests"],
            source_repo=source_repo,
            base_commit=spec.base_commit,
            test_prefix=spec.test_command,
            validator=validator,
            output_dir=attempt_dir,
            test_timeout=args.test_timeout,
            source_trajectory_ids=[],
        )
        validation_path = attempt_dir / "tasks" / candidate.candidate_id / "validation.json"
        validation_data = _read_json(validation_path)
        row["validation_runtime_sec"] = round(time.monotonic() - validation_started, 3)
        row["candidate_id"] = candidate.candidate_id
        row["validation"] = validation_data
        row["validation_passed"] = bool(validation_data.get("passed"))
        row["ablation_strict_repo_level"] = bool(
            task and task.get("strict_repo_level")
        )
        row["strict_repo_level"] = row["ablation_strict_repo_level"]

        if row["validation_passed"]:
            coupling_started = time.monotonic()
            row["behavioral_coupling"] = measure_behavioral_coupling(
                source_repo=source_repo,
                base_commit=spec.base_commit,
                bug_patch=bug_patch,
                full_f2p=list(validation_data.get("f2p_tests") or []),
                test_command=test_command,
                timeout=args.test_timeout,
                closed_loop=closed_loop,
            )
            row["coupling_runtime_sec"] = round(time.monotonic() - coupling_started, 3)
            coupling = row["behavioral_coupling"]
            row["strict_repo_level"] = bool(
                row["ablation_strict_repo_level"]
                and coupling.get("behavior_overlap_graph_connected")
                and not coupling.get("standalone_inert_files")
            )
        row["experiment_accepted"] = bool(
            row["strict_repo_level"] and row["actual_ratio_gate"]
        )

        if task and solver_adapter is not None and row["experiment_accepted"]:
            print(
                f"validation passed; running solver (strict={task.get('strict_repo_level')})",
                flush=True,
            )
            solver_result = closed_loop._run_solver(
                task=task,
                phase="rewrite_probe",
                source_repo=source_repo,
                agent_src=root / "initial_agent" / "src",
                adapter=solver_adapter,
                model=args.model,
                output_dir=attempt_dir,
                agent_timeout=args.agent_timeout,
                test_timeout=args.test_timeout,
            )
            row["solver_result"] = solver_result
        elif solver_adapter is not None:
            row["solver_skipped_reason"] = (
                "candidate_failed_ratio_or_semantic_coupling_gate"
            )

        row["result"] = "evaluated"
        row["quality_label"] = classify_quality(row)
        row["finished_at"] = _utc_now()
        checkpoint()
        print(
            f"quality={row['quality_label']} validation={row['validation_passed']} "
            f"strict={row['strict_repo_level']}",
            flush=True,
        )

    report["finished_at"] = _utc_now()
    checkpoint()
    print(f"\nReport: {output_dir / 'repo_lm_rewrite_report.json'}", flush=True)
    return 0


def mask_function_implementations(
    source: str,
    *,
    target_ratio: float,
) -> tuple[str, dict[str, Any]]:
    """Replace whole function bodies until target implementation coverage is met."""
    tree = ast.parse(source)
    parent: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    candidates: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        ancestor = parent.get(node)
        nested = False
        while ancestor is not None:
            if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
                nested = True
                break
            ancestor = parent.get(ancestor)
        if nested or not node.body:
            continue

        body = list(node.body)
        docstring_node: Optional[ast.AST] = None
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            docstring_node = body.pop(0)
        if not body:
            continue
        start = int(body[0].lineno)
        end = int(getattr(node, "end_lineno", body[-1].end_lineno or body[-1].lineno))
        implementation_lines = _code_line_numbers(source, start, end)
        if not implementation_lines:
            continue
        candidates.append(
            {
                "node": node,
                "start": start,
                "end": end,
                "implementation_lines": implementation_lines,
                "qualname": _qualname(node, parent),
                "docstring_preserved": docstring_node is not None,
            }
        )

    total_lines = sum(len(item["implementation_lines"]) for item in candidates)
    if not total_lines:
        raise ValueError("source has no rewritable function implementation lines")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in candidates:
        grouped.setdefault(item["qualname"], []).append(item)

    selected: list[dict[str, Any]] = []
    selected_lines = 0
    for group in sorted(
        grouped.values(),
        key=lambda values: (
            -sum(len(value["implementation_lines"]) for value in values),
            min(value["start"] for value in values),
        ),
    ):
        # Conditional class definitions often expose the same method through a
        # C-extension and pure-Python branch. Rewrite all variants together so
        # the selected code cannot be inactive in the current environment.
        selected.extend(group)
        selected_lines += sum(len(item["implementation_lines"]) for item in group)
        if selected_lines / total_lines + 1e-9 >= target_ratio:
            break

    lines = source.splitlines(keepends=True)
    for item in sorted(selected, key=lambda value: value["start"], reverse=True):
        original_line = lines[item["start"] - 1]
        indent = original_line[: len(original_line) - len(original_line.lstrip())]
        newline = "\r\n" if original_line.endswith("\r\n") else "\n"
        replacement = f'{indent}raise NotImplementedError("{STUB_MARKER}"){newline}'
        lines[item["start"] - 1 : item["end"]] = [replacement]
    masked_source = "".join(lines)
    ast.parse(masked_source)
    return masked_source, {
        "implementation_line_count": total_lines,
        "masked_implementation_line_count": selected_lines,
        "masked_ratio": selected_lines / total_lines,
        "selected_symbol_count": len({item["qualname"] for item in selected}),
        "selected_definition_count": len(selected),
        "selected_symbols": list(dict.fromkeys(item["qualname"] for item in selected)),
        "selected_ranges": [
            {
                "symbol": item["qualname"],
                "start": item["start"],
                "end": item["end"],
                "implementation_lines": len(item["implementation_lines"]),
                "docstring_preserved": item["docstring_preserved"],
            }
            for item in selected
        ],
    }


def compute_rewrite_metrics(original: str, rewritten: str) -> dict[str, Any]:
    """Measure how many original function-implementation lines changed."""
    implementation_lines = _all_function_implementation_lines(original)
    original_lines = original.splitlines()
    rewritten_lines = rewritten.splitlines()
    changed_original_lines: set[int] = set()
    matcher = difflib.SequenceMatcher(
        None,
        original_lines,
        rewritten_lines,
        autojunk=False,
    )
    for tag, old_start, old_end, _new_start, _new_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed_original_lines.update(range(old_start + 1, old_end + 1))
    changed_implementation = implementation_lines & changed_original_lines
    file_code_lines = _code_line_numbers(original, 1, len(original_lines))
    changed_file_code = file_code_lines & changed_original_lines
    ratio = (
        len(changed_implementation) / len(implementation_lines)
        if implementation_lines
        else 0.0
    )
    return {
        "original_line_count": len(original_lines),
        "rewritten_line_count": len(rewritten_lines),
        "implementation_line_count": len(implementation_lines),
        "file_code_line_count": len(file_code_lines),
        "changed_original_line_count": len(changed_original_lines),
        "changed_implementation_line_count": len(changed_implementation),
        "changed_implementation_ratio": ratio,
        "changed_file_code_line_count": len(changed_file_code),
        "changed_file_code_ratio": (
            len(changed_file_code) / len(file_code_lines) if file_code_lines else 0.0
        ),
    }


def transplant_selected_function_bodies(
    original: str,
    model_source: str,
    manifest: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Merge only selected generated bodies into the original file structure."""
    original_records = _outer_function_records(original)
    model_records = {
        (record["qualname"], record["occurrence"]): record
        for record in _outer_function_records(model_source)
    }
    selected_ranges = list(manifest.get("selected_ranges") or [])
    selected_keys: list[tuple[str, int]] = []
    replacements: list[tuple[int, int, str]] = []
    missing: list[str] = []

    for selected in selected_ranges:
        qualname = str(selected.get("symbol") or "")
        start = int(selected.get("start") or 0)
        original_record = next(
            (
                record
                for record in original_records
                if record["qualname"] == qualname and record["body_start"] == start
            ),
            None,
        )
        if original_record is None:
            missing.append(f"original:{qualname}@{start}")
            continue
        key = (qualname, int(original_record["occurrence"]))
        model_record = model_records.get(key)
        if model_record is None:
            missing.append(f"model:{qualname}#{key[1]}")
            continue
        generated_body = _record_body_source(model_source, model_record)
        if not generated_body or STUB_MARKER in generated_body:
            missing.append(f"body:{qualname}#{key[1]}")
            continue
        original_lines = original.splitlines(keepends=True)
        target_line = original_lines[int(original_record["body_start"]) - 1]
        indent = target_line[: len(target_line) - len(target_line.lstrip())]
        replacements.append(
            (
                int(original_record["body_start"]),
                int(original_record["body_end"]),
                _reindent_block(generated_body, indent),
            )
        )
        selected_keys.append(key)

    if missing:
        raise ValueError("cannot transplant selected bodies: " + ", ".join(missing))
    if len(replacements) != len(selected_ranges):
        raise ValueError("not every selected function body was transplanted")

    lines = original.splitlines(keepends=True)
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start - 1 : end] = replacement.splitlines(keepends=True)
    rewritten = "".join(lines)
    ast.parse(rewritten)
    return rewritten, {
        "transplanted_definition_count": len(replacements),
        "transplanted_keys": [f"{name}#{occurrence}" for name, occurrence in selected_keys],
        "preserved_non_body_structure": True,
    }


def build_rewrite_prompt(
    domain: dict[str, Any],
    masked_sources: dict[str, str],
    target_ratio: float,
) -> str:
    file_sections = []
    for path in domain["files"]:
        file_sections.append(
            f'<input_file path="{path}">\n{masked_sources[path]}</input_file>'
        )
    return (
        "You are performing SWE-smith-style LM Rewrite over one connected repository "
        "slice. The original implementations of selected functions were removed and "
        f"replaced by {STUB_MARKER}. Reimplement every stub correctly. Do not "
        "intentionally inject a bug; natural reimplementation mistakes are the mutation.\n\n"
        f"Shared repository contract:\n{domain['contract']}\n\n"
        "Requirements:\n"
        f"- You received exactly {len(domain['files'])} related production files.\n"
        f"- At least {target_ratio:.0%} of each file's function implementation lines "
        "were hidden before this request.\n"
        "- Remove every LM_REWRITE_STUB and preserve public names, signatures, imports, "
        "inheritance, module constants, and license headers.\n"
        "- Reconstruct behavior from signatures, docstrings, surrounding code, and the "
        "cross-file contract. Do not add tests, placeholders, compatibility shims, or "
        "explanations inside the files.\n"
        "- Return every complete file, including unchanged surrounding text. Never use "
        "ellipsis or say that content is unchanged.\n\n"
        "Output format (repeat once for every input path, with no Markdown fences):\n"
        '<file path="repository/relative/path.py">\nFULL FILE CONTENT\n</file>\n\n'
        + "\n\n".join(file_sections)
    )


def parse_file_blocks(response: str) -> tuple[dict[str, str], dict[str, Any]]:
    pattern = re.compile(
        r'<file\s+path=(?:"([^"]+)"|\'([^\']+)\')\s*>\s*\n?(.*?)\n?</file>',
        flags=re.DOTALL,
    )
    files: dict[str, str] = {}
    duplicates: list[str] = []
    for match in pattern.finditer(response):
        path = (match.group(1) or match.group(2) or "").strip()
        content = _strip_optional_fence(match.group(3))
        if path in files:
            duplicates.append(path)
        files[path] = content
    return files, {
        "parsed_block_count": len(files),
        "duplicate_paths": duplicates,
        "unclosed_file_tag": response.count("<file ") > response.count("</file>"),
    }


def request_rewrite(
    *,
    host: str,
    port: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    seed: int,
) -> tuple[str, dict[str, Any]]:
    import openai

    client = openai.OpenAI(
        base_url=f"http://{host}:{port}/v1",
        api_key="dummy",
        timeout=timeout,
    )
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You reimplement missing Python functions and emit complete files "
                    "in an exact machine-readable tagged format."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    try:
        response = client.chat.completions.create(
            **kwargs,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except Exception:
        response = client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    content = str(choice.message.content or "")
    usage = getattr(response, "usage", None)
    if hasattr(usage, "model_dump"):
        usage_data = usage.model_dump()
    else:
        usage_data = str(usage or "")
    return content, {
        "finish_reason": str(choice.finish_reason or ""),
        "usage": usage_data,
    }


def measure_behavioral_coupling(
    *,
    source_repo: Path,
    base_commit: str,
    bug_patch: str,
    full_f2p: list[str],
    test_command: str,
    timeout: int,
    closed_loop: Any,
) -> dict[str, Any]:
    from swesmith.repo_level import (
        RepositoryWorkspace,
        apply_repository_patch,
        split_patch_by_file,
    )

    blocks = split_patch_by_file(bug_patch)
    full_f2p_set = set(full_f2p)
    only_file_f2p: dict[str, list[str]] = {}
    setup_valid: dict[str, bool] = {}
    for path, block in blocks:
        with RepositoryWorkspace(str(source_repo), base_commit) as workspace:
            setup_valid[path] = apply_repository_patch(workspace, block)
            if not setup_valid[path]:
                only_file_f2p[path] = []
                continue
            result = closed_loop._run_command(workspace, test_command, timeout)
            _passed, failed = closed_loop._pytest_status_sets(result, workspace)
            only_file_f2p[path] = sorted(full_f2p_set & failed)

    paths = [path for path, _block in blocks]
    overlap_edges: list[dict[str, Any]] = []
    adjacency = {path: set() for path in paths}
    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            shared = sorted(set(only_file_f2p[left]) & set(only_file_f2p[right]))
            if shared:
                overlap_edges.append({"left": left, "right": right, "tests": shared})
                adjacency[left].add(right)
                adjacency[right].add(left)
    reached: set[str] = set()
    if paths:
        reached.add(paths[0])
        frontier = [paths[0]]
        while frontier:
            current = frontier.pop()
            for neighbor in adjacency[current] - reached:
                reached.add(neighbor)
                frontier.append(neighbor)
    shared_all = set(full_f2p)
    for path in paths:
        shared_all &= set(only_file_f2p[path])
    return {
        "only_file_f2p": only_file_f2p,
        "standalone_inert_files": [path for path in paths if not only_file_f2p[path]],
        "overlap_edges": overlap_edges,
        "behavior_overlap_graph_connected": bool(paths and len(reached) == len(paths)),
        "shared_f2p_across_all_files": sorted(shared_all),
        "setup_valid": setup_valid,
    }


def classify_quality(row: dict[str, Any]) -> str:
    if not row.get("generation_complete"):
        return "invalid_model_output"
    if not row.get("validation_passed"):
        return "not_a_valid_f2p_task"
    validation = row.get("validation") or {}
    clean_count = len(validation.get("clean_passed_tests") or [])
    f2p_count = len(validation.get("f2p_tests") or [])
    if clean_count and f2p_count / clean_count > 0.5:
        return "over_destructive_regression"
    coupling = row.get("behavioral_coupling") or {}
    if coupling.get("standalone_inert_files"):
        return "contains_behaviorally_inert_files"
    if not coupling.get("behavior_overlap_graph_connected"):
        return "independent_multi_bug_rewrite"
    if not row.get("strict_repo_level"):
        return "has_partial_repair_shortcut"
    if not row.get("actual_ratio_gate"):
        return "valid_task_below_requested_rewrite_ratio"
    solver_result = row.get("solver_result") or {}
    if solver_result:
        return "solver_resolved" if solver_result.get("resolved") else "solver_failed_strict_candidate"
    return "strict_candidate_not_solver_tested"


def _all_function_implementation_lines(source: str) -> set[int]:
    tree = ast.parse(source)
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or not node.body:
            continue
        body = list(node.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body.pop(0)
        if not body:
            continue
        start = int(body[0].lineno)
        end = int(getattr(node, "end_lineno", body[-1].end_lineno or body[-1].lineno))
        lines.update(_code_line_numbers(source, start, end))
    return lines


def _outer_function_records(source: str) -> list[dict[str, Any]]:
    tree = ast.parse(source)
    parent: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node
    records: list[dict[str, Any]] = []
    occurrences: dict[str, int] = {}
    nodes = sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ),
        key=lambda node: node.lineno,
    )
    for node in nodes:
        ancestor = parent.get(node)
        nested = False
        while ancestor is not None:
            if isinstance(ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)):
                nested = True
                break
            ancestor = parent.get(ancestor)
        if nested or not node.body:
            continue
        body = list(node.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body.pop(0)
        if not body:
            continue
        qualname = _qualname(node, parent)
        occurrence = occurrences.get(qualname, 0)
        occurrences[qualname] = occurrence + 1
        records.append(
            {
                "qualname": qualname,
                "occurrence": occurrence,
                "body_start": int(body[0].lineno),
                "body_end": int(
                    getattr(node, "end_lineno", body[-1].end_lineno or body[-1].lineno)
                ),
            }
        )
    return records


def _record_body_source(source: str, record: dict[str, Any]) -> str:
    lines = source.splitlines(keepends=True)
    start = int(record["body_start"])
    end = int(record["body_end"])
    return "".join(lines[start - 1 : end])


def _reindent_block(source: str, indent: str) -> str:
    dedented = textwrap.dedent(source).strip("\r\n")
    if not dedented:
        return ""
    lines = dedented.splitlines()
    return "\n".join(f"{indent}{line}" if line else "" for line in lines) + "\n"


def _code_line_numbers(source: str, start: int, end: int) -> set[int]:
    physical = source.splitlines()
    return {
        number
        for number in range(start, min(end, len(physical)) + 1)
        if physical[number - 1].strip()
        and not physical[number - 1].lstrip().startswith("#")
    }


def _qualname(node: ast.AST, parent: dict[ast.AST, ast.AST]) -> str:
    names = [str(getattr(node, "name", "<anonymous>"))]
    ancestor = parent.get(node)
    while ancestor is not None:
        if isinstance(ancestor, ast.ClassDef):
            names.append(ancestor.name)
        ancestor = parent.get(ancestor)
    return ".".join(reversed(names))


def _syntax_errors(sources: dict[str, str]) -> dict[str, str]:
    errors: dict[str, str] = {}
    for path, source in sources.items():
        try:
            ast.parse(source, filename=path)
        except SyntaxError as exc:
            errors[path] = f"line {exc.lineno}: {exc.msg}"
    return errors


def _dependency_evidence(
    paths: Iterable[str],
    sources: dict[str, str],
    closed_loop: Any,
) -> dict[str, Any]:
    ordered = list(paths)
    edges: list[dict[str, str]] = []
    adjacency = {path: set() for path in ordered}
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            edge = closed_loop._static_import_dependency(
                left,
                sources[left],
                right,
                sources[right],
            )
            if edge:
                edges.append(edge)
                adjacency[left].add(right)
                adjacency[right].add(left)
    reached: set[str] = set()
    if ordered:
        reached.add(ordered[0])
        frontier = [ordered[0]]
        while frontier:
            current = frontier.pop()
            for neighbor in adjacency[current] - reached:
                reached.add(neighbor)
                frontier.append(neighbor)
    return {
        "connected": bool(ordered and len(reached) == len(ordered)),
        "edges": edges,
    }


def _read_file_at_commit(repo: Path, commit: str, relative_path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative_path}"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Cannot read {relative_path} at {commit}: {result.stderr.strip()}")
    return result.stdout


def _normalize_source(source: str) -> str:
    source = source.replace("\r\n", "\n")
    return source if source.endswith("\n") else source + "\n"


def _strip_optional_fence(content: str) -> str:
    value = content.strip("\n")
    lines = value.splitlines()
    if lines and re.fullmatch(r"```(?:python|py)?", lines[0].strip(), re.IGNORECASE):
        lines.pop(0)
        if lines and lines[-1].strip() == "```":
            lines.pop()
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
