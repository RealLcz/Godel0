#!/usr/bin/env python3
"""Probe PR Mirror candidate generation against a vLLM endpoint.

This intentionally uses the existing single-file PRMirror implementation.
The PR context is derived from known-valid bug patches by inverting the hunk
signs so the prompt resembles a fixing PR.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


DEFAULT_SOURCES = [
    "output_proposer_200746/candidates/cand_fe4b7f157f49/candidate.json",
    "output_lm_rewrite_probe_200746/candidates/cand_f96507ce8f1b/candidate.json",
    "output_proposer_200746/candidates/cand_868d730e1c11/candidate.json",
]

MODULE_TEST_MAP = {
    "lib/ansible/module_utils/common/dict_transformations.py": [
        "test/units/module_utils/common/test_dict_transformations.py",
    ],
    "lib/ansible/utils/helpers.py": [
        "test/units/utils/test_helpers.py",
    ],
    "lib/ansible/utils/listify.py": [
        "test/units/utils/test_listify.py",
    ],
}


class VLLMPRMirrorAdapter:
    def __init__(self, host: str, port: str, model: str, max_tokens: int = 8192):
        import openai

        self.client = openai.OpenAI(base_url=f"http://{host}:{port}/v1", api_key="dummy")
        self.model = model
        self.max_tokens = max_tokens

    def mirror_pr(self, workspace_dir: str, target_file: str, request_text: str) -> str:
        from swesmith.lm_modify import extract_code_block

        target_path = Path(workspace_dir) / target_file
        original_source = target_path.read_text(encoding="utf-8")
        request = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a senior Python maintainer creating a bug-introducing "
                        "reverse patch for benchmark construction. Return code only."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{request_text}\n\n"
                        "Return exactly one fenced Python code block. The code block "
                        f"should contain the complete modified contents of {target_file}. "
                        "If you only change one function, still return the full file. "
                        "Do not return a diff, explanation, markdown prose, or comments "
                        "that reveal the bug."
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        try:
            response = self.client.chat.completions.create(
                **request,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except Exception:
            response = self.client.chat.completions.create(**request)
        text = response.choices[0].message.content or ""
        code = extract_code_block(text)
        modified_source = _materialize_response(original_source, code)
        target_path.write_text(modified_source, encoding="utf-8")
        return modified_source


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe PR Mirror candidate quality")
    parser.add_argument("--godel0-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--repo-pool", default="repo_pool")
    parser.add_argument("--repo-id", default="ansible")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--vllm-host", required=True)
    parser.add_argument("--vllm-port", default="8000")
    parser.add_argument("--output-dir", default=f"output_pr_mirror_probe_{int(time.time())}")
    parser.add_argument("--max-targets", type=int, default=3)
    parser.add_argument("--source-candidate", action="append", default=[])
    parser.add_argument("--test-python", default="")
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()

    os.environ.setdefault("HOME", "/tmp/godel0_home")
    os.environ.setdefault("ANSIBLE_LOCAL_TEMP", "/tmp/godel0_ansible_tmp")
    os.environ.setdefault("TMPDIR", "/tmp/godel0_tmp")
    Path(os.environ["HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["ANSIBLE_LOCAL_TEMP"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)

    root = Path(args.godel0_root).resolve()
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "initial_agent" / "src"))
    sys.path.insert(0, str(root))

    from godel0.proposer_trusted.candidate_validator import CandidateValidator
    from godel0.tasks.repo_pool import RepoPool
    from swesmith.engine import BugConstraints, BugGenerationPlan, RepoSpec as EngineRepoSpec, SWESmithEngine

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pool = RepoPool(root / args.repo_pool)
    spec = pool.get(args.repo_id)
    if spec is None:
        raise RuntimeError(f"repo not found in pool: {args.repo_id}")

    source_paths = args.source_candidate or DEFAULT_SOURCES
    source_candidates = [_load_candidate(root / p) for p in source_paths]
    source_candidates = source_candidates[: args.max_targets]

    agent = VLLMPRMirrorAdapter(args.vllm_host, args.vllm_port, args.model, max_tokens=args.max_tokens)
    engine = SWESmithEngine(agent_adapter=agent)
    repo_spec = EngineRepoSpec(
        repo_id=spec.repo_id,
        repo_path=str(spec.path),
        base_commit=spec.base_commit,
        test_command=spec.test_command,
    )
    validator = CandidateValidator(
        workspace_root=output_dir / "validator_ws",
        test_timeout_sec=90,
        max_patch_lines=160,
        forbid_test_file_edits=True,
    )

    results = []
    for idx, src in enumerate(source_candidates, start=1):
        target_file = src["target_file"]
        target_symbol = src.get("target_symbol", "")
        print(f"\n[{idx}/{len(source_candidates)}] PR Mirror {target_file}::{target_symbol}", flush=True)

        plan = BugGenerationPlan(
            plan_id=f"pr_mirror_probe_{idx:04d}",
            target_repo_id=spec.repo_id,
            target_base_commit=spec.base_commit,
            target_file=target_file,
            target_symbol=target_symbol,
            strategy="pr_mirror",
            operator="pr_mirror",
            constraints=BugConstraints(max_modified_lines=160),
            seed=3000 + idx,
        )
        plan.pr_diff = _invert_patch_hunks(src["bug_patch"])

        candidates = engine.generate(
            plan=plan,
            node_code_dir=str(output_dir),
            repo_spec=repo_spec,
            output_dir=str(output_dir / "candidates"),
        )
        if not candidates:
            print("  generated=0", flush=True)
            results.append(
                {
                    "source_candidate_id": src["candidate_id"],
                    "target": f"{target_file}::{target_symbol}",
                    "generated": 0,
                }
            )
            continue

        for cand in candidates:
            print(f"  generated {cand.candidate_id}: {len(cand.bug_patch)} chars", flush=True)
            print(cand.bug_patch[:2500], flush=True)
            test_files = MODULE_TEST_MAP.get(target_file, [])
            test_command = spec.test_command + " " + " ".join(test_files) if test_files else spec.test_command
            if args.test_python:
                test_command = test_command.replace("python3.11", args.test_python)

            report = validator.validate(
                candidate_patch=cand.bug_patch,
                repo_path=Path(spec.path),
                base_commit=spec.base_commit,
                test_command=test_command,
                candidate_id=cand.candidate_id,
                repo_id=spec.repo_id,
                target_file=target_file,
                target_symbol=target_symbol,
                operator="pr_mirror",
            )
            row = {
                "source_candidate_id": src["candidate_id"],
                "candidate_id": cand.candidate_id,
                "target": f"{target_file}::{target_symbol}",
                "patch_chars": len(cand.bug_patch),
                "patch": cand.bug_patch,
                "passed": report.passed,
                "patch_applied": report.patch_applied,
                "syntax_valid": report.syntax_valid,
                "f2p_tests": report.f2p_tests,
                "p2p_tests": report.p2p_tests,
                "reverse_restored": report.reverse_restored,
                "rejection_reasons": report.rejection_reasons,
            }
            results.append(row)
            print(
                f"  validation passed={report.passed} f2p={len(report.f2p_tests)} "
                f"p2p={len(report.p2p_tests)} reasons={report.rejection_reasons}",
                flush=True,
            )

    report_path = output_dir / "pr_mirror_probe_report.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nReport: {report_path}", flush=True)
    return 0


def _load_candidate(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = ["candidate_id", "target_file", "bug_patch"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"{path} is missing fields: {missing}")
    return data


def _invert_patch_hunks(patch: str) -> str:
    lines = []
    in_hunk = False
    for line in patch.splitlines():
        if line.startswith("@@"):
            in_hunk = True
            lines.append(line)
        elif in_hunk and line.startswith("+") and not line.startswith("+++"):
            lines.append("-" + line[1:])
        elif in_hunk and line.startswith("-") and not line.startswith("---"):
            lines.append("+" + line[1:])
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


def _materialize_response(original_source: str, code: str) -> str:
    code = _strip_markdown_language(code)
    if not code:
        return original_source
    if code.lstrip().startswith(("diff --git", "--- ", "+++ ", "@@")):
        return original_source

    try:
        compile(code, "<pr_mirror_response>", "exec")
    except SyntaxError:
        return original_source

    if _looks_like_full_file(original_source, code):
        return _ensure_trailing_newline(code)

    spliced = _splice_single_symbol(original_source, code)
    if spliced:
        return _ensure_trailing_newline(spliced)
    return _ensure_trailing_newline(code)


def _strip_markdown_language(code: str) -> str:
    code = code.strip()
    code = re.sub(r"^python\s*\n", "", code, flags=re.IGNORECASE)
    return code.strip()


def _looks_like_full_file(original_source: str, code: str) -> bool:
    original_lines = len(original_source.splitlines())
    code_lines = len(code.splitlines())
    if code_lines >= max(20, original_lines // 2):
        return True
    return bool(re.search(r"^(from|import)\s+", code, flags=re.MULTILINE))


def _splice_single_symbol(original_source: str, fragment: str) -> str:
    import ast

    try:
        original_tree = ast.parse(original_source)
        fragment_tree = ast.parse(fragment)
    except SyntaxError:
        return ""

    fragment_defs = [
        node
        for node in fragment_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    if len(fragment_defs) != 1:
        return ""

    target = fragment_defs[0]
    for node in ast.walk(original_tree):
        if not isinstance(node, type(target)):
            continue
        if getattr(node, "name", None) != target.name:
            continue
        start = getattr(node, "lineno", 0)
        end = getattr(node, "end_lineno", 0)
        if not start or not end:
            continue
        original_lines = original_source.splitlines()
        fragment_lines = fragment.splitlines()
        return "\n".join(original_lines[: start - 1] + fragment_lines + original_lines[end:])
    return ""


def _ensure_trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
