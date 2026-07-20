#!/usr/bin/env python3
"""Small LM Rewrite probe against an existing vLLM endpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


TARGETS = [
    ("lib/ansible/module_utils/common/dict_transformations.py", "dict_merge"),
    ("lib/ansible/module_utils/common/dict_transformations.py", "recursive_diff"),
    ("lib/ansible/module_utils/common/dict_transformations.py", "camel_dict_to_snake_dict"),
    ("lib/ansible/utils/helpers.py", "pct_to_int"),
    ("lib/ansible/utils/listify.py", "listify_lookup_plugin_terms"),
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


class VLLMAgentAdapter:
    def __init__(self, host: str, port: str, model: str):
        import openai

        self.client = openai.OpenAI(base_url=f"http://{host}:{port}/v1", api_key="dummy")
        self.model = model

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0, max_tokens: int = 4096) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe LM Rewrite candidate quality")
    parser.add_argument("--godel0-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--repo-pool", default="repo_pool")
    parser.add_argument("--repo-id", default="ansible")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--vllm-host", required=True)
    parser.add_argument("--vllm-port", default="8000")
    parser.add_argument("--output-dir", default=f"output_lm_rewrite_probe_{int(time.time())}")
    parser.add_argument("--max-targets", type=int, default=3)
    args = parser.parse_args()

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

    agent = VLLMAgentAdapter(args.vllm_host, args.vllm_port, args.model)
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
        max_patch_lines=120,
        forbid_test_file_edits=True,
    )

    results = []
    for idx, (target_file, target_symbol) in enumerate(TARGETS[: args.max_targets], start=1):
        print(f"\n[{idx}/{args.max_targets}] LM Rewrite {target_file}::{target_symbol}", flush=True)
        plan = BugGenerationPlan(
            plan_id=f"rewrite_probe_{idx:04d}",
            target_repo_id=spec.repo_id,
            target_base_commit=spec.base_commit,
            target_file=target_file,
            target_symbol=target_symbol,
            strategy="lm_rewrite",
            operator="lm_rewrite",
            constraints=BugConstraints(max_modified_lines=120),
            seed=1000 + idx,
        )
        candidates = engine.generate(
            plan=plan,
            node_code_dir=str(output_dir),
            repo_spec=repo_spec,
            output_dir=str(output_dir / "candidates"),
        )
        if not candidates:
            print("  generated=0", flush=True)
            results.append({"target": f"{target_file}::{target_symbol}", "generated": 0})
            continue

        for cand in candidates:
            print(f"  generated {cand.candidate_id}: {len(cand.bug_patch)} chars", flush=True)
            print(cand.bug_patch[:2000], flush=True)
            test_files = MODULE_TEST_MAP.get(target_file, [])
            test_command = spec.test_command + " " + " ".join(test_files) if test_files else spec.test_command
            report = validator.validate(
                candidate_patch=cand.bug_patch,
                repo_path=Path(spec.path),
                base_commit=spec.base_commit,
                test_command=test_command,
                candidate_id=cand.candidate_id,
                repo_id=spec.repo_id,
                target_file=target_file,
                target_symbol=target_symbol,
                operator="lm_rewrite",
            )
            row = {
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

    report_path = output_dir / "lm_rewrite_probe_report.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nReport: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
