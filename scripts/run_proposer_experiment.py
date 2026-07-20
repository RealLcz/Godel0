#!/usr/bin/env python3
"""Run the Gödel0 Proposer experiment with vLLM-served Qwen3.6-35B-A3B.

This version uses BOTH procedural and LM Modify strategies:
- Procedural: change_operator, invert_if (fast, no LLM needed)
- LM Modify: chaos monkey testing with Qwen3.6-35B-A3B (SWE-smith style prompt)
- P2P validation: requires both F2P and P2P tests
- Issue generation: LLM-based with test context
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


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
    "lib/ansible/utils/shlex.py": [
        "test/units/utils/test_shlex.py",
    ],
    "lib/ansible/module_utils/common/text/converters.py": [
        "test/units/module_utils/common/text/converters/test_to_bytes.py",
        "test/units/module_utils/common/text/converters/test_to_text.py",
    ],
}

TARGETS = [
    {
        "file": "lib/ansible/module_utils/common/dict_transformations.py",
        "symbols": ["dict_merge", "recursive_diff", "camel_dict_to_snake_dict",
                     "snake_dict_to_camel_dict", "_camel_to_snake", "_snake_to_camel"],
    },
    {
        "file": "lib/ansible/utils/helpers.py",
        "symbols": ["pct_to_int", "object_to_dict", "deduplicate_list"],
    },
    {
        "file": "lib/ansible/utils/listify.py",
        "symbols": ["listify_lookup_plugin_terms"],
    },
    {
        "file": "lib/ansible/utils/shlex.py",
        "symbols": ["shlex_split"],
    },
    {
        "file": "lib/ansible/module_utils/common/text/converters.py",
        "symbols": ["to_bytes", "to_text", "to_native"],
    },
]


class VLLMAgentAdapter:
    """Simple agent adapter that calls a vLLM OpenAI-compatible endpoint."""

    def __init__(self, host: str, port: str, model: str):
        import openai
        self.client = openai.OpenAI(
            base_url=f"http://{host}:{port}/v1",
            api_key="dummy",
        )
        self.model = model

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 1, max_tokens: int = 16384) -> str:
        """Call the LLM with system + user messages."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def run_task(self, prompt: str, system_message: str, model: str = "", workspace_dir: str = "") -> str:
        """Compatibility method for agent adapter protocol."""
        return self.chat(system_message, prompt, temperature=1)


def main():
    parser = argparse.ArgumentParser(description="Run Gödel0 Proposer experiment")
    parser.add_argument("--godel0-root", required=True)
    parser.add_argument("--repo-pool", required=True)
    parser.add_argument("--repo-id", default="ansible")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--vllm-host", default="127.0.0.1")
    parser.add_argument("--vllm-port", default="8000")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--use-lm-modify", action="store_true", default=True,
                        help="Use LM Modify strategy (requires vLLM)")
    parser.add_argument("--use-procedural", action="store_true", default=True,
                        help="Use procedural mutation strategy")
    args = parser.parse_args()

    godel0_root = Path(args.godel0_root)
    sys.path.insert(0, str(godel0_root / "src"))
    sys.path.insert(0, str(godel0_root / "initial_agent" / "src"))
    os.environ["VLLM_HOST"] = args.vllm_host
    os.environ["VLLM_PORT"] = args.vllm_port

    from godel0.tasks.repo_pool import RepoPool
    from godel0.proposer_trusted.candidate_validator import CandidateValidator
    from godel0.proposer_trusted.task_committer import TaskCommitter
    from godel0.tasks.store import TaskStore
    from godel0.proposer_trusted.safety import check_safety
    from godel0.proposer_trusted.duplicate_detector import DuplicateDetector
    from swesmith.engine import SWESmithEngine, BugGenerationPlan, RepoSpec as EngineRepoSpec, BugConstraints

    pool = RepoPool(Path(args.repo_pool))
    spec = pool.get(args.repo_id)
    if spec is None:
        print(f"ERROR: Repo '{args.repo_id}' not found in pool")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Gödel0 Proposer Experiment (SWE-smith style)")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"vLLM: http://{args.vllm_host}:{args.vllm_port}/v1")
    print(f"Repo: {spec.repo_id}")
    print(f"Strategies: {'LM Modify' if args.use_lm_modify else ''} {'+ Procedural' if args.use_procedural else ''}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test LLM
    print(f"\n--- LLM Connectivity ---")
    agent_adapter = VLLMAgentAdapter(args.vllm_host, args.vllm_port, args.model)
    try:
        resp = agent_adapter.chat("You are a helpful assistant.", "Reply with exactly: OK", temperature=0, max_tokens=5)
        print(f"  Response: {resp.strip()}")
    except Exception as e:
        print(f"  WARNING: LLM not available: {e}")
        agent_adapter = None

    # Create engine with agent adapter for LM Modify
    engine = SWESmithEngine(agent_adapter=agent_adapter)

    repo_spec = EngineRepoSpec(
        repo_id=spec.repo_id,
        repo_path=str(spec.path),
        base_commit=spec.base_commit,
        test_command=spec.test_command,
    )

    # Generate candidates
    print(f"\n--- Generating Bug Candidates ---")
    all_candidates = []
    seed_counter = 0

    for target in TARGETS:
        target_file = target["file"]
        file_path = Path(spec.path) / target_file
        if not file_path.exists():
            continue

        for symbol_name in target["symbols"]:
            if len(all_candidates) >= args.max_candidates:
                break

            # 1. Try LM Modify (chaos monkey style)
            if args.use_lm_modify and agent_adapter:
                seed_counter += 1
                plan = BugGenerationPlan(
                    plan_id=f"lm_plan_{seed_counter:04d}",
                    target_repo_id=spec.repo_id,
                    target_base_commit=spec.base_commit,
                    target_file=target_file,
                    target_symbol=symbol_name,
                    strategy="lm_modify",
                    operator="lm_modify",
                    constraints=BugConstraints(
                        max_modified_lines=15,
                        desired_behavior=f"Introduce a subtle bug in {symbol_name} that breaks at least one test",
                    ),
                    seed=42 + seed_counter,
                )
                try:
                    print(f"  [LM Modify] {symbol_name} in {target_file}...")
                    cands = engine.generate(
                        plan=plan, node_code_dir=str(output_dir),
                        repo_spec=repo_spec, output_dir=str(output_dir / "candidates"),
                    )
                    for c in cands:
                        all_candidates.append(c)
                        print(f"    -> Generated: {c.candidate_id} ({len(c.bug_patch)} chars)")
                except Exception as e:
                    print(f"    -> ERROR: {e}")

            # 2. Try procedural (change_operator, invert_if)
            if args.use_procedural and len(all_candidates) < args.max_candidates:
                for op_name in ["change_operator", "invert_if"]:
                    if len(all_candidates) >= args.max_candidates:
                        break
                    seed_counter += 1
                    plan = BugGenerationPlan(
                        plan_id=f"proc_plan_{seed_counter:04d}",
                        target_repo_id=spec.repo_id,
                        target_base_commit=spec.base_commit,
                        target_file=target_file,
                        target_symbol=symbol_name,
                        strategy="procedural",
                        operator=op_name,
                        constraints=BugConstraints(max_modified_lines=15),
                        seed=42 + seed_counter,
                    )
                    try:
                        cands = engine.generate(
                            plan=plan, node_code_dir=str(output_dir),
                            repo_spec=repo_spec, output_dir=str(output_dir / "candidates"),
                        )
                        for c in cands:
                            all_candidates.append(c)
                            print(f"  [Procedural/{op_name}] {symbol_name}: {c.candidate_id}")
                    except Exception:
                        pass

    print(f"\nTotal candidates: {len(all_candidates)}")

    # Validate candidates
    print(f"\n--- Validating Candidates (F2P + P2P Check) ---")
    validator = CandidateValidator(
        workspace_root=output_dir / "validator_ws",
        test_timeout_sec=60,
        max_patch_lines=80,
        forbid_test_file_edits=True,
    )
    dup_detector = DuplicateDetector()
    task_store = TaskStore(output_dir / "task_store")
    committer = TaskCommitter(task_store)

    results = {
        "total": len(all_candidates),
        "validated": 0, "passed": 0, "rejected": 0,
        "rejection_reasons": {},
        "tasks_committed": [],
        "candidate_details": [],
    }

    for i, cand in enumerate(all_candidates):
        target_file = cand.target_file
        strategy = cand.strategy
        print(f"\n  [{i+1}/{len(all_candidates)}] {cand.candidate_id} ({strategy})")
        print(f"    {cand.operator} on {target_file}::{cand.target_symbol}")

        # Safety check
        is_safe, safety_reasons = check_safety(cand.bug_patch)
        if not is_safe:
            print(f"    REJECTED (safety): {safety_reasons}")
            results["rejected"] += 1
            for r in safety_reasons:
                results["rejection_reasons"][r] = results["rejection_reasons"].get(r, 0) + 1
            results["candidate_details"].append({"candidate_id": cand.candidate_id, "strategy": strategy, "result": "rejected_safety", "reasons": safety_reasons})
            continue

        # Duplicate check
        if not dup_detector.check(cand.bug_patch, spec.repo_id, target_file, cand.target_symbol or "", cand.operator or ""):
            print(f"    REJECTED (duplicate)")
            results["rejected"] += 1
            results["rejection_reasons"]["duplicate"] = results["rejection_reasons"].get("duplicate", 0) + 1
            results["candidate_details"].append({"candidate_id": cand.candidate_id, "strategy": strategy, "result": "rejected_duplicate"})
            continue

        # Determine test file
        test_files = MODULE_TEST_MAP.get(target_file, [])
        test_cmd = spec.test_command + " " + " ".join(test_files) + " -v" if test_files else spec.test_command

        results["validated"] += 1
        report = validator.validate(
            candidate_patch=cand.bug_patch,
            repo_path=Path(spec.path),
            base_commit=spec.base_commit,
            test_command=test_cmd,
            candidate_id=cand.candidate_id,
            repo_id=spec.repo_id,
            target_file=target_file,
            target_symbol=cand.target_symbol or "",
            operator=cand.operator or "",
        )

        print(f"    applied={report.patch_applied} syntax={report.syntax_valid} f2p={len(report.f2p_tests)} p2p={len(report.p2p_tests)} reverse={report.reverse_restored} passed={report.passed}")

        if report.passed:
            results["passed"] += 1
            task = committer.commit_task(
                batch_id=f"batch_{int(time.time())}",
                proposer_node_id="root",
                repo_id=spec.repo_id,
                base_commit=spec.base_commit,
                bug_strategy=strategy,
                bug_patch=cand.bug_patch,
                problem_statement=f"Bug in {cand.target_symbol or 'module'} via {cand.operator}",
                f2p_tests=report.f2p_tests,
                baseline_test_command=test_cmd,
                modified_files=[target_file],
                modified_entities=[cand.target_symbol] if cand.target_symbol else [],
            )
            results["tasks_committed"].append(task.task_id)
            print(f"    TASK COMMITTED: {task.task_id}")
            results["candidate_details"].append({
                "candidate_id": cand.candidate_id, "strategy": strategy,
                "operator": cand.operator, "target": f"{target_file}::{cand.target_symbol}",
                "result": "passed",
                "f2p_tests": report.f2p_tests[:5],
                "p2p_count": len(report.p2p_tests),
                "task_id": task.task_id,
            })
        else:
            results["rejected"] += 1
            for r in report.rejection_reasons:
                results["rejection_reasons"][r] = results["rejection_reasons"].get(r, 0) + 1
            results["candidate_details"].append({
                "candidate_id": cand.candidate_id, "strategy": strategy,
                "operator": cand.operator, "target": f"{target_file}::{cand.target_symbol}",
                "result": "rejected_validation",
                "reasons": report.rejection_reasons,
                "f2p_count": len(report.f2p_tests),
                "p2p_count": len(report.p2p_tests),
            })

    # Report
    print(f"\n{'='*60}")
    print(f"PROPOSER QUALITY REPORT")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Repo: {args.repo_id}")
    print(f"Total candidates: {results['total']}")
    print(f"Validated: {results['validated']}")
    print(f"Passed (F2P+P2P): {results['passed']}")
    print(f"Rejected: {results['rejected']}")
    print(f"Tasks committed: {len(results['tasks_committed'])}")

    # Strategy breakdown
    lm_count = sum(1 for d in results["candidate_details"] if d.get("strategy") == "lm_modify")
    proc_count = sum(1 for d in results["candidate_details"] if d.get("strategy") == "procedural")
    lm_passed = sum(1 for d in results["candidate_details"] if d.get("strategy") == "lm_modify" and d["result"] == "passed")
    proc_passed = sum(1 for d in results["candidate_details"] if d.get("strategy") == "procedural" and d["result"] == "passed")
    print(f"\nStrategy breakdown:")
    print(f"  LM Modify: {lm_passed}/{lm_count} passed")
    print(f"  Procedural: {proc_passed}/{proc_count} passed")

    print(f"\nRejection breakdown:")
    for reason, count in sorted(results["rejection_reasons"].items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

    if results["total"] > 0:
        yield_rate = results["passed"] / results["total"] * 100
        print(f"\nValid yield: {yield_rate:.1f}%")

    # Save report
    report_path = output_dir / "proposer_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nReport: {report_path}")

    # Details
    print(f"\n{'='*60}")
    print(f"CANDIDATE DETAILS")
    print(f"{'='*60}")
    for d in results["candidate_details"]:
        status = "PASS" if d["result"] == "passed" else "FAIL"
        strat = d.get("strategy", "?")
        op = d.get("operator", "?")
        tgt = d.get("target", "?")
        print(f"  [{status}] {d['candidate_id']} ({strat}/{op}) on {tgt}")
        if d["result"] == "passed":
            print(f"         F2P: {d.get('f2p_tests', [])}  P2P: {d.get('p2p_count', 0)}")
        elif "reasons" in d:
            print(f"         Reasons: {d['reasons']}  F2P: {d.get('f2p_count', 0)}  P2P: {d.get('p2p_count', 0)}")

    print(f"\n{'='*60}")
    print(f"Experiment complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
