#!/usr/bin/env python3
"""Probe cross-file Combine candidates from already generated artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import combinations
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


def _load_passed_from_proposer(report_path: Path) -> set[str]:
    if not report_path.exists():
        return set()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        row["candidate_id"]
        for row in report.get("candidate_details", [])
        if row.get("result") == "passed"
    }


def _load_passed_from_lm_rewrite(report_path: Path) -> set[str]:
    if not report_path.exists():
        return set()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        row["candidate_id"]
        for row in report
        if row.get("passed") and row.get("candidate_id")
    }


def _load_candidate(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_candidate_dir"] = str(path.parent)
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe cross-file Combine quality")
    parser.add_argument("--godel0-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--repo-pool", default="repo_pool")
    parser.add_argument("--repo-id", default="ansible")
    parser.add_argument("--output-dir", default=f"output_combine_probe_{int(time.time())}")
    parser.add_argument("--max-combos", type=int, default=8)
    parser.add_argument("--test-python", default="")
    parser.add_argument(
        "--candidate-roots",
        nargs="+",
        default=["output_proposer_200746", "output_lm_rewrite_probe_200746"],
    )
    args = parser.parse_args()

    root = Path(args.godel0_root).resolve()
    os.environ.setdefault("HOME", "/tmp/godel0_home")
    os.environ.setdefault("ANSIBLE_LOCAL_TEMP", "/tmp/godel0_ansible_tmp")
    os.environ.setdefault("TMPDIR", "/tmp/godel0_tmp")
    for env_dir in ("HOME", "ANSIBLE_LOCAL_TEMP", "TMPDIR"):
        Path(os.environ[env_dir]).mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "initial_agent" / "src"))
    sys.path.insert(0, str(root))

    from godel0.proposer_trusted.candidate_validator import CandidateValidator
    from godel0.tasks.repo_pool import RepoPool
    from swesmith.combine import CombinedCandidateRef
    from swesmith.engine import SWESmithEngine
    from swesmith.patch_utils import extract_changed_files, count_modified_lines

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    passed_ids: set[str] = set()
    for candidate_root in args.candidate_roots:
        base = Path(candidate_root)
        passed_ids |= _load_passed_from_proposer(base / "proposer_report.json")
        passed_ids |= _load_passed_from_lm_rewrite(base / "lm_rewrite_probe_report.json")

    candidates: list[dict] = []
    for candidate_root in args.candidate_roots:
        for path in sorted(Path(candidate_root).glob("candidates/*/candidate.json")):
            data = _load_candidate(path)
            if data.get("candidate_id") not in passed_ids:
                continue
            files = extract_changed_files(data.get("bug_patch", ""))
            if not files:
                continue
            data["_changed_files"] = files
            candidates.append(data)

    print(f"Loaded passed candidates: {len(candidates)}", flush=True)
    for cand in candidates:
        print(
            f"  {cand['candidate_id']} {cand.get('strategy')} "
            f"{cand.get('target_file')}::{cand.get('target_symbol')} files={cand['_changed_files']}",
            flush=True,
        )

    pool = RepoPool(root / args.repo_pool)
    spec = pool.get(args.repo_id)
    if spec is None:
        raise RuntimeError(f"repo not found: {args.repo_id}")

    validator = CandidateValidator(
        workspace_root=output_dir / "validator_ws",
        test_timeout_sec=120,
        max_patch_lines=240,
        forbid_test_file_edits=True,
    )
    engine = SWESmithEngine()

    results = []
    attempted = 0
    for pair in combinations(candidates, 2):
        if attempted >= args.max_combos:
            break
        files_a = set(pair[0]["_changed_files"])
        files_b = set(pair[1]["_changed_files"])
        if not files_a.isdisjoint(files_b):
            continue

        refs = [
            CombinedCandidateRef(
                candidate_id=c["candidate_id"],
                plan_id=c.get("plan_id", c["candidate_id"]),
                bug_patch=c["bug_patch"],
                target_file=c.get("target_file", ""),
                target_symbol=c.get("target_symbol", ""),
                failure_signature="",
            )
            for c in pair
        ]
        combined_candidates = engine.combiner.generate(
            plan=type("Plan", (), {"constraints": type("Constraints", (), {"candidates": refs})()})(),
            node_code_dir=str(output_dir),
            repo_spec=None,
            output_dir=str(output_dir / "candidates"),
            candidates=refs,
        )
        if not combined_candidates:
            continue

        combined = combined_candidates[0]
        changed_files = extract_changed_files(combined.bug_patch)
        if len(set(changed_files)) < 2:
            continue

        attempted += 1
        test_files: list[str] = []
        for file_name in changed_files:
            for test_file in MODULE_TEST_MAP.get(file_name, []):
                if test_file not in test_files:
                    test_files.append(test_file)
        base_test_command = spec.test_command
        if args.test_python:
            base_test_command = base_test_command.replace("python3.11", args.test_python, 1)
        test_command = base_test_command + " " + " ".join(test_files) if test_files else base_test_command

        print(
            f"\n[{attempted}/{args.max_combos}] {combined.candidate_id} "
            f"from {[r.candidate_id for r in refs]} files={changed_files} lines={count_modified_lines(combined.bug_patch)}",
            flush=True,
        )
        print(combined.bug_patch[:3000], flush=True)

        report = validator.validate(
            candidate_patch=combined.bug_patch,
            repo_path=Path(spec.path),
            base_commit=spec.base_commit,
            test_command=test_command,
            candidate_id=combined.candidate_id,
            repo_id=spec.repo_id,
            target_file="",
            target_symbol="",
            operator=f"combine:{combined.candidate_id}",
        )
        row = {
            "candidate_id": combined.candidate_id,
            "source_candidate_ids": [r.candidate_id for r in refs],
            "changed_files": changed_files,
            "modified_lines": count_modified_lines(combined.bug_patch),
            "patch": combined.bug_patch,
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

    report_path = output_dir / "combine_probe_report.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nReport: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
