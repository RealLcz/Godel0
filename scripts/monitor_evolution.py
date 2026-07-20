#!/usr/bin/env python3
"""Audit a live Gödel0 run for mainline progress and invariant violations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    config = yaml.safe_load((run_dir / "config.resolved.yaml").read_text())
    expected_k = int(config["tasks"]["batch_size"])
    target_epochs = int(config["run"]["max_nodes"])

    latest: dict[str, dict] = {}
    archive = run_dir / "archive.jsonl"
    if archive.is_file():
        for line in archive.read_text(encoding="utf-8").splitlines():
            if line.strip():
                node = json.loads(line)
                latest[str(node["node_id"])] = node

    anomalies: list[str] = []
    complete_children = []
    status_counts: dict[str, int] = {}
    for node in latest.values():
        status = str(node.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        node_id = str(node["node_id"])
        node_dir = run_dir / "nodes" / node_id
        if status != "complete":
            continue
        generation_path = node_dir / "proposer" / "generation_summary.json"
        level2_path = node_dir / "level2" / "result.json"
        if not generation_path.is_file() or not level2_path.is_file():
            anomalies.append(f"{node_id}: complete node missing proposer or Level 2 result")
            continue
        generation = read_json(generation_path)
        level2 = read_json(level2_path)
        task_ids = list(generation.get("task_ids") or [])
        outcomes = list(level2.get("outcomes") or [])
        if not generation.get("complete") or len(task_ids) != expected_k:
            anomalies.append(
                f"{node_id}: proposer committed {len(task_ids)}/{expected_k} valid tasks"
            )
        if len(outcomes) != expected_k:
            anomalies.append(f"{node_id}: Level 2 has {len(outcomes)}/{expected_k} outcomes")
        if node.get("parent_node_id") is not None:
            complete_children.append(node_id)
            level1_path = node_dir / "level1" / "result.json"
            if not level1_path.is_file():
                anomalies.append(f"{node_id}: complete child missing Level 1 result")
            else:
                level1 = read_json(level1_path)
                if set(level1.get("evaluated_task_ids") or []) != set(
                    level1.get("parent_solved_task_ids") or []
                ):
                    anomalies.append(
                        f"{node_id}: Level 1 did not evaluate exactly the parent-solved set"
                    )
                if not level1.get("passed"):
                    anomalies.append(f"{node_id}: complete child has failed Level 1")

    report = {
        "run_dir": str(run_dir),
        "target_successful_epochs": target_epochs,
        "successful_epochs": len(complete_children),
        "remaining_epochs": max(0, target_epochs - len(complete_children)),
        "node_status_counts": status_counts,
        "anomalies": anomalies,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if anomalies else 0


if __name__ == "__main__":
    raise SystemExit(main())
