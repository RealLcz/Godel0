#!/usr/bin/env python3
"""Select the highest-scoring complete node and materialize its exact Git tree."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


def load_latest_nodes(archive_path: Path) -> dict[str, dict]:
    nodes: dict[str, dict] = {}
    for line in archive_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            nodes[str(record["node_id"])] = record
    return nodes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--agent-repo", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    agent_repo = args.agent_repo.resolve()
    output_dir = args.output_dir.resolve()
    nodes = load_latest_nodes(run_dir / "archive.jsonl")
    complete = [node for node in nodes.values() if node.get("status") == "complete"]
    if not complete:
        raise SystemExit("No complete node is available for SWE-bench evaluation")
    best = max(
        complete,
        key=lambda node: (
            float(node.get("node_score") or 0.0),
            float(node.get("solver_score") or 0.0),
            str(node.get("completed_at") or ""),
        ),
    )
    commit = str(best["code_commit"])
    subprocess.run(
        ["git", "-C", str(agent_repo), "cat-file", "-e", f"{commit}^{{commit}}"],
        check=True,
    )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="godel0_best_node_") as temp:
        archive = Path(temp) / "node.tar"
        subprocess.run(
            [
                "git",
                "-C",
                str(agent_repo),
                "archive",
                "--format=tar",
                f"--output={archive}",
                commit,
            ],
            check=True,
        )
        with tarfile.open(archive) as bundle:
            bundle.extractall(output_dir)

    metadata = {
        "node_id": best["node_id"],
        "code_commit": commit,
        "node_score": best.get("node_score"),
        "solver_score": best.get("solver_score"),
        "proposer_score": best.get("proposer_score"),
        "frontier_accuracy": best.get("frontier_accuracy"),
        "source_run_dir": str(run_dir),
        "agent_repo": str(agent_repo),
    }
    (output_dir / "godel0_export.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
