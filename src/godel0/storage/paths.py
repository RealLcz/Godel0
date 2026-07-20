"""Path management for run directories."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class RunPaths:
    """Manages all filesystem paths for a single run."""

    def __init__(self, runs_dir: Path, run_id: str):
        self.runs_dir = Path(runs_dir)
        self.run_id = run_id
        self.run_dir = self.runs_dir / run_id
        self.nodes_dir = self.run_dir / "nodes"
        self.root_dir = self.run_dir / "root"
        self.logs_dir = self.run_dir / "logs"

    @property
    def config_path(self) -> Path:
        return self.run_dir / "config.resolved.yaml"

    @property
    def run_json_path(self) -> Path:
        return self.run_dir / "run.json"

    @property
    def events_path(self) -> Path:
        return self.run_dir / "events.jsonl"

    @property
    def archive_path(self) -> Path:
        return self.run_dir / "archive.jsonl"

    def node_dir(self, node_id: str) -> Path:
        return self.nodes_dir / node_id

    def node_json(self, node_id: str) -> Path:
        return self.node_dir(node_id) / "node.json"

    def node_scores(self, node_id: str) -> Path:
        return self.node_dir(node_id) / "scores.json"

    def diagnosis_dir(self, node_id: str) -> Path:
        return self.node_dir(node_id) / "diagnosis"

    def cycle_summary_path(self, node_id: str) -> Path:
        return self.diagnosis_dir(node_id) / "cycle_summary.json"

    def special_alerts_path(self, node_id: str) -> Path:
        return self.diagnosis_dir(node_id) / "special_alerts.json"

    def evidence_bundle_path(self, node_id: str) -> Path:
        return self.diagnosis_dir(node_id) / "evidence_bundle.json"

    def diagnosis_path(self, node_id: str) -> Path:
        return self.diagnosis_dir(node_id) / "diagnosis.json"

    def problem_statement_path(self, node_id: str) -> Path:
        return self.diagnosis_dir(node_id) / "problem_statement.md"

    def self_evolve_dir(self, node_id: str) -> Path:
        return self.node_dir(node_id) / "self_evolve"

    def self_evolve_trajectory(self, node_id: str) -> Path:
        return self.self_evolve_dir(node_id) / "trajectory.jsonl"

    def self_evolve_patch(self, node_id: str) -> Path:
        return self.self_evolve_dir(node_id) / "final.patch"

    def gates_dir(self, node_id: str) -> Path:
        return self.node_dir(node_id) / "gates"

    def level1_dir(self, node_id: str) -> Path:
        return self.node_dir(node_id) / "level1"

    def level1_result(self, node_id: str) -> Path:
        return self.level1_dir(node_id) / "result.json"

    def proposer_dir(self, node_id: str) -> Path:
        return self.node_dir(node_id) / "proposer"

    def level2_dir(self, node_id: str) -> Path:
        return self.node_dir(node_id) / "level2"

    def level2_result(self, node_id: str) -> Path:
        return self.level2_dir(node_id) / "result.json"

    def mutation_manifest_path(self, node_id: str) -> Path:
        return self.node_dir(node_id) / "mutation_manifest.json"

    def ensure_dirs(self) -> None:
        """Create all top-level run directories."""
        for d in [self.run_dir, self.nodes_dir, self.root_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def ensure_node_dirs(self, node_id: str) -> None:
        """Create all directories for a node."""
        nd = self.node_dir(node_id)
        for d in [
            nd,
            self.diagnosis_dir(node_id),
            self.self_evolve_dir(node_id),
            self.gates_dir(node_id),
            self.level1_dir(node_id),
            self.proposer_dir(node_id),
            self.level2_dir(node_id),
            self.diagnosis_dir(node_id) / "raw_excerpts",
            self.level1_dir(node_id) / "outcomes",
            self.level2_dir(node_id) / "outcomes",
            self.proposer_dir(node_id) / "candidates",
        ]:
            d.mkdir(parents=True, exist_ok=True)
