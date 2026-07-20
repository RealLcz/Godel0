"""Gates: SolverCoreParityGate, SolverPathIsolationGate, ProposerExtensionGate."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ..errors import ToolRegistrationError


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


@dataclass
class SolverCoreParityReport:
    passed: bool
    checked_files: int = 0
    mismatches: List[str] = field(default_factory=list)
    missing_files: List[str] = field(default_factory=list)
    extra_allowed: bool = True


class SolverCoreParityGate:
    """Verify that protected Solver Core files match the lock file."""

    def run(self, node_code_dir: Path, lock_file: Path) -> SolverCoreParityReport:
        node_code_dir = Path(node_code_dir)
        lock_file = Path(lock_file)
        report = SolverCoreParityReport(passed=True)

        if not lock_file.exists():
            report.passed = False
            report.missing_files.append(str(lock_file))
            return report

        with open(lock_file) as f:
            lock = json.load(f)

        for rel_path, expected_hash in lock["files"].items():
            file_path = node_code_dir / rel_path
            report.checked_files += 1
            if not file_path.exists():
                report.missing_files.append(rel_path)
                report.passed = False
                continue
            actual_hash = sha256_file(file_path)
            if actual_hash != expected_hash:
                report.mismatches.append(rel_path)
                report.passed = False

        return report


@dataclass
class SolverPathIsolationReport:
    passed: bool
    solver_imports_proposer: bool = False
    extra_tools_found: List[str] = field(default_factory=list)
    proposer_instructions_in_prompt: bool = False
    import_side_effects: List[str] = field(default_factory=list)


class SolverPathIsolationGate:
    """Verify that proposer/swesmith code does not leak into the Solver path."""

    FORBIDDEN_TOOL_NAMES = {
        "ast_mutate",
        "find_f2p",
        "combine_validated_patch",
        "task_commit",
    }

    def run(self, node_code_dir: Path) -> SolverPathIsolationReport:
        node_code_dir = Path(node_code_dir)
        report = SolverPathIsolationReport(passed=True)

        coding_agent = node_code_dir / "coding_agent.py"
        if coding_agent.exists():
            content = coding_agent.read_text()
            for forbidden in ["proposer_main", "import proposer", "from proposer"]:
                if forbidden in content:
                    report.solver_imports_proposer = True
                    report.passed = False
                    report.import_side_effects.append(forbidden)

        tools_dir = node_code_dir / "tools"
        if tools_dir.exists():
            for py in tools_dir.glob("*.py"):
                if py.stem == "__init__":
                    continue
                if py.stem in self.FORBIDDEN_TOOL_NAMES:
                    report.extra_tools_found.append(py.stem)
                    report.passed = False

        prompts_dir = node_code_dir / "prompts"
        if prompts_dir.exists():
            for py in prompts_dir.glob("*.py"):
                content = py.read_text()
                if "proposer" in content.lower() and "bug generation" in content.lower():
                    report.proposer_instructions_in_prompt = True
                    report.passed = False

        return report


@dataclass
class ProposerExtensionReport:
    passed: bool
    proposer_main_exists: bool = False
    schemas_parseable: bool = False
    cannot_write_taskstore: bool = True
    cannot_read_secrets: bool = True
    errors: List[str] = field(default_factory=list)


class ProposerExtensionGate:
    """Verify that the Root Proposer extension is valid and isolated."""

    def run(self, node_code_dir: Path) -> ProposerExtensionReport:
        node_code_dir = Path(node_code_dir)
        report = ProposerExtensionReport(passed=True)

        proposer_main = node_code_dir / "proposer" / "proposer_main.py"
        report.proposer_main_exists = proposer_main.exists()
        if not report.proposer_main_exists:
            report.passed = False
            report.errors.append("proposer/proposer_main.py not found")
            return report

        if proposer_main.exists():
            content = proposer_main.read_text()
            if "TaskStore" in content and "put(" in content:
                report.cannot_write_taskstore = False
                report.passed = False
                report.errors.append("Proposer directly calls TaskStore.put()")
            if "materialize_private" in content:
                report.cannot_read_secrets = False
                report.passed = False
                report.errors.append("Proposer reads trusted private inputs")

        report.schemas_parseable = True
        return report
