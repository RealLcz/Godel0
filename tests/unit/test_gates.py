"""Unit tests for gates."""

from __future__ import annotations

from pathlib import Path

import pytest

from godel0.evolution.gates import (
    SolverCoreParityGate,
    SolverPathIsolationGate,
    ProposerExtensionGate,
)


class TestSolverCoreParityGate:
    def test_verify_passes(self):
        gate = SolverCoreParityGate()
        report = gate.run(
            Path("initial_agent/src"),
            Path("initial_agent/solver_core.lock.json"),
        )
        assert report.passed, f"Mismatches: {report.mismatches}, Missing: {report.missing_files}"

    def test_missing_lock_file(self, tmp_path):
        gate = SolverCoreParityGate()
        report = gate.run(tmp_path, tmp_path / "nonexistent.lock.json")
        assert not report.passed


class TestSolverPathIsolationGate:
    def test_isolation_passes(self):
        gate = SolverPathIsolationGate()
        report = gate.run(Path("initial_agent/src"))
        assert report.passed, f"Import side effects: {report.import_side_effects}, Extra tools: {report.extra_tools_found}"


class TestProposerExtensionGate:
    def test_extension_exists(self):
        gate = ProposerExtensionGate()
        report = gate.run(Path("initial_agent/src"))
        assert report.proposer_main_exists
        assert report.schemas_parseable
        assert report.passed, report.errors


class TestSchemaCompatibility:
    def test_initial_agent_schemas_accept_legacy_payload(self):
        from godel0.evolution.gates import validate_proposer_schema_compatibility

        errors = validate_proposer_schema_compatibility(Path("initial_agent/src"))
        assert errors == []

    def test_missing_required_class_fails(self, tmp_path: Path):
        from godel0.evolution.gates import validate_proposer_schema_compatibility

        schemas = tmp_path / "proposer"
        schemas.mkdir()
        (schemas / "schemas.py").write_text(
            "from pydantic import BaseModel\n"
            "class BugConstraints(BaseModel):\n"
            "    min_modified_files: int = 1\n"
            "    max_modified_files: int = 1\n"
            "    max_modified_lines: int = 20\n",
            encoding="utf-8",
        )
        errors = validate_proposer_schema_compatibility(tmp_path)
        assert any("missing schema class" in e for e in errors)

    def test_optional_field_addition_passes(self, tmp_path: Path):
        import shutil

        from godel0.evolution.gates import validate_proposer_schema_compatibility

        src = Path("initial_agent/src/proposer/schemas.py")
        dest_dir = tmp_path / "proposer"
        dest_dir.mkdir()
        text = src.read_text(encoding="utf-8")
        text = text.replace(
            "seed: int = 0\n",
            "seed: int = 0\n    extra_planning_hint: str = \"\"\n",
        )
        (dest_dir / "schemas.py").write_text(text, encoding="utf-8")
        errors = validate_proposer_schema_compatibility(tmp_path)
        assert errors == []
