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
