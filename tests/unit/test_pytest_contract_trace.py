"""Tests for per-test repository contract tracing."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_plugin_records_only_runtime_production_edges(tmp_path: Path):
    repo = tmp_path / "repo"
    package = repo / "lib" / "pipeline"
    tests = repo / "tests"
    package.mkdir(parents=True)
    tests.mkdir()
    (repo / "lib" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "producer.py").write_text(
        "def produce(value):\n"
        "    return value + 1\n",
        encoding="utf-8",
    )
    (package / "consumer.py").write_text(
        "from .producer import produce\n\n"
        "IMPORT_ONLY = produce\n\n"
        "def consume(value):\n"
        "    return produce(value) * 2\n",
        encoding="utf-8",
    )
    (tests / "test_pipeline.py").write_text(
        "from lib.pipeline.consumer import consume\n\n"
        "def test_contract():\n"
        "    assert consume(2) == 6\n",
        encoding="utf-8",
    )
    output = tmp_path / "trace.json"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join([str(ROOT), str(repo)]),
            "GODEL0_TRACE_OUTPUT": str(output),
            "GODEL0_TRACE_REPO_ROOT": str(repo),
            "GODEL0_TRACE_SOURCE_ROOTS": "lib",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-p",
            "scripts.pytest_contract_trace",
            "tests/test_pipeline.py::test_contract",
            "-q",
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["nodeid"] == "tests/test_pipeline.py::test_contract"
    assert {row["path"] for row in payload["files"]} == {
        "lib/pipeline/consumer.py",
        "lib/pipeline/producer.py",
    }
    assert payload["file_edges"] == [
        {
            "caller": "lib/pipeline/consumer.py",
            "callee": "lib/pipeline/producer.py",
            "call_count": 1,
        }
    ]
    assert payload["pytest_exception"] == ""
