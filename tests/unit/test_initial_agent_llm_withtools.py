"""Solver-core parity tests for the baseline tool-calling request."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_module():
    agent_src = Path(__file__).resolve().parents[2] / "initial_agent" / "src"
    sys.path.insert(0, str(agent_src))
    spec = importlib.util.spec_from_file_location(
        "initial_agent_llm_withtools", agent_src / "llm_withtools.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_solver_core_does_not_inject_godel0_only_request_options(monkeypatch):
    module = _load_module()
    captured = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(choices=[])

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=Completions()),
        models=SimpleNamespace(list=lambda: SimpleNamespace(data=[])),
    )
    monkeypatch.setenv("GODEL0_TOOL_MAX_TOKENS", "8192")

    module.get_response_withtools(
        client=client,
        model="Qwen/Qwen3.6-35B-A3B",
        messages=[{"role": "user", "content": "edit the code"}],
        tools=[],
        tool_choice="auto",
        logging=lambda _message: None,
    )

    # Qwen deployment policy belongs to the external vLLM service. Adding
    # these fields here changes the canonical DGM/HGM Solver Core.
    assert "max_tokens" not in captured
    assert "extra_body" not in captured
