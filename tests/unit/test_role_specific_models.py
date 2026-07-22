"""P1-1: role-specific models must reach chat(..., model=...)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from godel0.evolution.diagnose import CycleDiagnoser
from swesmith.repo_chain import RepoChainGenerator
from proposer.schemas import BugGenerationPlan


class RecordingChatAdapter:
    def __init__(self, default_model: str = ""):
        self.default_model = default_model
        self.calls = []

    def chat(self, system_prompt, user_prompt, temperature=0, max_tokens=8192, model=None):
        self.calls.append(
            {
                "system": system_prompt,
                "user": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "model": model,
            }
        )
        return '{"ok": true}'


class TestRepoChainPassesPlanModel:
    def test_chat_receives_plan_model(self):
        adapter = RecordingChatAdapter(default_model="adapter-default")
        gen = RepoChainGenerator(agent_adapter=adapter)
        plan = BugGenerationPlan(
            plan_id="p1",
            model="proposer-model-X",
            target_file="a.py",
            strategy="repo_chain",
        )
        gen._chat("sys", "user", max_tokens=100, model=plan.model)
        assert adapter.calls
        assert adapter.calls[0]["model"] == "proposer-model-X"

    def test_chat_falls_back_to_adapter_default_model(self):
        adapter = RecordingChatAdapter(default_model="proposer-from-request")
        gen = RepoChainGenerator(agent_adapter=adapter)
        gen._chat("sys", "user", max_tokens=50, model="")
        assert adapter.calls[0]["model"] == "proposer-from-request"


class TestCycleDiagnoserPassesDiagnoseModel:
    def test_diagnose_chat_uses_configured_model(self):
        adapter = RecordingChatAdapter()
        diagnoser = CycleDiagnoser(
            chat_adapter=adapter,
            model="diagnose-model-Y",
            max_retries=0,
        )
        # Force chat path; parsing will fail and fall back to deterministic,
        # but _call_chat must still have been invoked with the model.
        summary = MagicMock()
        evidence = MagicMock()
        evidence.alerts = []
        evidence.solver_failures = []
        evidence.proposer_failures = []
        # Stub helpers so we only exercise _call_chat.
        diagnoser._parse_llm_response = MagicMock(
            side_effect=ValueError("stop after chat")
        )
        diagnoser._diagnose_deterministic = MagicMock(
            return_value=SimpleNamespace(node_id="n")
        )
        diagnoser.diagnose("n", summary, evidence, agent_code_summary="")
        assert adapter.calls
        assert adapter.calls[0]["model"] == "diagnose-model-Y"
