"""Fake LLM client for deterministic testing."""

from __future__ import annotations

from typing import Any, List, Optional


class FakeLLMResponse:
    """A scripted LLM response."""

    def __init__(self, content: str, tool_calls: list = None):
        self.content = content
        self.tool_calls = tool_calls or []

    @property
    def choices(self):
        class Choice:
            def __init__(self, content, tool_calls):
                self.message = type("Message", (), {
                    "content": content,
                    "tool_calls": tool_calls,
                })()
        return [Choice(self.content, self.tool_calls)]


class FakeLLMClient:
    """Deterministic LLM client for testing.

    Returns pre-scripted responses in order.
    """

    def __init__(self, responses: List[str] = None):
        self._responses = list(responses or [])
        self._call_count = 0
        self.calls: List[dict] = []

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs) -> FakeLLMResponse:
        self._call_count += 1
        self.calls.append(kwargs)

        if self._responses:
            content = self._responses.pop(0)
        else:
            content = '{"primary_root_cause": "test root cause", "problem_statement": "test problem"}'

        return FakeLLMResponse(content)

    def models(self):
        class ModelsList:
            data = [type("Model", (), {"id": "fake-model"})()]
        return ModelsList()
