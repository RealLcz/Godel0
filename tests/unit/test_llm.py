"""Unit tests for multi-model LLM support."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Add initial_agent/src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "initial_agent" / "src"))

import llm
from llm import (
    create_client, is_deepseek_api_model, is_minimax_model,
    is_qwen_model, is_vllm_model, is_openai_model,
    AVAILABLE_LLMS, extract_json_between_markers,
)


class TestModelDispatch:
    def test_openai_detection(self):
        assert is_openai_model("gpt-5")
        assert is_openai_model("o4-mini")
        assert is_openai_model("o3")

    def test_deepseek_detection(self):
        assert is_deepseek_api_model("deepseek-chat")
        assert is_deepseek_api_model("deepseek-reasoner")
        assert is_deepseek_api_model("deepseek/deepseek-chat")

    def test_minimax_detection(self):
        assert is_minimax_model("minimax/abab6.5s-chat")
        assert is_minimax_model("minimax/MiniMax-Text-01")

    def test_qwen_detection(self):
        assert is_qwen_model("qwen/qwen-coder-32b")
        assert is_qwen_model("Qwen/Qwen2.5-Coder-32B")

    def test_vllm_detection(self):
        assert is_vllm_model("vllm-qwen-10.0.0.1")

    def test_available_llms_includes_supported(self):
        """All user-requested model families should be in AVAILABLE_LLMS."""
        model_families = [m.split("/")[0] if "/" in m else m.split("-")[0] for m in AVAILABLE_LLMS]
        assert "deepseek" in model_families or any("deepseek" in m for m in AVAILABLE_LLMS)
        assert "minimax" in model_families or any("minimax" in m for m in AVAILABLE_LLMS)
        assert "qwen" in model_families or any("qwen" in m.lower() for m in AVAILABLE_LLMS)

    def test_qwen_vllm_preserves_served_model_name(self, monkeypatch):
        calls = []

        def fake_openai(**kwargs):
            calls.append(kwargs)
            return object()

        monkeypatch.setattr(llm, "QWEN_API_BASE_URL", "")
        monkeypatch.setattr(llm.openai, "OpenAI", fake_openai)

        _client, client_model = create_client("Qwen/Qwen3.6-35B-A3B")

        assert client_model == "Qwen/Qwen3.6-35B-A3B"
        assert calls[0]["base_url"].endswith("/v1")


class TestJSONExtraction:
    def test_extract_from_code_block(self):
        text = 'Here is the result:\n```json\n{"key": "value"}\n```\nDone.'
        result = extract_json_between_markers(text)
        assert result == {"key": "value"}

    def test_extract_from_inline(self):
        text = 'The answer is {"key": "value"} here.'
        result = extract_json_between_markers(text)
        assert result == {"key": "value"}

    def test_no_json(self):
        text = "No JSON here."
        result = extract_json_between_markers(text)
        assert result is None

    def test_nested_json(self):
        text = '```json\n{"outer": {"inner": "value"}}\n```'
        result = extract_json_between_markers(text)
        assert result["outer"]["inner"] == "value"

    def test_json_with_special_chars(self):
        text = '```json\n{"path": "/usr/local/bin", "count": 42}\n```'
        result = extract_json_between_markers(text)
        assert result["path"] == "/usr/local/bin"
        assert result["count"] == 42
