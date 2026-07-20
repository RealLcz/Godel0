"""Tests for the high-coverage repository LM Rewrite probe."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_repo_lm_rewrite_probe.py"
SPEC = importlib.util.spec_from_file_location("repo_lm_rewrite_probe", SCRIPT)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_mask_reaches_ratio_and_preserves_signatures_and_docstrings():
    source = (
        "def large(value):\n"
        "    \"\"\"Keep this contract.\"\"\"\n"
        "    first = value + 1\n"
        "    second = first * 2\n"
        "    third = second - 3\n"
        "    return third\n\n"
        "def small(value):\n"
        "    return value\n"
    )

    masked, manifest = probe.mask_function_implementations(source, target_ratio=0.6)

    ast.parse(masked)
    assert manifest["masked_ratio"] >= 0.6
    assert manifest["selected_symbols"] == ["large"]
    assert "def large(value):" in masked
    assert '"""Keep this contract."""' in masked
    assert masked.count(probe.STUB_MARKER) == 1
    assert "def small(value):" in masked


def test_mask_rewrites_all_conditional_definitions_of_same_symbol():
    source = (
        "if FAST:\n"
        "    class Loader:\n"
        "        def load(self, value):\n"
        "            first = value + 1\n"
        "            return first\n"
        "else:\n"
        "    class Loader:\n"
        "        def load(self, value):\n"
        "            first = value - 1\n"
        "            second = first * 2\n"
        "            return second\n"
    )

    masked, manifest = probe.mask_function_implementations(source, target_ratio=0.6)

    ast.parse(masked)
    assert manifest["selected_symbols"] == ["Loader.load"]
    assert manifest["selected_definition_count"] == 2
    assert masked.count(probe.STUB_MARKER) == 2


def test_parse_file_blocks_accepts_plain_and_fenced_content():
    response = (
        'preface\n<file path="pkg/a.py">\nvalue = 1\n</file>\n'
        "<file path='pkg/b.py'>\n```python\nvalue = 2\n```\n</file>"
    )

    files, metadata = probe.parse_file_blocks(response)

    assert files == {"pkg/a.py": "value = 1", "pkg/b.py": "value = 2"}
    assert metadata["parsed_block_count"] == 2
    assert metadata["duplicate_paths"] == []
    assert metadata["unclosed_file_tag"] is False


def test_rewrite_metrics_distinguish_function_and_whole_file_coverage():
    original = (
        "import os\n\n"
        "CONSTANT = 1\n\n"
        "def calculate(value):\n"
        "    first = value + CONSTANT\n"
        "    second = first * 2\n"
        "    return second\n"
    )
    rewritten = (
        "import os\n\n"
        "CONSTANT = 1\n\n"
        "def calculate(value):\n"
        "    result = (value + CONSTANT) * 2\n"
        "    return result\n"
    )

    metrics = probe.compute_rewrite_metrics(original, rewritten)

    assert metrics["changed_implementation_ratio"] == 1.0
    assert 0 < metrics["changed_file_code_ratio"] < 1.0


def test_body_transplant_preserves_imports_signatures_and_unselected_code():
    original = (
        "from pkg import existing\n\n"
        "def selected(value):\n"
        "    \"\"\"Keep this.\"\"\"\n"
        "    first = existing(value)\n"
        "    second = first + 1\n"
        "    return second\n\n"
        "def untouched(value):\n"
        "    return value\n"
    )
    _masked, manifest = probe.mask_function_implementations(original, target_ratio=0.6)
    model_source = (
        "from pkg import existing\n"
        "from bad import circular\n\n"
        "def selected(value, changed_signature=False):\n"
        "    result = existing(value) * 2\n"
        "    return result\n\n"
        "def untouched(value):\n"
        "    return circular(value)\n"
    )

    rewritten, metadata = probe.transplant_selected_function_bodies(
        original,
        model_source,
        manifest,
    )

    ast.parse(rewritten)
    assert "from bad import circular" not in rewritten
    assert "def selected(value):" in rewritten
    assert '"""Keep this."""' in rewritten
    assert "result = existing(value) * 2" in rewritten
    assert "def untouched(value):\n    return value" in rewritten
    assert metadata["preserved_non_body_structure"] is True


def test_quality_classification_prioritizes_behavioral_validity_over_ratio():
    assert probe.classify_quality(
        {
            "generation_complete": True,
            "actual_ratio_gate": False,
            "validation_passed": False,
        }
    ) == "not_a_valid_f2p_task"
    assert probe.classify_quality(
        {
            "generation_complete": True,
            "actual_ratio_gate": False,
            "validation_passed": True,
            "strict_repo_level": True,
            "behavioral_coupling": {
                "standalone_inert_files": [],
                "behavior_overlap_graph_connected": True,
            },
        }
    ) == "valid_task_below_requested_rewrite_ratio"


def test_prompt_contains_every_connected_file_and_no_original_body_instruction():
    domain = {
        "files": ["pkg/a.py", "pkg/b.py"],
        "contract": "a feeds b",
    }
    prompt = probe.build_rewrite_prompt(
        domain,
        {
            "pkg/a.py": 'def a():\n    raise NotImplementedError("LM_REWRITE_STUB")\n',
            "pkg/b.py": 'def b():\n    raise NotImplementedError("LM_REWRITE_STUB")\n',
        },
        0.6,
    )

    assert '<input_file path="pkg/a.py">' in prompt
    assert '<input_file path="pkg/b.py">' in prompt
    assert "Do not intentionally inject a bug" in prompt
    assert "60%" in prompt
