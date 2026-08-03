"""Tests for initial_agent editor tool: str_replace and large-file edit guard."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

AGENT_TOOLS = (
    Path(__file__).resolve().parents[2] / "initial_agent" / "src" / "tools"
)
sys.path.insert(0, str(AGENT_TOOLS.parent))

from tools import edit as editor  # noqa: E402


def test_str_replace_updates_one_occurrence(tmp_path: Path):
    path = tmp_path / "mod.py"
    path.write_text("def a():\n    return 1\n\ndef b():\n    return 2\n")
    out = editor.tool_function(
        command="str_replace",
        path=str(path),
        old_str="def b():\n    return 2\n",
        new_str="def b():\n    return 3\n",
    )
    assert out.startswith("Successfully replaced")
    assert "return 3" in path.read_text()
    assert "return 1" in path.read_text()


def test_str_replace_rejects_ambiguous_old_str(tmp_path: Path):
    path = tmp_path / "mod.py"
    path.write_text("x = 1\nx = 1\n")
    out = editor.tool_function(
        command="str_replace",
        path=str(path),
        old_str="x = 1\n",
        new_str="x = 2\n",
    )
    assert out.startswith("Error:")
    assert "matched 2 times" in out
    assert path.read_text() == "x = 1\nx = 1\n"


def test_edit_rejects_large_existing_file(tmp_path: Path):
    path = tmp_path / "big.py"
    path.write_text("x" * (editor.FULL_FILE_EDIT_MAX_CHARS + 1))
    out = editor.tool_function(
        command="edit",
        path=str(path),
        file_text="tiny\n",
    )
    assert out.startswith("Error:")
    assert "str_replace" in out
    assert path.read_text().startswith("x")


def test_edit_rejects_truncated_rewrite_of_small_file(tmp_path: Path):
    path = tmp_path / "mid.py"
    path.write_text("a" * 1000)
    out = editor.tool_function(
        command="edit",
        path=str(path),
        file_text="a" * 100,
    )
    assert out.startswith("Error:")
    assert "truncated" in out.lower() or "str_replace" in out
    assert len(path.read_text()) == 1000


def test_edit_allows_full_rewrite_of_small_file(tmp_path: Path):
    path = tmp_path / "small.py"
    path.write_text("old\n")
    out = editor.tool_function(
        command="edit",
        path=str(path),
        file_text="new content\n",
    )
    assert "overwritten" in out
    assert path.read_text() == "new content\n"


def test_view_range_returns_slice(tmp_path: Path):
    path = tmp_path / "lines.py"
    path.write_text("\n".join(f"line{i}" for i in range(1, 6)) + "\n")
    out = editor.tool_function(
        command="view",
        path=str(path),
        view_range=[2, 4],
    )
    assert "line2" in out
    assert "line4" in out
    assert "line1" not in out
    assert "line5" not in out


def test_tool_info_advertises_str_replace():
    info = editor.tool_info()
    assert "str_replace" in info["input_schema"]["properties"]["command"]["enum"]
    assert "str_replace" in info["description"]
    assert "old_str" in info["input_schema"]["properties"]
