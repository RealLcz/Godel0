"""Tests for typed failure-signature patterns in code localization."""

from proposer.code_locator import CodeLocator, RepoIndex
from proposer.schemas import FailureSignature


def test_typed_function_pattern_selects_exact_symbol():
    index = RepoIndex(
        repo_id="ansible",
        symbols=[
            {
                "file_path": "lib/ansible/__main__.py",
                "symbol_name": "_short_name",
                "symbol_type": "function",
                "line_start": 1,
                "line_end": 3,
                "source": "def _short_name(value): return value",
            },
            {
                "file_path": "lib/ansible/playbook/helpers.py",
                "symbol_name": "load_list_of_blocks",
                "symbol_type": "function",
                "line_start": 20,
                "line_end": 45,
                "source": "def load_list_of_blocks(ds): return []",
            },
        ],
    )
    signature = FailureSignature(
        signature_id="sig",
        code_patterns=["function:load_list_of_blocks"],
    )

    targets = CodeLocator().locate(signature, index)

    assert targets[0].symbol_name == "load_list_of_blocks"
