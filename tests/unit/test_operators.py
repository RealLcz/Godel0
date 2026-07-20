"""Unit tests for procedural mutation operators."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

# Add initial_agent/src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "initial_agent" / "src"))

from swesmith.operators.base import MutationSite
from swesmith.operators.change_operator import ChangeOperator
from swesmith.operators.change_constant import ChangeConstant
from swesmith.operators.invert_if import InvertIfElse
from swesmith.operators.remove_assignment import RemoveAssignment
from swesmith.operators.remove_conditional import RemoveConditional
from swesmith.operators.remove_loop import RemoveLoop


SOURCE = """
def clamp(x, low, high):
    if x < low:
        return low
    elif x > high:
        return high
    else:
        return x

def calculate(a, b):
    result = a + b
    for i in range(10):
        result += i
    return result
"""


class TestChangeOperator:
    def test_enumerate_sites(self):
        op = ChangeOperator()
        sites = op.enumerate_sites(SOURCE, "clamp")
        assert len(sites) > 0

    def test_apply_produces_valid_python(self):
        op = ChangeOperator()
        sites = op.enumerate_sites(SOURCE, "clamp")
        if sites:
            result = op.apply(SOURCE, sites[0])
            try:
                ast.parse(result)
            except SyntaxError:
                pytest.fail("Applied mutation produced invalid Python")


class TestChangeConstant:
    def test_enumerate_sites(self):
        op = ChangeConstant()
        sites = op.enumerate_sites(SOURCE, "calculate")
        assert len(sites) > 0


class TestInvertIfElse:
    def test_enumerate_sites(self):
        op = InvertIfElse()
        sites = op.enumerate_sites(SOURCE, "clamp")
        assert len(sites) > 0

    def test_apply_swaps_branches(self):
        op = InvertIfElse()
        sites = op.enumerate_sites(SOURCE, "clamp")
        if sites:
            result = op.apply(SOURCE, sites[0])
            assert result != SOURCE


class TestRemoveAssignment:
    def test_enumerate_sites(self):
        op = RemoveAssignment()
        sites = op.enumerate_sites(SOURCE, "calculate")
        assert len(sites) > 0


class TestRemoveConditional:
    def test_enumerate_sites(self):
        op = RemoveConditional()
        sites = op.enumerate_sites(SOURCE, "clamp")
        assert len(sites) > 0


class TestRemoveLoop:
    def test_enumerate_sites(self):
        op = RemoveLoop()
        sites = op.enumerate_sites(SOURCE, "calculate")
        assert len(sites) > 0
