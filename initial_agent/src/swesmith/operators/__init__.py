from __future__ import annotations

from .base import (
    MutationSite,
    ProceduralOperator,
    build_parent_map,
    ast_path_for,
    get_source_segment,
    make_site_id,
    make_rng,
)
from .change_operator import ChangeOperator
from .change_constant import ChangeConstant
from .invert_if import InvertIfElse
from .remove_conditional import RemoveConditional
from .remove_loop import RemoveLoop
from .remove_assignment import RemoveAssignment
from .remove_wrapper import RemoveWrapper

OPERATORS = {
    "change_operator": ChangeOperator,
    "change_constant": ChangeConstant,
    "invert_if_else": InvertIfElse,
    "remove_conditional": RemoveConditional,
    "remove_loop": RemoveLoop,
    "remove_assignment": RemoveAssignment,
    "remove_wrapper": RemoveWrapper,
}


def get_operator(name: str):
    cls = OPERATORS.get(name)
    if cls is None:
        raise KeyError(f"Unknown operator: {name}. Available: {list(OPERATORS.keys())}")
    return cls()


__all__ = [
    "MutationSite",
    "ProceduralOperator",
    "ChangeOperator",
    "ChangeConstant",
    "InvertIfElse",
    "RemoveConditional",
    "RemoveLoop",
    "RemoveAssignment",
    "RemoveWrapper",
    "OPERATORS",
    "get_operator",
    "build_parent_map",
    "ast_path_for",
    "get_source_segment",
    "make_site_id",
    "make_rng",
]
