from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, runtime_checkable


@dataclass
class MutationSite:
    site_id: str
    ast_node_type: str
    ast_path: str
    line: int
    col: int
    seed: int = 0
    before_snippet: str = ""
    after_snippet: str = ""
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class ProceduralOperator(Protocol):
    name: str

    def enumerate_sites(
        self,
        source: str,
        target_symbol: str = "",
    ) -> List[MutationSite]:
        ...

    def apply(
        self,
        source: str,
        site: MutationSite,
    ) -> str:
        ...


def make_site_id(operator_name: str, ast_path: str, line: int, col: int) -> str:
    raw = f"{operator_name}:{ast_path}:{line}:{col}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
    return f"{operator_name}_{digest}"


def ast_path_for(node: ast.AST, parent_map: Optional[dict] = None) -> str:
    parts: List[str] = []
    current: Optional[ast.AST] = node
    seen: set = set()
    while current is not None:
        if id(current) in seen:
            break
        seen.add(id(current))
        cls = type(current).__name__
        idx = ""
        if parent_map is not None:
            parent_info = parent_map.get(id(current))
            if parent_info is not None:
                parent_node, field_name, position = parent_info
                idx = f"[{position}]" if position is not None else ""
                field_part = f".{field_name}" if field_name else ""
                parts.append(f"{cls}{field_part}{idx}")
            else:
                parts.append(cls)
        else:
            parts.append(cls)
        current = parent_map[id(current)][0] if parent_map and id(current) in parent_map else None
    parts.reverse()
    return "/".join(parts) if parts else type(node).__name__


def build_parent_map(tree: ast.AST) -> dict:
    parent_map: dict = {}

    def _visit(node: ast.AST, parent: Optional[ast.AST], field_name: str = "", position: Optional[int] = None):
        parent_map[id(node)] = (parent, field_name, position)
        for child_field, child_value in ast.iter_fields(node):
            if isinstance(child_value, list):
                for i, item in enumerate(child_value):
                    if isinstance(item, ast.AST):
                        _visit(item, node, child_field, i)
            elif isinstance(child_value, ast.AST):
                _visit(child_value, node, child_field, None)

    _visit(tree, None)
    return parent_map


def get_source_segment(source: str, node: ast.AST) -> str:
    try:
        seg = ast.get_source_segment(source, node)
        if seg is not None:
            return seg.strip()
    except Exception:
        pass
    try:
        return ast.unparse(node).strip()
    except Exception:
        return ""


def make_rng(seed: int):
    import random
    return random.Random(seed)
