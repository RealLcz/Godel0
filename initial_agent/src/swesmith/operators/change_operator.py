from __future__ import annotations

import ast
from typing import List, Optional

from .base import (
    MutationSite,
    ProceduralOperator,
    build_parent_map,
    ast_path_for,
    get_source_segment,
    make_site_id,
    make_rng,
)


COMPARE_SWAPS: dict = {
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Lt: ast.GtE,
    ast.GtE: ast.Lt,
    ast.Gt: ast.LtE,
    ast.LtE: ast.Gt,
}

BINOP_SWAPS: dict = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.Div,
    ast.Div: ast.Mult,
}

_OPERATOR_SYMBOLS: dict = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.GtE: ">=",
    ast.Gt: ">",
    ast.LtE: "<=",
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
}


class ChangeOperator:
    name: str = "change_operator"

    def enumerate_sites(self, source: str, target_symbol: str = "") -> List[MutationSite]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        parent_map = build_parent_map(tree)
        sites: List[MutationSite] = []
        module_nodes = list(ast.walk(tree))

        for node in module_nodes:
            if isinstance(node, ast.Compare):
                for op_idx, op in enumerate(node.ops):
                    if type(op) in COMPARE_SWAPS:
                        sites.append(self._make_site(node, op_idx, "compare", parent_map, source))
            elif isinstance(node, ast.BinOp):
                if type(node.op) in BINOP_SWAPS:
                    sites.append(self._make_site(node, 0, "binop", parent_map, source))

        if target_symbol:
            sites = self._filter_by_symbol(tree, sites, target_symbol, parent_map)

        return sites

    def _make_site(
        self,
        node: ast.AST,
        op_idx: int,
        kind: str,
        parent_map: dict,
        source: str,
    ) -> MutationSite:
        path = ast_path_for(node, parent_map)
        line = getattr(node, "lineno", 1)
        col = getattr(node, "col_offset", 0)
        op_type = ast.Compare if kind == "compare" else ast.BinOp
        if kind == "compare":
            op = node.ops[op_idx]
        else:
            op = node.op
        before = get_source_segment(source, node)
        seed = abs(hash((self.name, path, line, col, op_idx))) % (2 ** 31)
        return MutationSite(
            site_id=make_site_id(self.name, path, line, col),
            ast_node_type=type(node).__name__,
            ast_path=path,
            line=line,
            col=col,
            seed=seed,
            before_snippet=before,
            after_snippet="",
            metadata={"kind": kind, "op_index": op_idx, "operator": _OPERATOR_SYMBOLS.get(type(op), "?")},
        )

    def _filter_by_symbol(
        self,
        tree: ast.Module,
        sites: List[MutationSite],
        target_symbol: str,
        parent_map: dict,
    ) -> List[MutationSite]:
        symbol_nodes = self._collect_symbol_node_ids(tree, target_symbol)
        if not symbol_nodes:
            return sites
        filtered: List[MutationSite] = []
        for site in sites:
            site_node = self._find_node_by_path(tree, site.ast_path, parent_map)
            if site_node is not None and self._is_within_symbol(site_node, symbol_nodes, parent_map):
                filtered.append(site)
        return filtered

    def _collect_symbol_node_ids(self, tree: ast.Module, target_symbol: str) -> set:
        ids: set = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == target_symbol:
                    for child in ast.walk(node):
                        ids.add(id(child))
        return ids

    def _find_node_by_path(self, tree: ast.Module, path: str, parent_map: dict) -> Optional[ast.AST]:
        for node in ast.walk(tree):
            if ast_path_for(node, parent_map) == path:
                return node
        return None

    def _is_within_symbol(self, node: ast.AST, symbol_ids: set, parent_map: dict) -> bool:
        current: Optional[ast.AST] = node
        seen: set = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if id(current) in symbol_ids:
                return True
            entry = parent_map.get(id(current))
            current = entry[0] if entry else None
        return False

    def apply(self, source: str, site: MutationSite) -> str:
        """Apply the operator swap using line-level text replacement.

        Instead of using ast.unparse (which reformats the entire file),
        we modify only the specific line where the operator appears.
        This produces minimal diffs suitable for bug candidates.
        """
        lines = source.splitlines(keepends=True)
        if site.line < 1 or site.line > len(lines):
            return source

        line_idx = site.line - 1
        line = lines[line_idx]

        old_op = site.metadata.get("operator", "")
        new_op = self._get_swap_symbol(old_op)
        if not old_op or not new_op:
            return source

        # Replace the operator in the line (only the first occurrence at or after col)
        col = site.col
        # Find the operator in the line
        idx = line.find(old_op, col)
        if idx == -1:
            # Try finding it anywhere in the line
            idx = line.find(old_op)
        if idx == -1:
            return source

        new_line = line[:idx] + new_op + line[idx + len(old_op):]
        lines[line_idx] = new_line
        new_source = "".join(lines)

        # Verify the result is valid Python
        try:
            ast.parse(new_source)
        except SyntaxError:
            return source

        site.after_snippet = new_line.strip()
        return new_source

    def _get_swap_symbol(self, old_symbol: str) -> str:
        swaps = {
            "==": "!=",
            "!=": "==",
            "<": ">=",
            ">=": "<",
            ">": "<=",
            "<=": ">",
            "+": "-",
            "-": "+",
            "*": "/",
            "/": "*",
        }
        return swaps.get(old_symbol, "")
