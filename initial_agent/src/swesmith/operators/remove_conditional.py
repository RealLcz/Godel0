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
)


class RemoveConditional:
    name: str = "remove_conditional"

    def enumerate_sites(self, source: str, target_symbol: str = "") -> List[MutationSite]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        parent_map = build_parent_map(tree)
        sites: List[MutationSite] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            sites.append(self._make_site(node, parent_map, source))

        if target_symbol:
            sites = self._filter_by_symbol(tree, sites, target_symbol, parent_map)

        return sites

    def _make_site(self, node: ast.If, parent_map: dict, source: str) -> MutationSite:
        path = ast_path_for(node, parent_map)
        line = getattr(node, "lineno", 1)
        col = getattr(node, "col_offset", 0)
        before = get_source_segment(source, node)
        seed = abs(hash((self.name, path, line, col))) % (2 ** 31)
        return MutationSite(
            site_id=make_site_id(self.name, path, line, col),
            ast_node_type=type(node).__name__,
            ast_path=path,
            line=line,
            col=col,
            seed=seed,
            before_snippet=before,
            after_snippet="",
            metadata={"has_orelse": bool(node.orelse)},
        )

    def _filter_by_symbol(
        self,
        tree: ast.Module,
        sites: List[MutationSite],
        target_symbol: str,
        parent_map: dict,
    ) -> List[MutationSite]:
        symbol_ids = self._collect_symbol_node_ids(tree, target_symbol)
        if not symbol_ids:
            return sites
        filtered: List[MutationSite] = []
        for site in sites:
            site_node = self._find_node_by_position_and_path(tree, site, parent_map)
            if site_node is not None and self._is_within_symbol(site_node, symbol_ids, parent_map):
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

    def _find_node_by_position_and_path(
        self,
        tree: ast.Module,
        site: MutationSite,
        parent_map: dict,
    ) -> Optional[ast.AST]:
        for node in ast.walk(tree):
            if (
                getattr(node, "lineno", None) == site.line
                and getattr(node, "col_offset", None) == site.col
                and ast_path_for(node, parent_map) == site.ast_path
            ):
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
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source

        parent_map = build_parent_map(tree)
        target_node: Optional[ast.AST] = None
        for node in ast.walk(tree):
            if (
                getattr(node, "lineno", None) == site.line
                and getattr(node, "col_offset", None) == site.col
                and ast_path_for(node, parent_map) == site.ast_path
            ):
                target_node = node
                break

        if target_node is None or not isinstance(target_node, ast.If):
            return source

        if not target_node.body:
            return source

        replacement_stmts = list(target_node.body)
        self._replace_in_parent(tree, target_node, replacement_stmts, parent_map)

        try:
            new_source = ast.unparse(tree)
        except Exception:
            return source

        site.after_snippet = "\n".join(ast.unparse(s) for s in replacement_stmts)
        return new_source

    def _replace_in_parent(
        self,
        tree: ast.Module,
        target: ast.AST,
        replacement: List[ast.stmt],
        parent_map: dict,
    ) -> None:
        entry = parent_map.get(id(target))
        if entry is None:
            return
        parent, field_name, position = entry
        if parent is None or field_name is None:
            return

        attr = getattr(parent, field_name, None)
        if isinstance(attr, list) and position is not None:
            attr[position:position + 1] = replacement
        elif isinstance(attr, ast.AST) and replacement:
            setattr(parent, field_name, replacement[0])
