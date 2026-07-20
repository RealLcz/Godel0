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


class ChangeConstant:
    name: str = "change_constant"

    def enumerate_sites(self, source: str, target_symbol: str = "") -> List[MutationSite]:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        parent_map = build_parent_map(tree)
        sites: List[MutationSite] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant):
                continue
            if isinstance(node.value, bool):
                continue
            if isinstance(node.value, (int, float, complex)):
                sites.append(self._make_site(node, "number", parent_map, source))
            elif isinstance(node.value, str):
                sites.append(self._make_site(node, "str", parent_map, source))

        if target_symbol:
            sites = self._filter_by_symbol(tree, sites, target_symbol, parent_map)

        return sites

    def _make_site(
        self,
        node: ast.Constant,
        kind: str,
        parent_map: dict,
        source: str,
    ) -> MutationSite:
        path = ast_path_for(node, parent_map)
        line = getattr(node, "lineno", 1)
        col = getattr(node, "col_offset", 0)
        before = get_source_segment(source, node)
        seed = abs(hash((self.name, path, line, col, repr(node.value)))) % (2 ** 31)
        rng = make_rng(seed)
        mutation_choice = rng.choice(["negate", "plus_one", "zero"]) if kind == "number" else rng.choice(["append", "truncate"])
        return MutationSite(
            site_id=make_site_id(self.name, path, line, col),
            ast_node_type=type(node).__name__,
            ast_path=path,
            line=line,
            col=col,
            seed=seed,
            before_snippet=before,
            after_snippet="",
            metadata={"kind": kind, "value": repr(node.value), "mutation": mutation_choice},
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

    def _mutate_value(self, value, mutation: str):
        if isinstance(value, bool):
            return value
        if mutation == "negate":
            if isinstance(value, complex):
                return -value
            return -value
        elif mutation == "plus_one":
            return value + 1
        elif mutation == "zero":
            if isinstance(value, complex):
                return complex(0)
            return type(value)(0)
        elif mutation == "append":
            return str(value) + "x"
        elif mutation == "truncate":
            s = str(value)
            return s[:-1] if len(s) > 1 else s
        return value

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

        if target_node is None or not isinstance(target_node, ast.Constant):
            return source

        mutation = site.metadata.get("mutation", "")
        new_value = self._mutate_value(target_node.value, mutation)
        target_node.value = new_value

        try:
            new_source = ast.unparse(tree)
        except Exception:
            return source

        site.after_snippet = repr(new_value)
        return new_source
