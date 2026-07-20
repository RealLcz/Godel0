from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Entity:
    name: str
    kind: str
    qualname: str
    file: str
    line: int
    col: int
    end_line: int = 0
    end_col: int = 0
    args: List[str] = field(default_factory=list)
    docstring: str = ""
    source: str = ""
    decorators: List[str] = field(default_factory=list)


class EntityIndex:
    def __init__(self) -> None:
        self.entities: List[Entity] = []
        self._by_name: Dict[str, List[Entity]] = {}

    def index_file(self, file_path: str, source: Optional[str] = None) -> None:
        if source is None:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

        try:
            tree = ast.parse(source, filename=file_path)
        except SyntaxError:
            return

        rel_path = file_path
        for node in tree.body:
            self._index_node(node, rel_path, source, prefix="")

    def index_directory(self, dir_path: str, suffix: str = ".py") -> None:
        for root, dirs, files in os.walk(dir_path):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fname in sorted(files):
                if fname.endswith(suffix):
                    fpath = os.path.join(root, fname)
                    self.index_file(fpath)

    def _index_node(
        self,
        node: ast.AST,
        file_path: str,
        source: str,
        prefix: str,
    ) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualname = f"{prefix}.{node.name}" if prefix else node.name
            args = [a.arg for a in node.args.args]
            docstring = ast.get_docstring(node) or ""
            try:
                src_seg = ast.get_source_segment(source, node) or ""
            except Exception:
                src_seg = ""
            decorators = [ast.unparse(d) for d in node.decorator_list]
            entity = Entity(
                name=node.name,
                kind="async_function" if isinstance(node, ast.AsyncFunctionDef) else "function",
                qualname=qualname,
                file=file_path,
                line=node.lineno,
                col=node.col_offset,
                end_line=getattr(node, "end_lineno", node.lineno) or node.lineno,
                end_col=getattr(node, "end_col_offset", node.col_offset) or node.col_offset,
                args=args,
                docstring=docstring,
                source=src_seg,
                decorators=decorators,
            )
            self.entities.append(entity)
            self._by_name.setdefault(node.name, []).append(entity)
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    self._index_node(child, file_path, source, prefix=qualname)

        elif isinstance(node, ast.ClassDef):
            qualname = f"{prefix}.{node.name}" if prefix else node.name
            docstring = ast.get_docstring(node) or ""
            try:
                src_seg = ast.get_source_segment(source, node) or ""
            except Exception:
                src_seg = ""
            decorators = [ast.unparse(d) for d in node.decorator_list]
            entity = Entity(
                name=node.name,
                kind="class",
                qualname=qualname,
                file=file_path,
                line=node.lineno,
                col=node.col_offset,
                end_line=getattr(node, "end_lineno", node.lineno) or node.lineno,
                end_col=getattr(node, "end_col_offset", node.col_offset) or node.col_offset,
                docstring=docstring,
                source=src_seg,
                decorators=decorators,
            )
            self.entities.append(entity)
            self._by_name.setdefault(node.name, []).append(entity)
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    self._index_node(child, file_path, source, prefix=qualname)

    def find(self, name: str) -> List[Entity]:
        return self._by_name.get(name, [])

    def find_in_file(self, file_path: str, name: Optional[str] = None) -> List[Entity]:
        results = [e for e in self.entities if e.file == file_path]
        if name:
            results = [e for e in results if e.name == name]
        return results

    def get_source_for_entity(self, entity: Entity, source: Optional[str] = None) -> str:
        if source is None and entity.source:
            return entity.source
        if source is None:
            return entity.source
        try:
            tree = ast.parse(source, filename=entity.file)
        except SyntaxError:
            return entity.source
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and node.name == entity.name
                and node.lineno == entity.line
            ):
                seg = ast.get_source_segment(source, node)
                return seg or entity.source
        return entity.source

    def all_files(self) -> List[str]:
        seen: List[str] = []
        for e in self.entities:
            if e.file not in seen:
                seen.append(e.file)
        return seen

    def __len__(self) -> int:
        return len(self.entities)
