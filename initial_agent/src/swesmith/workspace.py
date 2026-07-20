from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class WorkspaceSpec:
    workspace_dir: str
    target_file: str = ""
    target_symbol: str = ""
    base_commit: str = ""
    repo_id: str = ""
    expose_full_source: bool = True
    restricted: bool = False
    extra_files: List[str] = field(default_factory=list)


class WorkspaceManager:
    def __init__(self, base_temp_dir: str = "/tmp/swesmith_workspaces") -> None:
        self.base_temp_dir = base_temp_dir

    def create_workspace(
        self,
        spec: WorkspaceSpec,
        source_content: Optional[str] = None,
        source_files: Optional[dict] = None,
    ) -> str:
        ws_dir = spec.workspace_dir
        os.makedirs(ws_dir, exist_ok=True)

        if spec.restricted:
            output_dir = os.path.join(ws_dir, "output")
            os.makedirs(output_dir, exist_ok=True)

        if source_files:
            for rel_path, content in source_files.items():
                dest = os.path.join(ws_dir, rel_path)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest, "w", encoding="utf-8") as f:
                    f.write(content)
        elif source_content is not None and spec.target_file:
            dest = os.path.join(ws_dir, spec.target_file)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "w", encoding="utf-8") as f:
                f.write(source_content)

        return ws_dir

    def create_clean_copy(
        self,
        source_repo_dir: str,
        dest_dir: str,
        ignore_patterns: Optional[List[str]] = None,
    ) -> str:
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        os.makedirs(os.path.dirname(dest_dir), exist_ok=True)

        ignore = shutil.ignore_patterns(
            "__pycache__", "*.pyc", ".git", "*.egg-info", ".pytest_cache"
        )
        if ignore_patterns:
            base_ignore = ignore
            extra = shutil.ignore_patterns(*ignore_patterns)

            def combined(src, names):
                ignored = base_ignore(src, names)
                ignored |= extra(src, names)
                return ignored

            shutil.copytree(source_repo_dir, dest_dir, ignore=combined)
        else:
            shutil.copytree(source_repo_dir, dest_dir, ignore=ignore)

        return dest_dir

    def write_file(self, workspace_dir: str, rel_path: str, content: str) -> str:
        dest = os.path.join(workspace_dir, rel_path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as f:
            f.write(content)
        return dest

    def read_file(self, workspace_dir: str, rel_path: str) -> str:
        path = os.path.join(workspace_dir, rel_path)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def write_request(self, workspace_dir: str, request_text: str) -> str:
        path = os.path.join(workspace_dir, "request.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(request_text)
        return path

    def cleanup(self, workspace_dir: str) -> None:
        if os.path.exists(workspace_dir):
            shutil.rmtree(workspace_dir)

    def list_files(self, workspace_dir: str, suffix: str = ".py") -> List[str]:
        result: List[str] = []
        for root, dirs, files in os.walk(workspace_dir):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fname in sorted(files):
                if fname.endswith(suffix):
                    result.append(os.path.relpath(os.path.join(root, fname), workspace_dir))
        return result
