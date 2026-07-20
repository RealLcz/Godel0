"""Atomic file write utilities."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Union

import yaml


def atomic_write_json(path: Union[str, Path], data: Any, indent: int = 2) -> None:
    """Write JSON atomically: write to .tmp, fsync, rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, indent=indent, default=str, ensure_ascii=False)
    _atomic_write_bytes(path, content.encode("utf-8"))


def atomic_write_yaml(path: Union[str, Path], data: Any) -> None:
    """Write YAML atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
    _atomic_write_bytes(path, content.encode("utf-8"))


def atomic_write_text(path: Union[str, Path], content: str) -> None:
    """Write text atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(path, content.encode("utf-8"))


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def read_json(path: Union[str, Path]) -> Any:
    """Read a JSON file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_yaml(path: Union[str, Path]) -> Any:
    """Read a YAML file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
