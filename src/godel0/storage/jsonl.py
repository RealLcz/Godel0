"""JSONL append/read utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator, Union


def append_jsonl(path: Union[str, Path], record: Any, flush: bool = True) -> None:
    """Append a single record as a JSON line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=str, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        if flush:
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass


def read_jsonl(path: Union[str, Path]) -> Iterator[Any]:
    """Yield records from a JSONL file."""
    path = Path(path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def read_all_jsonl(path: Union[str, Path]) -> list[Any]:
    """Read all records from a JSONL file into a list."""
    return list(read_jsonl(path))


def count_jsonl(path: Union[str, Path]) -> int:
    """Count lines in a JSONL file."""
    path = Path(path)
    if not path.exists():
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count
