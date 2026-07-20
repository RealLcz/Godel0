"""Pytest plugin that records production call edges during one test call.

The plugin is configured entirely through environment variables so it can be
loaded in an arbitrary target repository with ``pytest -p``.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import CodeType, FrameType
from typing import Any, Dict, Optional, Tuple

import pytest


TRACE_OUTPUT_ENV = "GODEL0_TRACE_OUTPUT"
TRACE_REPO_ENV = "GODEL0_TRACE_REPO_ROOT"
TRACE_SOURCE_ROOTS_ENV = "GODEL0_TRACE_SOURCE_ROOTS"


class RuntimeContractTrace:
    """Collect Python call edges whose callee belongs to production code."""

    def __init__(self, repo_root: Path, source_roots: list[Path], nodeid: str) -> None:
        self.repo_root = repo_root.resolve()
        self.source_roots = tuple(path.resolve() for path in source_roots)
        self.nodeid = nodeid
        self.started_at = 0.0
        self.elapsed_sec = 0.0
        self.calls: Counter[Tuple[str, str, int]] = Counter()
        self.edges: Counter[Tuple[str, str, str, str, int]] = Counter()
        self.entrypoints: Counter[Tuple[str, str, int]] = Counter()
        self._code_cache: Dict[CodeType, Optional[Tuple[str, str, int]]] = {}
        self._previous_sys_profile: Any = None
        self._previous_thread_profile: Any = None

    def start(self) -> None:
        self.started_at = time.monotonic()
        self._previous_sys_profile = sys.getprofile()
        get_thread_profile = getattr(threading, "getprofile", None)
        self._previous_thread_profile = (
            get_thread_profile() if get_thread_profile is not None else None
        )
        sys.setprofile(self._profile)
        threading.setprofile(self._profile)

    def stop(self) -> None:
        sys.setprofile(self._previous_sys_profile)
        threading.setprofile(self._previous_thread_profile)
        self.elapsed_sec = time.monotonic() - self.started_at

    def _profile(self, frame: FrameType, event: str, _arg: Any) -> None:
        if event != "call":
            return
        callee = self._code_ref(frame.f_code)
        if callee is None:
            return
        self.calls[callee] += 1
        caller_frame = frame.f_back
        caller = self._code_ref(caller_frame.f_code) if caller_frame else None
        if caller is None:
            self.entrypoints[callee] += 1
            return
        caller_path, caller_symbol, _caller_first_line = caller
        callee_path, callee_symbol, _callee_first_line = callee
        call_line = int(caller_frame.f_lineno) if caller_frame else 0
        self.edges[
            (caller_path, caller_symbol, callee_path, callee_symbol, call_line)
        ] += 1

    def _code_ref(self, code: CodeType) -> Optional[Tuple[str, str, int]]:
        if code in self._code_cache:
            return self._code_cache[code]
        try:
            path = Path(code.co_filename).resolve()
        except (OSError, RuntimeError):
            self._code_cache[code] = None
            return None
        if not any(_is_relative_to(path, root) for root in self.source_roots):
            self._code_cache[code] = None
            return None
        try:
            relative = path.relative_to(self.repo_root).as_posix()
        except ValueError:
            self._code_cache[code] = None
            return None
        symbol = str(getattr(code, "co_qualname", code.co_name))
        value = (relative, symbol, int(code.co_firstlineno))
        self._code_cache[code] = value
        return value

    def to_dict(self) -> dict[str, Any]:
        symbols_by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
        file_call_counts: Counter[str] = Counter()
        for (path, symbol, first_line), count in self.calls.items():
            symbols_by_file[path].append(
                {
                    "symbol": symbol,
                    "first_line": first_line,
                    "call_count": count,
                }
            )
            file_call_counts[path] += count

        file_edges: Counter[Tuple[str, str]] = Counter()
        function_edges: list[dict[str, Any]] = []
        for (caller_path, caller_symbol, callee_path, callee_symbol, line), count in self.edges.items():
            if caller_path != callee_path:
                file_edges[(caller_path, callee_path)] += count
            function_edges.append(
                {
                    "caller_path": caller_path,
                    "caller_symbol": caller_symbol,
                    "call_line": line,
                    "callee_path": callee_path,
                    "callee_symbol": callee_symbol,
                    "call_count": count,
                }
            )

        files = [
            {
                "path": path,
                "call_count": file_call_counts[path],
                "symbols": sorted(
                    symbols_by_file[path],
                    key=lambda row: (-row["call_count"], row["first_line"], row["symbol"]),
                ),
            }
            for path in sorted(symbols_by_file)
        ]
        return {
            "schema_version": 1,
            "nodeid": self.nodeid,
            "elapsed_sec": round(self.elapsed_sec, 6),
            "production_file_count": len(files),
            "files": files,
            "file_edges": [
                {"caller": caller, "callee": callee, "call_count": count}
                for (caller, callee), count in sorted(file_edges.items())
            ],
            "function_edges": sorted(
                function_edges,
                key=lambda row: (
                    row["caller_path"],
                    row["callee_path"],
                    row["caller_symbol"],
                    row["callee_symbol"],
                    row["call_line"],
                ),
            ),
            "entrypoints": [
                {
                    "path": path,
                    "symbol": symbol,
                    "first_line": first_line,
                    "call_count": count,
                }
                for (path, symbol, first_line), count in sorted(self.entrypoints.items())
            ],
        }


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _build_trace(nodeid: str) -> Optional[RuntimeContractTrace]:
    output = os.environ.get(TRACE_OUTPUT_ENV, "").strip()
    repo_value = os.environ.get(TRACE_REPO_ENV, "").strip()
    if not output or not repo_value:
        return None
    repo_root = Path(repo_value).resolve()
    configured = os.environ.get(TRACE_SOURCE_ROOTS_ENV, "lib").strip()
    source_roots = [
        repo_root / value
        for value in configured.split(os.pathsep)
        if value.strip()
    ]
    return RuntimeContractTrace(repo_root, source_roots, nodeid)


def _write_trace(trace: RuntimeContractTrace, outcome: Any) -> None:
    output = Path(os.environ[TRACE_OUTPUT_ENV]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = trace.to_dict()
    payload["pytest_exception"] = ""
    if outcome is not None and getattr(outcome, "excinfo", None) is not None:
        excinfo = outcome.excinfo
        payload["pytest_exception"] = str(excinfo[1])[:2000]
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(output)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_call(item: pytest.Item):
    trace = _build_trace(item.nodeid)
    if trace is None:
        yield
        return
    trace.start()
    outcome = None
    try:
        outcome = yield
    finally:
        trace.stop()
        _write_trace(trace, outcome)


__all__ = ["RuntimeContractTrace"]
