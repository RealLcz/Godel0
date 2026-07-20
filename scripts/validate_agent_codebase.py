#!/usr/bin/env python3
"""Validate an agent codebase for import-ability and basic structure."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def validate_agent_codebase(code_dir: Path) -> bool:
    """Check that the agent codebase can be imported and has required files."""
    code_dir = Path(code_dir)
    ok = True

    required_files = [
        "coding_agent.py",
        "llm_withtools.py",
        "llm.py",
        "tools/__init__.py",
        "tools/bash.py",
        "tools/edit.py",
    ]

    for f in required_files:
        p = code_dir / f
        if not p.exists():
            print(f"MISSING: {f}")
            ok = False
        else:
            print(f"OK: {f}")

    try:
        sys.path.insert(0, str(code_dir))
        import importlib
        mod = importlib.import_module("tools")
        tools = mod.load_all_tools(logging=print)
        print(f"\nTools loaded: {[t['name'] for t in tools]}")
        if len(tools) < 2:
            print("WARNING: fewer than 2 tools loaded")
            ok = False
    except Exception as e:
        print(f"Tool loading failed: {e}")
        ok = False

    proposer_main = code_dir / "proposer" / "proposer_main.py"
    if proposer_main.exists():
        print(f"OK: proposer/proposer_main.py")
    else:
        print(f"MISSING: proposer/proposer_main.py")
        ok = False

    return ok


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-dir", default="initial_agent/src")
    args = parser.parse_args()
    ok = validate_agent_codebase(Path(args.code_dir))
    exit(0 if ok else 1)


if __name__ == "__main__":
    main()
