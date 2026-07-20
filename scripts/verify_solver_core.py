#!/usr/bin/env python3
"""Verify Solver Core checksums against the lock file."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def verify_solver_core(code_dir: Path, lock_file: Path) -> bool:
    """Verify that all protected files match the lock file checksums."""
    code_dir = Path(code_dir)
    lock_file = Path(lock_file)

    if not lock_file.exists():
        print(f"Lock file not found: {lock_file}")
        return False

    with open(lock_file) as f:
        lock = json.load(f)

    all_ok = True
    for rel_path, expected_hash in lock["files"].items():
        file_path = code_dir / rel_path
        if not file_path.exists():
            print(f"MISSING: {rel_path}")
            all_ok = False
            continue
        actual_hash = sha256_file(file_path)
        if actual_hash != expected_hash:
            print(f"MISMATCH: {rel_path}")
            print(f"  expected: {expected_hash}")
            print(f"  actual:   {actual_hash}")
            all_ok = False
        else:
            print(f"OK: {rel_path}")

    if all_ok:
        print(f"\nAll {len(lock['files'])} protected files verified.")
    else:
        print("\nVerification FAILED.")
    return all_ok


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify Solver Core checksums")
    parser.add_argument("--code-dir", default="initial_agent/src")
    parser.add_argument("--lock-file", default="initial_agent/solver_core.lock.json")
    args = parser.parse_args()
    ok = verify_solver_core(Path(args.code_dir), Path(args.lock_file))
    exit(0 if ok else 1)


if __name__ == "__main__":
    main()
