#!/usr/bin/env python3
"""Export Solver Core from a DGM/HGM source repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


PROTECTED_PATHS = [
    "coding_agent.py",
    "llm_withtools.py",
    "llm.py",
    "tools",
    "prompts",
    "utils",
    "self_improve_step.py",
    "config.py",
    "config.yaml",
    "tree.py",
    "requirements.txt",
]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def export_solver_core(
    source_repo: Path,
    source_commit: str | None,
    output: Path,
) -> None:
    """Copy protected solver core files from source repo and generate lock file."""
    source = Path(source_repo)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    if source_commit is None:
        import subprocess
        result = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True, text=True,
        )
        source_commit = result.stdout.strip()

    files: dict[str, str] = {}
    for item in PROTECTED_PATHS:
        src = source / item
        if src.is_file():
            dst = output / item
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            files[item] = sha256_file(dst)
        elif src.is_dir():
            for f in sorted(src.rglob("*")):
                if f.is_file() and "__pycache__" not in str(f):
                    rel = f.relative_to(source)
                    dst = output / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dst)
                    files[str(rel)] = sha256_file(dst)

    lock = {
        "baseline_name": "dgm_hgm_solver_core",
        "source_repository": "metauto-ai/HGM",
        "source_commit": source_commit,
        "protected_paths": PROTECTED_PATHS,
        "files": files,
    }

    lock_path = output.parent / "solver_core.lock.json"
    with open(lock_path, "w") as f:
        json.dump(lock, f, indent=2)

    checksums_path = output.parent / "solver_core.checksums.sha256"
    with open(checksums_path, "w") as f:
        for path, hash_val in sorted(files.items()):
            f.write(f"{hash_val}  {path}\n")

    print(f"Exported {len(files)} files to {output}")
    print(f"Lock file: {lock_path}")
    print(f"Checksums: {checksums_path}")


def main():
    parser = argparse.ArgumentParser(description="Export Solver Core from HGM/DGM repo")
    parser.add_argument("--source-repo", required=True, help="Path to source repository")
    parser.add_argument("--source-commit", default=None, help="Source commit SHA")
    parser.add_argument("--output", default="initial_agent/src", help="Output directory")
    args = parser.parse_args()
    export_solver_core(Path(args.source_repo), args.source_commit, Path(args.output))


if __name__ == "__main__":
    main()
