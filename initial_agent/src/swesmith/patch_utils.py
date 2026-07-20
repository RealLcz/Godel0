from __future__ import annotations

import difflib
import re
from typing import List, Optional, Tuple


def make_diff(
    original: str,
    modified: str,
    filename: str = "source.py",
    original_label: str = "a",
    modified_label: str = "b",
) -> str:
    original_lines = original.splitlines(keepends=True)
    modified_lines = modified.splitlines(keepends=True)

    if original_lines and not original_lines[-1].endswith("\n"):
        original_lines[-1] += "\n"
    if modified_lines and not modified_lines[-1].endswith("\n"):
        modified_lines[-1] += "\n"

    diff = difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile=f"{original_label}/{filename}",
        tofile=f"{modified_label}/{filename}",
        lineterm="\n",
    )
    return "".join(diff)


def make_git_diff(
    original: str,
    modified: str,
    filename: str = "source.py",
) -> str:
    header = f"diff --git a/{filename} b/{filename}\n"
    body = make_diff(original, modified, filename=filename)
    if not body:
        return ""
    return header + body


def apply_patch_to_string(original: str, patch: str) -> str:
    lines = original.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"

    hunks = _parse_hunks(patch)
    result_lines = list(lines)

    for hunk_start, hunk_old_count, hunk_new_count, hunk_lines in hunks:
        new_block: List[str] = []
        consumed = 0
        for hline in hunk_lines:
            if hline.startswith("+"):
                new_block.append(hline[1:])
            elif hline.startswith("-"):
                consumed += 1
            elif hline.startswith(" "):
                new_block.append(hline[1:])
                consumed += 1
            elif hline.startswith("\\"):
                pass
            else:
                continue

        idx = hunk_start - 1
        end_idx = idx + consumed
        result_lines[idx:end_idx] = new_block

    return "".join(result_lines)


def _parse_hunks(patch: str) -> List[Tuple[int, int, int, List[str]]]:
    hunks: List[Tuple[int, int, int, List[str]]] = []
    current_lines: List[str] = []
    current_start = 0
    current_old = 0
    current_new = 0
    in_hunk = False

    for line in patch.splitlines():
        if line.startswith("@@"):
            if in_hunk:
                hunks.append((current_start, current_old, current_new, current_lines))
            match = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if match:
                current_start = int(match.group(1))
                current_old = int(match.group(2)) if match.group(2) else 1
                current_new = int(match.group(4)) if match.group(4) else 1
            else:
                current_start = 1
                current_old = 0
                current_new = 0
            current_lines = []
            in_hunk = True
        elif in_hunk:
            current_lines.append(line)

    if in_hunk:
        hunks.append((current_start, current_old, current_new, current_lines))

    return hunks


def extract_changed_files(patch: str) -> List[str]:
    files: List[str] = []
    for line in patch.splitlines():
        if line.startswith("diff --git"):
            match = re.match(r"^diff --git a/(.+?) b/(.+)$", line)
            if match:
                f = match.group(2)
                if f not in files:
                    files.append(f)
        elif line.startswith("+++ b/"):
            f = line[len("+++ b/"):].strip()
            if f and f != "/dev/null" and f not in files:
                files.append(f)
    return files


def patch_conflicts(patch_a: str, patch_b: str) -> bool:
    files_a = set(extract_changed_files(patch_a))
    files_b = set(extract_changed_files(patch_b))
    if files_a.isdisjoint(files_b):
        return False

    common = files_a & files_b
    for fname in common:
        hunks_a = _parse_hunks(_filter_patch_for_file(patch_a, fname))
        hunks_b = _parse_hunks(_filter_patch_for_file(patch_b, fname))
        for a_start, a_old, _, _ in hunks_a:
            a_end = a_start + a_old
            for b_start, b_old, _, _ in hunks_b:
                b_end = b_start + b_old
                if a_start < b_end and b_start < a_end:
                    return True
    return False


def _filter_patch_for_file(patch: str, filename: str) -> str:
    lines = patch.splitlines()
    filtered: List[str] = []
    in_block = False
    for line in lines:
        if line.startswith("diff --git"):
            in_block = f"b/{filename}" in line or f"a/{filename}" in line
        if in_block:
            filtered.append(line)
    return "\n".join(filtered)


def count_modified_lines(patch: str) -> int:
    count = 0
    in_hunk = False
    for line in patch.splitlines():
        if line.startswith("@@"):
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            count += 1
        elif line.startswith("-") and not line.startswith("---"):
            count += 1
    return count


def reverse_patch(patch: str) -> str:
    lines = patch.splitlines(keepends=True)
    reversed_lines: List[str] = []
    for line in lines:
        if line.startswith("diff --git"):
            reversed_lines.append(line)
        elif line.startswith("--- a/"):
            reversed_lines.append("--- " + line[len("--- "):].replace("a/", "b/"))
        elif line.startswith("+++ b/"):
            reversed_lines.append("+++ " + line[len("+++ "):].replace("b/", "a/"))
        elif line.startswith("+"):
            reversed_lines.append("-" + line[1:])
        elif line.startswith("-"):
            reversed_lines.append("+" + line[1:])
        else:
            reversed_lines.append(line)
    return "".join(reversed_lines)
