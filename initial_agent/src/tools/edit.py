# This file is adapted from https://github.com/jennyzzt/dgm.

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional, Sequence, Union

# Full-file overwrite is unsafe above this size: models routinely emit truncated
# file_text and then revert, leaving an empty evolution diff.
FULL_FILE_EDIT_MAX_CHARS = 8000


def tool_info():
    return {
        "name": "editor",
        "description": """Custom editing tool for viewing, creating, and editing files
* State is persistent across command calls and discussions with the user.
* Prefer `str_replace` for modifying existing files. Pass exact `old_str` / `new_str`.
* Use `create` only for new files. Keep new helpers small; wire them with `str_replace`.
* `edit` overwrites an entire existing file and is ONLY allowed for small files
  (at most {max_chars} characters). For larger files it returns an error — use
  `str_replace` instead. Never rewrite a large module by regenerating all of it.
* `view` shows a file or directory. Optional `view_range` [start_line, end_line]
  (1-indexed, inclusive) views a slice without loading the whole file.
* If a `command` generates a long output, it will be truncated and marked with
  `<response clipped>`.
* The `create` command cannot be used if the specified `path` already exists.
""".format(
            max_chars=FULL_FILE_EDIT_MAX_CHARS
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "edit", "str_replace"],
                    "description": (
                        "Command to run: `view`, `create`, `str_replace` (preferred "
                        "for edits), or `edit` (small files only)."
                    ),
                },
                "path": {
                    "description": "Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`.",
                    "type": "string",
                },
                "file_text": {
                    "description": (
                        "Required for `create` (and for `edit` on small files): full "
                        "file contents. Do not use this to rewrite large files."
                    ),
                    "type": "string",
                },
                "old_str": {
                    "description": (
                        "Required for `str_replace`: exact existing text to replace. "
                        "Must appear exactly once in the file."
                    ),
                    "type": "string",
                },
                "new_str": {
                    "description": (
                        "Required for `str_replace`: replacement text (may be empty "
                        "to delete `old_str`)."
                    ),
                    "type": "string",
                },
                "view_range": {
                    "description": (
                        "Optional for `view` on a file: [start_line, end_line], "
                        "1-indexed inclusive."
                    ),
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
            "required": ["command", "path"],
        },
    }


def maybe_truncate(content: str, max_length: int = 10000) -> str:
    """Truncate long content and add marker."""
    if len(content) > max_length:
        return content[:max_length] + "\n<response clipped>"
    return content


def validate_path(path: str, command: str) -> Path:
    """
    Validate the file path for each command:
      - 'view': path may be a file or directory; must exist.
      - 'create': path must not exist (for new file creation).
      - 'edit' / 'str_replace': path must exist and be a file.
    """
    path_obj = Path(path)

    if not path_obj.is_absolute():
        raise ValueError(
            f"The path {path} is not an absolute path (must start with '/')."
        )

    if command == "view":
        if not path_obj.exists():
            raise ValueError(f"The path {path} does not exist.")
    elif command == "create":
        if path_obj.exists():
            raise ValueError(f"Cannot create new file; {path} already exists.")
    elif command in ("edit", "str_replace"):
        if not path_obj.exists():
            raise ValueError(f"The file {path} does not exist.")
        if path_obj.is_dir():
            raise ValueError(f"{path} is a directory and cannot be edited as a file.")
    else:
        raise ValueError(f"Unknown or unsupported command: {command}")

    return path_obj


def format_output(content: str, path: str, init_line: int = 1) -> str:
    """Format output with line numbers (for file content)."""
    content = maybe_truncate(content)
    content = content.expandtabs()
    numbered_lines = [
        f"{i + init_line:6}\t{line}" for i, line in enumerate(content.split("\n"))
    ]
    return (
        f"Here's the result of running `cat -n` on {path}:\n"
        + "\n".join(numbered_lines)
        + "\n"
    )


def read_file(path: Path) -> str:
    """Read and return the entire file contents."""
    try:
        return path.read_text()
    except Exception as e:
        raise ValueError(f"Failed to read file: {e}")


def write_file(path: Path, content: str):
    """Write (overwrite) entire file contents."""
    try:
        path.write_text(content)
    except Exception as e:
        raise ValueError(f"Failed to write file: {e}")


def _normalize_view_range(
    view_range: Optional[Union[Sequence[int], str]],
) -> Optional[List[int]]:
    if view_range is None:
        return None
    if isinstance(view_range, str):
        text = view_range.strip().strip("[]")
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError("view_range must be [start_line, end_line].")
        start, end = int(parts[0]), int(parts[1])
    else:
        if len(view_range) != 2:
            raise ValueError("view_range must be [start_line, end_line].")
        start, end = int(view_range[0]), int(view_range[1])
    if start < 1 or end < start:
        raise ValueError(
            "view_range lines must be 1-indexed with end_line >= start_line."
        )
    return [start, end]


def view_path(
    path_obj: Path,
    view_range: Optional[Union[Sequence[int], str]] = None,
) -> str:
    """View file contents (optional line range) or a directory listing."""
    if path_obj.is_dir():
        if view_range is not None:
            raise ValueError("view_range is only supported when path is a file.")
        try:
            result = subprocess.run(
                ["find", str(path_obj), "-maxdepth", "2", "-not", "-path", "*/\\.*"],
                capture_output=True,
                text=True,
            )
            if result.stderr:
                return f"Error listing directory: {result.stderr}"
            return (
                f"Here's the files and directories up to 2 levels deep in {path_obj}, excluding hidden items:\n"
                + result.stdout
            )
        except Exception as e:
            raise ValueError(f"Failed to list directory: {e}")

    content = read_file(path_obj)
    lines = content.split("\n")
    normalized = _normalize_view_range(view_range)
    if normalized is None:
        return format_output(content, str(path_obj))

    start, end = normalized
    if start > len(lines):
        raise ValueError(
            f"view_range start_line {start} is past end of file ({len(lines)} lines)."
        )
    end = min(end, len(lines))
    sliced = "\n".join(lines[start - 1 : end])
    return format_output(sliced, str(path_obj), init_line=start)


def str_replace_in_file(path_obj: Path, old_str: str, new_str: str) -> str:
    """Replace exactly one occurrence of old_str with new_str."""
    if old_str is None:
        raise ValueError("Missing required `old_str` for 'str_replace' command.")
    if new_str is None:
        raise ValueError("Missing required `new_str` for 'str_replace' command.")
    if old_str == "":
        raise ValueError("`old_str` must be non-empty.")

    content = read_file(path_obj)
    count = content.count(old_str)
    if count == 0:
        raise ValueError(
            "`old_str` not found in file. Read the file (optionally with "
            "`view_range`) and pass an exact unique snippet."
        )
    if count > 1:
        raise ValueError(
            f"`old_str` matched {count} times; provide a larger unique snippet "
            "so exactly one occurrence is replaced."
        )

    write_file(path_obj, content.replace(old_str, new_str, 1))
    return f"Successfully replaced text in {path_obj}."


def tool_function(
    command: str,
    path: str,
    file_text: str = None,
    old_str: str = None,
    new_str: str = None,
    view_range=None,
    **_ignored,
) -> str:
    """
    Main tool function that handles:
      - 'view'        : View file/dir; optional view_range for files
      - 'create'      : Create a new file with file_text
      - 'str_replace' : Exact one-occurrence string replacement (preferred)
      - 'edit'        : Full overwrite; rejected for large files
    """
    try:
        path_obj = validate_path(path, command)

        if command == "view":
            return view_path(path_obj, view_range=view_range)

        if command == "create":
            if file_text is None:
                raise ValueError("Missing required `file_text` for 'create' command.")
            write_file(path_obj, file_text)
            return f"File created successfully at: {path}"

        if command == "str_replace":
            return str_replace_in_file(path_obj, old_str, new_str)

        if command == "edit":
            if file_text is None:
                raise ValueError("Missing required `file_text` for 'edit' command.")
            existing = read_file(path_obj)
            if len(existing) > FULL_FILE_EDIT_MAX_CHARS:
                raise ValueError(
                    f"Refusing full-file `edit` on {path}: file has "
                    f"{len(existing)} characters (limit "
                    f"{FULL_FILE_EDIT_MAX_CHARS}). Use `str_replace` with a "
                    "unique old_str/new_str snippet, or `create` a small new "
                    "helper and wire it in with `str_replace`. Do not regenerate "
                    "the entire file."
                )
            if len(file_text) < max(1, len(existing) // 2):
                raise ValueError(
                    f"Refusing truncated full-file `edit` on {path}: new "
                    f"file_text has {len(file_text)} characters but the existing "
                    f"file has {len(existing)}. Use `str_replace` instead of "
                    "rewriting the whole file."
                )
            write_file(path_obj, file_text)
            return f"File at {path} has been overwritten with new content."

        raise ValueError(f"Unknown command: {command}")

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    result = tool_function("view", str(Path(__file__).resolve()), view_range=[1, 10])
    print(result)
