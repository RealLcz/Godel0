"""Unit tests for atomic storage."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from godel0.storage.atomic import atomic_write_json, atomic_write_text, read_json
from godel0.storage.jsonl import append_jsonl, read_all_jsonl, count_jsonl


class TestAtomicWrite:
    def test_write_and_read_json(self, tmp_path):
        path = tmp_path / "test.json"
        data = {"key": "value", "num": 42}
        atomic_write_json(path, data)
        result = read_json(path)
        assert result == data

    def test_write_text(self, tmp_path):
        path = tmp_path / "test.txt"
        atomic_write_text(path, "hello world")
        assert path.read_text() == "hello world"

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "a" / "b" / "c" / "test.json"
        atomic_write_json(path, {"x": 1})
        assert path.exists()

    def test_overwrite(self, tmp_path):
        path = tmp_path / "test.json"
        atomic_write_json(path, {"v": 1})
        atomic_write_json(path, {"v": 2})
        assert read_json(path)["v"] == 2


class TestJSONL:
    def test_append_and_read(self, tmp_path):
        path = tmp_path / "events.jsonl"
        append_jsonl(path, {"event": "a"})
        append_jsonl(path, {"event": "b"})
        records = read_all_jsonl(path)
        assert len(records) == 2
        assert records[0]["event"] == "a"
        assert records[1]["event"] == "b"

    def test_count(self, tmp_path):
        path = tmp_path / "events.jsonl"
        append_jsonl(path, {"event": "a"})
        append_jsonl(path, {"event": "b"})
        assert count_jsonl(path) == 2

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        assert count_jsonl(path) == 0
        assert read_all_jsonl(path) == []

    def test_skip_empty_lines(self, tmp_path):
        path = tmp_path / "events.jsonl"
        path.write_text('{"a": 1}\n\n{"b": 2}\n')
        records = read_all_jsonl(path)
        assert len(records) == 2
