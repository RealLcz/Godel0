"""Regression tests for LM Modify response handling."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "initial_agent" / "src"))

from swesmith.engine import BugConstraints, BugGenerationPlan, RepoSpec
from swesmith.lm_modify import LMModify, extract_code_block


SOURCE = '''# module comment
import os


def clamp(x, low, high):
    """Clamp x to range."""
    if x < low:
        return low
    if x > high:
        return high
    return x


def untouched():
    return os.getcwd()
'''


class FakeAdapter:
    def __init__(self, response: str):
        self.response = response

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        return self.response


def _plan(max_lines: int = 20) -> BugGenerationPlan:
    return BugGenerationPlan(
        plan_id="lm_test",
        target_repo_id="repo",
        target_base_commit="abc",
        target_file="module.py",
        target_symbol="clamp",
        strategy="lm_modify",
        operator="lm_modify",
        constraints=BugConstraints(max_modified_lines=max_lines),
        seed=1,
    )


def test_extract_code_block_prefers_bugged_code_marker():
    text = """Thinking:
```python
def wrong():
    pass
```

Bugged Code:
```python
def clamp(x, low, high):
    return low
```
"""

    assert "def clamp" in extract_code_block(text)
    assert "def wrong" not in extract_code_block(text)


def test_replace_function_preserves_surrounding_file_text():
    lm = LMModify()
    replacement = '''def clamp(x, low, high):
    """Clamp x to range."""
    if x < low:
        return low
    if x > high:
        return low
    return x
'''

    modified = lm._replace_function_in_file(SOURCE, "clamp", replacement)

    assert "# module comment" in modified
    assert "import os" in modified
    assert "def untouched" in modified
    assert "return low" in modified


def test_generate_accepts_body_only_response(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "module.py").write_text(SOURCE)
    response = """Explanation:
body only

Bugged Code:
```python
if x < low:
    return low
if x > high:
    return low
return x
```
"""

    lm = LMModify(agent_adapter=FakeAdapter(response))
    candidates = lm.generate(
        plan=_plan(),
        node_code_dir=str(tmp_path),
        repo_spec=RepoSpec(repo_id="repo", repo_path=str(repo)),
        output_dir=str(tmp_path / "out"),
    )

    assert len(candidates) == 1
    assert "return low" in candidates[0].bug_patch
    assert "def untouched" not in candidates[0].bug_patch
