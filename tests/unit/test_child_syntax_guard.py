"""Tests for the trusted child Python syntax gate."""

from godel0.evolution.child_builder import validate_changed_python_syntax


def _new_file_patch(path: str) -> str:
    return f"""diff --git a/{path} b/{path}
new file mode 100644
--- /dev/null
+++ b/{path}
@@ -0,0 +1 @@
+content
"""


def test_syntax_guard_accepts_valid_changed_python(tmp_path):
    (tmp_path / "coding_agent.py").write_text("VALUE = 1\n")
    assert validate_changed_python_syntax(
        tmp_path, _new_file_patch("coding_agent.py")
    ) == []


def test_syntax_guard_rejects_invalid_changed_python(tmp_path):
    (tmp_path / "coding_agent.py").write_text("    def broken(:\n")
    errors = validate_changed_python_syntax(
        tmp_path, _new_file_patch("coding_agent.py")
    )
    assert len(errors) == 1
    assert "coding_agent.py:1" in errors[0]
