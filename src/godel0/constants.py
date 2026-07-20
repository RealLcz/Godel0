"""Project-wide constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

NODE_REF_PREFIX = "refs/godel0/nodes"

DEFAULT_BATCH_SIZE = 10
DEFAULT_REGRESSION_THRESHOLD = 0.8
DEFAULT_REGRESSION_WEIGHT = 0.5
DEFAULT_PROPOSER_TARGET_ACCURACY = 0.5
DEFAULT_MIN_PARENT_SOLVED_TASKS = 3

ALLOWED_PATCH_PREFIXES = (
    "coding_agent.py",
    "llm_withtools.py",
    "llm.py",
    "tools/",
    "prompts/",
    "utils/",
    "proposer/",
    "swesmith/",
    "tests/",
    "requirements.txt",
)

FORBIDDEN_PATCH_PATTERNS = (
    "../",
    "/.git",
    "symlink",
)

MAX_PATCH_LINES = 80
MAX_OUTPUT_TOKENS = 32768
MAX_LLM_CALLS = 100
MAX_TOOL_ERRORS = 5

ROOT_NODE_ID = "root"
