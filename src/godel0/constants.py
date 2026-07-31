"""Project-wide constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

NODE_REF_PREFIX = "refs/godel0/nodes"

DEFAULT_BATCH_SIZE = 10
DEFAULT_REGRESSION_THRESHOLD = 0.8
DEFAULT_REGRESSION_WEIGHT = 0.5
DEFAULT_PROPOSER_TARGET_ACCURACY = 0.5
DEFAULT_MIN_PARENT_SOLVED_TASKS = 3

# Role-specific allowlists for dual-phase PatchGuard.
# Proposer must not touch solver/shared runtime paths during its phase.
PROPOSER_ALLOWED_PATCH_PREFIXES = (
    "proposer/",
    "swesmith/",
)

SOLVER_ALLOWED_PATCH_PREFIXES = (
    "coding_agent.py",
    "llm.py",
    "llm_withtools.py",
    "tools/",
    "prompts/",
    "utils/",
    "requirements.txt",
)

# Union used by the legacy joint path and final cumulative checks.
ALLOWED_PATCH_PREFIXES = (
    *PROPOSER_ALLOWED_PATCH_PREFIXES,
    *SOLVER_ALLOWED_PATCH_PREFIXES,
)

FORBIDDEN_PATCH_PATTERNS = (
    "../",
    "/.git",
    "symlink",
    # Transport wire format is permanently frozen; schemas.py may evolve under
    # a compatibility gate.
    "proposer/request.py",
)

MAX_PATCH_LINES = 80
MAX_OUTPUT_TOKENS = 32768
MAX_LLM_CALLS = 100
MAX_TOOL_ERRORS = 5

ROOT_NODE_ID = "root"
