"""Custom exception hierarchy for Godel0."""


class Godel0Error(Exception):
    """Base error for all Godel0 failures."""


class ConfigError(Godel0Error):
    """Configuration loading or validation failed."""


class ToolRegistrationError(Godel0Error):
    """Tool discovery, schema, or registration failed."""


class PatchGuardError(Godel0Error):
    """A self-evolution patch violated the allowlist or safety rules."""


class AgentExecutionError(Godel0Error):
    """The coding agent failed to execute or produced invalid output."""


class CandidateValidationError(Godel0Error):
    """Trusted candidate validation failed irrecoverably."""


class EvaluationError(Godel0Error):
    """Solver evaluation or test harness error."""


class WorkspaceError(Godel0Error):
    """Workspace setup, cleanup, or isolation failure."""


class ResumeError(Godel0Error):
    """Resume state machine or checkpoint error."""


class GitRefError(Godel0Error):
    """Git node ref creation or lookup failed."""


class SchemaValidationError(Godel0Error):
    """A data structure failed schema validation."""


class BudgetExhaustedError(Godel0Error):
    """The evolution budget has been exhausted."""


class RetryableError(Godel0Error):
    """An error that can be retried (rate limit, transient failure)."""
