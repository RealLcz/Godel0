"""HGM-style entry-grounded diagnose → problem_statement for self-edit."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..errors import SchemaValidationError
from ..schemas.diagnosis import CycleDiagnosis
from .agent_code_dump import dump_agent_code, role_code_summary
from .entry_selector import ProposerFailureEntry, SolverFailureEntry


def clip_text(text: str, max_chars: int) -> str:
    """Keep head+tail when over budget (HGM clip style)."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars < 64:
        return text[:max_chars]
    marker = "\n... [log truncated: middle omitted] ...\n"
    keep = max_chars - len(marker)
    head = keep // 2
    tail = keep - head
    return text[:head] + marker + text[-tail:]


DIAGNOSE_SYSTEM_TEMPLATE = """You are diagnosing one observed failure of an
evolvable software agent.

Your goal is to identify exactly ONE general capability defect in the current
agent implementation and propose exactly ONE coherent improvement.

The observed failure is evidence for the diagnosis, not the target of the
modification. Do not propose a task-specific fix, repository-specific rule,
hard-coded identifier, expected patch, or benchmark-specific behavior.

Prefer the smallest coherent mechanism that could plausibly have changed the
observed trajectory. A general improvement may touch multiple production files
when necessary, but breadth is not a virtue and a new subsystem is not required.

Distinguish carefully between:

1. a capability failure in the evolvable agent;
2. an infrastructure or execution failure;
3. a rejection produced by a trusted evaluator.

Do not convert infrastructure failures into agent capability proposals.
Do not weaken, bypass, redefine, or imitate trusted evaluation behavior.

You will be shown a focused subset of the current agent implementation and one
failure entry. Ground the root cause and proposed edit in the supplied evidence.

{role_constraints}

# Relevant Agent Implementation
----- Agent Implementation Start -----
{code}
----- Agent Implementation End -----
"""

PROPOSER_DIAGNOSIS_CONSTRAINTS = """You are diagnosing the Proposer.

The trusted candidate validator defines the task specification. It is not an
obstacle for the Proposer to bypass.

Never propose or implement any of the following:

- marking a candidate as passed after an exception;
- accepting a candidate when causal isolation could not be executed;
- skipping causal isolation or causal ablation;
- lowering trusted acceptance thresholds;
- fabricating validation metadata or causal statistics;
- changing trusted pass/fail semantics;
- hiding answer leakage through superficial lexical rewriting alone;
- editing proposer/request.py;
- editing trusted-controller code;
- copying solver implementation code into the Proposer;
- weakening, bypassing, or imitating trusted validation to inflate acceptance.

A rejected candidate is negative evidence about upstream generation. Improve
target selection, planning, patch construction, mutation execution, issue
generation, or candidate robustness so that future candidates genuinely satisfy
the existing validation contract.

When companion solver-improvement context is provided for this mutation, the
Proposer should plan/generate tasks that exercise the Solver capability being
improved in this mutation, while still fixing exactly one primary Proposer
failure mechanism grounded in the failure entry.

Choose one primary failure mechanism. If the candidate has multiple rejection
reasons, select the earliest causal reason that explains the later failures.
"""

SOLVER_DIAGNOSIS_CONSTRAINTS = """You are diagnosing the Solver.

Improve the Solver's general coding behavior. Relevant edit surfaces may include but not limited to:

- the live workflow in coding_agent.py;
- prompts used by the live forward path;
- existing tool descriptions or implementations;
- context management and tool-loop behavior;
- skills or tools already present in the agent.

Do not:

- encode the observed task, repository, file, symbol, or expected patch;
- copy private-test behavior into the agent;
- force one particular tool on every task unless the evidence demonstrates a
  general workflow failure;
- build a large evaluation framework when a focused prompt, workflow, or
  existing-tool change addresses the observed defect;
- modify benchmark, trusted evaluation, or task-generation code.

Choose exactly one capability defect supported by the trajectory. Prefer a
focused change to the current live workflow over introducing a new subsystem.
"""

DIAGNOSE_USER_TEMPLATE = """{intro}

# Failure Case
----- Failure Case Start -----
{github_issue}
----- Failure Case End -----

# Agent Run Log
----- Agent Run Log Start -----
{md_log}
----- Agent Run Log End -----

# Generated / Predicted Patch
----- Patch Start -----
{predicted_patch}
----- Patch End -----

# Evaluation / Validation Result
----- Evaluation Start -----
{eval_log}
----- Evaluation End -----

Return one JSON object between the required JSON markers.

The JSON must contain:

- "failure_summary":
  Briefly state what happened in this run.

- "primary_root_cause":
  Exactly one agent capability defect supported by the evidence.

- "generalization":
  Explain why the defect is broader than this one instance.

- "single_improvement":
  Exactly one coherent capability improvement.

- "edit_scope":
  A list of the live files or components most likely to require changes.
  Keep this focused.

- "implementation_suggestion":
  A concrete implementation direction. Prefer adapting an existing workflow,
  prompt, or tool over creating a new subsystem.

- "expected_behavior_change":
  State what should be observably different in a future run.

- "problem_description":
  A concise GitHub-issue-style task that another coding agent can implement.

Do not include multiple alternative improvements. Do not include task-specific
identifiers as implementation constants.
"""

PROBLEM_DESCRIPTION_WRAP = """{code_summary}

# To Implement

{implementation_suggestion}

{problem_description}

---

**Constraints:**
- Wire changes into the live runtime path; do not leave unused helpers.
- Keep the improvement general; do not hard-code task-specific identifiers.
- Prefer adapting an existing workflow, prompt, or tool over a new subsystem.
"""

PROPOSER_INTRO = (
    "Here is the log for the Proposer agent trying to generate a valid repository "
    "task but failing validation / acceptance."
)

SOLVER_INTRO = (
    "Here is the log for the coding agent trying to solve a repository issue but failed."
)

DANGEROUS_DIAGNOSIS_PATTERNS = (
    "passed=true",
    "passed = true",
    "accept on exception",
    "accept after exception",
    "skip causal",
    "bypass validation",
    "lower the threshold",
    "reduce the threshold",
    "force accept",
    "fabricate",
    "proposer/request.py",
)

# Earliest causal rejection reasons preferred for proposer diagnosis.
_PRIMARY_REJECTION_PRIORITY = (
    "execution",
    "setup",
    "malformed",
    "non-applicable",
    "apply",
    "syntax",
    "import",
    "no_f2p",
    "fail_to_pass",
    "fail-to-pass",
    "f2p",
    "causal",
    "ablation",
    "statement",
    "leakage",
    "duplicate",
    "calibration",
    "diversity",
)


@dataclass
class HgmDiagnoseClips:
    md_log_clip_chars: int = 60_000
    eval_log_clip_chars: int = 30_000
    predicted_patch_clip_chars: int = 20_000
    code_dump_clip_chars: int = 80_000
    max_code_files: int = 12


def wrap_problem_statement(
    *,
    role: str,
    implementation_suggestion: str,
    problem_description: str,
) -> str:
    return PROBLEM_DESCRIPTION_WRAP.format(
        code_summary=role_code_summary(role).rstrip(),
        implementation_suggestion=implementation_suggestion.strip(),
        problem_description=problem_description.strip(),
    ).strip()


def select_primary_rejection_reason(reasons: Sequence[str]) -> str:
    """Pick the earliest causal rejection reason from a list."""
    cleaned = [str(r).strip() for r in reasons if str(r).strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]

    def priority(reason: str) -> int:
        lower = reason.lower()
        for idx, token in enumerate(_PRIMARY_REJECTION_PRIORITY):
            if token in lower:
                return idx
        return len(_PRIMARY_REJECTION_PRIORITY)

    return sorted(cleaned, key=priority)[0]


def _diagnosis_blob(data: Dict[str, Any]) -> str:
    parts = []
    for key in (
        "failure_summary",
        "primary_root_cause",
        "generalization",
        "single_improvement",
        "implementation_suggestion",
        "expected_behavior_change",
        "problem_description",
        "problem_statement",
        "edit_scope",
    ):
        value = data.get(key)
        if value is None:
            continue
        parts.append(str(value))
    return "\n".join(parts).lower()


def reject_dangerous_diagnosis(data: Dict[str, Any]) -> None:
    """Raise if the diagnosis proposes trusted-validator bypasses."""
    blob = _diagnosis_blob(data)
    for pattern in DANGEROUS_DIAGNOSIS_PATTERNS:
        if pattern in blob:
            raise SchemaValidationError(
                f"diagnosis proposes forbidden trusted-boundary change: {pattern}"
            )
    effect = str(data.get("trusted_boundary_effect") or "preserve").strip().lower()
    if effect and effect != "preserve":
        raise SchemaValidationError(
            "trusted_boundary_effect must be 'preserve'"
        )


def build_proposer_diagnose_messages(
    entry: ProposerFailureEntry,
    code_dump: str,
    clips: HgmDiagnoseClips,
    *,
    companion_solver_improvement: Optional[CycleDiagnosis] = None,
    companion_solver_patch: str = "",
) -> tuple[str, str]:
    reasons = []
    if entry.validation_report:
        raw = entry.validation_report.get("rejection_reasons") or []
        if isinstance(raw, list):
            reasons = [str(r) for r in raw]
    primary_reason = select_primary_rejection_reason(reasons) or entry.reason

    issue_parts = [
        f"candidate_id: {entry.candidate_id}",
        f"primary_rejection_reason: {primary_reason or '(none)'}",
    ]
    if reasons:
        issue_parts.append("all_rejection_reasons:\n- " + "\n- ".join(reasons))
    if entry.problem_statement.strip():
        issue_parts.append("generated_problem_statement:\n" + entry.problem_statement)
    elif entry.validation_report:
        issue_parts.append(
            "validation_report:\n"
            + json.dumps(entry.validation_report, indent=2, default=str)[:12_000]
        )
    github_issue = "\n\n".join(issue_parts)

    log_parts = []
    if entry.stdout_log.strip():
        log_parts.append("## proposer.stdout.log\n" + entry.stdout_log)
    if entry.stderr_log.strip():
        log_parts.append("## proposer.stderr.log\n" + entry.stderr_log)
    if not log_parts and entry.validation_report:
        log_parts.append(
            "## validation_report\n"
            + json.dumps(entry.validation_report, indent=2, default=str)
        )
    md_log = "\n\n".join(log_parts) or "(no generation logs available)"

    predicted = entry.mutation_diff or entry.bug_patch or "(no patch artifact)"
    eval_log = entry.failing_test_output.strip()
    if not eval_log:
        eval_log = (
            json.dumps(reasons, indent=2, default=str)
            if reasons
            else (entry.reason or "(no eval output)")
        )

    system = DIAGNOSE_SYSTEM_TEMPLATE.format(
        role_constraints=PROPOSER_DIAGNOSIS_CONSTRAINTS.strip(),
        code=clip_text(code_dump, clips.code_dump_clip_chars),
    )
    user = DIAGNOSE_USER_TEMPLATE.format(
        intro=PROPOSER_INTRO,
        github_issue=clip_text(github_issue, clips.md_log_clip_chars),
        md_log=clip_text(md_log, clips.md_log_clip_chars),
        predicted_patch=clip_text(predicted, clips.predicted_patch_clip_chars),
        eval_log=clip_text(eval_log, clips.eval_log_clip_chars),
    )

    companion_sections: list[str] = []
    solver_to_implement = ""
    if companion_solver_improvement is not None:
        solver_to_implement = (
            companion_solver_improvement.problem_statement or ""
        ).strip()
    if solver_to_implement:
        companion_sections.append(
            "# Companion Solver Improvement (this mutation)\n"
            "----- Solver To Implement Start -----\n"
            f"{clip_text(solver_to_implement, clips.md_log_clip_chars)}\n"
            "----- Solver To Implement End -----"
        )
    patch_text = (companion_solver_patch or "").strip()
    if patch_text:
        companion_sections.append(
            "# Solver Phase Diff (this mutation)\n"
            "----- Solver Patch Start -----\n"
            f"{clip_text(patch_text, clips.predicted_patch_clip_chars)}\n"
            "----- Solver Patch End -----"
        )
    if companion_sections:
        user = user.rstrip() + "\n\n" + "\n\n".join(companion_sections) + "\n"
    return system, user


def build_solver_diagnose_messages(
    entry: SolverFailureEntry,
    code_dump: str,
    clips: HgmDiagnoseClips,
) -> tuple[str, str]:
    github_issue = entry.problem_statement.strip() or (
        f"Solver failed on task_id={entry.task_id} (level={entry.level}). "
        "No problem_statement.md was found in the task store."
    )
    system = DIAGNOSE_SYSTEM_TEMPLATE.format(
        role_constraints=SOLVER_DIAGNOSIS_CONSTRAINTS.strip(),
        code=clip_text(code_dump, clips.code_dump_clip_chars),
    )
    user = DIAGNOSE_USER_TEMPLATE.format(
        intro=SOLVER_INTRO,
        github_issue=clip_text(github_issue, clips.md_log_clip_chars),
        md_log=clip_text(entry.trajectory_text or "(empty trajectory)", clips.md_log_clip_chars),
        predicted_patch=clip_text(
            entry.predicted_patch or "(empty predicted patch)",
            clips.predicted_patch_clip_chars,
        ),
        eval_log=clip_text(
            entry.eval_log or entry.failing_test_output or "(no eval log)",
            clips.eval_log_clip_chars,
        ),
    )
    return system, user


def parse_diagnose_json(response: str) -> Dict[str, Any]:
    from ..llm_compat import extract_json_between_markers

    data = extract_json_between_markers(response)
    if data is None:
        raise SchemaValidationError("Could not parse diagnose LLM response as JSON")
    if not isinstance(data, dict):
        raise SchemaValidationError("Diagnose LLM JSON must be an object")

    # Accept limited legacy aliases.
    if not str(data.get("primary_root_cause") or "").strip():
        legacy = data.get("improvement_proposal") or data.get("failure_summary")
        if legacy:
            data["primary_root_cause"] = legacy
    if not str(data.get("single_improvement") or "").strip():
        legacy = data.get("improvement_proposal")
        if legacy:
            data["single_improvement"] = legacy
    if not str(data.get("problem_description") or "").strip():
        if str(data.get("problem_statement") or "").strip():
            data["problem_description"] = data["problem_statement"]

    required = {
        "primary_root_cause",
        "single_improvement",
        "implementation_suggestion",
        "expected_behavior_change",
        "problem_description",
    }
    missing = [key for key in required if not str(data.get(key) or "").strip()]
    if missing:
        raise SchemaValidationError(
            "missing required diagnosis fields: " + ", ".join(missing)
        )

    edit_scope = data.get("edit_scope") or []
    if not isinstance(edit_scope, list):
        raise SchemaValidationError("diagnosis edit_scope must be a list")
    if len(edit_scope) > 4:
        raise SchemaValidationError(
            "diagnosis edit_scope must contain at most four focused components"
        )

    reject_dangerous_diagnosis(data)
    return data


class HgmEntryDiagnoser:
    """Diagnose one failure entry via diagnose_model (HGM Strategy A style)."""

    def __init__(
        self,
        chat_adapter=None,
        model: str = "",
        max_retries: int = 2,
        clips: Optional[HgmDiagnoseClips] = None,
    ):
        self.chat_adapter = chat_adapter
        self.model = str(model or "").strip()
        self.max_retries = max(0, int(max_retries))
        self.clips = clips or HgmDiagnoseClips()

    def diagnose_proposer(
        self,
        *,
        node_id: str,
        entry: ProposerFailureEntry,
        agent_repo: Path,
        companion_solver_improvement: Optional[CycleDiagnosis] = None,
        companion_solver_patch: str = "",
    ) -> Optional[CycleDiagnosis]:
        evidence_hints = self._proposer_evidence_hints(entry)
        code = dump_agent_code(
            agent_repo,
            "proposer",
            max_chars=self.clips.code_dump_clip_chars,
            max_files=self.clips.max_code_files,
            evidence_hints=evidence_hints,
        )
        system, user = build_proposer_diagnose_messages(
            entry,
            code,
            self.clips,
            companion_solver_improvement=companion_solver_improvement,
            companion_solver_patch=companion_solver_patch,
        )
        return self._run(
            node_id=node_id,
            role="proposer",
            system=system,
            user=user,
            evidence_id=entry.candidate_id,
        )

    def diagnose_solver(
        self,
        *,
        node_id: str,
        entry: SolverFailureEntry,
        agent_repo: Path,
    ) -> Optional[CycleDiagnosis]:
        evidence_hints = self._solver_evidence_hints(entry)
        code = dump_agent_code(
            agent_repo,
            "solver",
            max_chars=self.clips.code_dump_clip_chars,
            max_files=self.clips.max_code_files,
            evidence_hints=evidence_hints,
        )
        system, user = build_solver_diagnose_messages(entry, code, self.clips)
        return self._run(
            node_id=node_id,
            role="solver",
            system=system,
            user=user,
            evidence_id=entry.task_id,
        )

    def _proposer_evidence_hints(self, entry: ProposerFailureEntry) -> List[str]:
        blob = " ".join(
            [
                entry.reason or "",
                entry.stdout_log or "",
                entry.stderr_log or "",
                json.dumps(entry.validation_report or {}, default=str),
            ]
        )
        hints: List[str] = []
        for token in re.findall(r"[\w./-]+\.py", blob):
            if token.startswith(("proposer/", "swesmith/")):
                hints.append(token)
        lower = blob.lower()
        if "statement" in lower or "leak" in lower:
            hints.append("proposer/")
        if "causal" in lower or "ablation" in lower:
            hints.append("swesmith/")
        if "mutation" in lower or "patch" in lower:
            hints.append("swesmith/")
        return hints

    def _solver_evidence_hints(self, entry: SolverFailureEntry) -> List[str]:
        blob = " ".join(
            [
                entry.trajectory_text or "",
                entry.eval_log or "",
                entry.failing_test_output or "",
            ]
        )
        hints: List[str] = []
        for token in re.findall(r"(?:tools|prompts|utils)/[\w./-]+\.py", blob):
            hints.append(token)
        lower = blob.lower()
        if "llm.py" in lower or "model routing" in lower or "context" in lower:
            hints.append("llm.py")
        if "prompt" in lower:
            hints.append("prompts/")
        if "tool" in lower:
            hints.append("tools/")
        return hints

    def _run(
        self,
        *,
        node_id: str,
        role: str,
        system: str,
        user: str,
        evidence_id: str,
    ) -> Optional[CycleDiagnosis]:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._call_chat(system, user)
                data = parse_diagnose_json(response)
                problem_statement = wrap_problem_statement(
                    role=role,
                    implementation_suggestion=str(data["implementation_suggestion"]),
                    problem_description=str(data["problem_description"]),
                )
                edit_scope = data.get("edit_scope") or []
                if role == "proposer":
                    default_scopes = ["proposer_logic", "proposer_prompt"]
                else:
                    default_scopes = ["coding_agent", "solver_prompt", "tools"]
                allowed = {
                    "coding_agent",
                    "solver_prompt",
                    "proposer_prompt",
                    "proposer_logic",
                    "tools",
                    "llm_withtools",
                    "utils",
                    "requirements",
                }
                scopes = []
                for item in edit_scope:
                    token = str(item).strip()
                    if token in allowed:
                        scopes.append(token)
                    elif token.startswith("proposer/") or "proposer" in token:
                        scopes.append("proposer_logic")
                    elif token.startswith("tools/") or token == "tools":
                        scopes.append("tools")
                    elif "prompt" in token:
                        scopes.append(
                            "proposer_prompt" if role == "proposer" else "solver_prompt"
                        )
                    elif token in {"coding_agent.py", "llm_withtools.py", "llm.py"}:
                        scopes.append(
                            "llm_withtools" if "llm" in token else "coding_agent"
                        )
                    elif token.startswith("utils/"):
                        scopes.append("utils")
                if not scopes:
                    scopes = default_scopes
                # Deduplicate while preserving order.
                scopes = list(dict.fromkeys(scopes))[:4]
                return CycleDiagnosis(
                    node_id=node_id,
                    primary_root_cause=str(data["primary_root_cause"])[:2000],
                    selected_alert_id=None,
                    source_stages=[role],
                    recommended_edit_scopes=scopes[:4],
                    evidence_ids=[evidence_id],
                    expected_effects={
                        "expected_behavior_change": str(
                            data.get("expected_behavior_change") or ""
                        )[:2000]
                    },
                    non_goals=[
                        "Do not hardcode task-specific solutions",
                        "Do not invent unrelated refactors",
                        "Do not bypass trusted validation",
                    ],
                    validation_plan=[
                        "Re-run the failure class that triggered this diagnosis",
                    ],
                    problem_statement=problem_statement,
                    override_reason=None,
                )
            except (SchemaValidationError, TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
        # Diagnosis failure aborts the mutation; no generic fallback issue.
        return None

    def _call_chat(self, system: str, user: str) -> str:
        if self.chat_adapter is None or not callable(
            getattr(self.chat_adapter, "chat", None)
        ):
            raise SchemaValidationError("No chat adapter configured for HGM diagnose")
        kwargs: Dict[str, Any] = {"temperature": 0, "max_tokens": 4096}
        if self.model:
            kwargs["model"] = self.model
        chat = self.chat_adapter.chat
        try:
            return chat(system, user, **kwargs)
        except TypeError:
            kwargs.pop("model", None)
            try:
                return chat(system, user, **kwargs)
            except TypeError:
                return chat(system, user)
