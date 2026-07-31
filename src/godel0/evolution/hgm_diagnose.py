"""HGM-style entry-grounded diagnose → problem_statement for self-edit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

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


DIAGNOSE_SYSTEM_TEMPLATE = """Here is the implementation of the agent.

{code_summary}

# Agent Implementation
----- Agent Implementation Start -----
{code}
----- Agent Implementation End -----

Your task is to identify ONE detailed plan that would help the agent actually
handle failures like the ones shown in the logs. The improvement must be
GENERAL (no task-specific hardcoding).
"""

DIAGNOSE_USER_TEMPLATE = """{intro}

# Failure Case / Issue
----- Issue Start -----
{github_issue}
----- Issue End -----

# Agent Run Log
----- Agent Run Log Start -----
{md_log}
----- Agent Run Log End -----

# Predicted / Generated Patch
----- Patch Start -----
{predicted_patch}
----- Patch End -----

# Evaluation / Validation Results
----- Eval Start -----
{eval_log}
----- Eval End -----

Respond precisely in the following format including the JSON start and end markers:

```json
<JSON>
```

In <JSON>, provide a JSON response with the following fields:
- "log_summarization": Analyze the run log, patch, and eval results. What went wrong?
- "potential_improvements": Concrete improvements grounded in the evidence.
- "improvement_proposal": ONE high-impact improvement in detail.
- "implementation_suggestion": What to change in the agent code (wire into live paths; no hard-coded task constants).
- "problem_description": Phrase the proposal as a GitHub-issue-style description an engineer can implement.

Your response will be automatically parsed. Do NOT include the `<JSON>` tag in your output.
"""

PROBLEM_DESCRIPTION_WRAP = """{code_summary}

# To Implement

{implementation_suggestion}

{problem_description}

---

**REMINDER — implement the full diagnosis:**
- Do not leave helper functions or modules dead/unused. Wire changes into the live path.
- Focus on making the agent handle **this class of failures** correctly.
- FORBIDDEN: hard-coding task-specific identifiers (repo/file/module/instance names) as constants.
- Prefer data-driven mechanisms computed from the problem statement and repo state.
"""

PROPOSER_INTRO = (
    "Here is the log for the Proposer agent trying to generate a valid repository "
    "task but failing validation / acceptance."
)

SOLVER_INTRO = (
    "Here is the log for the coding agent trying to solve a repository issue but failed."
)


@dataclass
class HgmDiagnoseClips:
    md_log_clip_chars: int = 60_000
    eval_log_clip_chars: int = 30_000
    predicted_patch_clip_chars: int = 20_000
    code_dump_clip_chars: int = 200_000


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


def build_proposer_diagnose_messages(
    entry: ProposerFailureEntry,
    code_dump: str,
    clips: HgmDiagnoseClips,
) -> tuple[str, str]:
    issue_parts = [
        f"candidate_id: {entry.candidate_id}",
        f"rejection_reason: {entry.reason or '(none)'}",
    ]
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
        reasons = entry.validation_report.get("rejection_reasons") if entry.validation_report else None
        eval_log = (
            json.dumps(reasons, indent=2, default=str)
            if reasons
            else (entry.reason or "(no eval output)")
        )

    system = DIAGNOSE_SYSTEM_TEMPLATE.format(
        code_summary=role_code_summary("proposer"),
        code=clip_text(code_dump, clips.code_dump_clip_chars),
    )
    user = DIAGNOSE_USER_TEMPLATE.format(
        intro=PROPOSER_INTRO,
        github_issue=clip_text(github_issue, clips.md_log_clip_chars),
        md_log=clip_text(md_log, clips.md_log_clip_chars),
        predicted_patch=clip_text(predicted, clips.predicted_patch_clip_chars),
        eval_log=clip_text(eval_log, clips.eval_log_clip_chars),
    )
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
        code_summary=role_code_summary("solver"),
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
    if not str(data.get("implementation_suggestion") or "").strip():
        raise SchemaValidationError("missing implementation_suggestion")
    if not str(data.get("problem_description") or "").strip():
        # Accept problem_statement alias.
        if not str(data.get("problem_statement") or "").strip():
            raise SchemaValidationError("missing problem_description")
        data["problem_description"] = data["problem_statement"]
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
    ) -> CycleDiagnosis:
        code = dump_agent_code(
            agent_repo, "proposer", max_chars=self.clips.code_dump_clip_chars
        )
        system, user = build_proposer_diagnose_messages(entry, code, self.clips)
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
    ) -> CycleDiagnosis:
        code = dump_agent_code(
            agent_repo, "solver", max_chars=self.clips.code_dump_clip_chars
        )
        system, user = build_solver_diagnose_messages(entry, code, self.clips)
        return self._run(
            node_id=node_id,
            role="solver",
            system=system,
            user=user,
            evidence_id=entry.task_id,
        )

    def _run(
        self,
        *,
        node_id: str,
        role: str,
        system: str,
        user: str,
        evidence_id: str,
    ) -> CycleDiagnosis:
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
                scopes = (
                    ["proposer_logic", "proposer_prompt"]
                    if role == "proposer"
                    else ["coding_agent", "solver_prompt", "tools"]
                )
                return CycleDiagnosis(
                    node_id=node_id,
                    primary_root_cause=str(
                        data.get("improvement_proposal")
                        or data.get("problem_description")
                        or ""
                    )[:2000],
                    selected_alert_id=None,
                    source_stages=[role],
                    recommended_edit_scopes=scopes,
                    evidence_ids=[evidence_id],
                    expected_effects={},
                    non_goals=[
                        "Do not hardcode task-specific solutions",
                        "Do not invent unrelated refactors",
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
        # Deterministic fallback so expansion can still attempt a self-edit.
        return self._fallback(node_id=node_id, role=role, evidence_id=evidence_id, error=last_error)

    def _fallback(
        self,
        *,
        node_id: str,
        role: str,
        evidence_id: str,
        error: Optional[Exception],
    ) -> CycleDiagnosis:
        if role == "proposer":
            description = (
                f"Improve the Proposer so candidates like {evidence_id} pass validation. "
                "Focus on mutation planning / patch application / contract grounding. "
                f"Diagnose LLM fallback reason: {error}"
            )
            suggestion = (
                "Inspect proposer/ and swesmith/ paths that produce invalid chain plans "
                "or fragile patch application during ablation, and make the mechanism robust."
            )
            scopes = ["proposer_logic", "proposer_prompt"]
        else:
            description = (
                f"Improve the Solver so it can resolve failures like task {evidence_id}. "
                "Focus on localization, editing, and using failing-test evidence. "
                f"Diagnose LLM fallback reason: {error}"
            )
            suggestion = (
                "Inspect coding_agent.py forward() and tools/ to address the failure mode "
                "shown in the trajectory without hard-coding this task."
            )
            scopes = ["coding_agent", "solver_prompt", "tools"]
        return CycleDiagnosis(
            node_id=node_id,
            primary_root_cause=description[:500],
            source_stages=[role],
            recommended_edit_scopes=scopes,
            evidence_ids=[evidence_id],
            non_goals=["Do not hardcode task-specific solutions"],
            validation_plan=["Re-run the originating failure class"],
            problem_statement=wrap_problem_statement(
                role=role,
                implementation_suggestion=suggestion,
                problem_description=description,
            ),
        )

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
