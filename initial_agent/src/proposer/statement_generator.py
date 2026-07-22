from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

from .request import CandidateArtifact
from .schemas import BugGenerationPlan


ISSUE_SYSTEM = """You are a software engineer helping to create a realistic dataset of synthetic GitHub issues.

You will be given the following input:

1. Patch: A git diff output/pull request changes that introduces a bug (included in the <patch> tag).
2. Test output: The output of running the tests after the patch is applied (included in the <test_output> tag).
3. Test source code: Source code for one or more tests that failed (included in the <test_source_code> tag).

Output: A realistic GitHub issue for the patch.

Guidelines:

- DO NOT explain the fix/what caused the bug itself, focus on how to reproduce the issue it introduces.
- Do not mention pytest or what exact test failed. Instead, generate a realistic issue.
- If possible, include information about how to reproduce the issue. An ideal reproduction script should raise an error
  or print an unexpected output together with the expected output.
- DO NOT GIVE AWAY THE FIX! THE SOLUTION CODE SHOULD NEVER APPEAR IN YOUR RESPONSE.
- DO NOT SAY THAT EXISTING TEST(s) FAILED.
- DO NOT SUGGEST RUNNING ANY TESTING COMMANDS (e.g., pytest)."""


def build_issue_prompt(
    patch: str,
    test_output: str = "",
    test_source_code: str = "",
) -> str:
    """Build the issue generation prompt with test context."""
    user = f"""<patch>
{patch}
</patch>

<test_output>
{test_output or "(not available)"}
</test_output>

<test_source_code>
{test_source_code or "(not available)"}
</test_source_code>

**Issue Text**

<START WRITING>"""
    return ISSUE_SYSTEM, user


@dataclass
class IssueDraft:
    """A generated GitHub-style issue draft for a candidate."""
    candidate_id: str
    title: str
    body: str

    def to_markdown(self) -> str:
        return f"{self.title}\n\n{self.body}\n"


class StatementGenerator:
    """Generates GitHub-style issue drafts for accepted candidates.

    If an agent_adapter is available, uses the SWE-smith-style prompt
    with test context (patch + test output + test source code).
    Otherwise, falls back to a template-based issue.
    """

    def __init__(self, agent_adapter: Any = None):
        self.agent_adapter = agent_adapter

    def generate(
        self,
        candidate: CandidateArtifact,
        plan: Optional[BugGenerationPlan] = None,
        test_output: str = "",
        test_source_code: str = "",
    ) -> IssueDraft:
        # Try LLM-based generation if adapter is available
        if self.agent_adapter is not None and candidate.patch:
            try:
                system_prompt, user_prompt = build_issue_prompt(
                    patch=candidate.patch,
                    test_output=test_output,
                    test_source_code=test_source_code,
                )
                llm_response = self._call_agent(system_prompt, user_prompt)
                if llm_response and len(llm_response) > 50:
                    return IssueDraft(
                        candidate_id=candidate.candidate_id,
                        title=llm_response.split("\n")[0].strip(),
                        body=llm_response,
                    )
            except Exception:
                pass

        # Fallback: template-based
        return self._generate_template(candidate, plan)

    def write_issue_draft(
        self,
        candidate: CandidateArtifact,
        candidate_dir: str,
        plan: Optional[BugGenerationPlan] = None,
        test_output: str = "",
        test_source_code: str = "",
    ) -> str:
        draft = self.generate(candidate, plan, test_output, test_source_code)
        os.makedirs(candidate_dir, exist_ok=True)
        path = os.path.join(candidate_dir, "issue_draft.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(draft.to_markdown())
        candidate.issue_draft = draft.to_markdown()
        return path

    def _generate_template(
        self,
        candidate: CandidateArtifact,
        plan: Optional[BugGenerationPlan],
    ) -> IssueDraft:
        symbol = candidate.symbol_name or (plan.target_symbol if plan else "module")
        file_path = candidate.file_path or (plan.target_file if plan else "file")
        title = f"# Bug: incorrect behavior in `{symbol}`"

        body_parts = [
            f"A regression has been introduced in `{symbol}` that causes incorrect behavior.",
            f"\n**Affected file:** `{file_path}`",
        ]
        if plan and plan.constraints and plan.constraints.desired_behavior:
            body_parts.append(f"\n**Expected behavior:** {plan.constraints.desired_behavior}")
        body_parts.append("\n**How to reproduce:** Apply the provided patch and run the test suite.")

        return IssueDraft(
            candidate_id=candidate.candidate_id,
            title=title,
            body="\n".join(body_parts),
        )

    def _call_agent(self, system_prompt: str, user_prompt: str) -> str:
        if hasattr(self.agent_adapter, "chat"):
            model = str(getattr(self.agent_adapter, "default_model", "") or "")
            try:
                if model:
                    return self.agent_adapter.chat(
                        system_prompt, user_prompt, model=model
                    )
                return self.agent_adapter.chat(system_prompt, user_prompt)
            except TypeError:
                return self.agent_adapter.chat(system_prompt, user_prompt)
        if hasattr(self.agent_adapter, "run_task"):
            return self.agent_adapter.run_task(user_prompt, system_prompt, "", "")
        if hasattr(self.agent_adapter, "chat") and hasattr(self.agent_adapter, "completions"):
            response = self.agent_adapter.chat.completions.create(
                model=getattr(self.agent_adapter, "model", "default"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=2048,
            )
            return response.choices[0].message.content
        return ""


__all__ = ["StatementGenerator", "IssueDraft", "build_issue_prompt"]
