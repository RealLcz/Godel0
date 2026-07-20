from __future__ import annotations

import ast
import hashlib
import os
import random
import re
import textwrap
from typing import Any, List, Optional

from .candidate import CandidateArtifact
from .engine import BugGenerationPlan, RepoSpec
from .patch_utils import make_git_diff, count_modified_lines, extract_changed_files
from .workspace import WorkspaceManager, WorkspaceSpec


BUG_EXAMPLES = [
    "Alter calculation order for incorrect results: Rearrange the sequence of operations in a calculation to subtly change the output (e.g., change (a + b) * c to a + (b * c)).",
    "Introduce subtle data transformation errors: Modify data processing logic, such as flipping a sign, truncating a value, or applying the wrong transformation function.",
    "Change variable assignments to alter computation state: Assign a wrong or outdated value to a variable that affects subsequent logic.",
    "Mishandle edge cases for specific inputs: Change handling logic to ignore or improperly handle boundary cases, like an empty array or a null input.",
    "Modify logic in conditionals or loops: Adjust conditions or loop boundaries (e.g., replace <= with <) to change the control flow.",
    "Introduce off-by-one errors in indices or loop boundaries: Shift an index or iteration boundary by one, such as starting a loop at 1 instead of 0.",
    "Adjust default values or constants to affect behavior: Change a hardcoded value or default parameter that alters how the function behaves under normal use.",
    "Reorder operations while maintaining syntax: Rearrange steps in a process so the function produces incorrect intermediate results without breaking the code.",
    "Swallow exceptions or return defaults silently: Introduce logic that catches an error but doesn't log or handle it properly, leading to silent failures.",
]

TIPS = [
    "It should not cause compilation errors.",
    "It should not be a syntax error.",
    "It should be subtle and challenging to detect.",
    "It should not modify the function signature.",
    "It should not modify the documentation significantly.",
    "For longer functions, if there is an opportunity to introduce multiple bugs, please do!",
    "Please DO NOT INCLUDE COMMENTS IN THE CODE indicating the bug location or the bug itself.",
]


def build_lm_modify_prompt(
    source: str,
    target_file: str,
    target_symbol: str,
    desired_behavior: str,
    max_lines: int,
    seed: int = 42,
) -> str:
    """Build the LM Modify prompt following SWE-smith's chaos monkey design."""
    rng = random.Random(seed)
    selected_examples = rng.sample(BUG_EXAMPLES, min(3, len(BUG_EXAMPLES)))
    shuffled_tips = TIPS[:]
    rng.shuffle(shuffled_tips)

    bug_list = "\n".join(f"  - {ex}" for ex in selected_examples)
    tips_list = "\n".join(f"  - {tip}" for tip in shuffled_tips)

    system = f"""You are a software developer doing chaos monkey testing.
Your job is to rewrite a function such that it introduces a logical bug that will break existing unit test(s) in a codebase.

To this end, some kinds of bugs you might introduce include:
{bug_list}

Tips about the bug-introducing task:
{tips_list}

Your answer should be formatted as follows:

Explanation:
<explanation>

Bugged Code:
```
<bugged_code>
```"""

    user = f"""<INPUT>
{source}
</INPUT>

<IMPORTANT>As a reminder, Please DO NOT INCLUDE ANY COMMENTS IN THE CODE OR POINT OUT THE BUG IN ANY WAY.</IMPORTANT>
<IMPORTANT>Return exactly one Python code block under "Bugged Code". If a target symbol is provided, the code block must contain the complete modified definition for that symbol, including decorators, signature, and body. Do not return unrelated functions or partial snippets.</IMPORTANT>
<IMPORTANT>The returned code must make at least one behavioral change compared with the input while preserving the function/class signature.</IMPORTANT>

Target file: {target_file}
Target symbol: {target_symbol or "(entire file)"}
Max modified lines: {max_lines}
Desired behavior: {desired_behavior or "any realistic bug"}

OUTPUT:"""

    return system, user


def extract_code_block(text: str) -> str:
    """Extract code from the most likely answer code block.

    Qwen3.6 outputs thinking process first, which may contain code blocks.
    Prefer a fenced block after the "Bugged Code" marker, then fall back to
    the last fenced block.
    """
    marker_match = re.search(r"Bugged\s+Code\s*:?", text, flags=re.IGNORECASE)
    search_text = text[marker_match.end():] if marker_match else text
    block = _last_fenced_code_block(search_text)
    if block:
        return _clean_extracted_code(block)
    block = _last_fenced_code_block(text)
    if block:
        return _clean_extracted_code(block)
    return _clean_extracted_code(text)


def _last_fenced_code_block(text: str) -> str:
    lines = text.split("\n")
    in_block = False
    code_lines = []
    last_code_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```") and not in_block:
            in_block = True
            code_lines = []
            continue
        elif stripped.startswith("```") and in_block:
            in_block = False
            # Save this block as the potential answer
            last_code_lines = code_lines[:]
            continue
        elif in_block:
            code_lines.append(line)
    # Return the last code block found
    return "\n".join(last_code_lines) if last_code_lines else "\n".join(code_lines)


def _clean_extracted_code(code: str) -> str:
    code = code.strip()
    if code.startswith("<bugged_code>"):
        code = code[len("<bugged_code>"):]
    if code.endswith("</bugged_code>"):
        code = code[: -len("</bugged_code>")]
    return code.strip()


def extract_explanation(text: str) -> str:
    """Extract the explanation from LLM output."""
    if "Explanation:" in text:
        parts = text.split("Explanation:", 1)
        if len(parts) > 1:
            rest = parts[1]
            if "```" in rest:
                return rest.split("```")[0].strip()
            return rest.strip()
    return text.split("```")[0].strip()


class LMModify:
    """LM Modify strategy: ask the LLM to introduce a subtle bug.

    Follows SWE-smith's chaos monkey testing design:
    - System prompt with bug examples (randomly sampled) and tips
    - User prompt with source code in <INPUT> tags
    - Parses Explanation + Bugged Code from response
    """

    def __init__(self, agent_adapter: Any = None) -> None:
        self.agent_adapter = agent_adapter
        self.workspace_manager = WorkspaceManager()

    def generate(
        self,
        plan: BugGenerationPlan,
        node_code_dir: str,
        repo_spec: RepoSpec,
        output_dir: str,
    ) -> List[CandidateArtifact]:
        target_file = plan.target_file
        if not target_file:
            return []

        file_path = target_file
        if not os.path.isabs(file_path):
            file_path = os.path.join(repo_spec.repo_path or node_code_dir, target_file)

        if not os.path.exists(file_path):
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            original_source = f.read()

        prompt_source = self._extract_target_source(original_source, plan.target_symbol)
        if not prompt_source:
            prompt_source = original_source

        system_prompt, user_prompt = build_lm_modify_prompt(
            source=prompt_source,
            target_file=target_file,
            target_symbol=plan.target_symbol,
            desired_behavior=plan.constraints.desired_behavior,
            max_lines=plan.constraints.max_modified_lines,
            seed=plan.seed,
        )

        if self.agent_adapter is None:
            return []

        try:
            response_text = self._call_agent(system_prompt, user_prompt)
        except Exception as e:
            return []

        if not response_text:
            return []

        modified_fragment = extract_code_block(response_text)
        if not modified_fragment:
            # Fallback: try to find any Python code in the response
            lines = response_text.split("\n")
            code_lines = []
            in_code = False
            for line in lines:
                if line.strip().startswith("def ") or line.strip().startswith("class "):
                    in_code = True
                if in_code:
                    code_lines.append(line)
                    if line.strip() == "" and code_lines:
                        # Check if next non-empty line is still indented
                        pass
            if code_lines:
                modified_fragment = "\n".join(code_lines)

        if not modified_fragment:
            return []

        modified_source = self._materialize_modified_source(
            original_source,
            plan.target_symbol,
            modified_fragment,
        )
        if not modified_source or modified_source == original_source:
            return []

        try:
            import ast as _ast
            _ast.parse(modified_source)
        except SyntaxError:
            return []

        patch = make_git_diff(original_source, modified_source, filename=target_file)
        if not patch:
            return []

        changed_files = extract_changed_files(patch)
        if len(changed_files) > plan.constraints.max_modified_files:
            return []

        if count_modified_lines(patch) > plan.constraints.max_modified_lines:
            return []

        if not plan.constraints.allow_test_edits:
            for cf in changed_files:
                if "test" in cf.lower():
                    return []

        explanation = extract_explanation(response_text)
        candidate_id = self._make_candidate_id(plan)
        artifact = CandidateArtifact(
            candidate_id=candidate_id,
            plan_id=plan.plan_id,
            strategy="lm_modify",
            operator="lm_modify",
            target_file=target_file,
            target_symbol=plan.target_symbol,
            bug_patch=patch,
            mutation_site={"explanation": explanation},
            seed=plan.seed,
            before_snippet=original_source[:500],
            after_snippet=modified_source[:500],
            generation_metadata={
                "agent": str(type(self.agent_adapter).__name__),
                "explanation": explanation,
                "response": response_text[:2000],
            },
        )

        if output_dir:
            candidate_dir = os.path.join(output_dir, candidate_id)
            artifact.save(candidate_dir)

        return [artifact]

    def _call_agent(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM via the agent adapter or direct API call."""
        if hasattr(self.agent_adapter, "chat"):
            return self.agent_adapter.chat(system_prompt, user_prompt)

        if hasattr(self.agent_adapter, "complete"):
            return self.agent_adapter.complete(system_prompt + "\n\n" + user_prompt)

        if hasattr(self.agent_adapter, "run_task"):
            return self.agent_adapter.run_task(user_prompt, system_prompt, "", "")

        # Fallback: try direct OpenAI client
        if hasattr(self.agent_adapter, "chat") and hasattr(self.agent_adapter, "completions"):
            response = self.agent_adapter.chat.completions.create(
                model=getattr(self.agent_adapter, "model", "default"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=1,
                max_tokens=16384,
            )
            return response.choices[0].message.content

        return ""

    def _make_candidate_id(self, plan: BugGenerationPlan) -> str:
        raw = f"{plan.plan_id}:lm_modify:{plan.target_file}:{plan.target_symbol}"
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
        return f"cand_{digest}"

    def _replace_function_in_file(self, file_source: str, symbol_name: str, new_func_source: str) -> str:
        """Replace a function/class definition by line range, preserving file text."""
        return self._splice_symbol_definition(file_source, symbol_name, new_func_source)

    def _materialize_modified_source(
        self,
        original_source: str,
        target_symbol: str,
        modified_fragment: str,
    ) -> str:
        """Turn an LM code fragment into a full modified file."""
        if not target_symbol:
            return modified_fragment
        return self._splice_symbol_definition(original_source, target_symbol, modified_fragment)

    def _extract_target_source(self, file_source: str, symbol_name: str) -> str:
        if not symbol_name:
            return file_source
        node = self._find_symbol_node(file_source, symbol_name)
        if node is None:
            return ""
        lines = file_source.splitlines()
        start, end = _node_line_range(node)
        return textwrap.dedent("\n".join(lines[start - 1:end]))

    def _splice_symbol_definition(self, file_source: str, symbol_name: str, new_source: str) -> str:
        original_node = self._find_symbol_node(file_source, symbol_name)
        if original_node is None:
            return ""

        replacement = self._extract_replacement_definition(
            file_source,
            original_node,
            symbol_name,
            new_source,
        )
        if not replacement:
            return ""

        start, end = _node_line_range(original_node)
        original_lines = file_source.splitlines()
        indent = _line_indent(original_lines[start - 1])
        replacement = _indent_block(textwrap.dedent(replacement).strip("\n"), indent)
        new_lines = original_lines[: start - 1] + replacement.splitlines() + original_lines[end:]
        trailing_newline = "\n" if file_source.endswith("\n") else ""
        return "\n".join(new_lines) + trailing_newline

    def _extract_replacement_definition(
        self,
        file_source: str,
        original_node: ast.AST,
        symbol_name: str,
        new_source: str,
    ) -> str:
        dedented = textwrap.dedent(new_source).strip()
        new_node = _find_symbol_node_in_source(dedented, symbol_name)
        if new_node is not None:
            if not _signature_matches(original_node, new_node):
                return ""
            start, end = _node_line_range(new_node)
            return "\n".join(dedented.splitlines()[start - 1:end])

        body_replacement = self._body_only_replacement(file_source, original_node, dedented)
        return body_replacement

    def _body_only_replacement(
        self,
        file_source: str,
        original_node: ast.AST,
        body_source: str,
    ) -> str:
        """Build a complete definition when the LM returned only the body."""
        if not isinstance(original_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return ""
        try:
            parsed_body = ast.parse(body_source)
        except SyntaxError:
            return ""
        if not parsed_body.body:
            return ""

        original_lines = file_source.splitlines()
        start, end = _node_line_range(original_node)
        body_start = getattr(original_node.body[0], "lineno", start + 1)
        header_lines = original_lines[start - 1: body_start - 1]
        body_indent = _line_indent(original_lines[body_start - 1])
        body = _indent_block(textwrap.dedent(body_source).strip("\n"), body_indent)
        return "\n".join(header_lines + body.splitlines())

    def _find_symbol_node(self, source: str, symbol_name: str) -> Optional[ast.AST]:
        return _find_symbol_node_in_source(source, symbol_name)


def _find_symbol_node_in_source(source: str, symbol_name: str) -> Optional[ast.AST]:
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return None
    requested = (symbol_name or "").split(".")[-1]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == requested:
            return node
    return None


def _node_line_range(node: ast.AST) -> tuple[int, int]:
    start = getattr(node, "lineno", 0)
    decorators = getattr(node, "decorator_list", [])
    if decorators:
        start = min(getattr(dec, "lineno", start) for dec in decorators)
    end = getattr(node, "end_lineno", start)
    return start, end


def _line_indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _indent_block(block: str, indent: str) -> str:
    lines = block.splitlines()
    return "\n".join(indent + line if line.strip() else line for line in lines)


def _signature_matches(original_node: ast.AST, new_node: ast.AST) -> bool:
    if type(original_node) is not type(new_node):
        return False
    if getattr(original_node, "name", None) != getattr(new_node, "name", None):
        return False
    if isinstance(original_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ast.dump(original_node.args, include_attributes=False) == ast.dump(
            new_node.args,
            include_attributes=False,
        )
    return True
