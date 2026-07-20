from __future__ import annotations

import ast
import hashlib
import os
import subprocess
import textwrap
from typing import Any, List, Optional

from .candidate import CandidateArtifact
from .engine import BugGenerationPlan, RepoSpec
from .entity_index import EntityIndex, Entity
from .patch_utils import make_git_diff, count_modified_lines
from .workspace import WorkspaceManager, WorkspaceSpec
from .lm_modify import (
    extract_code_block,
    _find_symbol_node_in_source,
    _indent_block,
    _line_indent,
    _node_line_range,
    _signature_matches,
)


REWRITE_SYSTEM = """You are a software developer and you have been asked to implement a function.

You will be given the contents of an entire file, with one or more functions defined in it.
Please implement the function(s) that are missing.
Do NOT modify the function signature, including the function name, parameters, return types, or docstring if provided.
Do NOT change any other code in the file.
You should not use any external libraries."""


def build_lm_rewrite_prompt(
    func_signature: str,
    file_src_code: str,
    stub: str,
) -> str:
    """Build the LM Rewrite prompt (blank-body reimplementation)."""
    user = f"""Please implement the function `{func_signature}` in the following code:

```
{file_src_code}
```

Remember, you should not modify the function signature, including the function name, parameters, return types, or docstring if provided.
Do NOT change any other code in the file.
Format your output as:

<explanation>

```
{stub}
```"""
    return REWRITE_SYSTEM, user


class LMRewrite:
    """LM Rewrite strategy: blank out function body and ask LLM to reimplement.

    Unlike LM Modify (which asks for a bug), LM Rewrite asks the LLM to
    implement the function correctly. The LLM's natural imperfections
    (omissions, misunderstandings) produce realistic bugs.

    This follows SWE-smith's design:
    - Temperature 0 (deterministic)
    - n=1 (one rewrite per entity)
    - Provides entire file source with blanked function
    - Asks for correct implementation, not a bug
    """

    def __init__(self, agent_adapter: Any = None) -> None:
        self.agent_adapter = agent_adapter
        self.workspace_manager = WorkspaceManager()
        self.entity_index = EntityIndex()

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

        self.entity_index.index_file(file_path, source=original_source)
        entities = self.entity_index.find_in_file(file_path, name=plan.target_symbol or None)
        if not entities:
            return []

        target_entity = entities[0]

        # Create the blanked-out version of the source
        stub = self._make_stub(target_entity)
        blanked_source = self._blank_out_function(original_source, target_entity)
        if not blanked_source or blanked_source == original_source:
            return []

        func_signature = self._make_signature(target_entity)

        system_prompt, user_prompt = build_lm_rewrite_prompt(
            func_signature=func_signature,
            file_src_code=blanked_source,
            stub=stub,
        )

        if self.agent_adapter is None:
            return []

        try:
            response_text = self._call_agent(system_prompt, user_prompt)
        except Exception:
            return []

        if not response_text:
            return []

        new_body = extract_code_block(response_text)
        if not new_body:
            return []

        # Replace the blanked function with the LLM's reimplementation
        modified_source = self._replace_entity_body(original_source, target_entity, new_body)
        if not modified_source or modified_source == original_source:
            return []

        try:
            ast.parse(modified_source)
        except SyntaxError:
            return []

        patch = make_git_diff(original_source, modified_source, filename=target_file)
        if not patch:
            return []

        if count_modified_lines(patch) > plan.constraints.max_modified_lines:
            return []

        candidate_id = self._make_candidate_id(plan)
        artifact = CandidateArtifact(
            candidate_id=candidate_id,
            plan_id=plan.plan_id,
            strategy="lm_rewrite",
            operator="lm_rewrite",
            target_file=target_file,
            target_symbol=plan.target_symbol,
            bug_patch=patch,
            mutation_site={"entity": target_entity.qualname, "line": target_entity.line},
            seed=plan.seed,
            before_snippet=target_entity.source[:500] if target_entity.source else "",
            after_snippet=new_body[:500],
            generation_metadata={
                "agent": str(type(self.agent_adapter).__name__),
                "response": response_text[:2000],
            },
        )

        if output_dir:
            candidate_dir = os.path.join(output_dir, candidate_id)
            artifact.save(candidate_dir)

        return [artifact]

    def _make_stub(self, entity: Entity) -> str:
        """Create a stub version (signature + TODO)."""
        if entity.source:
            node = _find_symbol_node_in_source(entity.source, entity.name)
            if node is not None:
                return self._definition_with_stubbed_body(entity.source, node)
        return f"def {entity.name}(...):\n    ..."

    def _make_signature(self, entity: Entity) -> str:
        if entity.source:
            for line in entity.source.splitlines():
                stripped = line.strip()
                if stripped.startswith(("def ", "async def ", "class ")):
                    return stripped.rstrip(":")
        prefix = "async " if entity.kind == "async_function" else ""
        keyword = "class" if entity.kind == "class" else "def"
        args_str = ", ".join(entity.args)
        return f"{prefix}{keyword} {entity.name}({args_str})"

    def _blank_out_function(self, source: str, entity: Entity) -> str:
        """Replace the target body with a stub while preserving file text."""
        node = _find_entity_node(source, entity)
        if node is None:
            return source

        start, end = _node_line_range(node)
        original_lines = source.splitlines()
        replacement = self._definition_with_stubbed_body(
            "\n".join(original_lines[start - 1:end]),
            _find_symbol_node_in_source("\n".join(original_lines[start - 1:end]), entity.name),
        )
        if not replacement:
            return source

        indent = _line_indent(original_lines[start - 1])
        replacement = _indent_block(textwrap.dedent(replacement).strip("\n"), indent)
        trailing_newline = "\n" if source.endswith("\n") else ""
        new_lines = original_lines[: start - 1] + replacement.splitlines() + original_lines[end:]
        return "\n".join(new_lines) + trailing_newline

    def _replace_entity_body(self, source: str, entity: Entity, new_body: str) -> str:
        """Replace the target entity with the LLM-generated implementation."""
        target_node = _find_entity_node(source, entity)
        if target_node is None:
            return source

        replacement = self._materialize_rewrite(source, target_node, entity.name, new_body)
        if not replacement:
            return source

        start, end = _node_line_range(target_node)
        original_lines = source.splitlines()
        indent = _line_indent(original_lines[start - 1])
        replacement = _indent_block(textwrap.dedent(replacement).strip("\n"), indent)
        trailing_newline = "\n" if source.endswith("\n") else ""
        new_lines = original_lines[: start - 1] + replacement.splitlines() + original_lines[end:]
        return "\n".join(new_lines) + trailing_newline

    def _definition_with_stubbed_body(self, source: str, node: Optional[ast.AST]) -> str:
        if node is None or not hasattr(node, "body"):
            return ""
        lines = textwrap.dedent(source).splitlines()
        start, _ = _node_line_range(node)
        body = getattr(node, "body", [])
        if body:
            body_start = getattr(body[0], "lineno", start + 1)
            body_indent = _line_indent(lines[body_start - 1])
            keep_end = body_start - 1
            if _is_docstring_node(body[0]):
                keep_end = getattr(body[0], "end_lineno", body_start)
        else:
            body_indent = _line_indent(lines[getattr(node, "lineno", start) - 1]) + "    "
            keep_end = getattr(node, "lineno", start)
        return "\n".join(lines[:keep_end] + [f"{body_indent}pass"])

    def _materialize_rewrite(
        self,
        original_source: str,
        original_node: ast.AST,
        symbol_name: str,
        generated_source: str,
    ) -> str:
        dedented = textwrap.dedent(generated_source).strip()
        if not dedented:
            return ""

        generated_node = _find_symbol_node_in_source(dedented, symbol_name)
        if generated_node is not None:
            if not _signature_matches(original_node, generated_node):
                return ""
            start, end = _node_line_range(generated_node)
            return "\n".join(dedented.splitlines()[start - 1:end])

        try:
            parsed_body = ast.parse(dedented)
        except SyntaxError:
            return ""
        if not parsed_body.body:
            return ""

        original_lines = original_source.splitlines()
        start, _ = _node_line_range(original_node)
        body = getattr(original_node, "body", [])
        if body:
            body_start = getattr(body[0], "lineno", start + 1)
            body_indent = _line_indent(original_lines[body_start - 1])
            keep_end = body_start - 1
            if _is_docstring_node(body[0]) and not _module_starts_with_docstring(parsed_body):
                keep_end = getattr(body[0], "end_lineno", body_start)
        else:
            body_indent = _line_indent(original_lines[getattr(original_node, "lineno", start) - 1]) + "    "
            keep_end = getattr(original_node, "lineno", start)

        header = original_lines[start - 1:keep_end]
        body_text = _indent_block(dedented, body_indent)
        return "\n".join(header + body_text.splitlines())

    def _call_agent(self, system_prompt: str, user_prompt: str) -> str:
        """Call the LLM via the agent adapter."""
        if hasattr(self.agent_adapter, "chat"):
            try:
                return self.agent_adapter.chat(system_prompt, user_prompt, temperature=0)
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
                max_tokens=4096,
            )
            return response.choices[0].message.content

        return ""

    def _make_candidate_id(self, plan: BugGenerationPlan) -> str:
        raw = f"{plan.plan_id}:lm_rewrite:{plan.target_file}:{plan.target_symbol}"
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
        return f"cand_{digest}"


def _find_entity_node(source: str, entity: Entity) -> Optional[ast.AST]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == entity.name
            and getattr(node, "lineno", 0) == entity.line
        ):
            return node
    return None


def _is_docstring_node(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(getattr(node, "value", None), (ast.Constant, ast.Str))
        and isinstance(getattr(getattr(node, "value", None), "value", None), str)
    )


def _module_starts_with_docstring(module: ast.Module) -> bool:
    return bool(module.body and _is_docstring_node(module.body[0]))
