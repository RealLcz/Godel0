"""LLM compatibility utilities for the control layer."""

from __future__ import annotations

import json
import re
from typing import Any, Optional


def extract_json_between_markers(llm_output: str) -> Optional[Any]:
    """Extract JSON from LLM output (between ```json fences or fallback)."""
    inside_json_block = False
    json_lines = []

    for line in llm_output.split("\n"):
        striped_line = line.strip()

        if striped_line.startswith("```json"):
            inside_json_block = True
            continue

        if inside_json_block and striped_line.startswith("```"):
            inside_json_block = False
            break

        if inside_json_block:
            json_lines.append(line)

    if not json_lines:
        fallback_pattern = r"\{.*?\}"
        matches = re.findall(fallback_pattern, llm_output, re.DOTALL)
        for candidate in matches:
            candidate = candidate.strip()
            if candidate:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    candidate_clean = re.sub(r"[\x00-\x1F\x7F]", "", candidate)
                    try:
                        return json.loads(candidate_clean)
                    except json.JSONDecodeError:
                        continue
        return None

    json_string = "\n".join(json_lines).strip()

    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
        try:
            return json.loads(json_string_clean)
        except json.JSONDecodeError:
            return None


def get_llm_response(client, msg: str, system_message: str) -> str:
    """Get a response from an LLM client."""
    response = client.chat.completions.create(
        model=client.models.list().data[0].id if hasattr(client, "models") else "default",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        temperature=0.0,
        max_tokens=4096,
    )
    return response.choices[0].message.content
