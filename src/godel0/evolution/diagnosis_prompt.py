"""Diagnosis prompt builder."""

from __future__ import annotations

from ..schemas.cycle import CycleEvidenceBundle, NodeCycleSummary, SpecialAlert
from ..schemas.cycle import AlertPriority


DIAGNOSIS_SYSTEM_PROMPT = """You are diagnosing one complete self-evolving node, not an isolated Solver or Proposer.

Inspect the full task-generation-to-task-solving cycle.
Triggered special alerts have already been detected by trusted code.
Prioritize systemic Critical/High alerts.

Identify exactly ONE highest-impact root cause.

The root cause may require modifying Solver behavior, Proposer behavior,
shared tools, or common runtime. Do not choose a role first.

All proposed edits must address the same root cause.

Do not propose task-specific hardcoding.

Return one focused coding issue that can be implemented by the common
initial coding agent.
"""


def build_diagnosis_prompt(
    summary: NodeCycleSummary,
    evidence: CycleEvidenceBundle,
    agent_code_summary: str = "",
) -> str:
    """Build the diagnosis prompt from summary and evidence."""
    parts = []

    if agent_code_summary:
        parts.append(f"## Current Agent Code Summary\n{agent_code_summary}")

    parts.append(f"## Node Cycle Summary\n{summary.model_dump_json(indent=2)}")

    critical_high = [
        a for a in evidence.special_alerts
        if a.priority in (AlertPriority.CRITICAL, AlertPriority.HIGH)
    ]
    if critical_high:
        parts.append("## Triggered Special Alerts (Critical/High)")
        for alert in critical_high:
            parts.append(f"- {alert.alert_type}: {alert.recommended_attention} (severity={alert.severity})")

    if evidence.items:
        parts.append("## Representative Evidence")
        for item in evidence.items:
            parts.append(f"### {item.evidence_id} ({item.evidence_type})")
            parts.append(item.summary)

    parts.append("""
## Your Task

Analyze the above cycle and identify exactly ONE primary root cause.

Return your response as JSON with this structure:
```json
{
  "primary_root_cause": "...",
  "selected_alert_id": "...",
  "source_stages": ["solver", "proposer", "validation", "tools", "runtime"],
  "recommended_edit_scopes": ["coding_agent", "solver_prompt", "proposer_prompt", "proposer_logic", "tools", "llm_withtools", "utils", "requirements"],
  "evidence_ids": ["..."],
  "expected_effects": {"scope": "effect"},
  "non_goals": ["..."],
  "validation_plan": ["..."],
  "problem_statement": "..."
}
```

Rules:
- Select exactly ONE primary root cause.
- If Critical alerts exist, you must select one unless you provide an override_reason.
- All recommended edits must address the same root cause.
- Do not propose task-specific hardcoding.
""")

    return "\n\n".join(parts)
