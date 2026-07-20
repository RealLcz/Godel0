"""Cycle diagnoser: selects one primary root cause from evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..errors import SchemaValidationError
from ..schemas.cycle import CycleEvidenceBundle, NodeCycleSummary, SpecialAlert, AlertPriority
from ..schemas.diagnosis import CycleDiagnosis
from .diagnosis_prompt import build_diagnosis_prompt, DIAGNOSIS_SYSTEM_PROMPT


class CycleDiagnoser:
    """Selects one highest-value root cause from the cycle evidence.

    The diagnoser may use an LLM to analyze evidence, or use deterministic
    rules when no LLM is available.
    """

    def __init__(self, llm_client=None, chat_adapter=None, max_retries: int = 2):
        self.llm_client = llm_client
        self.chat_adapter = chat_adapter
        self.max_retries = max_retries

    def diagnose(
        self,
        node_id: str,
        summary: NodeCycleSummary,
        evidence: CycleEvidenceBundle,
        agent_code_summary: str = "",
    ) -> CycleDiagnosis:
        """Produce a cycle diagnosis with one primary root cause."""
        if self.chat_adapter is not None and callable(
            getattr(self.chat_adapter, "chat", None)
        ):
            return self._diagnose_with_chat(
                node_id, summary, evidence, agent_code_summary
            )
        if self.llm_client is not None:
            return self._diagnose_with_llm(node_id, summary, evidence, agent_code_summary)
        return self._diagnose_deterministic(node_id, summary, evidence)

    def _diagnose_with_chat(
        self,
        node_id: str,
        summary: NodeCycleSummary,
        evidence: CycleEvidenceBundle,
        agent_code_summary: str,
    ) -> CycleDiagnosis:
        """Use the same node-compatible chat path as other LLM decisions."""
        prompt = build_diagnosis_prompt(summary, evidence, agent_code_summary)
        for attempt in range(self.max_retries + 1):
            try:
                response = self.chat_adapter.chat(
                    DIAGNOSIS_SYSTEM_PROMPT,
                    prompt,
                    temperature=0,
                    max_tokens=4096,
                )
                diagnosis = self._parse_llm_response(response, node_id, evidence)
                self._validate(diagnosis, evidence)
                return diagnosis
            except (SchemaValidationError, json.JSONDecodeError, TypeError, ValueError):
                if attempt >= self.max_retries:
                    break
        return self._diagnose_deterministic(node_id, summary, evidence)

    def _diagnose_with_llm(
        self,
        node_id: str,
        summary: NodeCycleSummary,
        evidence: CycleEvidenceBundle,
        agent_code_summary: str,
    ) -> CycleDiagnosis:
        """Use LLM to produce diagnosis."""
        prompt = build_diagnosis_prompt(summary, evidence, agent_code_summary)

        for attempt in range(self.max_retries + 1):
            try:
                from ..llm_compat import get_llm_response
                response = get_llm_response(
                    self.llm_client, prompt, DIAGNOSIS_SYSTEM_PROMPT
                )
                diagnosis = self._parse_llm_response(response, node_id, evidence)
                self._validate(diagnosis, evidence)
                return diagnosis
            except (SchemaValidationError, json.JSONDecodeError) as e:
                if attempt >= self.max_retries:
                    break
                continue

        return self._diagnose_deterministic(node_id, summary, evidence)

    def _diagnose_deterministic(
        self,
        node_id: str,
        summary: NodeCycleSummary,
        evidence: CycleEvidenceBundle,
    ) -> CycleDiagnosis:
        """Produce a deterministic diagnosis without LLM."""
        critical = [
            a for a in evidence.special_alerts
            if a.priority == AlertPriority.CRITICAL and a.triggered
        ]
        high = [
            a for a in evidence.special_alerts
            if a.priority == AlertPriority.HIGH and a.triggered
        ]

        selected = None
        if critical:
            selected = max(critical, key=lambda a: a.severity)
        elif high:
            selected = max(high, key=lambda a: a.severity)

        if selected:
            root_cause = selected.recommended_attention
            source_stages = [selected.source.value]
            if "proposer" in selected.alert_type:
                source_stages.append("validation")
            edit_scopes = self._infer_scopes(selected)
        else:
            if summary.level2_accuracy is not None and summary.level2_accuracy < 0.4:
                root_cause = "Low frontier accuracy - solver may be too weak or tasks too hard"
                source_stages = ["solver", "proposer"]
                edit_scopes = ["coding_agent", "proposer_logic"]
            elif summary.proposer_valid_yield is not None and summary.proposer_valid_yield < 0.2:
                root_cause = "Low proposer valid yield - candidates not producing F2P"
                source_stages = ["proposer", "validation"]
                edit_scopes = ["proposer_logic", "tools"]
            else:
                root_cause = "General performance improvement needed"
                source_stages = ["solver"]
                edit_scopes = ["coding_agent"]

        return CycleDiagnosis(
            node_id=node_id,
            primary_root_cause=root_cause,
            selected_alert_id=selected.alert_id if selected else None,
            source_stages=source_stages,
            recommended_edit_scopes=edit_scopes,
            evidence_ids=[item.evidence_id for item in evidence.items],
            expected_effects={},
            non_goals=["Do not hardcode task-specific solutions"],
            validation_plan=["Run proposer candidate generation smoke test", "Compare metrics before and after"],
            problem_statement=root_cause,
        )

    def _infer_scopes(self, alert: SpecialAlert) -> list[str]:
        if alert.source.value == "solver":
            return ["coding_agent", "solver_prompt"]
        elif alert.source.value == "proposer":
            return ["proposer_logic", "proposer_prompt"]
        else:
            return ["tools"]

    def _parse_llm_response(
        self,
        response: str,
        node_id: str,
        evidence: CycleEvidenceBundle,
    ) -> CycleDiagnosis:
        from ..llm_compat import extract_json_between_markers
        data = extract_json_between_markers(response)
        if data is None:
            raise SchemaValidationError("Could not parse LLM response as JSON")
        return CycleDiagnosis(
            node_id=node_id,
            primary_root_cause=data.get("primary_root_cause", ""),
            selected_alert_id=data.get("selected_alert_id"),
            source_stages=data.get("source_stages", []),
            recommended_edit_scopes=data.get("recommended_edit_scopes", []),
            evidence_ids=data.get("evidence_ids", []),
            expected_effects=data.get("expected_effects", {}),
            non_goals=data.get("non_goals", []),
            validation_plan=data.get("validation_plan", []),
            problem_statement=data.get("problem_statement", ""),
            override_reason=data.get("override_reason"),
        )

    def _validate(self, diagnosis: CycleDiagnosis, evidence: CycleEvidenceBundle) -> None:
        if not diagnosis.primary_root_cause:
            raise SchemaValidationError("primary_root_cause is empty")
        if not diagnosis.problem_statement:
            raise SchemaValidationError("problem_statement is empty")

        critical = [
            a for a in evidence.special_alerts
            if a.priority == AlertPriority.CRITICAL and a.triggered
        ]
        if critical:
            selected_ids = {a.alert_id for a in critical}
            if diagnosis.selected_alert_id not in selected_ids and not diagnosis.override_reason:
                raise SchemaValidationError(
                    "Critical alert exists but not selected, and no override_reason provided"
                )
