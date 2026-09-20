"""
Conditional edge functions for the LangGraph workflow.

Graph routing summary
---------------------
classify_intent
  → "end_general"          END   (general intent answered inline)
  → "escalate"             escalate → END   (sensitive intent)
  → "retrieve"             retrieve → generate_response  (plain edge)

generate_response
  → "confirm_escalation"   confirm_escalation → END
  → "end_respond"          END
"""

from __future__ import annotations

import constant
from models.model import GraphState
from utils.logger import Logger


class ConditionalEdges:
    """All routing functions for the workflow graph."""

    def __init__(self) -> None:
        self.logger = Logger("conditional_edges")

    # ── After classify_intent ──────────────────────────────────────────────────
    def route_after_classify(self, state: GraphState) -> str:
        """
        Returns
        -------
        "end_general"  → intent == "general"
        "escalate"     → intent == "sensitive"
        "retrieve"     → intent == "policy"
        """
        try:
            intent = state.get("intent", "policy")
            self.logger.info(
                "route_after_classify",
                intent=intent,
                employee_id=state.get("employee_id"),
                session_id=state.get("session_id"),
            )
            if intent == constant.Intent.GENERAL.value:
                return "end_general"
            if intent == constant.Intent.SENSITIVE.value or state.get("is_sensitive_intent"):
                return "escalate"
            return "retrieve"

        except Exception as e:
            self.logger.warning(f"route_after_classify fallback: {e}")
            return "retrieve"

    # ── After generate_response ────────────────────────────────────────────────
    def route_after_generate(self, state: GraphState) -> str:
        """
        Routes on state["requires_escalation"] which is set explicitly by
        every path in generate_response (including failure and no-docs paths)
        so no stale value can leak in.

        Returns
        -------
        "confirm_escalation"  → requires_escalation == True
        "end_respond"         → requires_escalation == False
        """
        try:
            requires = state.get("requires_escalation", False)
            self.logger.info(
                "route_after_generate",
                employee_id=state.get("employee_id"),
                session_id=state.get("session_id"),
                requires_escalation=requires,
            )
            return "confirm_escalation" if requires else "end_respond"

        except Exception as e:
            self.logger.warning(f"route_after_generate fallback: {e}")
            return "end_respond"
