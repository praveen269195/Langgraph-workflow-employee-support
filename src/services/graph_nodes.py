"""
LangGraph node implementations.

Node map
--------
classify_intent    → LLM structured output (IntentClassification).
                     Writes to intent_messages only. Slice history to last 6.
retrieve           → Hybrid PgVector + BM25 search.
                     Sets retrieved_docs, retrieval_confidence,
                     is_low_confidence, retrieval_failed.
generate_response  → Single structured LLM call (GenerateDecision).
                     Picks GenerateResponse or LowConfidenceGenerate prompt
                     based on is_low_confidence. Handles retrieval_failed inline.
                     Writes to generate_messages only.
escalate           → Writes escalation row (sensitive intent path, no confirmation).
confirm_escalation → interrupt() OUTSIDE try. Pauses graph for employee yes/no.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

import constant
from models.model import GenerateDecision, GraphState, IntentClassification
from prompt.prompt import PromptBuilder
from repositories.escalation_repository import EscalationRepository
from services.llm_service import gemini_llm
from services.retrieval_service import RetrievalService
from utils.Exceptions.errorcodes import client_error
from utils.logger import Logger
from utils.mlflow_utils import (
    attach_trace_tags,
    log_escalation_decision,
    log_low_confidence_retrieval,
)

_prompts = PromptBuilder()

# ── Module-level constants ─────────────────────────────────────────────────────
ESCALATION_PROMPT = (
    "Would you like me to escalate this to the relevant team? (yes/no)"
)

_YES_WORDS = {
    "yes", "y", "yeah", "yep", "yup", "sure",
    "ok", "okay", "confirm", "true",
}


def _is_yes(answer) -> bool:
    """
    Accept str, bool, or {"confirm": bool} resume payloads.
    Strips punctuation and ignores case before checking.
    """
    if isinstance(answer, bool):
        return answer
    if isinstance(answer, dict):
        return bool(answer.get("confirm"))
    words = re.sub(r"[^a-z\s]", "", str(answer).lower()).split()
    return bool(words) and words[0] in _YES_WORDS


class GraphNodes:
    """All node callables wired into the LangGraph StateGraph."""

    def __init__(self, retrieval_service: RetrievalService) -> None:
        self.logger = Logger("graph_nodes")
        self.escalation_repo = EscalationRepository()
        self.retrieval_service = retrieval_service

    # ── Private helpers ────────────────────────────────────────────────────────

    def _safe_tag(self, **kwargs) -> None:
        """MLflow tag failures must never fail a node that already did its work."""
        try:
            attach_trace_tags(**kwargs)
        except Exception as e:
            self.logger.warning("attach_trace_tags failed", error=str(e))

    def _answer_update(
        self,
        human_msg: HumanMessage,
        response: str,
        *,
        requires_escalation: bool = False,
        reasoning: str | None = None,
        sources: list | None = None,
    ) -> dict:
        """
        Build the standard return dict for generate_response.

        The AIMessage stored in generate_messages contains only the policy
        response — NOT the escalation question — so conversation history
        stays clean. escalation_prompt is a separate state field read by
        the route handler to surface the question to the client.
        """
        return {
            "response": response,
            "sources": sources or [],
            "requires_escalation": requires_escalation,
            "escalation_reasoning": reasoning,
            # Escalation question carried separately — not mixed into history
            "escalation_prompt": ESCALATION_PROMPT if requires_escalation else None,
            "escalated": False,
            "awaiting_confirmation": False,
            # Store only the policy response in history (not the escalation prompt)
            "generate_messages": [human_msg, AIMessage(content=response)],
        }

    def _retrieval_query(self, state: GraphState) -> str:
        """
        Short follow-ups ("what about contractors?") borrow context from the
        previous question so the embedding has enough signal to retrieve well.
        """
        user_message = state["user_message"]
        if len(user_message.split()) >= 8:
            return user_message
        for m in reversed(state.get("generate_messages") or []):
            if isinstance(m, HumanMessage) and len(str(m.content).split()) > 3:
                return f"{m.content}\n{user_message}"
        return user_message

    # ══════════════════════════════════════════════════════════════════════════
    # Node 1 — classify_intent
    # Writes: intent_messages only
    # ══════════════════════════════════════════════════════════════════════════
    async def classify_intent(self, state: GraphState) -> dict:
        """
        LLM call → IntentClassification (structured output).
        - General messages answered inline; graph ends here.
        - Sensitive intent flagged; routes to escalate node.
        - Policy intent routes to retrieve.
        - History sliced to last 6 messages to bound context growth.
        - Writes only to intent_messages — never touches generate_messages.
        """
        try:
            user_message = state["user_message"]
            employee_id  = state["employee_id"]
            session_id   = state["session_id"]

            self.logger.info(
                "classify_intent started",
                employee_id=employee_id,
                session_id=session_id,
                message_length=len(user_message),
            )

            human_msg = HumanMessage(content=user_message)
            # Slice to last 6 to prevent unbounded context growth
            intent_history = list(state.get("intent_messages") or [])[-6:]
            messages_for_llm = intent_history + [human_msg]

            llm = await gemini_llm.fetch_llm(
                max_tokens=1000,
                temperature=constant.DECISION_TEMPERATURE,
            )
            structured = llm.with_structured_output(IntentClassification,method="json_schema",
                strict=True)
            result: IntentClassification = await structured.ainvoke(
                [SystemMessage(content=_prompts.fetch_system_prompt("ClassifyIntent"))]
                + messages_for_llm
            )
            if result is None:
                raise ValueError("Intent classifier returned no structured output")

            is_sensitive = result.is_sensitive_intent
            intent = "sensitive" if is_sensitive else result.intent

            self._safe_tag(
                employee_id=employee_id,
                session_id=session_id,
                intent=intent,
            )

            ai_content = (
                result.general_response
                if intent == "general" and result.general_response
                else json.dumps({"intent": intent})
            )
            ai_msg = AIMessage(content=ai_content)

            self.logger.info(
                "classify_intent ended",
                employee_id=employee_id,
                session_id=session_id,
                intent=intent,
                is_sensitive=is_sensitive,
                optimised_query=result.optimised_query,
            )

            update: dict = {
                "intent": intent,
                "is_sensitive_intent": is_sensitive,
                "optimised_query": result.optimised_query,
                "intent_messages": [human_msg, ai_msg],
            }

            if intent == "general":
                update["response"] = (
                    result.general_response or "Hello! How can I help you today?"
                )
                update["sources"] = []
                update["escalated"] = False
                update["awaiting_confirmation"] = False

            return update

        except Exception as e:
            await self.logger.error(
                "classify_intent failed",
                e,
                employee_id=state.get("employee_id"),
                session_id=state.get("session_id"),
            )
            raise client_error(str(e))

    # ══════════════════════════════════════════════════════════════════════════
    # Node 2 — retrieve
    # Writes: retrieved_docs, retrieval_confidence, is_low_confidence,
    #         retrieval_failed
    # ══════════════════════════════════════════════════════════════════════════
    async def retrieve(self, state: GraphState) -> dict:
        """
        Hybrid retrieval: semantic cosine + BM25 via RRF.
        Uses _retrieval_query() to augment short follow-up messages.
        Sets retrieval_failed=True on infrastructure exceptions so
        generate_response can return a graceful "try again" message
        without treating it as low confidence.
        """
        try:
            employee_id = state["employee_id"]
            session_id  = state["session_id"]

            self.logger.info(
                "retrieve started",
                employee_id=employee_id,
                session_id=session_id,
                message_length=len(state["user_message"]),
            )

            docs = await self.retrieval_service.hybrid_search(
                query=state.get("optimised_query") or self._retrieval_query(state),
                top_k=constant.RETRIEVAL_TOP_K,
            )

            if not docs:
                self.logger.warning(
                    "retrieve: no results from vector store",
                    employee_id=employee_id,
                    session_id=session_id,
                )
                return {
                    "retrieved_docs": [],
                    "retrieval_confidence": 0.0,
                    "is_low_confidence": True,
                    "retrieval_failed": False,
                }

            top_score = docs[0]["score"]
            is_low = top_score < constant.RETRIEVAL_CONFIDENCE_THRESHOLD

            self.logger.info(
                "retrieve ended",
                employee_id=employee_id,
                session_id=session_id,
                docs_returned=len(docs),
                top_score=round(top_score, 4),
                is_low_confidence=is_low,
            )
            return {
                "retrieved_docs": docs,
                "retrieval_confidence": top_score,
                "is_low_confidence": is_low,
                "retrieval_failed": False,
            }

        except Exception as e:
            await self.logger.error(
                "retrieve failed",
                e,
                employee_id=state.get("employee_id"),
                session_id=state.get("session_id"),
            )
            # Infrastructure failure — NOT the same as low confidence
            return {
                "retrieved_docs": [],
                "retrieval_confidence": 0.0,
                "is_low_confidence": False,
                "retrieval_failed": True,
            }

    # ══════════════════════════════════════════════════════════════════════════
    # Node 3 — generate_response
    # Single structured LLM call for all paths.
    # Writes: generate_messages only — never touches intent_messages.
    # ══════════════════════════════════════════════════════════════════════════
    async def generate_response(self, state: GraphState) -> dict:
        """
        Handles three paths with one structured GenerateDecision call:

        1. retrieval_failed=True  → no LLM call; return graceful "try again" message.
        2. no docs returned       → skip LLM; offer escalation directly.
        3. normal / low-confidence → structured LLM call with the appropriate prompt.

        Sources are validated against the actual docs in context to prevent
        the LLM from hallucinating source names.
        Writes only to generate_messages — completely separate from intent_messages.
        """
        try:
            employee_id  = state["employee_id"]
            session_id   = state["session_id"]
            user_message = state["user_message"]
            docs         = state.get("retrieved_docs") or []
            is_low       = state.get("is_low_confidence", False)

            human_msg        = HumanMessage(content=user_message)
            generate_history = list(state.get("generate_messages") or [])

            self.logger.info(
                "generate_response started",
                employee_id=employee_id,
                session_id=session_id,
                docs_in_context=len(docs),
                is_low_confidence=is_low,
                retrieval_failed=bool(state.get("retrieval_failed")),
            )

            # ── Path 1: infrastructure failure ─────────────────────────────────
            if state.get("retrieval_failed"):
                self.logger.warning(
                    "generate_response: retrieval_failed, skipping LLM call",
                    employee_id=employee_id,
                    session_id=session_id,
                )
                return self._answer_update(
                    human_msg,
                    "I'm having trouble searching the policy documents right now. "
                    "Please try again in a few minutes.",
                )

            # ── Path 2: empty result set ────────────────────────────────────────
            if not docs:
                self.logger.warning(
                    "generate_response: no docs, offering escalation",
                    employee_id=employee_id,
                    session_id=session_id,
                )
                return self._answer_update(
                    human_msg,
                    "I couldn't find anything relevant in the policy documents.",
                    requires_escalation=True,
                    reasoning="No relevant documents retrieved.",
                )

            # ── Path 3: low-confidence — log to MLflow before LLM call ─────────
            if is_low:
                log_low_confidence_retrieval(
                    query=user_message,
                    retrieved_docs=docs,
                    confidence=state.get("retrieval_confidence", 0.0),
                    session_id=session_id,
                )

            # ── Single structured LLM call (both normal and low-confidence) ─────
            prompt_name = "LowConfidenceGenerate" if is_low else "GenerateResponse"
            context_block = "\n\n---\n\n".join(
                f"[Source: {d['source']}]\n{d['text']}" for d in docs
            )

            llm = await gemini_llm.fetch_llm(
                max_tokens=constant.MAX_TOKENS,
                temperature=constant.DECISION_TEMPERATURE,
            )
            structured = llm.with_structured_output(GenerateDecision)
            result: GenerateDecision = await structured.ainvoke(
                [SystemMessage(content=_prompts.fetch_system_prompt(prompt_name))]
                + generate_history
                + [
                    HumanMessage(
                        content=(
                            f"Employee question: {user_message}\n\n"
                            f"<context>\n{context_block}\n</context>"
                        )
                    )
                ]
            )
            if result is None:
                raise ValueError("generate_response returned no structured output")

            log_escalation_decision(
                requires_escalation=result.requires_escalation,
                reasoning=result.reasoning,
                session_id=session_id,
            )

            # Validate sources — only keep names that were actually in the context
            doc_sources = {d["source"] for d in docs}
            sources = [s for s in (result.sources or []) if s in doc_sources]

            self.logger.info(
                "generate_response ended",
                employee_id=employee_id,
                session_id=session_id,
                prompt=prompt_name,
                requires_escalation=result.requires_escalation,
                sources=sources,
            )
            return self._answer_update(
                human_msg,
                result.response,
                requires_escalation=result.requires_escalation,
                reasoning=result.reasoning,
                sources=sources,
            )

        except Exception as e:
            await self.logger.error(
                "generate_response failed",
                e,
                employee_id=state.get("employee_id"),
                session_id=state.get("session_id"),
            )
            raise client_error(str(e))

    # ══════════════════════════════════════════════════════════════════════════
    # Node 4 — escalate  (sensitive intent, direct — no confirmation needed)
    # ══════════════════════════════════════════════════════════════════════════
    async def escalate(self, state: GraphState) -> dict:
        """Writes escalation row immediately. No employee confirmation required."""
        try:
            employee_id = state["employee_id"]
            session_id  = state["session_id"]

            self.logger.info(
                "escalate started",
                employee_id=employee_id,
                session_id=session_id,
                message_length=len(state["user_message"]),
            )

            await self.escalation_repo.create_escalation(
                employee_id=employee_id,
                session_id=session_id,
                message=state["user_message"],
                reason=constant.EscalationReason.SENSITIVE_INTENT.value,
                reasoning="Request classified as sensitive intent by classify_intent node.",
            )

            self._safe_tag(
                employee_id=employee_id,
                session_id=session_id,
                escalated=True,
                escalation_reason=constant.EscalationReason.SENSITIVE_INTENT.value,
            )

            optimised_query = state.get("optimised_query") or state["user_message"]
            response = (
                f"I've received your request: \"{optimised_query}\". "
                "This has been escalated to the appropriate team and someone "
                "will follow up with you directly."
            )

            self.logger.info(
                "escalate ended",
                employee_id=employee_id,
                session_id=session_id,
                escalated=True,
            )
            return {
                "response": response,
                "sources": [],
                "escalated": True,
                "awaiting_confirmation": False,
                "generate_messages": [
                    HumanMessage(content=state["user_message"]),
                    AIMessage(content=response),
                ],
            }

        except Exception as e:
            await self.logger.error(
                "escalate failed",
                e,
                employee_id=state.get("employee_id"),
                session_id=state.get("session_id"),
            )
            raise client_error(str(e))

    # ══════════════════════════════════════════════════════════════════════════
    # Node 5 — confirm_escalation  (human-in-the-loop)
    # interrupt() is OUTSIDE the try block.
    # GraphInterrupt must propagate freely — catching it breaks the mechanism.
    # ══════════════════════════════════════════════════════════════════════════
    async def confirm_escalation(self, state: GraphState) -> dict:
        """
        First pass  → interrupt() pauses the graph; client sees awaiting_confirmation=True.
        On resume   → _is_yes() checks the answer robustly (str / bool / dict).
                      yes: write escalation row, confirm to employee.
                      no:  return original policy answer so employee still gets it.
        """
        employee_id = state["employee_id"]
        session_id  = state["session_id"]

        self.logger.info(
            "confirm_escalation started",
            employee_id=employee_id,
            session_id=session_id,
        )

        # ── interrupt() OUTSIDE try — GraphInterrupt must not be caught ────────
        employee_answer = interrupt(
            {
                "answer": state.get("response", ""),
                "question": state.get("escalation_prompt") or ESCALATION_PROMPT,
                "reasoning": state.get("escalation_reasoning", ""),
            }
        )

        # ── Post-resume logic — safe to catch exceptions from here ─────────────
        try:
            confirmed = _is_yes(employee_answer)

            self.logger.info(
                "confirm_escalation resumed",
                employee_id=employee_id,
                session_id=session_id,
                confirmed=confirmed,
            )

            if confirmed:
                await self.escalation_repo.create_escalation(
                    employee_id=employee_id,
                    session_id=session_id,
                    message=state["user_message"],
                    reason=constant.EscalationReason.LLM_JUDGED_NO_SELF_SERVICE.value,
                    reasoning=state.get("escalation_reasoning"),
                )
                self._safe_tag(
                    employee_id=employee_id,
                    session_id=session_id,
                    escalated=True,
                    escalation_reason=constant.EscalationReason.LLM_JUDGED_NO_SELF_SERVICE.value,
                )
                response = (
                    "Understood. Your request has been escalated to the relevant team. "
                    "They will be in touch with you shortly."
                )
            else:
                # state["response"] now always holds the substantive policy answer
                # (never the escalation question — that lives in escalation_prompt).
                # Return it directly so the employee still gets the useful procedure.
                response = state.get("response", "")
                self._safe_tag(
                    employee_id=employee_id,
                    session_id=session_id,
                    escalated=False,
                )

            self.logger.info(
                "confirm_escalation ended",
                employee_id=employee_id,
                session_id=session_id,
                escalated=confirmed,
            )
            return {
                "response": response,
                "escalated": confirmed,
                "awaiting_confirmation": False,
                "generate_messages": [
                    HumanMessage(content="yes" if confirmed else "no"),
                    AIMessage(content=response),
                ],
            }

        except Exception as e:
            await self.logger.error(
                "confirm_escalation failed after resume",
                e,
                employee_id=employee_id,
                session_id=session_id,
            )
            raise client_error(str(e))
