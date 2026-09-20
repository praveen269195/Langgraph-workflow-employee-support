"""
AgentService — builds and runs the LangGraph workflow.

The compiled graph, checkpointer, and DB are all initialised ONCE at
application startup via the FastAPI lifespan and injected here through
the ``app.state`` object.  This avoids rebuilding the graph or opening
new DB/checkpointer connections on every request.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

import constant
from models.model import ChatRequest, ChatResponseData, GraphState
from repositories.session_repository import SessionRepository
from services.conditional_edges import ConditionalEdges
from services.graph_nodes import GraphNodes
from utils.Exceptions.errorcodes import ApplicationError, client_error
from utils.logger import Logger

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


class AgentService:
    """
    Builds the LangGraph workflow and executes chat turns.

    Parameters
    ----------
    compiled_graph:
        Pre-compiled StateGraph injected from lifespan.
    """

    def __init__(self, compiled_graph: CompiledStateGraph) -> None:
        self.graph = compiled_graph
        self.session_repo = SessionRepository()
        self.logger = Logger("agent_service")

    # ── Graph factory (called once at startup) ─────────────────────────────────
    @staticmethod
    def build_graph(
        checkpointer: "BaseCheckpointSaver",
        retrieval_service=None,
    ) -> CompiledStateGraph:
        """
        Assemble and compile the deterministic LangGraph workflow.

        Called once inside the FastAPI lifespan — never per-request.

        Parameters
        ----------
        checkpointer:       LangGraph checkpoint saver (AsyncPostgresSaver).
        retrieval_service:  Shared RetrievalService from app.state — injected so
                            GraphNodes never creates a duplicate VectorRepository.

        Workflow
        --------
        classify_intent
            ├─ "end_general"  → END  (general intent answered inline)
            ├─ "escalate"     → escalate → END  (sensitive intent)
            └─ "retrieve"     → retrieve (plain edge)
                                    └─ generate_response
                                           ├─ requires_escalation=True  → confirm_escalation → END
                                           └─ requires_escalation=False → END
        """
        from services.embedding_service import google_embedding
        from services.retrieval_service import RetrievalService as _RS
        from repositories.vector_repository import VectorRepository

        logger = Logger("agent_service")
        logger.info("Building LangGraph workflow")

        if retrieval_service is None:
            # Fallback for tests / standalone usage — creates its own instance
            retrieval_service = _RS(
                embedding_service=google_embedding,
                vector_repo=VectorRepository(),
            )

        nodes = GraphNodes(retrieval_service=retrieval_service)
        edges = ConditionalEdges()

        graph = StateGraph(GraphState)

        # ── Register nodes ─────────────────────────────────────────────────────
        graph.add_node(constant.NodeName.CLASSIFY_INTENT.value, nodes.classify_intent)
        graph.add_node(constant.NodeName.RETRIEVE.value, nodes.retrieve)
        graph.add_node(constant.NodeName.GENERATE_RESPONSE.value, nodes.generate_response)
        graph.add_node(constant.NodeName.ESCALATE.value, nodes.escalate)
        graph.add_node(constant.NodeName.CONFIRM_ESCALATION.value, nodes.confirm_escalation)

        # ── Entry edge ─────────────────────────────────────────────────────────
        graph.add_edge(START, constant.NodeName.CLASSIFY_INTENT.value)

        # ── Routing from classify_intent ───────────────────────────────────────
        graph.add_conditional_edges(
            constant.NodeName.CLASSIFY_INTENT.value,
            edges.route_after_classify,
            {
                "end_general": END,
                "escalate": constant.NodeName.ESCALATE.value,
                "retrieve": constant.NodeName.RETRIEVE.value,
            },
        )

        # ── Routing from retrieve → plain edge, always generate_response ─────
        graph.add_edge(
            constant.NodeName.RETRIEVE.value,
            constant.NodeName.GENERATE_RESPONSE.value,
        )

        # ── Terminal edge for escalate ─────────────────────────────────────────
        graph.add_edge(constant.NodeName.ESCALATE.value, END)

        # ── Routing from generate_response ─────────────────────────────────────
        graph.add_conditional_edges(
            constant.NodeName.GENERATE_RESPONSE.value,
            edges.route_after_generate,
            {
                "confirm_escalation": constant.NodeName.CONFIRM_ESCALATION.value,
                "end_respond": END,
            },
        )

        # ── Terminal edge for confirm_escalation ───────────────────────────────
        graph.add_edge(constant.NodeName.CONFIRM_ESCALATION.value, END)

        compiled = graph.compile(checkpointer=checkpointer, interrupt_before=[])
        logger.info("LangGraph workflow compiled successfully")
        return compiled

    # ── Main chat entry point ──────────────────────────────────────────────────
    async def chat(self, request: ChatRequest) -> ChatResponseData:
        """
        Execute one chat turn.

        Handles:
        - New session creation when session_id is absent.
        - Session ownership validation (403 on mismatch).
        - Resume via Command(resume=...) when the graph is awaiting confirmation.
        - interrupt detection → sets awaiting_confirmation=True in the response.
        """
        try:
            self.logger.info(
                "chat triggered",
                employee_id=request.employee_id,
                session_id=request.session_id,
            )

            employee_id = request.employee_id

            # ── Session resolution ─────────────────────────────────────────────
            if request.session_id is None:
                session_id = str(uuid.uuid4())
                await self.session_repo.create_session(session_id, employee_id)
                self.logger.info("New session created", session_id=session_id)
            else:
                session_id = request.session_id
                # validate_owner raises 404 if missing, 403 if wrong owner
                await self.session_repo.validate_owner(session_id, employee_id)

            thread_config: dict[str, Any] = {
                "configurable": {"thread_id": session_id}
            }

            # ── Check if graph is currently interrupted (awaiting confirmation) ─
            current_state = await self.graph.aget_state(thread_config)
            is_interrupted = bool(
                current_state and current_state.next and
                constant.NodeName.CONFIRM_ESCALATION.value in current_state.next
            )

            # ── Invoke graph ───────────────────────────────────────────────────
            if is_interrupted:
                # Resume the paused graph with the employee's yes/no answer
                result = await self.graph.ainvoke(
                    Command(resume=request.message),
                    config=thread_config,
                )
            else:
                # Fresh turn
                result = await self.graph.ainvoke(
                    {
                        "employee_id": employee_id,
                        "session_id": session_id,
                        "user_message": request.message,
                        "optimised_query": "",   # overwritten by classify_intent
                        # scratch defaults — overwritten by nodes
                        "intent": "",
                        "is_sensitive_intent": False,
                        "retrieved_docs": [],
                        "retrieval_confidence": 0.0,
                        "is_low_confidence": False,
                        "retrieval_failed": False,
                        "requires_escalation": False,
                        "escalation_reasoning": None,
                        "escalation_prompt": None,
                        "response": "",
                        "sources": [],
                        "escalated": False,
                        "awaiting_confirmation": False,
                    },
                    config=thread_config,
                )

            # ── Detect mid-graph interrupt (confirm_escalation first pass) ──────
            final_state = await self.graph.aget_state(thread_config)
            awaiting = bool(
                final_state and final_state.next and
                constant.NodeName.CONFIRM_ESCALATION.value in final_state.next
            )

            # When the graph is interrupted, ainvoke() returns the raw state values
            # dict from the checkpoint — read from final_state.values to be sure we
            # always get the latest persisted values regardless of interrupt vs normal.
            state_values: dict = (
                final_state.values if (final_state and final_state.values) else (result or {})
            )

            response_text: str = state_values.get("response", "")
            sources: list[str] = state_values.get("sources") or []
            escalated: bool = state_values.get("escalated", False)
            # Surface the escalation question so the client can display it.
            # Only populated when awaiting_confirmation=True (first interrupt pass).
            escalation_prompt: str | None = state_values.get("escalation_prompt") if awaiting else None

            # Touch last_active_at
            await self.session_repo.touch_session(session_id, employee_id)

            self.logger.info(
                "chat complete",
                session_id=session_id,
                escalated=escalated,
                awaiting_confirmation=awaiting,
            )

            return ChatResponseData(
                session_id=session_id,
                response=response_text,
                sources=sources,
                escalated=escalated,
                awaiting_confirmation=awaiting,
                escalation_prompt=escalation_prompt,
            )

        except ApplicationError:
            raise
        except Exception as e:
            await self.logger.error("Unexpected error in chat", e)
            raise client_error(str(e))
