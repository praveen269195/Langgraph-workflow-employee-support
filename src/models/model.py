from typing import Annotated, Any, Literal, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict


# ══════════════════════════════════════════════════════════════════════════════
# API envelope  (unchanged shape from previous project)
# ══════════════════════════════════════════════════════════════════════════════

class APIResponse(BaseModel):
    code: int
    status: str
    message: str
    data: Optional[Any] = None
    error: Optional[dict] = None
    request_id: str
    timestamp: str


class Error(BaseModel):
    function_name: str
    file_name: str
    error_message: str


# ══════════════════════════════════════════════════════════════════════════════
# API request / response models
# ══════════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    employee_id: str = Field(..., min_length=1, max_length=50)
    session_id: str | None = Field(default=None)
    message: str

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message cannot be blank or whitespace only")
        return v.strip()

    @field_validator("employee_id")
    @classmethod
    def employee_id_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("employee_id cannot be blank")
        return v.strip()


class ChatResponseData(BaseModel):
    session_id: str
    response: str
    sources: list[str] = Field(default_factory=list)
    escalated: bool = False
    awaiting_confirmation: bool = False
    escalation_prompt: str | None = Field(
        default=None,
        description=(
            "Populated when awaiting_confirmation=True. "
            "The question the client should display to the employee (e.g. 'Would you like me to escalate this?')."
        ),
    )


class UploadResponseData(BaseModel):
    filename: str
    chunks_indexed: int
    status: Literal["success", "failed"]


# ══════════════════════════════════════════════════════════════════════════════
# LangGraph structured-output schemas
# ══════════════════════════════════════════════════════════════════════════════

class IntentClassification(BaseModel):
    """Structured output from the classify_intent node."""

    intent: Literal["general", "policy", "sensitive"]
    is_sensitive_intent: bool
    optimised_query: str = Field(
        description=(
            "A clean, retrieval-ready rewrite of the employee's message. "
            "Always populated regardless of intent. "
            "Expands abbreviations, resolves pronouns from context, and uses "
            "formal HR/IT terminology so vector and keyword search work better."
        ),
    )
    general_response: str | None = Field(
        default=None,
        description=(
            "Populated only when intent='general'. "
            "Contains the direct conversational reply for greetings / chitchat."
        ),
    )


class GenerateDecision(BaseModel):
    """
    Structured output from the generate_response node.
    Used for both the normal RAG path and the low-confidence path.
    """

    requires_escalation: bool
    reasoning: str = Field(
        description="Why escalation is / is not required. Logged to MLflow."
    )
    approval_authority: str | None = Field(
        default=None,
        description="Who should handle this if escalation is required.",
    )
    response: str = Field(
        description="Plain-text answer to the employee. No markdown."
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Source document names used in this answer.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# LangGraph state
# ══════════════════════════════════════════════════════════════════════════════

class GraphState(TypedDict):
    # ── Identity ───────────────────────────────────────────────────────────────
    employee_id: str
    session_id: str

    # ── Per-turn input ─────────────────────────────────────────────────────────
    user_message: str
    optimised_query: str          # LLM-rewritten query from classify_intent; used by retrieve + escalate

    # ── Separate message histories (accumulate via reducer, never shared) ──────
    # classify_intent  reads/writes intent_messages only
    # generate_response reads/writes generate_messages only
    intent_messages:   Annotated[list[AnyMessage], add_messages]
    generate_messages: Annotated[list[AnyMessage], add_messages]

    # ── Per-turn scratch values (overwritten each run, never accumulated) ──────
    intent: str
    is_sensitive_intent: bool
    retrieved_docs: list[dict]        # {text, source, metadata, score}
    retrieval_confidence: float
    is_low_confidence: bool           # set by retrieve, read by generate_response
    retrieval_failed: bool            # True when the vector store call threw an exception
    requires_escalation: bool
    escalation_reasoning: str | None
    escalation_prompt: str | None     # confirmation question shown to the employee
    response: str
    sources: list[str]
    escalated: bool
    awaiting_confirmation: bool
