from enum import Enum

# ── Token / temperature constants ──────────────────────────────────────────────
MAX_TOKENS: int = 10_000
TEMPERATURE: float = 0.7
DECISION_TEMPERATURE: float = 0.0   # structured-output routing calls must be deterministic
SUMMARIZATION_TOKEN: int = 200
SUMMARIZATION_TEMPERATURE: float = 1.0
SUMMARY_TRIGGER: int = 8
SUMMARY_KEEP: int = 2

# ── Retrieval constants ────────────────────────────────────────────────────────
RETRIEVAL_TOP_K: int = 5
RETRIEVAL_CONFIDENCE_THRESHOLD: float = 0.015  # fused RRF score below this → generate_response uses LowConfidence prompt
                                               # Max possible RRF score ≈ 1/(60+1) + 1/(60+1) ≈ 0.033; 0.015 is ~45% of max
EMBEDDING_MODEL: str = "models/gemini-embedding-2"  # Google embedding model

# ── Logger ─────────────────────────────────────────────────────────────────────
LOGGER_NAME: str = "ops_assistant"


# ── Intent classification output values ────────────────────────────────────────
class Intent(str, Enum):
    GENERAL = "general"           # greetings / chitchat — answered inline by classify_intent
    POLICY = "policy"             # normal policy question → goes to retrieve
    SENSITIVE = "sensitive"       # immediately escalated without confirmation


# ── LangGraph node names ────────────────────────────────────────────────────────
class NodeName(str, Enum):
    CLASSIFY_INTENT    = "classify_intent"
    RETRIEVE           = "retrieve"
    GENERATE_RESPONSE  = "generate_response"
    ESCALATE           = "escalate"
    CONFIRM_ESCALATION = "confirm_escalation"


# ── Escalation reason tags (stored in escalations table + MLflow) ───────────────
class EscalationReason(str, Enum):
    SENSITIVE_INTENT = "sensitive_intent"
    LLM_JUDGED_NO_SELF_SERVICE = "llm_judged_no_self_service"
