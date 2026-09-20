from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


# ── ErrorLogger ────────────────────────────────────────────────────────────────
class ErrorLogger(Base):
    """Stores application-level error events for observability."""

    __tablename__ = "logger"
    __allow_unmapped__ = True

    logger_id   = Column(Integer, primary_key=True, autoincrement=True)
    function_name = Column(String(255), nullable=False)
    file_name   = Column(String(255), nullable=False)
    error_message = Column(Text, nullable=False)
    request_id  = Column(
        String(36),
        server_default=text("gen_random_uuid()::text"),
        unique=True,
        index=True,
        nullable=False,
    )
    # ── audit columns ──────────────────────────────────────────────────────────
    created_at  = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"))
    updated_at  = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"),
                         onupdate=lambda: datetime.now(timezone.utc))
    created_by  = Column(String(255), nullable=True)
    updated_by  = Column(String(255), nullable=True)
    is_active   = Column(Boolean, nullable=False, server_default=text("true"))


# ── Employee ───────────────────────────────────────────────────────────────────
class Employee(Base):
    """Represents an employee who interacts with the assistant."""

    __tablename__ = "employees"
    __allow_unmapped__ = True

    employee_id  = Column(String(50), primary_key=True)   # e.g. "emp_1023"
    name         = Column(String(255), nullable=False)
    email        = Column(String(255), nullable=False, unique=True, index=True)
    department   = Column(String(100), nullable=True)
    designation  = Column(String(100), nullable=True)
    # ── audit columns ──────────────────────────────────────────────────────────
    created_at   = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"))
    updated_at   = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"),
                          onupdate=lambda: datetime.now(timezone.utc))
    created_by   = Column(String(255), nullable=True)
    updated_by   = Column(String(255), nullable=True)
    is_active    = Column(Boolean, nullable=False, server_default=text("true"))


# ── Session ────────────────────────────────────────────────────────────────────
class Session(Base):
    """Maps a LangGraph thread (session_id) to the employee who owns it."""

    __tablename__ = "sessions"
    __allow_unmapped__ = True

    session_id   = Column(String(36), primary_key=True)   # UUID string
    employee_id  = Column(
        String(50),
        ForeignKey("employees.employee_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    last_active_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
    )
    # ── audit columns ──────────────────────────────────────────────────────────
    created_at   = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"))
    updated_at   = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"),
                          onupdate=lambda: datetime.now(timezone.utc))
    created_by   = Column(String(255), nullable=True)
    updated_by   = Column(String(255), nullable=True)
    is_active    = Column(Boolean, nullable=False, server_default=text("true"))


# ── Escalation ─────────────────────────────────────────────────────────────────
class Escalation(Base):
    """Records every escalation event for audit and human follow-up."""

    __tablename__ = "escalations"
    __allow_unmapped__ = True

    id           = Column(Integer, primary_key=True, autoincrement=True)
    employee_id  = Column(String(50), nullable=False, index=True)
    session_id   = Column(String(36), nullable=False, index=True)
    message      = Column(Text, nullable=False)      # triggering user message
    reason       = Column(String(50), nullable=False) # EscalationReason enum value
    reasoning    = Column(Text, nullable=True)        # LLM reasoning string when applicable
    status       = Column(String(20), nullable=False, server_default=text("'open'"))
    # ── audit columns ──────────────────────────────────────────────────────────
    created_at   = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"))
    updated_at   = Column(DateTime(timezone=True), nullable=False, server_default=text("now()"),
                          onupdate=lambda: datetime.now(timezone.utc))
    created_by   = Column(String(255), nullable=True)
    updated_by   = Column(String(255), nullable=True)
    is_active    = Column(Boolean, nullable=False, server_default=text("true"))
