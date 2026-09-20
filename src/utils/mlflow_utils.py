"""
MLflow observability helpers.

Usage
-----
Call ``setup_mlflow()`` once at application startup (inside lifespan).
Inside graph nodes use ``attach_trace_tags(...)`` to add business-level
tags that make it easy to filter traces by outcome in the MLflow UI.

Design note
-----------
``mlflow.langchain.autolog()`` is enabled once here.  Individual graph
nodes must NOT be decorated with ``@mlflow.trace`` — that fragments one
graph run into disconnected traces.  Autolog covers the whole LangGraph
execution as a single trace automatically.

The one exception is document ingestion which is NOT a LangGraph node,
so VectorRepository.ingest_document carries its own ``@mlflow.trace``.
"""

from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient

from settings import config
from utils.logger import Logger

_logger = Logger("mlflow_utils")


def setup_mlflow() -> None:
    """
    Configure the tracking server, create the experiment if needed,
    and enable LangChain / LangGraph autologging.

    Call once at startup — idempotent.

    Order matters per official docs:
      1. set_tracking_uri
      2. set_experiment
      3. langchain.autolog  ← must come last
    """
    try:
        # ── 1. Point to the tracking server ───────────────────────────────────
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)

        # ── 2. Create experiment if it doesn't exist, then activate it ─────────
        client = MlflowClient()
        experiment = client.get_experiment_by_name(config.mlflow_experiment_name)
        if experiment is None:
            client.create_experiment(config.mlflow_experiment_name)
            _logger.info(
                "MLflow experiment created",
                experiment=config.mlflow_experiment_name,
            )

        mlflow.set_experiment(config.mlflow_experiment_name)

        # ── 3. Enable autolog AFTER experiment is set ──────────────────────────
        # run_tracer_inline=True is required when using ainvoke() (async) so
        # that manually traced spans nest correctly within the autolog hierarchy.
        mlflow.langchain.autolog(
            log_traces=True,
            run_tracer_inline=True,    # required for async ainvoke() context propagation
        )

        _logger.info(
            "MLflow initialised",
            tracking_uri=config.mlflow_tracking_uri,
            experiment=config.mlflow_experiment_name,
        )
    except Exception as e:
        # MLflow failures must never crash the application.
        _logger.warning(f"MLflow setup failed (observability degraded): {e}")


def _has_active_trace() -> bool:
    """Return True when there is an active MLflow span to attach tags to."""
    try:
        return mlflow.get_current_active_span() is not None
    except Exception:
        return False


def attach_trace_tags(
    employee_id: str,
    session_id: str,
    intent: str | None = None,
    escalated: bool | None = None,
    escalation_reason: str | None = None,
) -> None:
    """
    Attach business-level tags to the currently active MLflow trace.

    Safe to call from inside any graph node. Silently skipped when no
    active trace exists (e.g. autolog hasn't opened one yet, or MLflow
    is not configured).
    """
    if not _has_active_trace():
        return
    try:
        tags: dict[str, str] = {
            "employee_id": employee_id,
            "session_id": session_id,
        }
        if intent is not None:
            tags["intent"] = intent
        if escalated is not None:
            tags["escalated"] = str(escalated).lower()
        if escalation_reason is not None:
            tags["escalation_reason"] = escalation_reason

        mlflow.update_current_trace(tags=tags)
    except Exception:
        pass


def log_low_confidence_retrieval(
    query: str,
    retrieved_docs: list[dict],
    confidence: float,
    session_id: str,
) -> None:
    """
    Log full raw retrieval detail to the current MLflow trace.

    The user-facing response intentionally omits raw chunk text —
    this is where the detail belongs (spec §7).
    """
    if not _has_active_trace():
        return
    try:
        mlflow.update_current_trace(
            tags={
                "low_confidence_retrieval": "true",
                "retrieval_confidence": str(round(confidence, 4)),
                "session_id": session_id,
            }
        )
        for idx, doc in enumerate(retrieved_docs[:5]):
            mlflow.log_param(
                f"low_conf_chunk_{idx}_source", doc.get("source", "unknown")
            )
            mlflow.log_param(
                f"low_conf_chunk_{idx}_score", str(doc.get("score", 0))
            )
    except Exception:
        pass


def log_escalation_decision(
    requires_escalation: bool,
    reasoning: str,
    session_id: str,
) -> None:
    """Log escalation decision tags to the current trace."""
    if not _has_active_trace():
        return
    try:
        mlflow.update_current_trace(
            tags={
                "requires_escalation": str(requires_escalation).lower(),
                "escalation_reasoning_preview": reasoning[:200],
                "session_id": session_id,
            }
        )
    except Exception:
        pass
