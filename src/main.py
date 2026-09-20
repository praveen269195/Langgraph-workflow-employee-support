"""
Application entry point.

Startup sequence (lifespan):
  1. Initialise Database singleton (engine + session factory).
  2. Run migrations (create tables + pgvector extension).
  3. Create VectorRepository (document_chunks table).
  4. Initialise AsyncPostgresSaver checkpointer (runs DDL once).
  5. Compile LangGraph workflow (once — never per request).
  6. Set up MLflow autologging.
  7. Attach all shared objects to app.state so routes can access them.

Shutdown sequence (lifespan):
  8. Dispose SQLAlchemy engine (closes connection pool).
  9. Close checkpointer connection.

"""

from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from uvicorn import Config, Server

import uuid
from datetime import datetime, timezone

from repositories.Database import Database
from routes.routes import router
from settings import config
from utils.Exceptions.errorcodes import ErrorCode
from utils.logger import Logger

logger = Logger("main")


# ══════════════════════════════════════════════════════════════════════════════
# Lifespan — startup and shutdown
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Async context manager for application lifecycle.

    Everything inside the ``try`` block runs at startup.
    Everything inside ``finally`` runs at shutdown.
    Objects are attached to ``app.state`` so all routes can access them
    without re-initialising anything.
    """
    logger.info("=== Application startup ===")

    # ── 1. Database ────────────────────────────────────────────────────────────
    db = Database()
    app.state.db = db
    logger.info("Database engine initialised")

    # ── 2. Migrations ──────────────────────────────────────────────────────────
    try:
        from migration.migration import Migration
        migration = Migration()
        await migration.create_tables()
        logger.info("Database migrations complete")
    except Exception as e:
        await logger.error("Migration failed — continuing without migration", e)

    # ── 3. VectorRepository + DocumentService + RetrievalService ──────────────
    try:
        from repositories.vector_repository import VectorRepository
        from services.embedding_service import google_embedding
        from services.document_service import DocumentService
        from services.retrieval_service import RetrievalService

        vector_repo = VectorRepository()
        await vector_repo.create_table()
        app.state.vector_repo = vector_repo

        document_service = DocumentService(
            embedding_service=google_embedding,
            vector_repo=vector_repo,
        )
        app.state.document_service = document_service

        retrieval_service = RetrievalService(
            embedding_service=google_embedding,
            vector_repo=vector_repo,
        )
        app.state.retrieval_service = retrieval_service

        logger.info("VectorRepository, DocumentService, RetrievalService initialised")
    except Exception as e:
        await logger.error("Vector/Document/Retrieval service init failed", e)
        app.state.vector_repo = None
        app.state.document_service = None
        app.state.retrieval_service = None

    # ── 4. LangGraph Checkpointer ──────────────────────────────────────────────
    try:
        from psycopg import AsyncConnection
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        pg_conn = await AsyncConnection.connect(config.db_uri(), autocommit=True)
        checkpointer = AsyncPostgresSaver(pg_conn)
        # Setup runs DDL once — safe to call at startup only
        await checkpointer.setup()
        app.state.checkpointer = checkpointer
        app.state._pg_conn = pg_conn
        logger.info("LangGraph checkpointer initialised")
    except Exception as e:
        await logger.error("Checkpointer init failed — chat will be unavailable", e)
        app.state.checkpointer = None
        app.state._pg_conn = None
        pg_conn = None
        checkpointer = None

    # ── 5. Compile LangGraph workflow ──────────────────────────────────────────
    if checkpointer is not None:
        from services.agent_service import AgentService
        compiled_graph = AgentService.build_graph(checkpointer, retrieval_service=app.state.retrieval_service)
        agent_service = AgentService(compiled_graph)
        app.state.agent_service = agent_service
        logger.info("LangGraph workflow compiled and ready")
    else:
        app.state.agent_service = None
        logger.warning("LangGraph workflow NOT compiled — checkpointer unavailable")

    # ── 6. MLflow ──────────────────────────────────────────────────────────────
    try:
        from utils.mlflow_utils import setup_mlflow
        setup_mlflow()
    except Exception as e:
        await logger.error("MLflow setup failed — observability degraded", e)

    logger.info("=== Application ready ===")

    yield  # ← application runs here

    # ══════════════════════════════════════════════════════════════════════════
    # Shutdown
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=== Application shutdown ===")

    try:
        await db.engine.dispose()
        logger.info("Database engine disposed")
    except Exception as e:
        await logger.error("Error disposing DB engine", e)

    try:
        if pg_conn is not None:
            await pg_conn.close()
            logger.info("Checkpointer connection closed")
    except Exception as e:
        await logger.error("Error closing checkpointer connection", e)


# ══════════════════════════════════════════════════════════════════════════════
# Exception handlers
# ══════════════════════════════════════════════════════════════════════════════

async def validation_exception_handler(
    request: Request, exc: ValidationError
) -> JSONResponse:
    """Handle Pydantic model validation errors (422)."""
    error_details = []
    for error in exc.errors():
        field_path = " -> ".join(str(loc) for loc in error["loc"])
        msg = error["msg"]
        if "missing" in msg:
            message = f"Field '{field_path}' is required."
        elif "empty" in msg or "whitespace" in msg:
            message = f"Field '{field_path}' cannot be empty or whitespace only."
        else:
            message = f"{field_path}: {msg}"
        await logger.error("Pydantic ValidationError", exc)
        error_details.append({"code": ErrorCode.VALIDATION_ERROR.value, "message": message})

    return JSONResponse(
        status_code=422,
        content={
            "code": 422,
            "status": "Error",
            "message": "Pydantic Validation",
            "request_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "errors": error_details,
        },
    )


async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle FastAPI request validation errors (400)."""
    error_details = []
    for error in exc.errors():
        field_path = " -> ".join(str(loc) for loc in error["loc"])
        msg = error["msg"]
        if "missing" in msg:
            message = f"Field '{field_path}' is required."
        elif "empty" in msg or "whitespace" in msg:
            message = f"Field '{field_path}' cannot be empty or whitespace only."
        else:
            message = f"{field_path}: {msg}"
        await logger.error("RequestValidationError", exc)
        error_details.append({"code": ErrorCode.VALIDATION_ERROR.value, "message": message})

    return JSONResponse(
        status_code=400,
        content={
            "code": 400,
            "status": "Error",
            "message": "Request validation failed",
            "request_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "errors": error_details,
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# App factory
# ══════════════════════════════════════════════════════════════════════════════

def create_app() -> FastAPI:
    app = FastAPI(
        title="Intelligent Operations Assistant",
        description="HR/IT policy chatbot — LangGraph + PgVector + Gemini",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],           # restrict to known domains in production
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
        allow_headers=[
            "Accept", "Accept-Language", "Content-Language", "Content-Type",
            "Authorization", "accessToken", "deviceIdentifier",
            "X-RequestID", "X-Requested-With", "Origin", "Cache-Control", "Pragma",
        ],
        expose_headers=["*"],
        max_age=3600,
    )

    app.add_exception_handler(ValidationError, validation_exception_handler)
    app.add_exception_handler(RequestValidationError, request_validation_exception_handler)

    app.include_router(router)

    return app


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

async def run_server() -> None:
    app = create_app()
    port = int(config.port)
    host = config.host

    logger.info(f"Starting server on {host}:{port}")
    logger.info(
        f"Database: {config.db_host}:{config.db_port}/{config.db_name}"
    )

    uv_config = Config(
        app=app,
        host=host,
        port=port,
        reload=False,
        access_log=True,
    )
    server = Server(uv_config)
    await server.serve()


if __name__ == "__main__":
    try:
        if sys.platform == "win32":
            # Windows requires SelectorEventLoop for asyncio + asyncpg compatibility
            import selectors
            asyncio.run(
                run_server(),
                loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector()),
            )
        else:
            asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        # Cannot use await here — log synchronously then exit
        logger.warning(f"Server startup failed: {e}")
        raise SystemExit(1)
