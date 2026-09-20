"""
API routes.

All endpoints use the same APIResponse envelope from the previous project.

Routes:
  POST /api/v1/chat               — main conversation endpoint
  POST /api/v1/documents/upload   — knowledge-base document ingestion
  GET  /api/v1/health             — service health check
"""

from __future__ import annotations

import os
import uuid
from typing import TYPE_CHECKING

from fastapi import APIRouter, Request, Response, UploadFile

from models.model import APIResponse, ChatRequest
from utils.Exceptions.errorcodes import ApplicationError, client_error, validation_error
from utils.logger import Logger
from utils.response_utils import ResponseFormatter

if TYPE_CHECKING:
    from services.agent_service import AgentService
    from services.document_service import DocumentService
    from services.retrieval_service import RetrievalService
    from repositories.Database import Database

# Maximum accepted upload size (10 MB)
MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024


class OpsRouter:
    """Registers all API routes on a single APIRouter."""

    def __init__(self) -> None:
        self.router = APIRouter(prefix="/api/v1")
        self.logger = Logger("routes")
        self.response_formatter = ResponseFormatter()

        self.router.add_api_route(
            "/chat",
            self.chat_route,
            methods=["POST"],
            response_model=APIResponse,
            summary="Send a message to the assistant",
        )
        self.router.add_api_route(
            "/documents/upload",
            self.upload_document_route,
            methods=["POST"],
            response_model=APIResponse,
            summary="Upload a policy document into the knowledge base",
        )
        self.router.add_api_route(
            "/health",
            self.health_route,
            methods=["GET"],
            response_model=APIResponse,
            summary="Service health check",
        )

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _get_agent_service(self, request: Request) -> "AgentService":
        return request.app.state.agent_service

    def _get_document_service(self, request: Request) -> "DocumentService":
        return request.app.state.document_service

    def _get_retrieval_service(self, request: Request) -> "RetrievalService":
        return request.app.state.retrieval_service

    def _get_db(self, request: Request) -> "Database":
        return request.app.state.db

    # ── POST /api/v1/chat ──────────────────────────────────────────────────────
    async def chat_route(
        self,
        request: Request,
        response: Response,
    ) -> APIResponse:
        request_id = str(uuid.uuid4())
        try:
            self.logger.info("chat_route started", request_id=request_id)

            raw_data = await request.json()
            chat_request = ChatRequest(**raw_data)

            agent_service = self._get_agent_service(request)
            result = await agent_service.chat(chat_request)

            response_message = self.response_formatter.format_success_response(
                code=200,
                message="LLM response",
                data=result.model_dump(),
                request_id=request_id,
            )
            response.status_code = response_message.code
            self.logger.info(
                "chat_route ended",
                request_id=request_id,
                session_id=result.session_id,
                escalated=result.escalated,
                awaiting_confirmation=result.awaiting_confirmation,
            )
            return response_message

        except ApplicationError as e:
            await self.logger.error("chat_route ApplicationError", e, request_id=request_id)
            response_message = self.response_formatter.format_error_response(e, request_id)
            response.status_code = response_message.code
            return response_message
        except Exception as e:
            await self.logger.error("chat_route unexpected error", e, request_id=request_id)
            err = client_error("An unexpected error occurred. Please try again.")
            response_message = self.response_formatter.format_error_response(err, request_id)
            response.status_code = response_message.code
            return response_message

    # ── POST /api/v1/documents/upload ──────────────────────────────────────────
    async def upload_document_route(
        self,
        request: Request,
        response: Response,
    ) -> APIResponse:
        request_id = str(uuid.uuid4())
        try:
            # ── Extract file from multipart form ───────────────────────────────
            # add_api_route() on a bound method does not support FastAPI's
            # File(...) / UploadFile DI. Pull the file directly from the form data.
            form = await request.form()
            file: UploadFile | None = form.get("file")  # type: ignore[assignment]

            if file is None or not hasattr(file, "read"):
                raise validation_error("file", "A file must be provided in the 'file' form field.")

            self.logger.info(
                "upload_document_route started",
                request_id=request_id,
                filename=file.filename,
            )

            # ── Extension guard ────────────────────────────────────────────────
            # Validate before reading the full body — gives a clean 400 immediately
            # instead of falling through to a generic 500 from DocumentService.
            _SUPPORTED = {"pdf", "docx", "txt"}
            _, _raw_ext = os.path.splitext(file.filename or "")
            _ext = _raw_ext.lstrip(".").lower()
            if not _ext or _ext not in _SUPPORTED:
                raise validation_error(
                    "file",
                    f"Unsupported file type '.{_ext}'. Supported: {', '.join(sorted(_SUPPORTED))}",
                )

            # ── Size + empty guard ─────────────────────────────────────────────
            # Read one byte beyond the limit so we can detect oversized files
            # without loading the whole file into memory first.
            content = await file.read(MAX_UPLOAD_BYTES + 1)

            if not content:
                raise validation_error("file", "Uploaded file is empty.")

            if len(content) > MAX_UPLOAD_BYTES:
                raise validation_error(
                    "file",
                    f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
                )

            document_service = self._get_document_service(request)
            chunks_indexed = await document_service.ingest_document(
                file.filename, content
            )

            response_message = self.response_formatter.format_success_response(
                code=200,
                message="Document ingested",
                data={
                    "filename":       file.filename,
                    "chunks_indexed": chunks_indexed,
                    "status":         "success",
                },
                request_id=request_id,
            )
            response.status_code = response_message.code
            self.logger.info(
                "upload_document_route ended",
                request_id=request_id,
                filename=file.filename,
                chunks_indexed=chunks_indexed,
            )
            return response_message

        except ApplicationError as e:
            await self.logger.error(
                "upload_document_route ApplicationError", e, request_id=request_id
            )
            response_message = self.response_formatter.format_error_response(e, request_id)
            response.status_code = response_message.code
            return response_message
        except Exception as e:
            await self.logger.error(
                "upload_document_route unexpected error", e, request_id=request_id
            )
            # Return a generic message — never expose str(e) to the client
            err = client_error("Document ingestion failed. Please try again.")
            response_message = self.response_formatter.format_error_response(err, request_id)
            response.status_code = response_message.code
            return response_message

    # ── GET /api/v1/health ─────────────────────────────────────────────────────
    async def health_route(
        self,
        request: Request,
        response: Response,
    ) -> APIResponse:
        request_id = str(uuid.uuid4())
        try:
            self.logger.info("health_route started", request_id=request_id)

            db: "Database" = self._get_db(request)
            db_ok = await db.test_connection()

            vector_ok = False
            try:
                retrieval_service = self._get_retrieval_service(request)
                await retrieval_service.hybrid_search("health check", top_k=1)
                vector_ok = True
            except Exception:
                vector_ok = False

            status_code = 200 if (db_ok and vector_ok) else 500
            response_message = self.response_formatter.format_success_response(
                code=status_code,
                message="Service is healthy" if db_ok else "Service is degraded",
                data={
                    "db":           "ok" if db_ok else "error",
                    "vector_store": "ok" if vector_ok else "error",
                },
                request_id=request_id,
            )
            response.status_code = status_code
            self.logger.info(
                "health_route ended",
                request_id=request_id,
                db=db_ok,
                vector_store=vector_ok,
            )
            return response_message

        except Exception as e:
            await self.logger.error(
                "health_route unexpected error", e, request_id=request_id
            )
            err = client_error("Health check failed.")
            response_message = self.response_formatter.format_error_response(err, request_id)
            response.status_code = response_message.code
            return response_message


ops_controller = OpsRouter()
router = ops_controller.router
