"""
Document service — parsing, chunking, embedding, and ingestion orchestration.

Responsibility (service layer only — no DB calls):
  - Validate file extension and binary signature.
  - Parse bytes → raw page texts (off the event loop via asyncio.to_thread).
  - Split into overlapping chunks; filter whitespace-only results.
  - Embed in batches of EMBED_BATCH_SIZE (Google API cap ~100).
  - Build ChunkRecords with deterministic IDs, page numbers, chunk index.
  - Hand records to VectorRepository.replace_chunks() (atomic dedup).

All DB work is done by VectorRepository.
All vector work is done by GoogleEmbedding.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
import uuid

import mlflow
from langchain_text_splitters import RecursiveCharacterTextSplitter

from repositories.vector_repository import ChunkRecord, VectorRepository
from services.embedding_service import GoogleEmbedding
from utils.Exceptions.errorcodes import client_error, validation_error
from utils.logger import Logger

# ── Constants ──────────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({"pdf", "docx", "txt"})
CHUNK_SIZE:      int = 800
CHUNK_OVERLAP:   int = 120
EMBED_BATCH_SIZE: int = 50   # Google embedding API batch cap

# Binary signatures — catches renamed files before parsing
_SIGNATURES: dict[str, bytes] = {
    "pdf":  b"%PDF-",
    "docx": b"PK\x03\x04",
}


class DocumentService:
    """
    Orchestrates the full document ingestion pipeline.

    Flow
    ----
    ingest_document(filename, bytes)
        ├─ _validate_extension()       → ext
        ├─ _check_signature()          → guard
        ├─ _parse_document()           → raw page texts  (asyncio.to_thread)
        ├─ _split_text()               → chunks (whitespace filtered)
        ├─ embedding_service (batched) → vectors (contextual prefix embedded)
        ├─ build ChunkRecords          → deterministic uuid5 IDs + page numbers
        └─ vector_repo.replace_chunks()→ atomic DELETE + INSERT
    """

    def __init__(
        self,
        embedding_service: GoogleEmbedding,
        vector_repo: VectorRepository,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_repo       = vector_repo
        self.logger            = Logger("document_service")
        self._splitter         = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            # Remove ". " to avoid chunks starting with a stray dot
            separators=["\n\n", "\n", " ", ""],
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════════════════════

    async def ingest_document(self, filename: str, content: bytes) -> int:
        """
        Full ingestion pipeline for one document.

        Uses mlflow.start_span so only filename + byte-count are recorded —
        the raw file bytes never enter the MLflow tracking store.

        Parameters
        ----------
        filename: Original filename including extension (basename is used as source).
        content:  Raw file bytes (already size-checked by the route).

        Returns
        -------
        Number of chunks indexed.
        """
        # Use basename so a client-supplied path never becomes the source label
        source = os.path.basename(filename or "")

        self.logger.info(
            "ingest_document started",
            filename=source,
            size_bytes=len(content),
        )

        with mlflow.start_span(name="document_ingestion") as span:
            # Record only metadata — never the file content bytes
            span.set_inputs({"filename": source, "size_bytes": len(content)})

            try:
                ext = self._validate_extension(source)
                self._check_signature(ext, content)

                # Content hash — used in chunk_id to prevent different files with
                # the same filename from silently overwriting each other's chunks.
                content_hash = hashlib.sha256(content).hexdigest()[:16]

                # ── Parse (off the event loop) ─────────────────────────────────
                raw_pages = await self._parse_document(source, content, ext)

                # ── Split + filter ─────────────────────────────────────────────
                chunks_with_pages = self._split_text(raw_pages)
                # Filter out whitespace-only chunks
                chunks_with_pages = [
                    (text, page) for text, page in chunks_with_pages
                    if text.strip()
                ]
                if not chunks_with_pages:
                    raise client_error(
                        f"No text could be extracted from '{source}'. "
                        "Scanned (image-only) files need OCR preprocessing."
                    )

                chunk_texts = [text for text, _ in chunks_with_pages]
                chunk_pages = [page for _, page in chunks_with_pages]

                self.logger.debug(
                    "ingest_document: split complete",
                    filename=source,
                    chunks=len(chunk_texts),
                )

                # ── Embed in batches ───────────────────────────────────────────
                # Embed contextual prefix (source + text) for better retrieval,
                # but store raw text as content so answers stay clean.
                contextual_texts = [
                    f"{source}\n{text}" for text in chunk_texts
                ]
                embeddings: list[list[float]] = []
                for i in range(0, len(contextual_texts), EMBED_BATCH_SIZE):
                    batch = contextual_texts[i: i + EMBED_BATCH_SIZE]
                    self.logger.debug(
                        "ingest_document: embedding batch",
                        filename=source,
                        batch_start=i,
                        batch_size=len(batch),
                    )
                    embeddings.extend(
                        await self.embedding_service.embed_documents(batch)
                    )

                # Guard against API returning a different count
                if len(embeddings) != len(chunk_texts):
                    raise RuntimeError(
                        f"Embedding count mismatch: got {len(embeddings)} vectors "
                        f"for {len(chunk_texts)} chunks in '{source}'"
                    )

                # ── Build ChunkRecords ─────────────────────────────────────────
                # Deterministic chunk_id via uuid5 — includes a content hash so
                # two different files sharing the same filename get distinct IDs
                # and replace_chunks() won't silently delete the other file's data.
                records: list[ChunkRecord] = [
                    ChunkRecord(
                        chunk_id=str(
                            uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{content_hash}:{i}")
                        ),
                        source=source,
                        content=chunk_texts[i],   # raw text, not contextual prefix
                        embedding=embeddings[i],
                        chunk_index=i,
                        page=chunk_pages[i],
                    )
                    for i in range(len(chunk_texts))
                ]

                # ── Atomic replace in DB ───────────────────────────────────────
                await self.vector_repo.replace_chunks(source, records)

                chunk_count = len(records)
                span.set_outputs({"chunks_indexed": chunk_count})
                # update_current_trace is safe because we are inside start_span
                mlflow.update_current_trace(
                    tags={
                        "ingestion.filename":    source,
                        "ingestion.chunk_count": str(chunk_count),
                    }
                )

                self.logger.info(
                    "ingest_document ended",
                    filename=source,
                    chunks_indexed=chunk_count,
                )
                return chunk_count

            except Exception as e:
                await self.logger.error("ingest_document failed", e, filename=source)
                raise

    # ══════════════════════════════════════════════════════════════════════════
    # Private helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _validate_extension(self, filename: str) -> str:
        """
        Return the lowercase extension or raise a 400 validation error.
        Uses os.path.splitext so filenames with no dot are rejected properly.
        """
        self.logger.debug("_validate_extension started", filename=filename)
        _, raw_ext = os.path.splitext(filename or "")
        ext = raw_ext.lstrip(".").lower()
        if not ext or ext not in SUPPORTED_EXTENSIONS:
            raise validation_error(
                "file",
                f"Unsupported or missing file extension '.{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
            )
        self.logger.debug("_validate_extension ended", ext=ext)
        return ext

    def _check_signature(self, ext: str, content: bytes) -> None:
        """
        Verify the file's binary signature matches the declared extension.
        Catches renamed files (e.g. a .exe renamed to .pdf) before parsing.
        """
        self.logger.debug("_check_signature started", ext=ext)
        sig = _SIGNATURES.get(ext)
        if sig and not content.startswith(sig):
            raise validation_error(
                "file",
                f"File content does not match the declared .{ext} format. "
                "The file may be corrupt or have the wrong extension.",
            )
        self.logger.debug("_check_signature ended")

    def _split_text(self, raw_pages: list[tuple[str, int | None]]) -> list[tuple[str, int | None]]:
        """
        Split each (text, page) pair into overlapping chunks.
        Returns list of (chunk_text, page_number).
        """
        self.logger.debug("_split_text started", pages=len(raw_pages))
        result: list[tuple[str, int | None]] = []
        for text, page in raw_pages:
            for chunk in self._splitter.split_text(text):
                result.append((chunk, page))
        self.logger.debug("_split_text ended", chunks=len(result))
        return result

    async def _parse_document(
        self,
        filename: str,
        content: bytes,
        ext: str,
    ) -> list[tuple[str, int | None]]:
        """
        Dispatch to the correct parser and return list of (text, page) tuples.
        """
        self.logger.info("_parse_document started", filename=filename, ext=ext)
        if ext == "pdf":
            pages = await self._parse_pdf(content)
        elif ext == "docx":
            pages = await self._parse_docx(content)
        else:  # txt
            pages = await self._parse_txt(content)
        self.logger.info(
            "_parse_document ended",
            filename=filename,
            sections=len(pages),
        )
        return pages

    async def _parse_pdf(self, content: bytes) -> list[tuple[str, int | None]]:
        """
        Parse PDF bytes → list of (page_text, page_number).
        Runs PyPDFLoader.load() off the event loop via asyncio.to_thread.
        Raises validation_error (400) for corrupt / password-protected files.
        """
        self.logger.debug("_parse_pdf started")
        tmp_path = ""
        try:
            from langchain_community.document_loaders import PyPDFLoader

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            loader = PyPDFLoader(tmp_path)
            try:
                # PyPDFLoader.load() is synchronous — move off the event loop
                docs = await asyncio.to_thread(loader.load)
            except Exception as parse_err:
                raise validation_error(
                    "file",
                    "Could not read the PDF. "
                    "It may be corrupt, password-protected, or contain no text layer.",
                ) from parse_err

            pages = [
                (doc.page_content, doc.metadata.get("page"))
                for doc in docs
                if doc.page_content.strip()
            ]
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass  # Windows: file handle not yet released — temp file will be cleaned up on process exit

        self.logger.debug("_parse_pdf ended", pages=len(pages))
        return pages

    async def _parse_docx(self, content: bytes) -> list[tuple[str, int | None]]:
        """
        Parse DOCX bytes → list of (text, None).
        Runs Docx2txtLoader.load() off the event loop via asyncio.to_thread.
        Raises validation_error (400) for corrupt files.
        """
        self.logger.debug("_parse_docx started")
        tmp_path = ""
        try:
            from langchain_community.document_loaders import Docx2txtLoader

            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            loader = Docx2txtLoader(tmp_path)
            try:
                docs = await asyncio.to_thread(loader.load)
            except Exception as parse_err:
                raise validation_error(
                    "file",
                    "Could not read the DOCX file. It may be corrupt.",
                ) from parse_err

            pages = [
                (doc.page_content, None)
                for doc in docs
                if doc.page_content.strip()
            ]
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass  # Windows: file handle not yet released — temp file will be cleaned up on process exit

        self.logger.debug("_parse_docx ended", sections=len(pages))
        return pages

    async def _parse_txt(self, content: bytes) -> list[tuple[str, int | None]]:
        """
        Decode TXT bytes using utf-8-sig (strips BOM if present).
        Returns a single (text, None) tuple.
        """
        self.logger.debug("_parse_txt started")
        text = content.decode("utf-8-sig", errors="replace")
        self.logger.debug("_parse_txt ended", chars=len(text))
        return [(text, None)]
