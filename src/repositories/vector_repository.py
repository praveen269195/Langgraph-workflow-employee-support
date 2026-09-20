"""
Vector repository — pure DB I/O for document chunks.

Responsibility:
  - replace_chunks(): DELETE old chunks for a source + INSERT new ones atomically.
  - similarity_search(): cosine-distance query against stored embeddings.
  - fetch_all_chunks(): full corpus for BM25 scoring (content only, no vectors).

No parsing, splitting, or embedding logic lives here.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from sqlalchemy import Column, Index, Integer, String, Text, delete, text
from sqlalchemy.orm import declarative_base

from pgvector.sqlalchemy import Vector

from repositories.Database import Database
from utils.Exceptions.errorcodes import client_error
from utils.logger import Logger

VectorBase = declarative_base()

# Google text-embedding-004 output dimension
EMBEDDING_DIM: int = 768


# ── ORM model ──────────────────────────────────────────────────────────────────
class DocumentChunk(VectorBase):
    """One text chunk with its pgvector embedding, page number, and chunk index."""

    __tablename__ = "document_chunks"
    __allow_unmapped__ = True

    id          = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id    = Column(String(36), unique=True, nullable=False,
                         default=lambda: str(uuid.uuid4()))
    source      = Column(String(500), nullable=False, index=True)
    content     = Column(Text, nullable=False)          # raw chunk text (no prefix)
    page        = Column(Integer, nullable=True)         # 0-based page number from loader
    chunk_index = Column(Integer, nullable=False, server_default=text("0"))  # position within source
    metadata_   = Column("metadata", Text, nullable=True)
    embedding   = Column(Vector(EMBEDDING_DIM), nullable=False)


# HNSW index for fast approximate nearest-neighbour (cosine distance)
Index(
    "ix_document_chunks_embedding_cosine",
    DocumentChunk.embedding,
    postgresql_using="hnsw",
    postgresql_with={"m": 16, "ef_construction": 64},
    postgresql_ops={"embedding": "vector_cosine_ops"},
)


# ── Transfer object (service → repository) ────────────────────────────────────
@dataclass
class ChunkRecord:
    """
    Built by DocumentService, stored by VectorRepository.
    chunk_id uses uuid5 so re-uploading the same file produces identical IDs.
    """
    chunk_id:    str
    source:      str
    content:     str              # raw text (displayed in answers)
    embedding:   list[float]      # vector of contextual text (source + content)
    chunk_index: int = 0
    page:        int | None = field(default=None)
    metadata:    str | None = field(default=None)


# ── Repository ─────────────────────────────────────────────────────────────────
class VectorRepository:
    """Pure DB I/O for the document_chunks table."""

    def __init__(self) -> None:
        self.db     = Database()
        self.logger = Logger("vector_repository")

    # ── Table creation ─────────────────────────────────────────────────────────
    async def create_table(self) -> None:
        """Create table + HNSW index if they do not exist."""
        self.logger.info("create_table started")
        async with self.db.engine.begin() as conn:
            await conn.run_sync(VectorBase.metadata.create_all)
        self.logger.info("create_table ended")

    # ── Atomic replace (dedup fix) ─────────────────────────────────────────────
    async def replace_chunks(self, source: str, records: list[ChunkRecord]) -> None:
        """
        Delete all existing chunks for *source* and insert the new ones in a
        single transaction.  A failed upload therefore keeps the previous version
        intact rather than leaving the table in a half-updated state.

        Parameters
        ----------
        source:  Basename of the document (used as the DELETE predicate).
        records: New chunks built by DocumentService.
        """
        self.logger.info(
            "replace_chunks started",
            source=source,
            incoming_chunks=len(records),
        )
        try:
            async with self.db.get_session() as session:
                # 1. Delete old chunks for this source
                await session.execute(
                    delete(DocumentChunk).where(DocumentChunk.source == source)
                )
                # 2. Insert new chunks
                for record in records:
                    row = DocumentChunk(
                        chunk_id=record.chunk_id,
                        source=record.source,
                        content=record.content,
                        page=record.page,
                        chunk_index=record.chunk_index,
                        embedding=record.embedding,
                        metadata_=record.metadata,
                    )
                    session.add(row)
                # commit handled by Database.get_session on clean exit
            self.logger.info(
                "replace_chunks ended",
                source=source,
                saved=len(records),
            )
        except Exception as e:
            await self.logger.error("replace_chunks failed", e, source=source)
            raise client_error(str(e))

    # ── Read: semantic search ──────────────────────────────────────────────────
    async def similarity_search(
        self,
        query_vector: list[float],
        top_k: int,
    ) -> list[dict]:
        """
        Return the top-k chunks closest to query_vector by cosine distance.

        Returns
        -------
        List of dicts: {id, content, source, page, chunk_index, metadata, score}
        """
        self.logger.info("similarity_search started", top_k=top_k)
        try:
            async with self.db.get_session() as session:
                rows = await session.execute(
                    text(
                        """
                        SELECT id,
                               content,
                               source,
                               page,
                               chunk_index,
                               metadata,
                               1 - (embedding <=> CAST(:vec AS vector)) AS score
                        FROM document_chunks
                        ORDER BY embedding <=> CAST(:vec AS vector)
                        LIMIT :limit
                        """
                    ),
                    {"vec": str(query_vector), "limit": top_k},
                )
                results = [
                    {
                        "id":          row.id,
                        "content":     row.content,
                        "source":      row.source,
                        "page":        row.page,
                        "chunk_index": row.chunk_index,
                        "metadata":    row.metadata,
                        "score":       float(row.score),
                    }
                    for row in rows.fetchall()
                ]
            self.logger.info("similarity_search ended", results=len(results))
            return results
        except Exception as e:
            await self.logger.error("similarity_search failed", e)
            raise client_error(str(e))

    # ── Read: ParadeDB BM25 keyword search ─────────────────────────────────────
    async def keyword_search(self, query: str, top_k: int) -> list[dict]:
        """
        Full-text keyword search using ParadeDB's built-in BM25 scoring.

        Uses the ||| match operator with pdb.score(id) for relevance scoring.
        The query text is always a bound parameter — never interpolated into SQL.

        Parameters
        ----------
        query: Raw search query text from the user.
        top_k: Number of results to return.

        Returns
        -------
        List of dicts: {id, content, source, page, chunk_index, metadata, bm25_score}
        bm25_score is a positive float — higher means more relevant.
        """
        self.logger.info("keyword_search started", top_k=top_k)
        try:
            async with self.db.get_session() as session:
                rows = await session.execute(
                    text(
                        """
                        SELECT id,
                               content,
                               source,
                               page,
                               chunk_index,
                               metadata,
                               pdb.score(id) AS bm25_score
                        FROM document_chunks
                        WHERE content ||| :query
                        ORDER BY bm25_score DESC
                        LIMIT :limit
                        """
                    ),
                    {"query": query, "limit": top_k},
                )
                results = [
                    {
                        "id":          row.id,
                        "content":     row.content,
                        "source":      row.source,
                        "page":        row.page,
                        "chunk_index": row.chunk_index,
                        "metadata":    row.metadata,
                        "bm25_score":  float(row.bm25_score),
                    }
                    for row in rows.fetchall()
                ]
            self.logger.info(
                "keyword_search ended",
                results=len(results),
            )
            return results
        except Exception as e:
            await self.logger.error("keyword_search failed", e)
            raise client_error(str(e))
