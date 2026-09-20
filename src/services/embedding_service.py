"""
Embedding service — Google gemini-embedding-2 client.

Responsibility: produce float vectors from text strings.
Nothing else — no chunking, no DB writes, no retrieval logic.
"""

from __future__ import annotations

import constant
from settings import config
from utils.Exceptions.errorcodes import client_error
from utils.logger import Logger


class GoogleEmbedding:
    """Wraps GoogleGenerativeAIEmbeddings with start/end logging."""

    def __init__(self) -> None:
        self.logger = Logger("embedding_service")
        self._client = None

    # ── Private helpers ────────────────────────────────────────────────────────
    def _get_client(self):
        """Lazily create the embedding client (one instance per process)."""
        self.logger.debug("_get_client started")
        if self._client is None:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            api_key = config.gemini_api_key
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY is not set. Required for embeddings."
                )
            self._client = GoogleGenerativeAIEmbeddings(
                model=constant.EMBEDDING_MODEL,
                google_api_key=api_key,
                output_dimensionality=768,
            )
            self.logger.info(
                "_get_client: embedding client created",
                model=constant.EMBEDDING_MODEL,
            )
        self.logger.debug("_get_client ended")
        return self._client

    # ── Public API ─────────────────────────────────────────────────────────────
    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of document texts.

        Used during ingestion to produce vectors stored in PgVector.

        Parameters
        ----------
        texts: List of text strings to embed.

        Returns
        -------
        List of float vectors, one per input text.
        """
        self.logger.info("embed_documents started", count=len(texts))
        try:
            client = self._get_client()
            vectors: list[list[float]] = await client.aembed_documents(texts)
            self.logger.info("embed_documents ended", vectors_produced=len(vectors))
            return vectors
        except Exception as e:
            await self.logger.error("embed_documents failed", e)
            raise client_error(str(e))

    async def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string.

        Used during retrieval to compare against stored chunk vectors.

        Parameters
        ----------
        query: The search query text.

        Returns
        -------
        A single float vector.
        """
        self.logger.info("embed_query started", query_preview=query[:60])
        try:
            client = self._get_client()
            vector: list[float] = await client.aembed_query(query)
            self.logger.info("embed_query ended", dims=len(vector))
            return vector
        except Exception as e:
            await self.logger.error("embed_query failed", e)
            raise client_error(str(e))


# Module-level singleton — import this everywhere rather than instantiating.
google_embedding = GoogleEmbedding()
