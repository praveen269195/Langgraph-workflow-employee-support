"""
Retrieval service — hybrid semantic + ParadeDB BM25 search via RRF.

Responsibility (service layer):
  - Embed the query via GoogleEmbedding.
  - Run cosine similarity search via VectorRepository.similarity_search().
  - Run ParadeDB BM25 keyword search via VectorRepository.keyword_search().
  - Fuse both ranked lists via Reciprocal Rank Fusion (RRF).
  - Return ordered, scored result dicts to the caller.

No in-process BM25 library is used — scoring is done entirely inside
ParadeDB using the ||| operator and pdb.score(id).
"""

from __future__ import annotations

import constant
from repositories.vector_repository import VectorRepository
from services.embedding_service import GoogleEmbedding
from utils.Exceptions.errorcodes import client_error
from utils.logger import Logger


class RetrievalService:
    """
    Hybrid retrieval: semantic cosine similarity + ParadeDB BM25, fused via RRF.

    Parameters
    ----------
    embedding_service: GoogleEmbedding instance for query embedding.
    vector_repo:       VectorRepository instance for DB queries.
    """

    def __init__(
        self,
        embedding_service: GoogleEmbedding,
        vector_repo: VectorRepository,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_repo       = vector_repo
        self.logger            = Logger("retrieval_service")

    async def hybrid_search(
        self,
        query: str,
        top_k: int = constant.RETRIEVAL_TOP_K,
        rrf_k: int = 60,
    ) -> list[dict]:
        """
        Hybrid search: semantic (PgVector cosine) + BM25 (ParadeDB), fused via RRF.

        Steps
        -----
        1. Embed query via GoogleEmbedding.embed_query().
        2. Semantic top-(top_k * 2) via cosine distance.
        3. ParadeDB BM25 top-(top_k * 2) via ||| operator + pdb.score(id).
        4. RRF fusion of both ranked lists.
        5. Return top_k results ordered by fused score (desc).

        Returns
        -------
        List of dicts: {text, source, page, chunk_index, metadata, score}
        """
        self.logger.info(
            "hybrid_search started",
            query_preview=query[:80],
            top_k=top_k,
        )

        try:
            # ── Step 1: embed query ────────────────────────────────────────────
            self.logger.debug("hybrid_search: embedding query")
            query_vector = await self.embedding_service.embed_query(query)

            # ── Step 2: semantic search ────────────────────────────────────────
            self.logger.debug("hybrid_search: running semantic search")
            semantic_results = await self.vector_repo.similarity_search(
                query_vector=query_vector,
                top_k=top_k * 2,
            )
            if not semantic_results:
                self.logger.warning("hybrid_search: semantic search returned no results — continuing with BM25 only")

            # ── Step 3: ParadeDB BM25 keyword search ───────────────────────────
            self.logger.debug("hybrid_search: running ParadeDB BM25 keyword search")
            keyword_results = await self.vector_repo.keyword_search(
                query=query,
                top_k=top_k * 2,
            )
            # bm25_score is positive and higher = better — no filtering needed
            # but guard against zero scores (non-matching rows on small tables)
            keyword_results = [r for r in keyword_results if r["bm25_score"] > 0]

            self.logger.debug(
                "hybrid_search: keyword results",
                keyword_hits=len(keyword_results),
            )

            # If both searches returned nothing, there is nothing to fuse
            if not semantic_results and not keyword_results:
                self.logger.warning("hybrid_search: both semantic and keyword searches returned no results")
                return []

            # ── Step 4: build rank maps ────────────────────────────────────────
            # id → semantic rank (1-based, lower is better)
            semantic_rank: dict[int, int] = {
                row["id"]: rank
                for rank, row in enumerate(semantic_results, start=1)
            }

            # id → BM25 rank (1-based, lower is better)
            bm25_rank: dict[int, int] = {
                row["id"]: rank
                for rank, row in enumerate(keyword_results, start=1)
            }

            # ── Step 5: Reciprocal Rank Fusion ─────────────────────────────────
            self.logger.debug("hybrid_search: fusing ranks via RRF")
            all_ids  = set(semantic_rank) | set(bm25_rank)
            fallback = top_k * 4
            rrf_scores: dict[int, float] = {
                cid: (
                    1.0 / (rrf_k + semantic_rank.get(cid, fallback))
                    + 1.0 / (rrf_k + bm25_rank.get(cid, fallback))
                )
                for cid in all_ids
            }

            top_ids = sorted(
                rrf_scores, key=lambda x: rrf_scores[x], reverse=True
            )[:top_k]

            # ── Build result list ──────────────────────────────────────────────
            # Merge both result sets so we can look up any id
            id_to_chunk: dict[int, dict] = {
                row["id"]: row for row in keyword_results
            }
            id_to_chunk.update({row["id"]: row for row in semantic_results})

            results: list[dict] = []
            for cid in top_ids:
                chunk = id_to_chunk.get(cid)
                if chunk is None:
                    continue
                results.append(
                    {
                        "text":        chunk["content"],
                        "source":      chunk["source"],
                        "page":        chunk.get("page"),
                        "chunk_index": chunk.get("chunk_index"),
                        "metadata":    chunk.get("metadata"),
                        "score":       round(rrf_scores[cid], 6),
                    }
                )

            self.logger.info(
                "hybrid_search ended",
                results=len(results),
                top_score=results[0]["score"] if results else 0.0,
            )
            return results

        except Exception as e:
            await self.logger.error("hybrid_search failed", e)
            raise client_error(str(e))

    # ── Semantic-only fallback ─────────────────────────────────────────────────
    def _format_semantic_only(self, rows: list[dict]) -> list[dict]:
        """Format semantic-only rows into the standard result shape."""
        self.logger.debug("_format_semantic_only started", count=len(rows))
        results = [
            {
                "text":        row["content"],
                "source":      row["source"],
                "page":        row.get("page"),
                "chunk_index": row.get("chunk_index"),
                "metadata":    row.get("metadata"),
                "score":       row["score"],
            }
            for row in rows
        ]
        self.logger.debug("_format_semantic_only ended")
        return results
