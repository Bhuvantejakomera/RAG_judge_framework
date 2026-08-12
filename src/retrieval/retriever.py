from __future__ import annotations

import logging
import re
import time
from typing import Sequence

from src.embeddings.embedder import EmbeddingEngine
from src.models.data_models import DocumentChunk, RetrievalResult
from src.retrieval.base import BaseRetriever
from src.storage.vector_store import VectorStore


class DenseFAISSRetriever(BaseRetriever):
    """Retrieves relevant chunks from a FAISS vector index."""

    def __init__(
        self,
        index_directory: str | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self._logger = logging.getLogger(__name__)
        self._vector_store = vector_store or VectorStore(index_directory)
        self._embedder: EmbeddingEngine | None = None
        self._last_retrieval_time: float | None = None

    def _get_embedder(self) -> EmbeddingEngine:
        if self._embedder is None:
            self._embedder = EmbeddingEngine()
        return self._embedder

    @property
    def last_retrieval_time(self) -> float | None:
        """Return the duration of the most recent retrieval call."""
        return self._last_retrieval_time

    def _normalize_text(self, text: str) -> str:
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return re.sub(r"\s+", " ", cleaned).strip()

    def _matches_query_title(self, query: str, article_title: str) -> bool:
        query_norm = self._normalize_text(query)
        title_norm = self._normalize_text(article_title)
        if not title_norm:
            return False
        if title_norm in query_norm:
            return True
        base_title = re.sub(r"\s*\(.*\)$", "", article_title).strip()
        return base_title and self._normalize_text(base_title) in query_norm

    def retrieve(self, query: str, top_k: int = 5) -> Sequence[RetrievalResult]:
        """Retrieve the most relevant chunks for a given query."""
        start_time = time.perf_counter()
        encode_start = time.perf_counter()
        self._logger.debug("START encode")
        embeddings = self._get_embedder().encode([query])
        self._logger.debug("END encode (%.3fs)", time.perf_counter() - encode_start)

        search_start = time.perf_counter()
        self._logger.debug("START search")
        scores, results = self._vector_store.search(embeddings, top_k)
        self._logger.debug("END search (%.3fs)", time.perf_counter() - search_start)

        if results and results[0]:
            metadata_start = time.perf_counter()
            self._logger.debug("START metadata")
            for item in results[0]:
                metadata = item.get("metadata", {})
                article_title = str(metadata.get("article_title", ""))
                if self._matches_query_title(query, article_title):
                    item["score"] = float(item["score"]) + 0.75
            results[0].sort(key=lambda row: row["score"], reverse=True)
            self._logger.debug("END metadata (%.3fs)", time.perf_counter() - metadata_start)

        duration = time.perf_counter() - start_time
        self._last_retrieval_time = duration
        self._logger.info("Retrieved %d chunks in %.3f seconds.", len(results[0]) if results else 0, duration)

        retrieved_results: list[RetrievalResult] = []
        if results and results[0]:
            for item in results[0]:
                metadata = item.get("metadata", {})
                retrieved_results.append(
                    RetrievalResult(
                        chunk=DocumentChunk(
                            chunk_id=item["id"],
                            article_title=str(metadata.get("article_title", "")),
                            chunk_number=int(metadata.get("chunk_number", 0)),
                            text=str(metadata.get("text", "")),
                            source_url=str(metadata.get("source_url", "")),
                            token_count=int(metadata.get("token_count", 0)),
                        ),
                        score=item["score"],
                    )
                )
        return retrieved_results
