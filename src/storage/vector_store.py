from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Sequence

from src.runtime_bootstrap import configure_native_threading

configure_native_threading()

import faiss
import numpy as np

from src.runtime_bootstrap import configure_faiss_threading

configure_faiss_threading(faiss)

from src.storage.base import BaseVectorStore


class VectorStore(BaseVectorStore):
    """FAISS-backed vector store for embedding index persistence."""

    def __init__(self, index_directory: str | Path | None = None) -> None:
        self._index_directory = Path(index_directory or Path.cwd() / "index")
        self._index_directory.mkdir(parents=True, exist_ok=True)
        self._index_path = self._index_directory / "faiss.index"
        self._metadata_path = self._index_directory / "metadata.pkl"
        self._article_ids_path = self._index_directory / "article_ids.pkl"
        self._article_map_path = self._index_directory / "articles.pkl"
        self._config_path = self._index_directory / "config.json"
        self._index: faiss.IndexFlatIP | None = None
        self._metadata: list[dict[str, Any]] = []
        self._article_ids: list[str] = []
        self._article_to_chunk_ids: dict[str, list[str]] = {}
        self._dimension = 0
        self._logger = logging.getLogger(__name__)

    def exists(self) -> bool:
        return (
            self._index_path.exists()
            and self._metadata_path.exists()
            and self._article_ids_path.exists()
            and self._config_path.exists()
        )

    def clear(self) -> None:
        for path in [
            self._index_path,
            self._metadata_path,
            self._article_ids_path,
            self._index_directory / "article_to_chunk_ids.pkl",
            self._config_path,
        ]:
            if path.exists():
                path.unlink()
        self._index = None
        self._metadata = []
        self._article_ids = []
        self._article_to_chunk_ids = {}
        self._dimension = 0
        self._logger.info("Vector store cleared from %s", self._index_directory)

    def index_size(self) -> int:
        return int(self._index.ntotal) if self._index is not None else 0

    def get_metadata_count(self) -> int:
        return len(self._metadata)

    def get_dimension(self) -> int:
        return self._dimension

    def create_new_index(self, dimension: int) -> None:
        self._dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        self._metadata = []
        self._article_ids = []
        self._article_to_chunk_ids = {}

    def add_embeddings(self, ids: Sequence[str], embeddings: np.ndarray, metadata: Sequence[dict[str, Any]]) -> None:
        if self._index is None:
            raise ValueError("Index has not been initialized.")
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != self._dimension:
            raise ValueError("Embedding dimension mismatch.")
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        self._index.add(embeddings)
        self._article_ids.extend(ids)
        self._metadata.extend(metadata)

        for chunk in metadata:
            article_title = chunk.get("article_title")
            chunk_id = chunk.get("chunk_id") or chunk.get("chunk_number")
            if not article_title or not chunk_id:
                continue
            self._article_to_chunk_ids.setdefault(article_title, []).append(str(chunk_id))

    def save(self) -> None:
        if self._index is None:
            raise ValueError("Cannot save an uninitialized index.")
        faiss.write_index(self._index, str(self._index_path))
        self._metadata_path.write_bytes(pickle.dumps(self._metadata))
        self._article_ids_path.write_bytes(pickle.dumps(self._article_ids))
        (self._index_directory / "article_to_chunk_ids.pkl").write_bytes(pickle.dumps(self._article_to_chunk_ids))
        self._config_path.write_text(json.dumps({"dimension": self._dimension}, indent=2), encoding="utf-8")
        self._logger.info("Vector store saved to %s", self._index_directory)

    def load(self) -> None:
        if not self.exists():
            raise FileNotFoundError("Vector store files are missing.")
        self._index = faiss.read_index(str(self._index_path))
        self._metadata = pickle.loads(self._metadata_path.read_bytes())
        self._article_ids = pickle.loads(self._article_ids_path.read_bytes())
        article_map_path = self._index_directory / "article_to_chunk_ids.pkl"
        if article_map_path.exists():
            self._article_to_chunk_ids = pickle.loads(article_map_path.read_bytes())
        else:
            self._article_to_chunk_ids = {}
        config = json.loads(self._config_path.read_text(encoding="utf-8"))
        self._dimension = int(config["dimension"])
        self._logger.info("Vector store loaded from %s", self._index_directory)

    def search(self, embeddings: np.ndarray, top_k: int) -> tuple[np.ndarray, list[list[dict[str, Any]]]]:
        if self._index is None:
            if self.exists():
                self._logger.info("Vector store exists on disk; loading index into memory for search.")
                self.load()
            else:
                raise ValueError("FAISS index files missing on disk. Run --rebuild-index first.")
        if self._index is None:
            raise ValueError("Index has not been loaded or created.")
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        scores, indices = self._index.search(embeddings, top_k)
        results: list[list[dict[str, Any]]] = []

        for row_id, row_indices in enumerate(indices):
            row_results: list[dict[str, Any]] = []
            for column_id, index_position in enumerate(row_indices):
                if index_position < 0 or index_position >= len(self._metadata):
                    continue
                row_results.append(
                    {
                        "id": self._article_ids[index_position],
                        "metadata": self._metadata[index_position],
                        "score": float(scores[row_id, column_id]),
                    }
                )
            results.append(row_results)

        return scores, results

    def get_article_chunk_map(self) -> dict[str, list[str]]:
        return self._article_to_chunk_ids.copy()
