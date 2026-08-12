from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Iterable

from src.runtime_bootstrap import configure_native_threading

configure_native_threading()

import numpy as np
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeRemainingColumn

from src.embeddings.embedder import EmbeddingEngine
from src.indexing.chunker import TokenChunker
from src.models.data_models import DocumentChunk
from src.storage.vector_store import VectorStore
from src.wikipedia.wiki_fetcher import WikipediaFetcher
from src.config.settings import ConfigManager


class IndexBuilder:
    """Builds or loads a persistent embedding index from article text."""

    def __init__(
        self,
        index_directory: str | Path | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self._logger = logging.getLogger(__name__)
        self._config = ConfigManager()
        self._vector_store = vector_store or VectorStore(index_directory)
        self._chunker: TokenChunker | None = None
        self._embedder: EmbeddingEngine | None = None

    def _get_chunker(self) -> TokenChunker:
        if self._chunker is None:
            self._chunker = TokenChunker(
                tokenizer_name=self._config.settings.embedding.model,
                chunk_size=self._config.settings.chunking.size,
                overlap=self._config.settings.chunking.overlap,
            )
        return self._chunker

    def _get_embedder(self) -> EmbeddingEngine:
        if self._embedder is None:
            self._embedder = EmbeddingEngine(model_name=self._config.settings.embedding.model)
        return self._embedder

    def _load_documents(self, article_texts: dict[str, str]) -> list[DocumentChunk]:
        chunker = self._get_chunker()

        def chunk_document(title: str, text: str) -> list[DocumentChunk]:
            source_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            return chunker.chunk_text(article_title=title, text=text, source_url=source_url)

        chunks: list[DocumentChunk] = []
        if not article_texts:
            return chunks

        max_workers = self._config.settings.indexing.max_workers
        with Progress(
            SpinnerColumn(),
            "Chunking Documents...",
            TaskProgressColumn(),
            BarColumn(),
            TimeRemainingColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("chunk", total=len(article_texts))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(chunk_document, title, text): title for title, text in article_texts.items()}
                for future in as_completed(futures):
                    try:
                        chunks.extend(future.result())
                    except Exception as exc:
                        self._logger.exception("Failed to chunk document: %s", exc)
                    progress.advance(task)
        return chunks

    def _create_embeddings(self, chunks: Iterable[DocumentChunk]) -> tuple[list[str], list[dict[str, str]], list[str], np.ndarray]:
        texts = [chunk.text for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        metadata = [
            {
                "article_title": chunk.article_title,
                "chunk_id": chunk.chunk_id,
                "chunk_number": chunk.chunk_number,
                "source_url": chunk.source_url,
                "token_count": chunk.token_count,
                "text": chunk.text,
            }
            for chunk in chunks
        ]

        batch_size = 32
        embedding_batches: list[np.ndarray] = []
        embedder = self._get_embedder()
        with Progress(
            SpinnerColumn(),
            "Computing BGE Embeddings...",
            TaskProgressColumn(),
            BarColumn(),
            TimeRemainingColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("embeddings", total=len(texts))
            for offset in range(0, len(texts), batch_size):
                batch_texts = texts[offset : offset + batch_size]
                batch_embeddings = embedder.encode(batch_texts)
                embedding_batches.append(batch_embeddings)
                progress.advance(task, len(batch_texts))
        embeddings = np.vstack(embedding_batches)
        return ids, metadata, texts, embeddings

    def build_index(self, article_texts: dict[str, str], force_rebuild: bool = False) -> None:
        """Build or load the vector index for the supplied article texts."""
        try:
            if self._vector_store.exists() and not force_rebuild:
                self._logger.info("Index exists, loading existing index.")
                self._vector_store.load()
                return

            chunks = self._load_documents(article_texts)
            if not chunks:
                self._logger.warning("No document chunks available to index.")
                return

            ids, metadata, _, embeddings = self._create_embeddings(chunks)
            self._vector_store.create_new_index(embeddings.shape[1])
            self._vector_store.add_embeddings(ids, embeddings, metadata)
            self._vector_store.save()
            self._vector_store.load()
        except Exception as exc:
            self._logger.exception("Failed to build or save index: %s", exc)
            raise

    def build_or_load(self, dataset_tsv: str | Path = Path("data/test.tsv"), force_rebuild: bool = False) -> None:
        """Load existing index or build it automatically from dataset articles."""
        if self._vector_store.exists() and not force_rebuild:
            self._logger.info("Existing index detected; loading index.")
            self._vector_store.load()
            return
        if force_rebuild and self._vector_store.exists():
            self._logger.info("Force rebuild requested; clearing existing index files.")
            self._vector_store.clear()

        dataset_path = Path(dataset_tsv)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        fetcher = WikipediaFetcher(config=self._config)
        all_titles = fetcher.get_unique_titles(dataset_path)
        total_titles = len(all_titles)
        max_articles = self._config.settings.indexing.max_articles
        if max_articles is None or max_articles == 0:
            titles = all_titles
        else:
            titles = all_titles[:max_articles]

        self._logger.info(
            "Starting index build: total unique titles=%d, max_articles=%s, titles_to_process=%d",
            total_titles,
            max_articles,
            len(titles),
        )

        downloaded = 0
        failed = 0
        article_texts: dict[str, str] = {}
        with Progress(
            SpinnerColumn(),
            TextColumn("Downloading Wikipedia Articles..."),
            TaskProgressColumn(),
            BarColumn(),
            TimeRemainingColumn(),
            transient=True,
        ) as progress:
            progress_task = progress.add_task("download", total=len(titles))
            with ThreadPoolExecutor(max_workers=self._config.settings.indexing.max_workers) as executor:
                futures = {executor.submit(fetcher.fetch_title, title): title for title in titles}
                for future in as_completed(futures):
                    title = futures[future]
                    try:
                        content = future.result()
                        if content is not None:
                            article_texts[title] = content
                            downloaded += 1
                        else:
                            failed += 1
                    except Exception as exc:
                        self._logger.exception("Failed to fetch Wikipedia article %s: %s", title, exc)
                        failed += 1
                    finally:
                        progress.advance(progress_task)

        self._logger.info(
            "Wikipedia download complete: total_titles=%d, successful=%d, failed=%d",
            total_titles,
            downloaded,
            failed,
        )

        if not article_texts:
            raise RuntimeError("No article texts were fetched for index building.")

        start_time = perf_counter()
        self.build_index(article_texts, force_rebuild=force_rebuild)
        elapsed = perf_counter() - start_time
        if self._vector_store._index is None and self._vector_store.exists():
            self._vector_store.load()
        self._logger.info(
            "Index build finished in %.2f seconds: articles_indexed=%d, chunks=%d, vectors=%d",
            elapsed,
            len(article_texts),
            self._vector_store.get_metadata_count(),
            self._vector_store.index_size(),
        )

    def search(self, query: str, top_k: int) -> tuple[list[dict[str, object]], list[float]]:
        """Search the index for the nearest neighbors to the query text."""
        embedder = self._get_embedder()
        embeddings = embedder.encode([query])
        scores, results = self._vector_store.search(embeddings, top_k)
        row_results = results[0] if results else []
        return row_results, [item["score"] for item in row_results]
