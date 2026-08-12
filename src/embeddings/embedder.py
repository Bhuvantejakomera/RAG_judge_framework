from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np

from src.config.settings import ConfigManager
from src.embeddings.base import BaseEmbeddingEngine
from src.runtime_bootstrap import configure_native_threading, configure_torch_threading


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str):
    """Load and cache SentenceTransformer models one time per model name."""
    configure_native_threading()
    configure_torch_threading()
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class EmbeddingEngine(BaseEmbeddingEngine):
    """Embedding engine that produces normalized float32 vectors."""

    def __init__(self, model_name: str | None = None) -> None:
        self._settings = ConfigManager().settings
        self._model_name = model_name or self._settings.embedding.model
        self._model = _load_sentence_transformer(self._model_name)
        self._tokenizer = getattr(self._model, "tokenizer", None)
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, use_fast=True)
        self.max_sequence_length = int(getattr(self._model, "max_seq_length", 512) or 512)

    def _prepare_texts(self, texts: Sequence[str]) -> list[str]:
        """Trim texts to the model context window before encoding."""
        prepared: list[str] = []
        for text in texts:
            token_ids = self._tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) > self.max_sequence_length:
                token_ids = token_ids[: self.max_sequence_length]
                text = self._tokenizer.decode(token_ids, skip_special_tokens=True)
            prepared.append(text)
        return prepared

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts into normalized float32 embeddings."""
        prepared_texts = self._prepare_texts(texts)
        embeddings = self._model.encode(
            prepared_texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return np.ascontiguousarray(embeddings, dtype=np.float32)
