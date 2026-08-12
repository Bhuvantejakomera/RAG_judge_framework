from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class BaseVectorStore(ABC):
    """Abstract base interface for vector stores."""

    @abstractmethod
    def exists(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def create_new_index(self, dimension: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_embeddings(self, ids: Sequence[str], embeddings: np.ndarray, metadata: Sequence[dict[str, Any]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, embeddings: np.ndarray, top_k: int) -> tuple[np.ndarray, list[list[dict[str, Any]]]]:
        raise NotImplementedError
