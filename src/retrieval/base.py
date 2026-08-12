from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from src.models.data_models import DocumentChunk, RetrievalResult


class BaseRetriever(ABC):
    """Abstract base class for retrievers."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> Sequence[RetrievalResult]:
        """Retrieve top-k chunks for the provided query."""
        raise NotImplementedError
