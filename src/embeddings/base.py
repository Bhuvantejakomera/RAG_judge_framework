from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class BaseEmbeddingEngine(ABC):
    """Abstract base class for embedding engines."""

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a sequence of texts into normalized embeddings."""
        raise NotImplementedError
