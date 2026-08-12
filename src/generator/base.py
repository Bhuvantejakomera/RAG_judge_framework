from __future__ import annotations

from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    """Abstract base class for answer generators."""

    @abstractmethod
    def generate(self, query: str, context: str) -> str:
        """Generate an answer from a query and retrieved context."""
        raise NotImplementedError
