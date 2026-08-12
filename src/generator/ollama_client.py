from __future__ import annotations

import logging
import time

import ollama
from src.config.settings import ConfigManager
from src.generator.base import BaseGenerator


class LocalOllamaGenerator(BaseGenerator):
    """Generates answers using a local Ollama model."""

    SYSTEM_PROMPT = (
        "Answer ONLY using the retrieved context. "
        "If context is insufficient reply 'I don't know based on the retrieved context.'"
    )
    MAX_CONTEXT_CHARS = 12000

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        self._settings = ConfigManager().settings
        self._model = self._settings.llm.local_model

    def _truncate_context(self, context: str) -> str:
        """Keep prompts bounded so oversized retrieval contexts stay safe."""
        if len(context) <= self.MAX_CONTEXT_CHARS:
            return context

        head = self.MAX_CONTEXT_CHARS // 2
        tail = self.MAX_CONTEXT_CHARS - head
        return f"{context[:head].rstrip()}\n\n[truncated]\n\n{context[-tail:].lstrip()}"

    def generate(self, query: str, context: str) -> str:
        """Generate an answer given a query and retrieved context."""
        if not context.strip():
            self._logger.warning("Empty retrieval context provided to generator.")
            return "I don't know based on the retrieved context."

        prompt_start = time.perf_counter()
        self._logger.debug("START prompt")
        prompt_text = self._build_prompt(query, context)
        self._logger.debug("END prompt (%.3fs)", time.perf_counter() - prompt_start)

        ollama_start = time.perf_counter()
        self._logger.debug("START ollama")
        try:
            response = ollama.generate(
                model=self._model,
                prompt=prompt_text,
                system=self.SYSTEM_PROMPT,
                options={
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                    "num_predict": 512,
                },
            )
            text = getattr(response, "response", None)
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Ollama returned empty text.")
            self._logger.debug("END ollama (%.3fs)", time.perf_counter() - ollama_start)
            return text.strip()
        except Exception as exc:
            self._logger.exception("Ollama generation failed: %s", exc)
            self._logger.debug("END ollama (%.3fs)", time.perf_counter() - ollama_start)
            return "I don't know based on the retrieved context."

    def _build_prompt(self, query: str, context: str) -> str:
        context = self._truncate_context(context)
        return (
            f"{self.SYSTEM_PROMPT}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
