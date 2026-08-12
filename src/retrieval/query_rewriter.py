from __future__ import annotations

import logging
import time

import ollama

from src.config.settings import ConfigManager


class QueryRewriter:
    """Rewrites conversational queries into standalone queries using Ollama."""

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        self._settings = ConfigManager().settings
        self._model = self._settings.llm.local_model

    def rewrite(self, question: str, history: list[dict[str, str]]) -> str:
        """Rewrite a follow-up question into a standalone search query."""
        prompt_start = time.perf_counter()
        self._logger.debug("START prompt")
        prompt = self._build_prompt(question, history)
        self._logger.debug("END prompt (%.3fs)", time.perf_counter() - prompt_start)

        ollama_start = time.perf_counter()
        self._logger.debug("START ollama")
        try:
            response = ollama.generate(
                model=self._model,
                prompt=prompt,
                options={
                    "temperature": 0.0,
                    "top_p": 0.9,
                    "num_predict": 128,
                },
            )
            text = getattr(response, "response", None)
            if isinstance(text, str) and text.strip():
                self._logger.debug("END ollama (%.3fs)", time.perf_counter() - ollama_start)
                return text.strip()
        except Exception as exc:
            self._logger.warning("Ollama rewrite failed, using raw question: %s", exc)
        self._logger.debug("END ollama (%.3fs)", time.perf_counter() - ollama_start)
        return question

    def _build_prompt(self, question: str, history: list[dict[str, str]]) -> str:
        messages = ["Convert the user question into a standalone search query."]
        if history:
            messages.append("Conversation history:")
            for turn in history:
                messages.append(f"Q: {turn.get('question')}\nA: {turn.get('generated_answer')}" )
        messages.append(f"Follow-up question: {question}")
        messages.append("Standalone query:")
        return "\n\n".join(messages)
