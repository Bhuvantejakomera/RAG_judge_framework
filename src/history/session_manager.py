from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.settings import ConfigManager
from src.models.data_models import ConversationTurn


@dataclass
class SessionManager:
    """Manages a session history with persistence and sliding window behavior."""

    session_root: Path | None = None
    _max_turns: int = 5
    _history: deque[ConversationTurn] = field(default_factory=deque)
    _completed_history: list[ConversationTurn] = field(default_factory=list)
    _logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))

    def __post_init__(self) -> None:
        settings = ConfigManager().settings
        self._max_turns = settings.memory.sliding_window_turns
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        self.session_root = Path(self.session_root or Path.cwd() / "sessions" / f"session_{timestamp}")
        self.session_root.mkdir(parents=True, exist_ok=True)
        self._logger.info("Created session directory %s", self.session_root)

    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a conversation turn and persist the updated session state."""
        self._completed_history.append(turn)
        self._history.append(turn)
        while len(self._history) > self._max_turns:
            self._history.popleft()
        self.persist()
        self._logger.debug("Added turn for question: %s", turn.question)

    @property
    def completed_turn_count(self) -> int:
        """Return the number of completed question-answer interactions."""
        return len(self._completed_history)

    def get_completed_turns(self) -> list[ConversationTurn]:
        """Return the full completed session history."""
        return list(self._completed_history)

    def get_completed_history(self) -> list[dict[str, Any]]:
        """Return the full completed session history as JSON-serializable data."""
        return [self._serialize_turn(turn) for turn in self._completed_history]

    def get_sliding_history(self) -> list[dict[str, Any]]:
        """Return the current sliding window history as JSON-serializable data."""
        return [self._serialize_turn(turn) for turn in list(self._history)]

    def clear_history(self) -> None:
        """Clear the session history and persist the empty state."""
        self._history.clear()
        self._completed_history.clear()
        self.persist()
        self._logger.debug("Cleared session history.")

    def persist(self) -> None:
        """Persist session history and configuration snapshot to disk."""
        history_path = self.session_root / "history.json"
        completed_history_path = self.session_root / "completed_history.json"
        config_path = self.session_root / "config_snapshot.json"
        history_path.write_text(json.dumps(self.get_sliding_history(), indent=2), encoding="utf-8")
        completed_history_path.write_text(json.dumps(self.get_completed_history(), indent=2), encoding="utf-8")
        config_snapshot = ConfigManager().settings.model_dump()
        config_path.write_text(json.dumps(config_snapshot, indent=2), encoding="utf-8")
        self._logger.info("Persisted session history and configuration snapshot.")

    def _serialize_turn(self, turn: ConversationTurn) -> dict[str, Any]:
        return {
            "question": turn.question,
            "rewritten_question": turn.rewritten_question,
            "retrieved_chunks": [
                {
                    "chunk_id": result.chunk.chunk_id,
                    "article_title": result.chunk.article_title,
                    "chunk_number": result.chunk.chunk_number,
                    "source_url": result.chunk.source_url,
                    "score": result.score,
                }
                for result in turn.retrieved_chunks
            ],
            "generated_answer": turn.generated_answer,
            "ground_truth": turn.ground_truth,
            "retrieval_time": turn.retrieval_time,
            "generation_time": turn.generation_time,
            "timestamp": turn.timestamp.isoformat(),
        }
