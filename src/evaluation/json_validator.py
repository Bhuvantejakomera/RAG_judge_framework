from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from pydantic import BaseModel, ValidationError

try:  # pragma: no cover - optional dependency.
    import json_repair as _json_repair_module
except Exception:  # noqa: BLE001 - optional dependency may not be installed.
    _json_repair_module = None


class JudgeSchema(BaseModel):
    groundedness: float
    correctness: float
    completeness: float
    fluency: float
    hallucination: float
    supported_by_context: bool
    matches_ground_truth: bool
    reasoning: str
    final_verdict: str


class JSONValidator:
    """Validates JSON responses against a Pydantic judge schema."""

    def __init__(self, max_attempts: int = 3, base_backoff: float = 1.0) -> None:
        self._logger = logging.getLogger(__name__)
        self._max_attempts = max_attempts
        self._base_backoff = base_backoff
        self._last_timing_profile: dict[str, Any] = {}

    def validate(self, raw_text: str) -> JudgeSchema:
        """Validate raw JSON text and retry when malformed."""
        text = raw_text.strip()
        parse_elapsed_total = 0.0
        validation_elapsed_total = 0.0
        total_start = time.perf_counter()
        self._last_timing_profile = {}
        parse_attempts = 0

        parse_start = time.perf_counter()
        try:
            parsed = json.loads(text)
            parse_attempts += 1
        except json.JSONDecodeError as exc:
            parse_attempts += 1
            parse_elapsed_total += time.perf_counter() - parse_start
            self._logger.warning("Validation attempt %d failed: %s", parse_attempts, exc)
            repaired_candidates = self._build_repair_candidates(text)
            last_json_error: json.JSONDecodeError | None = exc
            last_validation_error: ValidationError | None = None

            for candidate in repaired_candidates:
                parse_start = time.perf_counter()
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError as candidate_exc:
                    parse_attempts += 1
                    parse_elapsed_total += time.perf_counter() - parse_start
                    last_json_error = candidate_exc
                    self._logger.warning("Validation attempt %d failed: %s", parse_attempts, candidate_exc)
                    continue

                parse_attempts += 1
                parse_elapsed_total += time.perf_counter() - parse_start
                validation_start = time.perf_counter()
                try:
                    schema = JudgeSchema(**parsed)
                except ValidationError as candidate_exc:
                    validation_elapsed_total += time.perf_counter() - validation_start
                    last_validation_error = candidate_exc
                    self._logger.warning("Validation attempt %d failed: %s", parse_attempts, candidate_exc)
                    continue

                validation_elapsed_total += time.perf_counter() - validation_start
                self._last_timing_profile = {
                    "json_parse_elapsed": parse_elapsed_total,
                    "validation_elapsed": validation_elapsed_total,
                    "validation_attempts": parse_attempts,
                    "validation_total_elapsed": time.perf_counter() - total_start,
                }
                return schema

            self._last_timing_profile = {
                "json_parse_elapsed": parse_elapsed_total,
                "validation_elapsed": validation_elapsed_total,
                "validation_attempts": parse_attempts,
                "validation_total_elapsed": time.perf_counter() - total_start,
            }
            if last_validation_error is not None:
                raise last_validation_error
            if last_json_error is not None:
                raise last_json_error
            raise ValueError("Unable to validate JSON after retries.")
        else:
            validation_start = time.perf_counter()
            try:
                schema = JudgeSchema(**parsed)
            except ValidationError as exc:
                validation_elapsed_total += time.perf_counter() - validation_start
                self._last_timing_profile = {
                    "json_parse_elapsed": parse_elapsed_total,
                    "validation_elapsed": validation_elapsed_total,
                    "validation_attempts": parse_attempts,
                    "validation_total_elapsed": time.perf_counter() - total_start,
                }
                self._logger.warning("Validation attempt %d failed: %s", parse_attempts, exc)
                raise

            validation_elapsed_total += time.perf_counter() - validation_start
            self._last_timing_profile = {
                "json_parse_elapsed": parse_elapsed_total,
                "validation_elapsed": validation_elapsed_total,
                "validation_attempts": parse_attempts,
                "validation_total_elapsed": time.perf_counter() - total_start,
            }
            return schema

    @staticmethod
    def _strip_invalid_control_characters(text: str) -> str:
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    @classmethod
    def _build_repair_candidates(cls, text: str) -> list[str]:
        cleaned = cls._strip_invalid_control_characters(text).strip()
        candidates: list[str] = []
        seen: set[str] = set()

        def add_candidate(candidate: str | None) -> None:
            if not candidate:
                return
            normalized = candidate.strip()
            if not normalized or normalized in seen:
                return
            candidates.append(normalized)
            seen.add(normalized)

        add_candidate(cleaned if cleaned != text else None)

        fragments = cls._extract_json_fragments(cleaned)
        for fragment in fragments:
            add_candidate(fragment if fragment != cleaned else None)
            add_candidate(cls._close_truncated_json(fragment))
            add_candidate(cls._repair_with_library(fragment))

        add_candidate(cls._repair_with_library(cleaned))
        return candidates

    @classmethod
    def _extract_json_fragments(cls, text: str) -> list[str]:
        fragments: list[str] = []
        seen: set[str] = set()
        for match in re.finditer(r"[\{\[]", text):
            fragment = cls._extract_balanced_fragment(text, match.start())
            if not fragment:
                continue
            normalized = fragment.strip()
            if normalized and normalized not in seen:
                fragments.append(normalized)
                seen.add(normalized)
        if fragments:
            return fragments

        opening = min((index for index, char in enumerate(text) if char in "{["), default=-1)
        if opening >= 0:
            return [text[opening:].strip()]
        return []

    @classmethod
    def _extract_balanced_fragment(cls, text: str, start: int) -> str | None:
        if start < 0 or start >= len(text) or text[start] not in "{[":
            return None

        stack = [text[start]]
        in_string = False
        escaped = False
        for index in range(start + 1, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char in "{[":
                stack.append(char)
            elif char in "}]":
                if not stack:
                    return None
                opener = stack.pop()
                if (opener == "{" and char != "}") or (opener == "[" and char != "]"):
                    return None
                if not stack:
                    return text[start : index + 1]

        return text[start:]

    @staticmethod
    def _close_truncated_json(fragment: str) -> str:
        text = fragment.strip()
        if not text:
            return text

        stack: list[str] = []
        in_string = False
        escaped = False
        for char in text:
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char in "{[":
                stack.append(char)
            elif char in "}]":
                if stack:
                    opener = stack.pop()
                    if (opener == "{" and char != "}") or (opener == "[" and char != "]"):
                        return text

        repaired = text
        if in_string:
            if repaired and repaired[-1] in "}]":
                repaired = repaired[:-1].rstrip()
            repaired = JSONValidator._strip_trailing_escape(repaired)
            repaired += '"'

        while stack:
            opener = stack.pop()
            repaired += "}" if opener == "{" else "]"
        return repaired

    @staticmethod
    def _strip_trailing_escape(text: str) -> str:
        stripped = text.rstrip()
        if not stripped:
            return stripped

        backslashes = 0
        index = len(stripped) - 1
        while index >= 0 and stripped[index] == "\\":
            backslashes += 1
            index -= 1

        if backslashes % 2 == 1:
            return stripped[:-1]
        return stripped

    @staticmethod
    def _repair_with_library(text: str) -> str | None:
        if _json_repair_module is None:
            return None

        repair_function = None
        for name in ("repair_json", "loads", "repair"):
            candidate = getattr(_json_repair_module, name, None)
            if callable(candidate):
                repair_function = candidate
                break

        if repair_function is None:
            return None

        try:
            repaired = repair_function(text)
        except Exception:  # noqa: BLE001 - optional best-effort repair.
            return None

        if repaired is None:
            return None
        if isinstance(repaired, str):
            repaired_text = repaired.strip()
        else:
            repaired_text = json.dumps(repaired, ensure_ascii=False)
        return repaired_text or None
