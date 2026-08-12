from __future__ import annotations

import json
import logging
import os
import re
import time
from abc import ABC
from typing import Any, Sequence

import ollama
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from pydantic import ValidationError

from src.config.settings import ConfigManager
from src.evaluation.json_validator import JSONValidator, JudgeSchema
from src.models.data_models import EvaluationSample, JudgeEvaluation, RetrievalResult, Verdict


class _GeminiAPIError(RuntimeError):
    """Wraps Gemini API failures with structured metadata for fallback decisions."""

    def __init__(self, message: str, status_code: int | None = None, error_code: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


class _GeminiFallbackRequired(RuntimeError):
    """Signals that Gemini should be abandoned in favor of the local judge."""


class BaseJudge(ABC):
    """Abstract base class for judge implementations."""

    def evaluate(self, question: str, answer: str, ground_truth: str, context: str) -> JudgeEvaluation:
        raise NotImplementedError


class OpenAIJudge(BaseJudge):
    """Evaluates answers using Gemini first and Ollama as an automatic fallback."""

    SYSTEM_PROMPT = (
        "You are an evaluation model. Respond with valid JSON matching the schema: "
        "groundedness, correctness, completeness, fluency, hallucination, supported_by_context, "
        "matches_ground_truth, reasoning, final_verdict. "
        "Final verdict should be PASS, PARTIAL, or FAIL. "
        "Return JSON only."
    )
    LOCAL_SYSTEM_PROMPT = (
        "Return one JSON object only with keys groundedness, correctness, completeness, fluency, "
        "hallucination, supported_by_context, matches_ground_truth, reasoning, final_verdict. "
        "Scores are integers 1-5; supported_by_context and matches_ground_truth are booleans; "
        "reasoning should be one short sentence; final_verdict is PASS, PARTIAL, or FAIL."
    )
    LOCAL_JUDGE_MAX_PREDICT = 88
    LOCAL_PROMPT_TOKEN_TARGET = 650
    LOCAL_CONTEXT_TOKEN_BUDGET = 240
    LOCAL_CONTEXT_MIN_TOKEN_BUDGET = 120
    LOCAL_CONTEXT_MAX_CHARS_PER_CHUNK = 140
    LOCAL_CONTEXT_MAX_SENTENCES_PER_CHUNK = 1
    LOCAL_CONTEXT_MAX_CHUNKS = 4
    LOCAL_CONTEXT_STOPWORDS = {
        "about",
        "answer",
        "based",
        "context",
        "final",
        "from",
        "ground",
        "groundtruth",
        "hallucination",
        "json",
        "matches",
        "question",
        "reasoning",
        "retrieved",
        "return",
        "supported",
        "text",
        "truth",
        "using",
        "valid",
    }
    GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
    GEMINI_MAX_ATTEMPTS = 3
    LOCAL_MAX_ATTEMPTS = 3

    def __init__(self) -> None:
        load_dotenv(override=False)
        self._settings = ConfigManager().settings
        self._logger = logging.getLogger(__name__)
        self._validator = JSONValidator()
        self._gemini_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self._using_local_judge = False
        self._local_model_ready = False
        self._last_timing_profile: dict[str, Any] = {}
        self._print_startup_banner()

    def evaluate(self, question: str, answer: str, ground_truth: str, context: str) -> JudgeEvaluation:
        return self._evaluate_with_prompt(
            sample_id="legacy",
            question=question,
            rewritten_question=question,
            answer=answer,
            ground_truth=ground_truth,
            context=context,
        )

    def evaluate_sample(self, sample: EvaluationSample) -> JudgeEvaluation:
        """Evaluate a structured sample without changing the legacy public API."""
        return self._evaluate_with_prompt(
            sample_id=sample.sample_id,
            question=sample.question,
            rewritten_question=sample.rewritten_question,
            answer=sample.generated_answer,
            ground_truth=sample.ground_truth,
            context=sample.retrieved_chunks,
            retrieval_metadata=sample.retrieval_metadata,
            generation_metadata=sample.generation_metadata,
        )

    def _evaluate_with_prompt(
        self,
        sample_id: str,
        question: str,
        rewritten_question: str,
        answer: str,
        ground_truth: str,
        context: str | Sequence[RetrievalResult],
        retrieval_metadata: dict[str, Any] | None = None,
        generation_metadata: dict[str, Any] | None = None,
    ) -> JudgeEvaluation:
        self._last_timing_profile = {}
        prompt_start = time.perf_counter()
        prompt = self._build_prompt(
            question=question,
            rewritten_question=rewritten_question,
            answer=answer,
            ground_truth=ground_truth,
            context=context,
            retrieval_metadata=retrieval_metadata or {},
            generation_metadata=generation_metadata or {},
        )
        prompt_end = time.perf_counter()
        self._last_timing_profile = {
            "prompt_build_start": prompt_start,
            "prompt_build_end": prompt_end,
            "prompt_build_elapsed": prompt_end - prompt_start,
            "prompt_length": len(prompt),
            "prompt_characters": len(prompt),
            "estimated_prompt_tokens": self._estimate_token_count(prompt),
        }
        try:
            if self._using_local_judge:
                evaluation = self._evaluate_with_local_judge(sample_id=sample_id, prompt=prompt)
                if evaluation.judge_available:
                    self._record_last_timing_profile(provider_name="Ollama")
                return evaluation

            evaluation = self._evaluate_with_gemini(sample_id=sample_id, prompt=prompt)
            self._record_last_timing_profile(provider_name="Gemini")
            return evaluation
        except _GeminiFallbackRequired as exc:
            self._logger.warning("Gemini judge unavailable: %s", exc)
            try:
                self._switch_to_local_judge()
                evaluation = self._evaluate_with_local_judge(sample_id=sample_id, prompt=prompt)
                if evaluation.judge_available:
                    self._record_last_timing_profile(provider_name="Ollama")
                return evaluation
            except Exception as local_exc:  # noqa: BLE001 - fallback failure should degrade to an unavailable result.
                self._logger.warning("Local judge unavailable after Gemini fallback: %s", local_exc)
                return self._build_unavailable_evaluation()
        except Exception as exc:  # noqa: BLE001 - judge failures must not terminate evaluation.
            self._logger.warning("Judge evaluation unavailable for %s: %s", sample_id, exc)
            return self._build_unavailable_evaluation()

    def _evaluate_with_gemini(self, sample_id: str, prompt: str) -> JudgeEvaluation:
        attempt = 0
        use_response_format = True
        while attempt < self.GEMINI_MAX_ATTEMPTS:
            try:
                start_time = time.perf_counter()
                raw_text = self._request_gemini_completion(prompt, use_response_format=use_response_format)
                return self._finalize_evaluation(
                    provider_name="Gemini",
                    sample_id=sample_id,
                    raw_text=raw_text,
                    start_time=start_time,
                )
            except _GeminiAPIError as exc:
                attempt += 1
                self._logger.warning("Gemini judge attempt %d failed: %s", attempt, exc)
                if self._should_use_local_fallback(exc):
                    raise _GeminiFallbackRequired(str(exc)) from exc
                if use_response_format and self._should_disable_response_format(exc):
                    use_response_format = False
                if attempt >= self.GEMINI_MAX_ATTEMPTS:
                    raise _GeminiFallbackRequired(str(exc)) from exc
                time.sleep(2**attempt)
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                attempt += 1
                self._logger.warning("Gemini judge attempt %d produced invalid JSON: %s", attempt, exc)
                if attempt >= self.GEMINI_MAX_ATTEMPTS:
                    raise _GeminiFallbackRequired("Gemini returned invalid JSON.") from exc
                time.sleep(2**attempt)

        raise _GeminiFallbackRequired("Gemini evaluation exhausted its retry budget.")

    def _evaluate_with_local_judge(self, sample_id: str, prompt: str) -> JudgeEvaluation:
        self._ensure_local_judge_ready()
        attempt = 0
        while attempt < self.LOCAL_MAX_ATTEMPTS:
            try:
                start_time = time.perf_counter()
                raw_text = self._request_local_completion(prompt)
                self._record_last_timing_profile(provider_name="Ollama")
                return self._finalize_evaluation(
                    provider_name="Ollama",
                    sample_id=sample_id,
                    raw_text=raw_text,
                    start_time=start_time,
                )
            except (json.JSONDecodeError, ValidationError, ValueError, RuntimeError) as exc:
                attempt += 1
                self._logger.warning("Local judge attempt %d failed: %s", attempt, exc)
                if attempt >= self.LOCAL_MAX_ATTEMPTS:
                    self._record_last_timing_profile(provider_name="Ollama")
                    return self._build_unavailable_evaluation()
                time.sleep(2**attempt)
            except Exception as exc:  # noqa: BLE001 - fallback judge should fail loudly but clearly.
                attempt += 1
                self._logger.warning("Local judge attempt %d failed: %s", attempt, exc)
                if attempt >= self.LOCAL_MAX_ATTEMPTS:
                    self._record_last_timing_profile(provider_name="Ollama")
                    return self._build_unavailable_evaluation()
                time.sleep(2**attempt)

        self._record_last_timing_profile(provider_name="Ollama")
        return self._build_unavailable_evaluation()

    def _request_gemini_completion(self, prompt: str, use_response_format: bool) -> str:
        api_key = self._gemini_api_key.strip() if self._gemini_api_key else ""
        if not api_key:
            raise _GeminiAPIError("Gemini API key is missing.", status_code=401, error_code="authentication")

        gemini_model = self._settings.judge.model.removeprefix("models/")
        try:
            client = genai.Client(api_key=api_key)
            config_kwargs: dict[str, Any] = {
                "temperature": self._settings.judge.temperature,
                "max_output_tokens": self._settings.judge.max_tokens,
            }
            response = client.models.generate_content(
                model=gemini_model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(**config_kwargs),
            )
        except genai_errors.APIError as exc:
            response_json = getattr(exc, "response_json", None)
            error_payload: Any = response_json
            if isinstance(response_json, dict):
                error_payload = response_json.get("error", response_json)

            error_code: str | None = None
            message = getattr(exc, "message", None) or str(exc)
            if isinstance(error_payload, dict):
                status_value = error_payload.get("status")
                if isinstance(status_value, str) and status_value.strip():
                    normalized_status = status_value.strip().lower()
                    error_code = {
                        "resource_exhausted": "quota_exceeded",
                        "unauthenticated": "authentication",
                        "permission_denied": "permission_denied",
                        "not_found": "not_found",
                    }.get(normalized_status, normalized_status)
                message_value = error_payload.get("message")
                if isinstance(message_value, str) and message_value.strip():
                    message = message_value.strip()

            status_code = getattr(exc, "code", None)
            if status_code is None:
                status_code = getattr(exc, "status", None)
            raise _GeminiAPIError(message, status_code=status_code, error_code=error_code) from exc
        except Exception as exc:  # noqa: BLE001 - Gemini client errors are mapped to existing fallback handling.
            raise _GeminiAPIError(str(exc), error_code="network_error") from exc

        text = response.text
        if isinstance(text, str) and text.strip():
            return text.strip()
        raise ValueError("Gemini judge returned an empty response.")

    def _request_local_completion(self, prompt: str) -> str:
        request_start = time.perf_counter()
        response_stream = ollama.generate(
            model=self._settings.judge.fallback_model,
            prompt=prompt,
            system=self.LOCAL_SYSTEM_PROMPT,
            format="json",
            stream=True,
            options={
                "temperature": 0.0,
                "num_predict": min(self._settings.judge.max_tokens, self.LOCAL_JUDGE_MAX_PREDICT),
            },
        )

        first_response_time: float | None = None
        request_end = request_start
        response_parts: list[str] = []
        final_response: Any = None

        for chunk in response_stream:
            chunk_time = time.perf_counter()
            if first_response_time is None:
                first_response_time = chunk_time
            text = self._response_field(chunk, "response")
            if isinstance(text, str) and text:
                response_parts.append(text)
            final_response = chunk
            request_end = chunk_time

        if first_response_time is None:
            first_response_time = request_end

        text = "".join(response_parts).strip()
        if not text:
            raise ValueError("Ollama judge returned an empty response.")

        self._last_timing_profile.update(
            self._build_local_timing_profile(
                prompt=prompt,
                request_start=request_start,
                first_response_time=first_response_time,
                request_end=request_end,
                final_response=final_response,
                response_text=text,
            )
        )
        return text

    def _extract_gemini_text(self, response_payload: dict[str, Any]) -> str:
        candidates = response_payload.get("candidates")
        if isinstance(candidates, list):
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                content = candidate.get("content")
                if not isinstance(content, dict):
                    continue
                parts = content.get("parts")
                if not isinstance(parts, list):
                    continue
                text_parts = []
                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    text = part.get("text")
                    if isinstance(text, str) and text.strip():
                        text_parts.append(text.strip())
                if text_parts:
                    return "\n".join(text_parts).strip()

        text = response_payload.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        raise ValueError("Gemini judge returned an empty response.")

    def _build_gemini_error(self, status_code: int, error_body: str) -> _GeminiAPIError:
        message = error_body.strip() or f"Gemini request failed with HTTP {status_code}."
        error_code: str | None = None
        if error_body.strip():
            try:
                parsed = json.loads(error_body)
                error_payload = parsed.get("error", parsed) if isinstance(parsed, dict) else parsed
                if isinstance(error_payload, dict):
                    code = error_payload.get("code")
                    if isinstance(code, str):
                        error_code = code
                    message_value = error_payload.get("message")
                    if isinstance(message_value, str) and message_value.strip():
                        message = message_value.strip()
            except json.JSONDecodeError:
                pass
        return _GeminiAPIError(message=message, status_code=status_code, error_code=error_code)

    @staticmethod
    def _judge_response_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "groundedness": {"type": "number"},
                "correctness": {"type": "number"},
                "completeness": {"type": "number"},
                "fluency": {"type": "number"},
                "hallucination": {"type": "number"},
                "supported_by_context": {"type": "boolean"},
                "matches_ground_truth": {"type": "boolean"},
                "reasoning": {"type": "string"},
                "final_verdict": {
                    "type": "string",
                    "enum": ["PASS", "PARTIAL", "FAIL"],
                },
            },
            "required": [
                "groundedness",
                "correctness",
                "completeness",
                "fluency",
                "hallucination",
                "supported_by_context",
                "matches_ground_truth",
                "reasoning",
                "final_verdict",
            ],
            "additionalProperties": False,
        }

    def _should_use_local_fallback(self, exc: _GeminiAPIError) -> bool:
        if exc.status_code in {401, 403, 404, 429}:
            return True

        error_code = (exc.error_code or "").lower()
        message = str(exc).lower()
        fallback_codes = {
            "authentication",
            "permission_denied",
            "quota_exceeded",
            "rate_limit_exceeded",
            "failed_precondition",
            "model_not_found",
            "not_found",
        }
        if error_code in fallback_codes:
            return True

        fallback_markers = (
            "billing",
            "invalid api key",
            "authentication failure",
            "authentication",
            "resource_exhausted",
            "quota exceeded",
        )
        return any(marker in message for marker in fallback_markers)

    @staticmethod
    def _should_disable_response_format(exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            "response_format" in message
            or "responseformat" in message
            or "response format" in message
            or "response_mime_type" in message
            or "response mime type" in message
            or "json_object" in message
            or "json object" in message
            or "json mode" in message
        )

    def _print_startup_banner(self) -> None:
        provider_label = self._settings.judge.provider.capitalize()
        print(f"Judge Provider : {provider_label}")
        print(f"Judge Model    : {self._settings.judge.model}")

    def _switch_to_local_judge(self) -> None:
        if self._using_local_judge:
            return
        print("Gemini has no credits.")
        print("Switching to local judge...")
        provider_label = self._settings.judge.fallback_provider.capitalize()
        print(f"Judge Provider : {provider_label}")
        print(f"Judge Model    : {self._settings.judge.fallback_model}")
        self._ensure_local_judge_ready()
        self._using_local_judge = True

    def _ensure_local_judge_ready(self) -> None:
        if self._local_model_ready:
            return

        model_name = self._settings.judge.fallback_model
        if self._local_model_is_installed(model_name):
            self._local_model_ready = True
            return

        print("Local judge not found.")
        print(f"Downloading {model_name}...")
        try:
            self._download_local_model(model_name)
        except Exception as exc:  # noqa: BLE001 - local judge preparation should surface explicit errors.
            raise RuntimeError(f"Unable to download local judge model {model_name}.") from exc

        self._local_model_ready = True
        print("Local judge ready.")

    def _download_local_model(self, model_name: str) -> None:
        """Stream Ollama pull events into a live progress bar."""
        console = Console()
        progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        )
        latest_total: int | None = None
        latest_completed: int | None = None

        with progress:
            task_id = progress.add_task(f"Downloading {model_name}", total=1)
            for event in ollama.pull(model_name, stream=True):
                status = self._progress_event_value(event, "status")
                completed = self._progress_event_value(event, "completed")
                total = self._progress_event_value(event, "total")

                updates: dict[str, Any] = {}
                if status:
                    updates["description"] = f"Downloading {model_name} - {status}"
                if total is not None:
                    latest_total = total
                    updates["total"] = total
                if completed is not None:
                    latest_completed = completed
                    updates["completed"] = completed

                if updates:
                    progress.update(task_id, **updates)

            if latest_total is not None and latest_completed is not None and latest_completed < latest_total:
                progress.update(task_id, completed=latest_total)

    @staticmethod
    def _progress_event_value(event: Any, key: str) -> int | str | None:
        if isinstance(event, dict):
            return event.get(key)
        return getattr(event, key, None)

    def _local_model_is_installed(self, model_name: str) -> bool:
        try:
            response = ollama.list()
        except Exception as exc:  # noqa: BLE001 - best-effort detection before pulling a local model.
            self._logger.warning("Unable to query local Ollama models: %s", exc)
            return False

        models = getattr(response, "models", None)
        if models is None and isinstance(response, dict):
            models = response.get("models", [])

        for model in models or []:
            candidate = None
            if isinstance(model, dict):
                candidate = model.get("model")
            else:
                candidate = getattr(model, "model", None)
            if candidate == model_name:
                return True
        return False

    def _build_prompt(
        self,
        question: str,
        rewritten_question: str,
        answer: str,
        ground_truth: str,
        context: str | Sequence[RetrievalResult],
        retrieval_metadata: dict[str, Any],
        generation_metadata: dict[str, Any],
    ) -> str:
        normalized_question = self._normalize_whitespace(question)
        normalized_rewritten_question = self._normalize_whitespace(rewritten_question)
        normalized_answer = self._normalize_whitespace(answer)
        normalized_ground_truth = self._normalize_whitespace(ground_truth)

        sections = [f"Q: {normalized_question}"]
        if normalized_rewritten_question and normalized_rewritten_question != normalized_question:
            sections.append(f"RW: {normalized_rewritten_question}")
        sections.extend(
            [
                f"A: {normalized_answer}",
                f"GT: {normalized_ground_truth}",
            ]
        )
        context_budget = self._local_context_token_budget(
            question=normalized_question,
            rewritten_question=normalized_rewritten_question,
            answer=normalized_answer,
            ground_truth=normalized_ground_truth,
        )
        compact_context = self._render_context(
            context,
            question=normalized_question,
            rewritten_question=normalized_rewritten_question,
            answer=normalized_answer,
            ground_truth=normalized_ground_truth,
            token_budget=context_budget,
        )
        if compact_context:
            sections.append(f"CTX: {compact_context}")
        sections.append("JSON only.")
        return "\n".join([self.LOCAL_SYSTEM_PROMPT, *sections])

    def _record_last_timing_profile(self, provider_name: str) -> None:
        judge_config = self._settings.judge
        model_name = judge_config.fallback_model if provider_name == "Ollama" else judge_config.model
        self._last_timing_profile.update(
            {
                "judge_provider": provider_name,
                "judge_model": model_name,
            }
        )

    @staticmethod
    def _response_field(response: Any, key: str) -> Any:
        if isinstance(response, dict):
            return response.get(key)
        return getattr(response, key, None)

    @staticmethod
    def _estimate_token_count(text: str) -> int:
        stripped = text.strip()
        if not stripped:
            return 0
        return max(1, int(round(len(stripped) / 4)))

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        if not compact:
            return ""
        if len(compact) <= max_chars:
            return compact

        shortened = compact[:max_chars].rstrip()
        if len(shortened) < len(compact):
            shortened = shortened.rstrip(" ,;:") + "..."
        return shortened

    @classmethod
    def _extract_context_terms(cls, *texts: str) -> set[str]:
        terms: set[str] = set()
        for text in texts:
            if not text:
                continue
            for token in re.findall(r"[A-Za-z0-9']+", text.lower()):
                if len(token) < 4 or token in cls.LOCAL_CONTEXT_STOPWORDS:
                    continue
                terms.add(token)
        return terms

    def _local_context_token_budget(
        self,
        question: str,
        rewritten_question: str,
        answer: str,
        ground_truth: str,
    ) -> int:
        base_prompt = "\n".join(
            [
                self.LOCAL_SYSTEM_PROMPT,
                f"Q: {question}",
                f"RW: {rewritten_question}" if rewritten_question and rewritten_question != question else "",
                f"A: {answer}",
                f"GT: {ground_truth}",
                "JSON only.",
            ]
        )
        estimated_total = self._estimate_token_count(base_prompt)
        remaining_budget = self.LOCAL_PROMPT_TOKEN_TARGET - estimated_total
        return max(
            self.LOCAL_CONTEXT_MIN_TOKEN_BUDGET,
            min(self.LOCAL_CONTEXT_TOKEN_BUDGET, remaining_budget),
        )

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _normalize_sentence(sentence: str) -> str:
        return re.sub(r"\s+", " ", sentence).strip().lower()

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        normalized = text.strip()
        if not normalized:
            return []
        parts = re.split(r"(?<=[.!?])\s+", normalized)
        return [part.strip() for part in parts if part.strip()]

    def _truncate_sentences(
        self,
        text: str,
        seen_sentences: set[str],
        max_chars: int,
        max_sentences: int,
    ) -> str:
        if not text.strip():
            return ""

        sentences = self._split_sentences(text)
        selected: list[str] = []
        char_count = 0

        for sentence in sentences:
            normalized_sentence = self._normalize_sentence(sentence)
            if not normalized_sentence or normalized_sentence in seen_sentences:
                continue

            candidate = self._normalize_whitespace(sentence)
            if not candidate:
                continue

            if not selected and len(candidate) > max_chars:
                shortened = self._truncate_text(candidate, max_chars)
                selected.append(shortened)
                seen_sentences.add(normalized_sentence)
                return shortened

            additional_chars = len(candidate) if not selected else len(candidate) + 1
            if selected and char_count + additional_chars > max_chars:
                break

            selected.append(candidate)
            seen_sentences.add(normalized_sentence)
            char_count += additional_chars

            if len(selected) >= max_sentences:
                break

        if not selected:
            return self._truncate_text(text, max_chars)

        compact = " ".join(selected)
        return self._truncate_text(compact, max_chars)

    def _render_context(
        self,
        context: str | Sequence[RetrievalResult],
        question: str = "",
        rewritten_question: str = "",
        answer: str = "",
        ground_truth: str = "",
        token_budget: int | None = None,
    ) -> str:
        prompt_terms = self._extract_context_terms(question, rewritten_question, answer, ground_truth)
        budget = token_budget or self.LOCAL_CONTEXT_TOKEN_BUDGET

        if isinstance(context, str):
            compact = self._normalize_whitespace(context)
            if not compact:
                return ""
            if self._estimate_token_count(compact) <= budget:
                return compact
            return self._select_context_sentences(
                compact,
                prompt_terms=prompt_terms,
                seen_sentences=set(),
                max_chars=budget * 4,
                max_sentences=self.LOCAL_CONTEXT_MAX_SENTENCES_PER_CHUNK * 2,
            )

        if not context:
            return ""

        ordered_results = sorted(
            enumerate(context),
            key=lambda item: (
                -(item[1].score if isinstance(item[1].score, (int, float)) else float("-inf")),
                item[0],
            ),
        )
        seen_sentences: set[str] = set()
        blocks: list[str] = []
        estimated_tokens = 0

        for _, result in ordered_results[: self.LOCAL_CONTEXT_MAX_CHUNKS]:
            chunk = result.chunk
            title = self._normalize_whitespace(chunk.article_title) or "Untitled"
            chunk_text = self._select_context_sentences(
                chunk.text,
                prompt_terms=prompt_terms,
                seen_sentences=seen_sentences,
                max_chars=self.LOCAL_CONTEXT_MAX_CHARS_PER_CHUNK,
                max_sentences=self.LOCAL_CONTEXT_MAX_SENTENCES_PER_CHUNK,
            )
            if not chunk_text:
                continue

            score = result.score
            score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
            block = f"[{title} | chunk {chunk.chunk_number} | score {score_text}] {chunk_text}"
            block_tokens = self._estimate_token_count(block)
            remaining_budget = budget - estimated_tokens
            if block_tokens > remaining_budget:
                if blocks or remaining_budget <= 0:
                    break
                max_chars = max(80, remaining_budget * 4 - len(block) + len(chunk_text))
                chunk_text = self._truncate_text(chunk_text, max_chars)
                block = f"[{title} | chunk {chunk.chunk_number} | score {score_text}] {chunk_text}"
                block_tokens = self._estimate_token_count(block)
                if block_tokens > remaining_budget:
                    break

            blocks.append(block)
            estimated_tokens += block_tokens
            if estimated_tokens >= budget:
                break

        if not blocks:
            return ""

        if len(blocks) < len(ordered_results):
            blocks.append(f"[{len(ordered_results) - len(blocks)} additional retrieved chunk(s) omitted]")

        return "\n".join(blocks)

    def _select_context_sentences(
        self,
        text: str,
        prompt_terms: set[str],
        seen_sentences: set[str],
        max_chars: int,
        max_sentences: int,
    ) -> str:
        sentences = self._split_sentences(text)
        if not sentences:
            return ""

        ranked_sentences: list[tuple[int, int, str, str]] = []
        for index, sentence in enumerate(sentences):
            normalized_sentence = self._normalize_whitespace(sentence)
            if not normalized_sentence:
                continue

            normalized_key = self._normalize_sentence(normalized_sentence)
            if not normalized_key or normalized_key in seen_sentences:
                continue

            sentence_terms = self._extract_context_terms(normalized_sentence)
            overlap = len(sentence_terms & prompt_terms) if prompt_terms else 0
            ranked_sentences.append((overlap, index, normalized_sentence, normalized_key))

        if not ranked_sentences:
            return self._truncate_text(text, max_chars)

        ranked_sentences.sort(key=lambda item: (-item[0], item[1]))
        selected = ranked_sentences[:max_sentences]
        selected.sort(key=lambda item: item[1])

        snippets: list[str] = []
        for _, _, sentence, normalized_key in selected:
            snippet = self._truncate_text(sentence, max_chars)
            if not snippet:
                continue
            snippets.append(snippet)
            seen_sentences.add(normalized_key)

        if not snippets:
            return self._truncate_text(text, max_chars)

        compact = " ".join(snippets)
        return self._truncate_text(compact, max_chars)

    def _build_local_timing_profile(
        self,
        prompt: str,
        request_start: float,
        first_response_time: float,
        request_end: float,
        final_response: Any,
        response_text: str,
    ) -> dict[str, Any]:
        prompt_eval_count = self._response_field(final_response, "prompt_eval_count")
        eval_count = self._response_field(final_response, "eval_count")
        profile = {
            "ollama_request_start": request_start,
            "ollama_request_end": request_end,
            "ollama_request_elapsed": first_response_time - request_start,
            "time_to_first_response_elapsed": first_response_time - request_start,
            "generation_start": first_response_time,
            "generation_end": request_end,
            "generation_elapsed": request_end - first_response_time,
            "request_total_elapsed": request_end - request_start,
            "prompt_characters": len(prompt),
            "estimated_prompt_tokens": int(prompt_eval_count)
            if isinstance(prompt_eval_count, int)
            else self._estimate_token_count(prompt),
            "response_characters": len(response_text),
            "response_tokens": int(eval_count) if isinstance(eval_count, int) else None,
            "response_prompt_tokens": int(prompt_eval_count) if isinstance(prompt_eval_count, int) else None,
            "response_total_duration": self._response_field(final_response, "total_duration"),
            "response_load_duration": self._response_field(final_response, "load_duration"),
            "response_prompt_eval_duration": self._response_field(final_response, "prompt_eval_duration"),
            "response_eval_duration": self._response_field(final_response, "eval_duration"),
        }
        return profile

    @staticmethod
    def _format_timing_line(label: str, elapsed: float, detail: str | None = None) -> str:
        value = f"{elapsed:.2f} s"
        if detail:
            value = f"{value} | {detail}"
        dots = "." * max(3, 24 - len(label))
        return f"{label:<24} {dots} {value}"

    def _print_ollama_profile(self) -> None:
        profile = self._last_timing_profile
        print("=" * 60)
        print("Ollama Judge Profile")
        print("=" * 60)
        print(self._format_timing_line("Judge Prompt Build", float(profile.get("prompt_build_elapsed", 0.0) or 0.0)))
        print(self._format_timing_line("Ollama Request", float(profile.get("ollama_request_elapsed", 0.0) or 0.0)))
        print(self._format_timing_line("Generation", float(profile.get("generation_elapsed", 0.0) or 0.0)))
        print(self._format_timing_line("JSON Parse", float(profile.get("json_parse_elapsed", 0.0) or 0.0)))
        print(self._format_timing_line("Validation", float(profile.get("validation_elapsed", 0.0) or 0.0)))
        print(self._format_timing_line("Judge Total", float(profile.get("judge_total_elapsed", 0.0) or 0.0)))
        print(f"{'Prompt characters':<24} {profile.get('prompt_characters', 0)}")
        print(f"{'Estimated prompt tokens':<24} {profile.get('estimated_prompt_tokens', 0)}")
        print(f"{'Response characters':<24} {profile.get('response_characters', 0)}")
        response_tokens = profile.get("response_tokens")
        print(f"{'Response tokens':<24} {response_tokens if response_tokens is not None else 'N/A'}")
        print("=" * 60)

    def _finalize_evaluation(
        self,
        provider_name: str,
        sample_id: str,
        raw_text: str,
        start_time: float,
    ) -> JudgeEvaluation:
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError(f"{provider_name} judge returned an empty response.")

        judge_schema = self._validator.validate(raw_text)
        validator_profile = dict(getattr(self._validator, "_last_timing_profile", {}) or {})
        self._last_timing_profile.update(validator_profile)
        total_start = float(self._last_timing_profile.get("prompt_build_start", start_time) or start_time)
        judge_total_elapsed = time.perf_counter() - total_start
        self._last_timing_profile["judge_total_elapsed"] = judge_total_elapsed
        self._logger.info("%s judge request successful for %s.", provider_name, sample_id)
        self._logger.info("%s judge JSON parsed successfully for %s.", provider_name, sample_id)
        self._logger.info(
            "%s judge evaluation completed for %s in %.3f seconds.",
            provider_name,
            sample_id,
            judge_total_elapsed,
        )
        if provider_name == "Ollama":
            self._print_ollama_profile()
        return self._build_evaluation(judge_schema)

    def _build_evaluation(self, schema: JudgeSchema) -> JudgeEvaluation:
        try:
            verdict = Verdict(schema.final_verdict)
        except ValueError:
            verdict = Verdict.PARTIAL
        return JudgeEvaluation(
            groundedness=schema.groundedness,
            correctness=schema.correctness,
            completeness=schema.completeness,
            fluency=schema.fluency,
            hallucination=schema.hallucination,
            supported_by_context=schema.supported_by_context,
            matches_ground_truth=schema.matches_ground_truth,
            reasoning=schema.reasoning,
            final_verdict=verdict,
        )

    @staticmethod
    def _build_unavailable_evaluation() -> JudgeEvaluation:
        return JudgeEvaluation.unavailable()
