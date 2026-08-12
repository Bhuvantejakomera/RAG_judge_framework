from __future__ import annotations

import ast
import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass
from collections import Counter
from typing import Any, Sequence
from urllib.parse import unquote, urlparse

from bert_score import score as bert_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

from src.config.settings import ConfigManager
from src.models.data_models import RetrievalResult


class _NullBertScore:
    """Sentinel that behaves like 0.0 for arithmetic but serializes as null."""

    __slots__ = ()

    def __add__(self, other: object) -> object:
        return other

    def __radd__(self, other: object) -> object:
        return other

    def __float__(self) -> float:
        return 0.0

    def __deepcopy__(self, memo: dict[int, object]) -> _NullBertScore:
        return self


_BERT_SCORE_NULL = _NullBertScore()
_JSON_NULL_PATCHED = False
_ANSWER_ARTICLES = {"a", "an", "the"}
_ANSWER_CUE_PATTERNS: tuple[tuple[str, ...], ...] = (
    ("answer",),
    ("answer", "is"),
    ("final", "answer"),
    ("final", "answer", "is"),
    ("the", "answer"),
    ("the", "answer", "is"),
    ("it", "is"),
    ("it", "was"),
    ("it", "were"),
    ("is",),
    ("was",),
    ("were",),
    ("are",),
    ("be",),
    ("located", "in"),
    ("located", "at"),
    ("based", "in"),
    ("from",),
    ("in",),
    ("by",),
    ("produced",),
    ("written", "by"),
    ("directed", "by"),
    ("authored", "by"),
    ("called",),
    ("named",),
    ("born", "in"),
    ("died", "in"),
    ("capital", "of"),
    ("genre", "is"),
    ("occupation", "is"),
    ("author", "is"),
    ("director", "is"),
    ("screenwriter", "is"),
    ("father", "is"),
    ("mother", "is"),
)


@dataclass(frozen=True)
class AliasPredictionAnalysis:
    """Captures how a generated answer was interpreted for alias-aware scoring."""

    raw_prediction: str
    extracted_answer: str | None
    normalized_prediction: str
    references: tuple[str, ...]


def _install_json_null_patch() -> None:
    """Make the null sentinel serialize to JSON null without affecting other values."""
    global _JSON_NULL_PATCHED
    if _JSON_NULL_PATCHED:
        return

    original_default = json.JSONEncoder.default

    def default(self: json.JSONEncoder, obj: object) -> object:  # type: ignore[override]
        if isinstance(obj, _NullBertScore):
            return None
        return original_default(self, obj)

    json.JSONEncoder.default = default  # type: ignore[assignment]
    _JSON_NULL_PATCHED = True


def _coerce_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _normalize_reference_aliases(references: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for reference in references:
        normalized_reference = normalize_answer(reference)
        if not normalized_reference or normalized_reference in seen:
            continue
        normalized.append(reference)
        seen.add(normalized_reference)
    return tuple(normalized)


def _parse_serialized_aliases(text: str) -> list[str] | None:
    if not text:
        return None

    parsers = (
        json.loads,
        ast.literal_eval,
    )
    for parser in parsers:
        try:
            parsed = parser(text)
        except Exception:  # noqa: BLE001 - best-effort compatibility parsing only.
            continue

        if isinstance(parsed, str):
            parsed_text = parsed.strip()
            return [parsed_text] if parsed_text else []
        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip() for item in parsed if str(item).strip()]

    if "|" in text:
        parts = [part.strip() for part in text.split("|")]
        if len(parts) > 1:
            return [part for part in parts if part]

    if ";" in text:
        parts = [part.strip() for part in text.split(";")]
        if len(parts) > 1:
            return [part for part in parts if part]

    return None


def extract_ground_truth_answers(ground_truth: Any) -> list[str]:
    """Return every valid PopQA answer alias as a normalized list."""
    if ground_truth is None:
        return []

    if isinstance(ground_truth, (list, tuple, set)):
        raw_answers = [str(item).strip() for item in ground_truth]
        return _dedupe_preserve_order(raw_answers)

    text = _coerce_string(ground_truth)
    if not text:
        return []

    parsed = _parse_serialized_aliases(text)
    if parsed is None:
        return [text]
    return _dedupe_preserve_order(parsed)


def _collapse_single_letter_tokens(tokens: list[str]) -> list[str]:
    collapsed: list[str] = []
    buffer: list[str] = []
    for token in tokens:
        if len(token) == 1 and token.isalpha():
            buffer.append(token)
            continue
        if buffer:
            collapsed.append("".join(buffer))
            buffer = []
        collapsed.append(token)

    if buffer:
        collapsed.append("".join(buffer))
    return collapsed


def normalize_answer(text: Any) -> str:
    """Normalize PopQA answers without changing the meaning of the response."""
    normalized = unicodedata.normalize("NFKC", _coerce_string(text))
    if not normalized:
        return ""

    normalized = normalized.casefold()
    normalized = re.sub(r"^[\"'“”‘’`]+|[\"'“”‘’`]+$", "", normalized)
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return ""

    tokens = normalized.split()
    if len(tokens) > 1 and tokens[0] in _ANSWER_ARTICLES:
        tokens = tokens[1:]
    tokens = _collapse_single_letter_tokens(tokens)
    return " ".join(tokens)


def normalize_entity_title(text: Any) -> str:
    """Normalize a Wikipedia title or source URL for relevance matching."""
    raw = _coerce_string(text)
    if not raw:
        return ""

    parsed_url = urlparse(raw)
    if parsed_url.scheme and parsed_url.netloc and parsed_url.path:
        raw = parsed_url.path.rsplit("/", 1)[-1]

    raw = unquote(raw.replace("_", " ").strip())
    raw = re.sub(r"\s*\((?:[^()]|\([^()]*\))*\)\s*$", "", raw).strip()
    return normalize_answer(raw)


def _token_f1_from_tokens(pred_tokens: Sequence[str], ref_tokens: Sequence[str]) -> float:
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    common = sum(min(pred_counts[token], ref_counts[token]) for token in pred_counts)
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _has_confident_answer_cue(prefix_tokens: Sequence[str]) -> bool:
    if not prefix_tokens:
        return False

    prefix = tuple(prefix_tokens)
    return any(len(prefix) >= len(cue) and prefix[-len(cue) :] == cue for cue in _ANSWER_CUE_PATTERNS)


def _extract_confident_prediction_answer(
    prediction_tokens: Sequence[str],
    normalized_reference_map: dict[str, str],
) -> str | None:
    if not prediction_tokens or not normalized_reference_map:
        return None

    full_pred = " ".join(prediction_tokens)
    if full_pred in normalized_reference_map:
        return normalized_reference_map[full_pred]

    sorted_aliases = sorted(normalized_reference_map.keys(), key=lambda item: (-len(item.split()), -len(item)))
    for alias in sorted_aliases:
        alias_tokens = alias.split()
        alias_length = len(alias_tokens)
        if not alias_tokens or alias_length > len(prediction_tokens):
            continue

        for start in range(len(prediction_tokens) - alias_length + 1):
            if prediction_tokens[start : start + alias_length] != alias_tokens:
                continue

            prefix = prediction_tokens[:start]
            suffix = prediction_tokens[start + alias_length :]

            if _has_confident_answer_cue(prefix):
                if not suffix:
                    return normalized_reference_map[alias]

            if not prefix and suffix and suffix[0] in {"is", "was", "are", "were"}:
                return normalized_reference_map[alias]

    return None




def analyze_prediction_for_alias_scoring(prediction: Any, references: Sequence[str] | None = None) -> AliasPredictionAnalysis:
    """Separate the raw prediction from a conservative extracted answer candidate."""
    raw_prediction = _coerce_string(prediction)
    raw_normalized_prediction = normalize_answer(raw_prediction)
    normalized_references = _normalize_reference_aliases(references or [])
    normalized_reference_map: dict[str, str] = {}
    for reference in normalized_references:
        normalized_reference = normalize_answer(reference)
        if normalized_reference and normalized_reference not in normalized_reference_map:
            normalized_reference_map[normalized_reference] = reference

    extracted_answer = _extract_confident_prediction_answer(
        raw_normalized_prediction.split(),
        normalized_reference_map,
    )

    if extracted_answer is None and raw_prediction:
        clauses = [clause.strip() for clause in re.split(r"[,;:.!?\n]+", raw_prediction) if clause.strip()]
        for clause in clauses:
            clause_normalized = normalize_answer(clause)
            if not clause_normalized:
                continue
            candidate = _extract_confident_prediction_answer(
                clause_normalized.split(),
                normalized_reference_map,
            )
            if candidate is not None:
                extracted_answer = candidate
                break

    normalized_prediction = normalize_answer(extracted_answer or raw_prediction)
    return AliasPredictionAnalysis(
        raw_prediction=raw_prediction,
        extracted_answer=extracted_answer,
        normalized_prediction=normalized_prediction,
        references=normalized_references,
    )



def extract_prediction_answer(prediction: str, references: Sequence[str] | None = None) -> str:
    """Return a conservative answer candidate or the raw prediction if no extraction is confident."""
    analysis = analyze_prediction_for_alias_scoring(prediction, references)
    return analysis.extracted_answer or analysis.normalized_prediction


class TraditionalMetricsCalculator:
    """Calculates traditional QA and text similarity metrics."""

    DEFAULT_BERT_SCORE_MODEL = "roberta-base"

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        self._settings = ConfigManager().settings
        _install_json_null_patch()
        configured_model = self._settings.evaluation.bert_score_model.strip()
        self._bert_score_model = configured_model or self.DEFAULT_BERT_SCORE_MODEL
        self._bert_score_banner_printed = False

    @staticmethod
    def normalize_text(text: str) -> str:
        return " ".join(text.lower().strip().split())

    def exact_match(self, prediction: str, ground_truth: str) -> float:
        normalized_prediction = self.normalize_text(prediction)
        normalized_ground_truth = self.normalize_text(ground_truth)
        return 1.0 if normalized_prediction == normalized_ground_truth else 0.0

    def analyze_alias_prediction(self, prediction: str, ground_truth: Any) -> AliasPredictionAnalysis:
        references = extract_ground_truth_answers(ground_truth)
        return analyze_prediction_for_alias_scoring(prediction, references)

    def token_f1(self, prediction: str, ground_truth: str) -> float:
        pred_tokens = self.normalize_text(prediction).split()
        gt_tokens = self.normalize_text(ground_truth).split()
        return _token_f1_from_tokens(pred_tokens, gt_tokens)

    def bleu(self, prediction: str, ground_truth: str) -> float:
        reference = [self.normalize_text(ground_truth).split()]
        hypothesis = self.normalize_text(prediction).split()
        if not hypothesis or not reference[0]:
            return 0.0
        smoothing = SmoothingFunction().method2
        return sentence_bleu(reference, hypothesis, smoothing_function=smoothing)

    def rouge_l(self, prediction: str, ground_truth: str) -> float:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(self.normalize_text(ground_truth), self.normalize_text(prediction))
        return scores["rougeL"].fmeasure

    def alias_exact_match(self, prediction: str, ground_truth: Any) -> float:
        analysis = self.analyze_alias_prediction(prediction, ground_truth)
        if not analysis.normalized_prediction or not analysis.references:
            return 0.0

        normalized_references = {
            normalized_reference
            for normalized_reference in (normalize_answer(reference) for reference in analysis.references)
            if normalized_reference
        }
        return 1.0 if analysis.normalized_prediction in normalized_references else 0.0

    def alias_token_f1(self, prediction: str, ground_truth: Any) -> float:
        analysis = self.analyze_alias_prediction(prediction, ground_truth)
        if not analysis.normalized_prediction or not analysis.references:
            return 0.0

        pred_tokens = analysis.normalized_prediction.split()
        if not pred_tokens:
            return 0.0

        best_score = 0.0
        for reference in analysis.references:
            ref_tokens = normalize_answer(reference).split()
            score = _token_f1_from_tokens(pred_tokens, ref_tokens)
            if score > best_score:
                best_score = score
        return best_score

    def bert_score(self, prediction: str, ground_truth: Any) -> float:
        references = extract_ground_truth_answers(ground_truth)
        if not prediction.strip() or not references:
            return 0.0

        self._print_bert_score_model()
        try:
            return self._compute_bert_score(prediction, references, self._bert_score_model)
        except Exception:  # noqa: BLE001 - never crash the evaluation pipeline on metric failures.
            self._logger.exception("BERTScore computation failed.")
            return _BERT_SCORE_NULL

    def retrieval_metrics(
        self,
        retrieved_chunks: Sequence[RetrievalResult],
        relevant_titles: Any,
        top_k: int,
    ) -> dict[str, float | int | None]:
        reference_titles = extract_ground_truth_answers(relevant_titles)
        normalized_references = {
            normalized_reference
            for normalized_reference in (normalize_entity_title(reference) for reference in reference_titles)
            if normalized_reference
        }
        if not normalized_references:
            return {
                "available": False,
                "hit_rate": None,
                "recall": None,
                "mrr": None,
                "relevant_rank": None,
            }

        relevant_rank: int | None = None
        for index, result in enumerate(retrieved_chunks[:top_k], start=1):
            candidate_titles = {
                normalized_candidate
                for normalized_candidate in (
                    normalize_entity_title(result.chunk.article_title),
                    normalize_entity_title(result.chunk.source_url),
                )
                if normalized_candidate
            }
            if candidate_titles & normalized_references:
                relevant_rank = index
                break

        hit = 1.0 if relevant_rank is not None else 0.0
        reciprocal_rank = 1.0 / relevant_rank if relevant_rank is not None else 0.0
        return {
            "available": True,
            "hit_rate": hit,
            "recall": hit,
            "mrr": reciprocal_rank,
            "relevant_rank": relevant_rank,
        }

    def _print_bert_score_model(self) -> None:
        if self._bert_score_banner_printed:
            return
        print(f"BERTScore Model : {self._bert_score_model}")
        self._bert_score_banner_printed = True

    def _compute_bert_score(self, prediction: str, references: Sequence[str], model_type: str) -> float:
        print(
            "[BERTScore] Starting computation...\n"
            f"Model : {model_type}\n"
            f"Prediction length : {len(prediction)}\n"
            f"Reference count : {len(references)}\n"
            "--------------------------------------------------"
        )
        start = time.perf_counter()
        try:
            predictions = [prediction] * len(references)
            _, _, f1 = bert_score(
                predictions,
                list(references),
                model_type=model_type,
                lang="en",
                rescale_with_baseline=False,
            )
            elapsed = time.perf_counter() - start
            score = max(float(score.item()) for score in f1)
            print(
                "[BERTScore] Finished\n"
                f"Elapsed : {elapsed:.2f} seconds\n"
                f"Score : {score:.4f}\n"
                "--------------------------------------------------"
            )
            return score
        except Exception as exc:
            elapsed = time.perf_counter() - start
            print(
                "[BERTScore] FAILED\n"
                f"Elapsed before failure : {elapsed:.2f} seconds\n"
                f"Exception : {exc}\n"
                "--------------------------------------------------"
            )
            raise
