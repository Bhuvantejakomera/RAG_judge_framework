from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class Verdict(str, Enum):
    """Enumeration for judgment outcomes."""

    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"


@dataclass(frozen=True)
class DocumentChunk:
    """Represents a chunk of text extracted from an article."""

    chunk_id: str
    article_title: str
    chunk_number: int
    text: str
    source_url: str
    token_count: int


@dataclass(frozen=True)
class RetrievalResult:
    """Encapsulates a document chunk and its retrieval score."""

    chunk: DocumentChunk
    score: float


@dataclass(frozen=True)
class EvaluationSample:
    """Structured input passed into the judge for a single evaluation item."""

    sample_id: str
    question: str
    rewritten_question: str
    ground_truth: str
    generated_answer: str
    ground_truth_answers: List[str] = field(default_factory=list)
    retrieved_chunks: List[RetrievalResult] = field(default_factory=list)
    relevant_titles: List[str] = field(default_factory=list)
    retrieval_metadata: Dict[str, Any] = field(default_factory=dict)
    generation_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationTurn:
    """Tracks a single conversational interaction and retrieval context."""

    question: str
    rewritten_question: Optional[str] = None
    retrieved_chunks: List[RetrievalResult] = field(default_factory=list)
    generated_answer: Optional[str] = None
    ground_truth: Optional[str] = None
    retrieval_time: Optional[float] = None
    generation_time: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class JudgeEvaluation:
    """Captures judge model evaluation scores and reasoning."""

    groundedness: float | None
    correctness: float | None
    completeness: float | None
    fluency: float | None
    hallucination: float | None
    supported_by_context: bool | None
    matches_ground_truth: bool | None
    reasoning: str | None
    final_verdict: Verdict | None
    judge_available: bool = True

    @classmethod
    def unavailable(cls) -> JudgeEvaluation:
        """Return a sentinel result when neither judge provider can produce a valid score."""
        return cls(
            groundedness=None,
            correctness=None,
            completeness=None,
            fluency=None,
            hallucination=None,
            supported_by_context=None,
            matches_ground_truth=None,
            reasoning=None,
            final_verdict=None,
            judge_available=False,
        )


@dataclass
class TraditionalMetricsResult:
    """Stores conventional QA evaluation metrics."""

    exact_match: float
    token_f1: float
    bleu: float
    rouge_l: float
    bert_score: float
    alias_exact_match: float
    alias_token_f1: float


@dataclass
class QuestionEvaluationRecord:
    """Stores a complete record of question evaluation and metadata."""

    question: str
    answer: str
    ground_truth: Optional[str]
    judge_evaluation: JudgeEvaluation
    metrics: TraditionalMetricsResult
    retrieved_context: List[RetrievalResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, str] = field(default_factory=dict)
