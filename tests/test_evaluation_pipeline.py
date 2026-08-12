from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evaluation.pipeline import EvaluationPipeline

from src.history.session_manager import SessionManager
from src.models.data_models import (
    ConversationTurn,
    DocumentChunk,
    JudgeEvaluation,
    RetrievalResult,
    TraditionalMetricsResult,
    Verdict,
)


def _make_retrieval_result(index: int) -> RetrievalResult:
    chunk = DocumentChunk(
        chunk_id=f"chunk-{index}",
        article_title=f"Article {index}",
        chunk_number=index,
        text=f"Retrieved text {index}",
        source_url=f"https://example.com/{index}",
        token_count=10,
    )
    return RetrievalResult(chunk=chunk, score=0.9 - (index * 0.01))


def _make_turn(index: int) -> ConversationTurn:
    return ConversationTurn(
        question=f"Question {index}",
        rewritten_question=f"Rewritten question {index}",
        retrieved_chunks=[_make_retrieval_result(index)],
        generated_answer=f"Answer {index}",
        ground_truth=f"Answer {index}",
    )


class _FakeRetriever:
    def __init__(self) -> None:
        self.calls = 0

    def retrieve(self, query: str, top_k: int):  # noqa: ANN001 - simple test double
        self.calls += 1
        return [_make_retrieval_result(self.calls)]


class _FakeRewriter:
    def __init__(self) -> None:
        self.calls = 0

    def rewrite(self, question: str, history: list[dict[str, str]]):  # noqa: ANN001 - simple test double
        self.calls += 1
        return f"{question} rewritten"


class _FakeGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, query: str, context: str):  # noqa: ANN001 - simple test double
        self.calls += 1
        return f"generated answer {self.calls}"


class _FakeJudge:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate_sample(self, sample):  # noqa: ANN001 - simple test double
        self.calls += 1
        return JudgeEvaluation(
            groundedness=1.0,
            correctness=1.0,
            completeness=1.0,
            fluency=1.0,
            hallucination=0.0,
            supported_by_context=True,
            matches_ground_truth=True,
            reasoning=f"reason {self.calls}",
            final_verdict=Verdict.PASS,
        )


class _FakeMetrics:
    def __init__(self) -> None:
        self.calls = 0

    def analyze_alias_prediction(self, prediction: str, ground_truth: object):  # noqa: ANN001 - simple test double
        return SimpleNamespace(
            raw_prediction=prediction,
            extracted_answer=None,
            normalized_prediction=prediction,
            references=tuple(ground_truth) if isinstance(ground_truth, list) else tuple(),
        )

    def exact_match(self, prediction: str, ground_truth: str) -> float:
        self.calls += 1
        return 1.0

    def token_f1(self, prediction: str, ground_truth: str) -> float:
        return 1.0

    def alias_exact_match(self, prediction: str, ground_truth: object) -> float:  # noqa: ANN001 - simple test double
        return 1.0

    def alias_token_f1(self, prediction: str, ground_truth: object) -> float:  # noqa: ANN001 - simple test double
        return 1.0

    def bleu(self, prediction: str, ground_truth: str) -> float:
        return 1.0

    def rouge_l(self, prediction: str, ground_truth: str) -> float:
        return 1.0

    def bert_score(self, prediction: str, ground_truth: str) -> float:
        return 1.0

    def retrieval_metrics(self, retrieved_chunks, relevant_titles, top_k):  # noqa: ANN001 - simple test double
        return {
            "available": True,
            "hit_rate": 1.0,
            "recall": 1.0,
            "mrr": 1.0,
            "relevant_rank": 1,
        }


def test_session_mode_uses_completed_turns(tmp_path: Path) -> None:
    session_manager = SessionManager(session_root=tmp_path / "session")
    for index in range(5):
        session_manager.add_turn(_make_turn(index))

    retriever = _FakeRetriever()
    rewriter = _FakeRewriter()
    generator = _FakeGenerator()
    judge = _FakeJudge()
    metrics = _FakeMetrics()

    pipeline = EvaluationPipeline(
        session_manager=session_manager,
        retriever=retriever,
        rewriter=rewriter,
        generator=generator,
        judge=judge,
        metrics=metrics,
        dataset_path=tmp_path / "test.tsv",
    )
    pipeline.run(evaluate=True)

    output = json.loads((session_manager.session_root / "evaluation.json").read_text(encoding="utf-8"))
    assert output["evaluation_mode"] == "session"
    assert output["questions_evaluated"] == 5
    assert len(output["results"]) == 5
    assert retriever.calls == 0
    assert rewriter.calls == 0
    assert generator.calls == 0
    assert judge.calls == 5
    assert metrics.calls == 5


def test_dataset_mode_samples_rows_when_session_is_short(tmp_path: Path) -> None:
    dataset_path = tmp_path / "test.tsv"
    with dataset_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question", "possible_answers"], delimiter="\t")
        writer.writeheader()
        for index in range(4):
            writer.writerow(
                {
                    "question": f"Dataset question {index}",
                    "possible_answers": f"Dataset answer {index}",
                }
            )

    session_manager = SessionManager(session_root=tmp_path / "short_session")
    retriever = _FakeRetriever()
    rewriter = _FakeRewriter()
    generator = _FakeGenerator()
    judge = _FakeJudge()
    metrics = _FakeMetrics()

    pipeline = EvaluationPipeline(
        session_manager=session_manager,
        retriever=retriever,
        rewriter=rewriter,
        generator=generator,
        judge=judge,
        metrics=metrics,
        dataset_path=dataset_path,
    )
    pipeline._evaluation_settings = pipeline._evaluation_settings.model_copy(
        update={"dataset_sample_size": 2, "random_seed": 7}
    )
    pipeline.run(evaluate=True)

    output = json.loads((session_manager.session_root / "evaluation.json").read_text(encoding="utf-8"))
    assert output["evaluation_mode"] == "dataset"
    assert output["questions_evaluated"] == 2
    assert len(output["results"]) == 2
    assert retriever.calls == 2
    assert rewriter.calls == 2
    assert generator.calls == 2
    assert judge.calls == 2
    assert metrics.calls == 2


def test_average_nested_metrics_ignores_none() -> None:
    results = [
        {"judge_evaluation": {"correctness": 5.0}},
        {"judge_evaluation": {"correctness": 5.0}},
        {"judge_evaluation": {"correctness": None}},
        {"judge_evaluation": {"correctness": 4.0}},
    ]
    avg = EvaluationPipeline._average_nested_metrics(results, "judge_evaluation", "correctness")
    assert avg == pytest.approx(14 / 3)


def test_evaluation_pipeline_sample_size_override(tmp_path: Path) -> None:
    dataset_path = tmp_path / "test.tsv"
    with dataset_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question", "possible_answers"], delimiter="\t")
        writer.writeheader()
        for index in range(10):
            writer.writerow(
                {
                    "question": f"Dataset question {index}",
                    "possible_answers": f"Dataset answer {index}",
                }
            )

    session_manager = SessionManager(session_root=tmp_path / "sample_size_session")
    pipeline = EvaluationPipeline(
        session_manager=session_manager,
        retriever=_FakeRetriever(),
        rewriter=_FakeRewriter(),
        generator=_FakeGenerator(),
        judge=_FakeJudge(),
        metrics=_FakeMetrics(),
        dataset_path=dataset_path,
        sample_size=3,
    )
    pipeline.run(evaluate=True)
    output = json.loads((session_manager.session_root / "evaluation.json").read_text(encoding="utf-8"))
    assert output["questions_evaluated"] == 3


def test_normalize_args() -> None:
    from main import _normalize_args

    assert _normalize_args(["--evaluate", "-", "5"]) == ["--evaluate", "--sample-size", "5"]
    assert _normalize_args(["--evaluate", "5"]) == ["--evaluate", "--sample-size", "5"]
    assert _normalize_args(["--evaluate", "-5"]) == ["--evaluate", "--sample-size", "5"]
    assert _normalize_args(["--evaluate"]) == ["--evaluate"]
    assert _normalize_args(["--sample-size", "5", "--evaluate"]) == ["--sample-size", "5", "--evaluate"]


