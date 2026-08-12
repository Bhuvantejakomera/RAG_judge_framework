from __future__ import annotations

import csv
import json
import logging
import random
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from rich import box
from rich.console import Console
from rich.table import Table

from src.config.settings import ConfigManager
from src.evaluation.judge import OpenAIJudge
from src.evaluation.metrics import AliasPredictionAnalysis, TraditionalMetricsCalculator, extract_ground_truth_answers
from src.history.session_manager import SessionManager
from src.models.data_models import ConversationTurn, EvaluationSample, RetrievalResult, TraditionalMetricsResult
from src.generator.ollama_client import LocalOllamaGenerator
from src.retrieval.query_rewriter import QueryRewriter
from src.retrieval.retriever import DenseFAISSRetriever


class EvaluationPipeline:
    """Runs evaluation over session history or a sample dataset."""

    def __init__(
        self,
        session_manager: SessionManager | None = None,
        retriever: DenseFAISSRetriever | None = None,
        rewriter: QueryRewriter | None = None,
        generator: LocalOllamaGenerator | None = None,
        judge: OpenAIJudge | None = None,
        metrics: TraditionalMetricsCalculator | None = None,
        dataset_path: str | Path | None = None,
        sample_size: int | None = None,
    ) -> None:
        self._console = Console()
        self._logger = logging.getLogger(__name__)
        self._settings = ConfigManager().settings
        self._evaluation_settings = self._settings.evaluation.model_copy()
        if sample_size is not None and sample_size > 0:
            self._evaluation_settings.dataset_sample_size = sample_size
        self._session_manager = session_manager or SessionManager()
        self._retriever = retriever or DenseFAISSRetriever()
        self._rewriter = rewriter or QueryRewriter()
        self._generator = generator or LocalOllamaGenerator()
        self._judge = judge or OpenAIJudge()
        self._metrics = metrics or TraditionalMetricsCalculator()
        self._dataset_path = Path(dataset_path or Path.cwd() / "data" / "test.tsv")
        self._evaluation_sample_index = 0
        self._evaluation_sample_total = 0
        self._sample_stage_timings: dict[str, dict[str, dict[str, Any]]] = {}
        self._sample_timing_records: list[dict[str, Any]] = []

    def run(self, evaluate: bool = False, sample_size: int | None = None) -> None:
        if not evaluate:
            self._logger.info("Evaluation pipeline was not triggered.")
            return

        if sample_size is not None and sample_size > 0:
            self._evaluation_settings.dataset_sample_size = sample_size

        self._reset_timing_state()
        mode = self._select_evaluation_mode()
        if mode == "session":
            turns = self._session_manager.get_completed_turns()
            reason = (
                f"Session contains {len(turns)} completed interactions. "
                "Using session evaluation."
            )
            self._print_mode_banner(mode, len(turns), reason)
            results = self._evaluate_session_history(turns)
        else:
            sampled_rows = self._sample_dataset_rows()
            reason = (
                f"Session contains fewer than {self._evaluation_settings.session_threshold} completed interactions. "
                "Using dataset evaluation."
            )
            self._print_mode_banner(mode, len(sampled_rows), reason)
            results = self._evaluate_dataset_samples(sampled_rows)

        self._report_results(mode, results)

    def _select_evaluation_mode(self) -> str:
        if self._session_manager.completed_turn_count >= self._evaluation_settings.session_threshold:
            return "session"
        return "dataset"

    def _evaluate_session_history(self, history: Sequence[ConversationTurn]) -> list[dict[str, Any]]:
        self._evaluation_sample_index = 0
        self._evaluation_sample_total = len(history)
        results = []
        for index, turn in enumerate(history, start=1):
            if not turn.ground_truth:
                self._logger.warning(
                    "Session turn %d (%s) has no ground truth label; metrics will use an empty reference.",
                    index,
                    turn.question,
                )
            sample = self._build_session_sample(turn=turn, index=index)
            results.append(self._evaluate_sample(sample))
        return results

    def _evaluate_dataset_samples(self, rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
        self._evaluation_sample_index = 0
        self._evaluation_sample_total = len(rows)
        results = []
        for index, row in enumerate(rows, start=1):
            sample = self._build_dataset_sample(row=row, index=index)
            results.append(self._evaluate_sample(sample))
        return results

    def _sample_dataset_rows(self) -> list[dict[str, str]]:
        if not self._dataset_path.exists():
            raise FileNotFoundError("data/test.tsv not found for evaluation sample.")

        rows = list(self._load_tsv(self._dataset_path))
        if not rows:
            raise RuntimeError(f"No rows were found in {self._dataset_path}.")

        sample_size = min(self._evaluation_settings.dataset_sample_size, len(rows))
        strategy = self._evaluation_settings.sampling_strategy.lower().strip()
        if strategy == "random":
            rng = random.Random(self._evaluation_settings.random_seed)
            return rng.sample(rows, sample_size)

        self._logger.warning("Unsupported sampling strategy '%s'; using the first %d rows instead.", strategy, sample_size)
        return rows[:sample_size]

    def _load_tsv(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            return [row for row in reader]

    def _build_relevant_titles(self, row: dict[str, str]) -> list[str]:
        titles = [
            row.get("s_wiki_title", ""),
            row.get("subj", ""),
            *extract_ground_truth_answers(row.get("s_aliases", "")),
        ]
        seen: set[str] = set()
        relevant_titles: list[str] = []
        for title in titles:
            normalized = title.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            relevant_titles.append(normalized)
        return relevant_titles

    def _build_context(self, retrieved: Sequence[RetrievalResult]) -> str:
        return "\n\n".join(result.chunk.text for result in retrieved) if retrieved else ""

    def _reset_timing_state(self) -> None:
        self._sample_stage_timings = {}
        self._sample_timing_records = []

    def _build_session_sample(self, turn: ConversationTurn, index: int) -> EvaluationSample:
        sample_id = f"session-{index:03d}"
        self._sample_stage_timings[sample_id] = {
            "query_rewrite": {
                "start": None,
                "end": None,
                "elapsed": 0.0,
            },
            "retrieval": {
                "start": None,
                "end": None,
                "elapsed": self._duration_or_zero(turn.retrieval_time),
                "retrieved_chunk_count": len(turn.retrieved_chunks),
            },
            "generation": {
                "start": None,
                "end": None,
                "elapsed": self._duration_or_zero(turn.generation_time),
                "generated_answer_length": len(turn.generated_answer or ""),
            },
        }
        return EvaluationSample(
            sample_id=sample_id,
            question=turn.question,
            rewritten_question=turn.rewritten_question or turn.question,
            ground_truth=turn.ground_truth or "",
            ground_truth_answers=extract_ground_truth_answers(turn.ground_truth or ""),
            generated_answer=turn.generated_answer or "",
            retrieved_chunks=list(turn.retrieved_chunks),
            relevant_titles=[],
            retrieval_metadata={
                "source": "session",
                "retrieval_time": turn.retrieval_time,
                "retrieved_chunk_count": len(turn.retrieved_chunks),
                "timestamp": turn.timestamp.isoformat(),
            },
            generation_metadata={
                "source": "session",
                "generation_time": turn.generation_time,
                "timestamp": turn.timestamp.isoformat(),
            },
        )

    def _build_dataset_sample(self, row: dict[str, str], index: int) -> EvaluationSample:
        question = row["question"]
        ground_truth = row.get("possible_answers", "")
        ground_truth_answers = extract_ground_truth_answers(ground_truth)
        relevant_titles = self._build_relevant_titles(row)
        sample_id = f"dataset-{index:03d}"

        rewrite_start = time.perf_counter()
        rewritten_question = self._rewriter.rewrite(question, [])
        rewrite_end = time.perf_counter()

        retrieval_start = time.perf_counter()
        retrieved = self._retriever.retrieve(rewritten_question, self._settings.retrieval.top_k)
        retrieval_end = time.perf_counter()
        context = self._build_context(retrieved)

        generation_start = time.perf_counter()
        generated_answer = self._generator.generate(rewritten_question, context)
        generation_end = time.perf_counter()

        self._sample_stage_timings[sample_id] = {
            "query_rewrite": {
                "start": rewrite_start,
                "end": rewrite_end,
                "elapsed": rewrite_end - rewrite_start,
            },
            "retrieval": {
                "start": retrieval_start,
                "end": retrieval_end,
                "elapsed": retrieval_end - retrieval_start,
                "retrieved_chunk_count": len(retrieved),
            },
            "generation": {
                "start": generation_start,
                "end": generation_end,
                "elapsed": generation_end - generation_start,
                "generated_answer_length": len(generated_answer),
            },
        }

        return EvaluationSample(
            sample_id=sample_id,
            question=question,
            rewritten_question=rewritten_question,
            ground_truth=ground_truth,
            ground_truth_answers=ground_truth_answers,
            generated_answer=generated_answer,
            retrieved_chunks=list(retrieved),
            relevant_titles=relevant_titles,
            retrieval_metadata={
                "source": "dataset",
                "row_index": index,
                "rewrite_time": rewrite_end - rewrite_start,
                "retrieval_time": retrieval_end - retrieval_start,
                "retrieved_chunk_count": len(retrieved),
                "top_k": self._settings.retrieval.top_k,
                "relevant_titles": relevant_titles,
            },
            generation_metadata={
                "source": "dataset",
                "generation_time": generation_end - generation_start,
                "model": self._settings.llm.local_model,
            },
        )

    def _calculate_metrics(
        self,
        sample: EvaluationSample,
    ) -> tuple[TraditionalMetricsResult, dict[str, dict[str, float]], AliasPredictionAnalysis]:
        answer = sample.generated_answer
        ground_truth = sample.ground_truth
        ground_truth_answers = sample.ground_truth_answers or extract_ground_truth_answers(ground_truth)

        exact_match_start = time.perf_counter()
        exact_match = self._metrics.exact_match(answer, ground_truth)
        exact_match_end = time.perf_counter()

        f1_start = time.perf_counter()
        token_f1 = self._metrics.token_f1(answer, ground_truth)
        f1_end = time.perf_counter()

        bleu_start = time.perf_counter()
        bleu = self._metrics.bleu(answer, ground_truth)
        bleu_end = time.perf_counter()

        rouge_start = time.perf_counter()
        rouge_l = self._metrics.rouge_l(answer, ground_truth)
        rouge_end = time.perf_counter()

        alias_analysis = self._metrics.analyze_alias_prediction(answer, ground_truth_answers)
        alias_exact_match = self._metrics.alias_exact_match(answer, ground_truth_answers)
        alias_token_f1 = self._metrics.alias_token_f1(answer, ground_truth_answers)

        bert_score_start = time.perf_counter()
        bert_score = self._metrics.bert_score(answer, ground_truth_answers)
        bert_score_end = time.perf_counter()

        return (
            TraditionalMetricsResult(
                exact_match=exact_match,
                token_f1=token_f1,
                bleu=bleu,
                rouge_l=rouge_l,
                bert_score=bert_score,
                alias_exact_match=alias_exact_match,
                alias_token_f1=alias_token_f1,
            ),
            {
                "exact_match": {
                    "start": exact_match_start,
                    "end": exact_match_end,
                    "elapsed": exact_match_end - exact_match_start,
                },
                "f1": {
                    "start": f1_start,
                    "end": f1_end,
                    "elapsed": f1_end - f1_start,
                },
                "bleu": {
                    "start": bleu_start,
                    "end": bleu_end,
                    "elapsed": bleu_end - bleu_start,
                },
                "rouge": {
                    "start": rouge_start,
                    "end": rouge_end,
                    "elapsed": rouge_end - rouge_start,
                },
                "bert_score": {
                    "start": bert_score_start,
                    "end": bert_score_end,
                    "elapsed": bert_score_end - bert_score_start,
                },
            },
            alias_analysis,
        )

    def _evaluate_sample(self, sample: EvaluationSample) -> dict[str, Any]:
        total_start = time.perf_counter()
        self._evaluation_sample_index += 1
        sample_number = self._evaluation_sample_index
        sample_total = self._evaluation_sample_total or sample_number

        sample_stage_timings = dict(self._sample_stage_timings.pop(sample.sample_id, {}))

        judge_start = time.perf_counter()
        try:
            judge_eval = self._judge.evaluate_sample(sample)
        except Exception as exc:  # noqa: BLE001 - judge failures must not terminate evaluation.
            self._logger.warning("Judge evaluation failed unexpectedly for %s: %s", sample.sample_id, exc)
            from src.models.data_models import JudgeEvaluation
            judge_eval = JudgeEvaluation.unavailable()
        judge_end = time.perf_counter()
        judge_elapsed = judge_end - judge_start
        judge_profile = dict(getattr(self._judge, "_last_timing_profile", {}) or {})


        metrics, metric_timings, alias_analysis = self._calculate_metrics(sample)
        retrieval_metrics = self._metrics.retrieval_metrics(
            sample.retrieved_chunks,
            sample.relevant_titles,
            self._settings.retrieval.top_k,
        )
        total_end = time.perf_counter()
        total_elapsed = total_end - total_start

        self._print_sample_report(
            sample_number=sample_number,
            sample_total=sample_total,
            sample=sample,
            alias_analysis=alias_analysis,
            retrieval_metrics=retrieval_metrics,
            metrics=metrics,
            judge_eval=judge_eval,
        )

        timing_record = {
            "sample_id": sample.sample_id,
            "query_rewrite": sample_stage_timings.get(
                "query_rewrite",
                {
                    "start": None,
                    "end": None,
                    "elapsed": 0.0,
                },
            ),
            "retrieval": sample_stage_timings.get(
                "retrieval",
                {
                    "start": None,
                    "end": None,
                    "elapsed": self._duration_or_zero(sample.retrieval_metadata.get("retrieval_time")),
                    "retrieved_chunk_count": len(sample.retrieved_chunks),
                },
            ),
            "prompt_build": {
                "start": judge_profile.get("prompt_build_start"),
                "end": judge_profile.get("prompt_build_end"),
                "elapsed": float(judge_profile.get("prompt_build_elapsed", 0.0) or 0.0),
                "prompt_length": int(judge_profile.get("prompt_length", 0) or 0),
            },
            "generation": sample_stage_timings.get(
                "generation",
                {
                    "start": None,
                    "end": None,
                    "elapsed": self._duration_or_zero(sample.generation_metadata.get("generation_time")),
                    "generated_answer_length": len(sample.generated_answer),
                },
            ),
            "judge": {
                "start": judge_start,
                "end": judge_end,
                "elapsed": judge_elapsed,
                "provider": self._judge_provider(judge_profile),
                "model": self._judge_model(judge_profile),
            },
            "exact_match": metric_timings["exact_match"],
            "f1": metric_timings["f1"],
            "bleu": metric_timings["bleu"],
            "rouge": metric_timings["rouge"],
            "bert_score": metric_timings["bert_score"],
            "total": {
                "start": total_start,
                "end": total_end,
                "elapsed": total_elapsed,
            },
        }
        self._sample_timing_records.append(timing_record)
        self._print_sample_timing(
            sample_number=sample_number,
            sample_total=sample_total,
            timing_record=timing_record,
        )
        return self._assemble_record(sample, judge_eval, metrics, retrieval_metrics)

    @staticmethod
    def _extract_stage_duration(metadata: dict[str, Any], key: str) -> float:
        value = metadata.get(key, 0.0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _duration_or_zero(value: float | None) -> float:
        return float(value) if value is not None else 0.0

    def _print_sample_report(
        self,
        sample_number: int,
        sample_total: int,
        sample: EvaluationSample,
        alias_analysis: AliasPredictionAnalysis,
        retrieval_metrics: dict[str, Any],
        metrics: TraditionalMetricsResult,
        judge_eval: Any,
    ) -> None:
        top_k = self._settings.retrieval.top_k
        print("\n" + "=" * 60)
        print(f"Sample {sample_number} / {sample_total}")
        print("=" * 60)
        print()
        print("Question:")
        print(sample.question)
        print()
        print("Ground Truth:")
        print(sample.ground_truth)
        if sample.ground_truth_answers:
            print("References:")
            alias_preview = sample.ground_truth_answers[:5]
            if len(sample.ground_truth_answers) > 5:
                alias_preview.append(f"+{len(sample.ground_truth_answers) - 5} more")
            print(", ".join(alias_preview))
        print()
        print("Generated Answer:")
        print(sample.generated_answer)
        print()
        print("Prediction Analysis:")
        print(f"{'Raw Prediction':<18} : {alias_analysis.raw_prediction or 'N/A'}")
        print(
            f"{'Extracted Answer':<18} : "
            f"{alias_analysis.extracted_answer if alias_analysis.extracted_answer is not None else 'N/A'}"
        )
        print(f"{'Normalized Prediction':<18} : {alias_analysis.normalized_prediction or 'N/A'}")
        if alias_analysis.references:
            reference_preview = list(alias_analysis.references[:5])
            if len(alias_analysis.references) > 5:
                reference_preview.append(f"+{len(alias_analysis.references) - 5} more")
            print(f"{'References':<18} : {', '.join(reference_preview)}")
        if sample.retrieved_chunks:
            print()
            print("Retrieved Documents:")
            for index, result in enumerate(sample.retrieved_chunks[:3], start=1):
                print(f"{index}. {result.chunk.article_title} ({result.score:.4f})")

        print()
        print("-------------------------------")
        print("Retrieval")
        print("-------------------------------")
        hit_value = retrieval_metrics.get("hit_rate")
        relevant_rank = retrieval_metrics.get("relevant_rank")
        print(f"{f'Hit@{top_k}':<18} : {int(hit_value or 0.0)}")
        print(f"{'Relevant Rank':<18} : {relevant_rank if relevant_rank is not None else 'N/A'}")

        print()
        print("-------------------------------")
        print("Answer Metrics")
        print("-------------------------------")
        print(f"Alias Exact Match  : {metrics.alias_exact_match:.2f}")
        print(f"Alias Token F1     : {metrics.alias_token_f1:.2f}")
        print(f"BERTScore F1       : {metrics.bert_score:.4f}")

        print()
        print("-------------------------------")
        print("Legacy Metrics")
        print("-------------------------------")
        print(f"Raw Exact Match    : {metrics.exact_match:.2f}")
        print(f"Raw Token F1       : {metrics.token_f1:.4f}")
        print(f"BLEU               : {metrics.bleu:.4f}")
        print(f"ROUGE-L            : {metrics.rouge_l:.4f}")

        print()
        print("-------------------------------")
        print("LLM Judge")
        print("-------------------------------")
        print(f"{'Judge Available':<18} : {str(getattr(judge_eval, 'judge_available', True))}")
        print(f"{'Final Verdict':<18} : {self._format_optional_text(getattr(judge_eval, 'final_verdict', None))}")
        print(f"{'Correctness':<18} : {self._format_optional_float(judge_eval.correctness)}")
        print(f"{'Groundedness':<18} : {self._format_optional_float(judge_eval.groundedness)}")
        print(f"{'Completeness':<18} : {self._format_optional_float(judge_eval.completeness)}")
        print(f"{'Hallucination':<18} : {self._format_optional_float(judge_eval.hallucination)}")
        print(f"{'Supported Context':<18} : {self._format_optional_bool(judge_eval.supported_by_context)}")
        print(f"{'Matches GroundTruth':<18} : {self._format_optional_bool(judge_eval.matches_ground_truth)}")
        print()

    def _judge_provider(self, judge_profile: dict[str, Any]) -> str:
        provider = judge_profile.get("judge_provider")
        if isinstance(provider, str) and provider.strip():
            return provider

        judge_settings = getattr(self._judge, "_settings", None)
        judge_config = getattr(judge_settings, "judge", None)
        if getattr(self._judge, "_using_local_judge", False):
            fallback_provider = getattr(judge_config, "fallback_provider", "ollama")
            return str(fallback_provider).capitalize()
        configured_provider = getattr(judge_config, "provider", "gemini")
        return str(configured_provider).capitalize()

    def _judge_model(self, judge_profile: dict[str, Any]) -> str:
        model = judge_profile.get("judge_model")
        if isinstance(model, str) and model.strip():
            return model

        judge_settings = getattr(self._judge, "_settings", None)
        judge_config = getattr(judge_settings, "judge", None)
        if getattr(self._judge, "_using_local_judge", False):
            return str(getattr(judge_config, "fallback_model", "unknown"))
        return str(getattr(judge_config, "model", "unknown"))

    @staticmethod
    def _format_timing_line(label: str, elapsed: float, detail: str | None = None) -> str:
        value = f"{elapsed:.2f} s"
        if detail:
            value = f"{value} | {detail}"
        dots = "." * max(3, 18 - len(label))
        return f"{label:<18} {dots} {value}"

    def _print_sample_timing(
        self,
        sample_number: int,
        sample_total: int,
        timing_record: dict[str, Any],
    ) -> None:
        print("=" * 60)
        print()
        print(f"Sample {sample_number} / {sample_total}")
        print()
        print("=" * 60)
        print()
        print(self._format_timing_line("Query Rewrite", timing_record["query_rewrite"]["elapsed"]))
        print(
            self._format_timing_line(
                "Retrieve",
                timing_record["retrieval"]["elapsed"],
                f'{timing_record["retrieval"]["retrieved_chunk_count"]} chunks',
            )
        )
        print(
            self._format_timing_line(
                "Prompt Build",
                timing_record["prompt_build"]["elapsed"],
                f'{timing_record["prompt_build"]["prompt_length"]} chars',
            )
        )
        print(
            self._format_timing_line(
                "Generation",
                timing_record["generation"]["elapsed"],
                f'{timing_record["generation"]["generated_answer_length"]} chars',
            )
        )
        print(
            self._format_timing_line(
                "Judge",
                timing_record["judge"]["elapsed"],
                f'{timing_record["judge"]["provider"]} | {timing_record["judge"]["model"]}',
            )
        )
        print(self._format_timing_line("Exact Match", timing_record["exact_match"]["elapsed"]))
        print(self._format_timing_line("F1", timing_record["f1"]["elapsed"]))
        print(self._format_timing_line("BLEU", timing_record["bleu"]["elapsed"]))
        print(self._format_timing_line("ROUGE", timing_record["rouge"]["elapsed"]))
        print(self._format_timing_line("BERTScore", timing_record["bert_score"]["elapsed"]))
        print("-" * 44)
        print(self._format_timing_line("TOTAL", timing_record["total"]["elapsed"]))
        print("=" * 60)

    def _assemble_record(
        self,
        sample: EvaluationSample,
        judge_eval: Any,
        metrics: TraditionalMetricsResult,
        retrieval_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        final_verdict = getattr(judge_eval.final_verdict, "value", judge_eval.final_verdict)
        return {
            "sample_id": sample.sample_id,
            "question": sample.question,
            "rewritten_question": sample.rewritten_question,
            "answer": sample.generated_answer,
            "ground_truth": sample.ground_truth,
            "ground_truth_answers": sample.ground_truth_answers,
            "relevant_titles": sample.relevant_titles,
            "judge_evaluation": {
                **asdict(judge_eval),
                "final_verdict": final_verdict,
            },
            "metrics": asdict(metrics),
            "retrieval_metrics": retrieval_metrics,
            "retrieved_context": [
                {
                    "chunk_id": result.chunk.chunk_id,
                    "article_title": result.chunk.article_title,
                    "chunk_number": result.chunk.chunk_number,
                    "source_url": result.chunk.source_url,
                    "score": result.score,
                }
                for result in sample.retrieved_chunks
            ],
            "retrieval_metadata": sample.retrieval_metadata,
            "generation_metadata": sample.generation_metadata,
        }

    def _print_mode_banner(self, evaluation_mode: str, questions: int, reason: str) -> None:
        self._console.print("=" * 37)
        self._console.print("Evaluation Mode")
        self._console.print(evaluation_mode.capitalize())
        self._console.print("Questions")
        self._console.print(str(questions))
        self._console.print("=" * 37)
        self._console.print(reason)

    def _report_results(self, evaluation_mode: str, results: list[dict[str, Any]]) -> None:
        output_path = self._session_manager.session_root / "evaluation.json"
        payload = {
            "evaluation_mode": evaluation_mode,
            "questions_evaluated": len(results),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_name": str(self._dataset_path),
            "session_id": self._session_manager.session_root.name,
            "random_seed": self._evaluation_settings.random_seed,
            "results": results,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._logger.info("Saved evaluation results to %s", output_path)
        self._display_summary(results)

    def _display_summary(self, results: list[dict[str, Any]]) -> None:
        if not results:
            self._console.print("[red]No evaluation results to summarize.[/red]")
            return

        timing_records = self._sample_timing_records[: len(results)]
        if timing_records:
            self._print_timing_summary(timing_records)

        top_k = self._settings.retrieval.top_k
        retrieval_hit = self._average_nested_metrics(results, "retrieval_metrics", "hit_rate")
        retrieval_recall = self._average_nested_metrics(results, "retrieval_metrics", "recall")
        retrieval_mrr = self._average_nested_metrics(results, "retrieval_metrics", "mrr")

        alias_em = self._average_nested_metrics(results, "metrics", "alias_exact_match")
        alias_f1 = self._average_nested_metrics(results, "metrics", "alias_token_f1")
        bert_score = self._average_nested_metrics(results, "metrics", "bert_score")

        raw_em = self._average_nested_metrics(results, "metrics", "exact_match")
        raw_f1 = self._average_nested_metrics(results, "metrics", "token_f1")
        bleu = self._average_nested_metrics(results, "metrics", "bleu")
        rouge = self._average_nested_metrics(results, "metrics", "rouge_l")

        judge_correctness = self._average_nested_metrics(results, "judge_evaluation", "correctness")
        judge_groundedness = self._average_nested_metrics(results, "judge_evaluation", "groundedness")
        judge_completeness = self._average_nested_metrics(results, "judge_evaluation", "completeness")
        judge_hallucination = self._average_nested_metrics(results, "judge_evaluation", "hallucination")
        judge_supported = self._average_nested_metrics(results, "judge_evaluation", "supported_by_context")
        judge_matches = self._average_nested_metrics(results, "judge_evaluation", "matches_ground_truth")
        pass_count = sum(1 for item in results if item["judge_evaluation"]["final_verdict"] == "PASS")
        partial_count = sum(1 for item in results if item["judge_evaluation"]["final_verdict"] == "PARTIAL")
        fail_count = sum(1 for item in results if item["judge_evaluation"]["final_verdict"] == "FAIL")

        print("=" * 60)
        print()
        print("FINAL EVALUATION RESULTS")
        print()
        print("=" * 60)
        print()
        print(f"{'Samples Evaluated':<18} : {len(results)}")
        print()
        print("RETRIEVAL METRICS")
        print("--------------------------------")
        print(f"{f'Hit Rate@{top_k}':<18} : {self._format_summary_value(retrieval_hit, percent=True)}")
        print(f"{f'Recall@{top_k}':<18} : {self._format_summary_value(retrieval_recall, percent=True)}")
        print(f"{f'MRR@{top_k}':<18} : {self._format_summary_value(retrieval_mrr)}")
        print()
        print("ANSWER METRICS")
        print("--------------------------------")
        print(f"{'Alias Exact Match':<18} : {self._format_summary_value(alias_em, percent=True)}")
        print(f"{'Alias Token F1':<18} : {self._format_summary_value(alias_f1)}")
        print(f"{'BERTScore F1':<18} : {self._format_summary_value(bert_score)}")
        print()
        print("LEGACY METRICS")
        print("--------------------------------")
        print(f"{'Raw Exact Match':<18} : {self._format_summary_value(raw_em, percent=True)}")
        print(f"{'Raw Token F1':<18} : {self._format_summary_value(raw_f1)}")
        print(f"{'BLEU':<18} : {self._format_summary_value(bleu)}")
        print(f"{'ROUGE-L':<18} : {self._format_summary_value(rouge)}")
        print()
        print("LLM JUDGE")
        print("--------------------------------")
        judge_available_count = sum(1 for item in results if item.get("judge_evaluation", {}).get("judge_available", True))
        print(f"{'Judge Samples Available':<18} : {judge_available_count} / {len(results)}")
        print(f"{'Correctness':<18} : {self._format_summary_value(judge_correctness)}")
        print(f"{'Groundedness':<18} : {self._format_summary_value(judge_groundedness)}")
        print(f"{'Completeness':<18} : {self._format_summary_value(judge_completeness)}")
        print(f"{'Hallucination':<18} : {self._format_summary_value(judge_hallucination)}")
        print(f"{'Supported Context':<18} : {self._format_summary_value(judge_supported, percent=True)}")
        print(f"{'Matches GroundTruth':<18} : {self._format_summary_value(judge_matches, percent=True)}")
        print()
        print(f"{'PASS':<18} : {pass_count}")
        print(f"{'PARTIAL':<18} : {partial_count}")
        print(f"{'FAIL':<18} : {fail_count}")
        print()

    @staticmethod
    def _format_summary_value(value: float | None, percent: bool = False) -> str:
        if value is None:
            return "N/A"
        if percent:
            return f"{value * 100:.2f}%"
        return f"{value:.4f}"

    @staticmethod
    def _format_optional_float(value: float | None) -> str:
        if value is None:
            return "N/A"
        return f"{value:.4f}"

    @staticmethod
    def _format_optional_bool(value: bool | None) -> str:
        if value is None:
            return "N/A"
        return str(value)

    @staticmethod
    def _format_optional_text(value: Any) -> str:
        if value is None:
            return "N/A"
        if hasattr(value, "value"):
            return str(value.value)
        return str(value)

    @staticmethod
    def _average_nested_metrics(results: list[dict[str, Any]], *keys: str) -> float | None:
        values: list[float] = []
        for item in results:
            current: Any = item
            for key in keys:
                if not isinstance(current, dict):
                    current = None
                    break
                current = current.get(key)
            if current is None:
                continue
            try:
                values.append(float(current))
            except (TypeError, ValueError):
                continue
        if not values:
            return None
        return sum(values) / len(values)

    def _print_timing_summary(self, timing_records: list[dict[str, Any]]) -> None:
        stage_averages = self._stage_averages(timing_records)
        stage_labels = {
            "query_rewrite": "Query Rewrite",
            "retrieval": "Retrieval",
            "prompt_build": "Prompt Build",
            "generation": "Generation",
            "judge": "Judge",
            "exact_match": "Exact Match",
            "f1": "F1",
            "bleu": "BLEU",
            "rouge": "ROUGE",
            "bert_score": "BERTScore",
        }
        slowest_stage = max(stage_averages.items(), key=lambda item: item[1])
        fastest_stage = min(stage_averages.items(), key=lambda item: item[1])

        print("=" * 60)
        print()
        print("Evaluation Summary")
        print()
        print("=" * 60)
        print()
        print(f"{'Samples':<22} {len(timing_records)}")
        print(f"{'Average Generation Time':<22} {stage_averages['generation']:.2f} s")
        print(f"{'Average Judge Time':<22} {stage_averages['judge']:.2f} s")
        print(f"{'Average Retrieval Time':<22} {stage_averages['retrieval']:.2f} s")
        print(f"{'Average BERTScore Time':<22} {stage_averages['bert_score']:.2f} s")
        print(f"{'Average Total Sample Time':<22} {self._average_total_time(timing_records):.2f} s")
        print(f"{'Slowest Stage':<22} {stage_labels[slowest_stage[0]]} ({slowest_stage[1]:.2f} s)")
        print(f"{'Fastest Stage':<22} {stage_labels[fastest_stage[0]]} ({fastest_stage[1]:.2f} s)")
        print("=" * 60)

    def _stage_averages(self, timing_records: list[dict[str, Any]]) -> dict[str, float]:
        stage_keys = [
            "query_rewrite",
            "retrieval",
            "prompt_build",
            "generation",
            "judge",
            "exact_match",
            "f1",
            "bleu",
            "rouge",
            "bert_score",
        ]
        return {
            stage: sum(record[stage]["elapsed"] for record in timing_records) / len(timing_records)
            for stage in stage_keys
        }

    def _average_total_time(self, timing_records: list[dict[str, Any]]) -> float:
        return sum(record["total"]["elapsed"] for record in timing_records) / len(timing_records)
