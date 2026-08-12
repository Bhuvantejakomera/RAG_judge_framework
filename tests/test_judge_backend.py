from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.config.settings import DEFAULT_GEMINI_JUDGE_MODEL, ConfigManager
from src.evaluation import judge as judge_module
from src.evaluation.judge import OpenAIJudge
from src.models.data_models import DocumentChunk, EvaluationSample, RetrievalResult, Verdict


def _judge_payload(verdict: str = "PASS") -> str:
    return json.dumps(
        {
            "groundedness": 1.0,
            "correctness": 1.0,
            "completeness": 1.0,
            "fluency": 1.0,
            "hallucination": 0.0,
            "supported_by_context": True,
            "matches_ground_truth": True,
            "reasoning": "Looks good.",
            "final_verdict": verdict,
        }
    )


def _judge_payload_missing_final_verdict() -> str:
    payload = json.loads(_judge_payload())
    payload.pop("final_verdict", None)
    return json.dumps(payload)


def _local_stream_response(payload: str, prompt_eval_count: int = 123, eval_count: int = 27):
    def _stream():  # noqa: ANN202 - simple streamed test double
        yield SimpleNamespace(
            response=payload,
            prompt_eval_count=prompt_eval_count,
            eval_count=eval_count,
            total_duration=1.0,
            load_duration=0.1,
            prompt_eval_duration=0.2,
            eval_duration=0.7,
        )

    return _stream()


def test_judge_config_is_loaded() -> None:
    ConfigManager().reload()
    settings = ConfigManager().settings

    assert settings.judge.provider == "gemini"
    assert settings.judge.model == DEFAULT_GEMINI_JUDGE_MODEL
    assert settings.judge.fallback_provider == "ollama"
    assert settings.judge.fallback_model == "qwen3:8b"
    assert settings.judge.temperature == 0.0
    assert settings.judge.max_tokens == 1024


def test_local_judge_prompt_is_compact_and_deduplicated(monkeypatch) -> None:
    monkeypatch.setattr(OpenAIJudge, "_print_startup_banner", lambda self: None)
    monkeypatch.setattr(OpenAIJudge, "_ensure_local_judge_ready", lambda self: None)
    judge = OpenAIJudge()
    judge._using_local_judge = True
    judge._local_model_ready = True

    captured: dict[str, object] = {}

    def fake_generate(**kwargs):  # noqa: ANN001 - simple streamed test double
        captured.update(kwargs)
        return _local_stream_response(_judge_payload(), prompt_eval_count=91, eval_count=24)

    monkeypatch.setattr(judge_module.ollama, "generate", fake_generate)

    duplicate_sentence = "Shared evidence sentence."
    chunk_one = DocumentChunk(
        chunk_id="chunk-1",
        article_title="Example Article",
        chunk_number=1,
        text=(
            f"{duplicate_sentence} Unique alpha detail. "
            "Additional context that should be trimmed when the prompt is compact."
        ),
        source_url="https://example.com/1",
        token_count=64,
    )
    chunk_two = DocumentChunk(
        chunk_id="chunk-2",
        article_title="Example Article",
        chunk_number=2,
        text=(
            f"{duplicate_sentence} Unique beta detail. "
            "More supporting context that should not bloat the prompt."
        ),
        source_url="https://example.com/2",
        token_count=64,
    )
    sample = EvaluationSample(
        sample_id="sample-compact",
        question="Who leads the example group?",
        rewritten_question="Who leads the example group?",
        ground_truth="The example group is led by Alice Example.",
        generated_answer="Alice Example leads the group.",
        retrieved_chunks=[
            RetrievalResult(chunk=chunk_one, score=0.98),
            RetrievalResult(chunk=chunk_two, score=0.97),
        ],
    )

    evaluation = judge.evaluate_sample(sample)
    prompt = str(captured["prompt"])

    assert evaluation.final_verdict == Verdict.PASS
    assert captured["system"] == judge.LOCAL_SYSTEM_PROMPT
    assert captured["format"] == "json"
    assert captured["stream"] is True
    assert captured["options"] == {"temperature": 0.0, "num_predict": judge.LOCAL_JUDGE_MAX_PREDICT}
    assert "Q: Who leads the example group?" in prompt
    assert "A: Alice Example leads the group." in prompt
    assert "GT: The example group is led by Alice Example." in prompt
    assert "CTX:" in prompt
    assert prompt.count(duplicate_sentence) == 1
    assert judge._estimate_token_count(prompt) < judge.LOCAL_PROMPT_TOKEN_TARGET


def test_judge_prints_startup_banner(capsys) -> None:
    judge = OpenAIJudge()

    output = capsys.readouterr().out
    assert "Judge Provider : Gemini" in output
    assert f"Judge Model    : {DEFAULT_GEMINI_JUDGE_MODEL}" in output
    assert judge._using_local_judge is False


def test_gemini_success_path_uses_primary_judge(monkeypatch) -> None:
    monkeypatch.setattr(OpenAIJudge, "_print_startup_banner", lambda self: None)
    judge = OpenAIJudge()
    monkeypatch.setattr(judge, "_request_gemini_completion", lambda prompt, use_response_format: _judge_payload())

    evaluation = judge.evaluate(
        question="What is the answer?",
        answer="The answer is 42.",
        ground_truth="42",
        context="Relevant context.",
    )

    assert evaluation.final_verdict == Verdict.PASS
    assert evaluation.correctness == 1.0
    assert judge._using_local_judge is False


def test_gemini_quota_exceeded_switches_to_local_judge(monkeypatch, capsys) -> None:
    monkeypatch.setattr(OpenAIJudge, "_print_startup_banner", lambda self: None)
    judge = OpenAIJudge()

    def fake_gemini_completion(prompt: str, use_response_format: bool) -> str:  # noqa: ARG001
        raise judge_module._GeminiAPIError("Quota exceeded", status_code=429, error_code="quota_exceeded")

    monkeypatch.setattr(judge, "_request_gemini_completion", fake_gemini_completion)
    monkeypatch.setattr(
        judge_module.ollama,
        "list",
        lambda: SimpleNamespace(models=[SimpleNamespace(model="qwen3:8b")]),
    )

    pull_calls: list[str] = []
    monkeypatch.setattr(judge_module.ollama, "pull", lambda model, stream=False: pull_calls.append(model))
    monkeypatch.setattr(judge_module.ollama, "generate", lambda **kwargs: _local_stream_response(_judge_payload()))

    evaluation = judge.evaluate(
        question="What is the answer?",
        answer="The answer is 42.",
        ground_truth="42",
        context="Relevant context.",
    )

    output = capsys.readouterr().out
    assert "Gemini has no credits." in output
    assert "Switching to local judge..." in output
    assert "Judge Provider : Ollama" in output
    assert "Judge Model    : qwen3:8b" in output
    assert "Local judge not found." not in output
    assert pull_calls == []
    assert evaluation.final_verdict == Verdict.PASS
    assert judge._using_local_judge is True


def test_missing_local_model_is_pulled_once(monkeypatch, capsys) -> None:
    monkeypatch.setattr(OpenAIJudge, "_print_startup_banner", lambda self: None)
    judge = OpenAIJudge()

    def fake_gemini_completion(prompt: str, use_response_format: bool) -> str:  # noqa: ARG001
        raise judge_module._GeminiAPIError("Quota exceeded", status_code=429, error_code="quota_exceeded")

    monkeypatch.setattr(judge, "_request_gemini_completion", fake_gemini_completion)
    monkeypatch.setattr(judge_module.ollama, "list", lambda: SimpleNamespace(models=[]))

    pull_calls: list[str] = []
    pull_kwargs: list[dict[str, object]] = []

    def fake_pull(model: str, stream: bool = False):  # noqa: ANN001 - simple streamed test double
        pull_calls.append(model)
        pull_kwargs.append({"model": model, "stream": stream})
        yield SimpleNamespace(status="pulling manifest", completed=0, total=3)
        yield SimpleNamespace(status="downloading", completed=1, total=3)
        yield SimpleNamespace(status="success", completed=3, total=3)

    class _FakeProgress:
        instances: list["_FakeProgress"] = []

        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003 - simple test double
            self.added_tasks: list[tuple[str, int]] = []
            self.updates: list[tuple[int, dict[str, object]]] = []
            self.__class__.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001, ANN002, ANN003 - simple test double
            return None

        def add_task(self, description: str, total: int = 0) -> int:
            self.added_tasks.append((description, total))
            return 1

        def update(self, task_id: int, **kwargs) -> None:
            self.updates.append((task_id, kwargs))

    monkeypatch.setattr(judge_module.ollama, "pull", fake_pull)
    monkeypatch.setattr(judge_module, "Progress", _FakeProgress)
    monkeypatch.setattr(judge_module.ollama, "generate", lambda **kwargs: _local_stream_response(_judge_payload()))

    evaluation = judge.evaluate(
        question="What is the answer?",
        answer="The answer is 42.",
        ground_truth="42",
        context="Relevant context.",
    )

    output = capsys.readouterr().out
    assert "Local judge not found." in output
    assert "Downloading qwen3:8b..." in output
    assert "Local judge ready." in output
    assert pull_calls == ["qwen3:8b"]
    assert pull_kwargs == [{"model": "qwen3:8b", "stream": True}]
    assert len(_FakeProgress.instances) == 1
    assert _FakeProgress.instances[0].added_tasks == [("Downloading qwen3:8b", 1)]
    assert any(update[1].get("description", "").startswith("Downloading qwen3:8b - ") for update in _FakeProgress.instances[0].updates)
    assert any(update[1].get("total") == 3 for update in _FakeProgress.instances[0].updates)
    assert any(update[1].get("completed") == 3 for update in _FakeProgress.instances[0].updates)
    assert evaluation.final_verdict == Verdict.PASS
    assert judge._using_local_judge is True


@pytest.mark.parametrize(
    "payload",
    [
        _judge_payload_missing_final_verdict(),
        "{\"groundedness\": 1.0, \"correctness\": 1.0",
    ],
)
def test_local_judge_invalid_json_returns_unavailable(monkeypatch, payload: str) -> None:
    monkeypatch.setattr(OpenAIJudge, "_print_startup_banner", lambda self: None)
    monkeypatch.setattr(OpenAIJudge, "_ensure_local_judge_ready", lambda self: None)
    judge = OpenAIJudge()
    judge._using_local_judge = True
    judge._local_model_ready = True

    monkeypatch.setattr(judge_module.ollama, "generate", lambda **kwargs: _local_stream_response(payload))

    evaluation = judge.evaluate(
        question="What is the answer?",
        answer="The answer is 42.",
        ground_truth="42",
        context="Relevant context.",
    )

    assert evaluation.judge_available is False
    assert evaluation.final_verdict is None
    assert evaluation.correctness is None
    assert evaluation.groundedness is None
    assert evaluation.completeness is None
    assert evaluation.fluency is None
    assert evaluation.hallucination is None
    assert evaluation.supported_by_context is None
    assert evaluation.matches_ground_truth is None
    assert evaluation.reasoning is None


def test_gemini_and_ollama_failure_returns_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(OpenAIJudge, "_print_startup_banner", lambda self: None)
    judge = OpenAIJudge()

    def fake_gemini_completion(prompt: str, use_response_format: bool) -> str:  # noqa: ARG001
        raise judge_module._GeminiAPIError("Quota exceeded", status_code=429, error_code="quota_exceeded")

    monkeypatch.setattr(judge, "_request_gemini_completion", fake_gemini_completion)
    monkeypatch.setattr(OpenAIJudge, "_ensure_local_judge_ready", lambda self: None)
    monkeypatch.setattr(judge_module.ollama, "generate", lambda **kwargs: _local_stream_response(_judge_payload_missing_final_verdict()))

    evaluation = judge.evaluate(
        question="What is the answer?",
        answer="The answer is 42.",
        ground_truth="42",
        context="Relevant context.",
    )

    assert evaluation.judge_available is False
    assert evaluation.final_verdict is None
