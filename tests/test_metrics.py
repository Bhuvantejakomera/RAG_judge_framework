from __future__ import annotations

import json
import logging

import pytest

from src.config.settings import ConfigManager
from src.evaluation import metrics as metrics_module
from src.evaluation.metrics import TraditionalMetricsCalculator


class _DummyScore:
    def __init__(self, value: float) -> None:
        self._value = value

    def item(self) -> float:
        return self._value


def test_bert_score_model_defaults_to_roberta_base() -> None:
    ConfigManager().reload()
    assert ConfigManager().settings.evaluation.bert_score_model == "roberta-base"


def test_bert_score_uses_configured_model_and_prints_banner(monkeypatch, capsys) -> None:
    calls: list[str] = []

    def fake_bert_score(predictions, references, model_type, lang, rescale_with_baseline):  # noqa: ANN001
        calls.append(model_type)
        assert rescale_with_baseline is False
        return None, None, [_DummyScore(0.91)]

    monkeypatch.setattr(metrics_module, "bert_score", fake_bert_score)

    calculator = TraditionalMetricsCalculator()
    score = calculator.bert_score("prediction", "ground truth")

    output = capsys.readouterr().out
    assert "BERTScore Model : roberta-base" in output
    assert score == 0.91
    assert calls == ["roberta-base"]


def test_bert_score_failure_returns_json_null_and_continues(monkeypatch, caplog) -> None:
    def fake_bert_score(predictions, references, model_type, lang, rescale_with_baseline):  # noqa: ANN001
        assert rescale_with_baseline is False
        raise RuntimeError("OOM")

    monkeypatch.setattr(metrics_module, "bert_score", fake_bert_score)

    calculator = TraditionalMetricsCalculator()
    caplog.set_level(logging.ERROR)
    score = calculator.bert_score("prediction", "ground truth")

    payload = json.dumps({"bert_score": score})

    assert "BERTScore computation failed." in caplog.text
    assert payload == '{"bert_score": null}'
    assert sum([score, score], 0) == 0.0


@pytest.mark.parametrize(
    ("references", "prediction", "expected_em", "expected_f1", "expected_extracted"),
    [
        (["Poland"], "Poland", 1.0, 1.0, "Poland"),
        (["Poland"], "Wilcza Jama is located in Poland.", 1.0, 1.0, "Poland"),
        (["Poland"], "Poland-based organization", 0.0, pytest.approx(0.5), None),
        (["slasher"], "fantasy slasher", 0.0, pytest.approx(2 / 3), None),
        (["Ralph Waldo Emerson"], "Ralph Walco Emerson", 0.0, pytest.approx(2 / 3), None),
        (["Poland"], "Germany", 0.0, 0.0, None),
        (["Harold Pinter"], "Moira Buffini is the author of Silence.", 0.0, 0.0, None),
        (
            ["Tony Bennett", "Anthony Dominick Benedetto", "Anthony Benedetto"],
            "Tony Bennett",
            1.0,
            1.0,
            "Tony Bennett",
        ),
        (
            ["Tony Bennett", "Anthony Dominick Benedetto", "Anthony Benedetto"],
            "The author of New Keywords is Tony Bennett.",
            1.0,
            1.0,
            "Tony Bennett",
        ),
        (
            ["horror film", "horror movie", "slasher film", "slasher movie", "slasher"],
            "The genre of the movie \"Killjoy\" is fantasy slasher.",
            0.0,
            pytest.approx(0.4),
            None,
        ),
        (
            ["Poland"],
            "Wilcza Jama is located in Poland, within Sokółka County, Podlaskie Voivodeship.",
            1.0,
            1.0,
            "Poland",
        ),
        (
            ["Paul Simon", "Paul Frederic Simon", "Jerry Landis"],
            "Paul Simon is the composer of America.",
            1.0,
            1.0,
            "Paul Simon",
        ),
    ],
)
def test_alias_metrics_are_conservative(
    references: list[str],
    prediction: str,
    expected_em: float,
    expected_f1: object,
    expected_extracted: str | None,
) -> None:
    calculator = TraditionalMetricsCalculator()

    analysis = calculator.analyze_alias_prediction(prediction, references)
    exact_match = calculator.alias_exact_match(prediction, references)
    token_f1 = calculator.alias_token_f1(prediction, references)

    assert exact_match == expected_em
    assert token_f1 == expected_f1
    assert analysis.raw_prediction == prediction
    assert analysis.references == tuple(references)
    assert analysis.normalized_prediction

    if expected_extracted is None:
        assert analysis.extracted_answer is None
    else:
        assert analysis.extracted_answer == expected_extracted
