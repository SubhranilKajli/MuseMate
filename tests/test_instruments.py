import numpy as np
import pytest

from src.audio_analysis.instruments import analyze_instruments
from src.audio_analysis.model_runtime import LabelPrediction


def test_analyze_instruments_returns_multiple_ranked_labels(monkeypatch):
    monkeypatch.setattr(
        "src.audio_analysis.instruments.predict_instrument_probabilities",
        lambda embeddings, *, models: {"drums": 0.84, "piano": 0.78, "voice": 0.42, "flute": 0.1},
    )

    result = analyze_instruments(np.ones((2, 200)), models=object(), threshold=0.4, max_labels=3)

    assert result == [
        LabelPrediction("drums", 0.84),
        LabelPrediction("piano", 0.78),
        LabelPrediction("voice", 0.42),
    ]


def test_analyze_instruments_honours_max_labels(monkeypatch):
    monkeypatch.setattr(
        "src.audio_analysis.instruments.predict_instrument_probabilities",
        lambda embeddings, *, models: {"drums": 0.9, "piano": 0.8},
    )

    assert analyze_instruments(np.ones(1), models=object(), max_labels=1) == [
        LabelPrediction("drums", 0.9)
    ]


def test_analyze_instruments_returns_empty_when_no_score_passes_threshold(monkeypatch):
    monkeypatch.setattr(
        "src.audio_analysis.instruments.predict_instrument_probabilities",
        lambda embeddings, *, models: {"guitar": 0.2, "piano": 0.1},
    )

    assert analyze_instruments(np.ones((1, 200)), models=object(), threshold=0.35) == []


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_analyze_instruments_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        analyze_instruments(np.ones(1), models=object(), threshold=threshold)


def test_analyze_instruments_rejects_invalid_max_labels():
    with pytest.raises(ValueError, match="max_labels"):
        analyze_instruments(np.ones(1), models=object(), max_labels=0)


def test_analyze_instruments_rejects_empty_embeddings():
    with pytest.raises(ValueError, match="embeddings"):
        analyze_instruments(np.array([]), models=object())
