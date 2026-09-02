import numpy as np
import pytest

from src.audio_analysis.genre import analyze_genre
from src.audio_analysis.model_runtime import LabelPrediction


def test_analyze_genre_returns_only_predictions_above_threshold(monkeypatch):
    monkeypatch.setattr(
        "src.audio_analysis.genre.predict_genre_probabilities",
        lambda embeddings, *, models: {"rock": 0.62, "jazz": 0.93, "pop": 0.35, "noise": 0.2},
    )

    result = analyze_genre(np.ones((2, 200)), models=object(), threshold=0.35, max_labels=2)

    assert result == [LabelPrediction("jazz", 0.93), LabelPrediction("rock", 0.62)]


def test_analyze_genre_returns_top_three_ranked_fallback_when_none_pass_threshold(monkeypatch):
    monkeypatch.setattr(
        "src.audio_analysis.genre.predict_genre_probabilities",
        lambda embeddings, *, models: {
            "rock": 0.2607,
            "pop": 0.2450,
            "easylistening": 0.1398,
            "ambient": 0.1160,
        },
    )

    assert analyze_genre(np.ones((1, 200)), models=object(), threshold=0.35) == [
        LabelPrediction("rock", 0.2607),
        LabelPrediction("pop", 0.2450),
        LabelPrediction("easylistening", 0.1398),
    ]


def test_analyze_genre_limits_threshold_predictions_to_max_labels(monkeypatch):
    monkeypatch.setattr(
        "src.audio_analysis.genre.predict_genre_probabilities",
        lambda embeddings, *, models: {"first": 0.9, "second": 0.8, "third": 0.7},
    )

    assert analyze_genre(np.ones((1, 200)), models=object(), max_labels=2) == [
        LabelPrediction("first", 0.9),
        LabelPrediction("second", 0.8),
    ]


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_analyze_genre_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        analyze_genre(np.ones(1), models=object(), threshold=threshold)


def test_analyze_genre_rejects_empty_embeddings():
    with pytest.raises(ValueError, match="embeddings"):
        analyze_genre(np.array([]), models=object())
