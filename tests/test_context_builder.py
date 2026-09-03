import numpy as np
from pathlib import Path

from src.audio_analysis.model_runtime import LabelPrediction
from src.context.context_builder import build_musical_context


def test_build_musical_context_reuses_embeddings_and_serializes_predictions(monkeypatch):
    models = object()
    embeddings = np.ones((2, 1280))
    calls = []

    monkeypatch.setattr(
        "src.context.context_builder.prepare_tagging_embeddings",
        lambda audio_path, *, models: calls.append((audio_path, models)) or embeddings,
    )
    monkeypatch.setattr(
        "src.context.context_builder.analyze_genre",
        lambda received_embeddings, **_: [LabelPrediction("rock", 0.2607)]
        if received_embeddings is embeddings
        else [],
    )
    monkeypatch.setattr(
        "src.context.context_builder.analyze_instruments",
        lambda received_embeddings, **_: [LabelPrediction("guitar", 0.5935)]
        if received_embeddings is embeddings
        else [],
    )

    context = build_musical_context("data/guitar_sample.mp3", models=models)

    assert calls == [(Path("data/guitar_sample.mp3"), models)]
    assert context["genre"] == [{"label": "rock", "confidence": 0.2607}]
    assert context["instruments"] == [{"label": "guitar", "confidence": 0.5935}]
    assert context["audio"] == {"filename": "guitar_sample.mp3"}
    assert context["structure"] == {"sections": []}


def test_build_musical_context_loads_models_when_not_provided(monkeypatch):
    models = object()
    monkeypatch.setattr("src.context.context_builder.get_audio_tagging_models", lambda: models)
    monkeypatch.setattr(
        "src.context.context_builder.prepare_tagging_embeddings",
        lambda audio_path, *, models: np.ones((1, 1280)),
    )
    monkeypatch.setattr("src.context.context_builder.analyze_genre", lambda *args, **kwargs: [])
    monkeypatch.setattr("src.context.context_builder.analyze_instruments", lambda *args, **kwargs: [])

    context = build_musical_context("demo.wav")

    assert context["genre"] == []
    assert context["instruments"] == []