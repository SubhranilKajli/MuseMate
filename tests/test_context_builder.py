import numpy as np
from pathlib import Path

from src.audio_analysis.model_runtime import LabelPrediction
from src.audio_analysis.key import KeyPrediction
from src.audio_analysis.tempo import TempoPrediction
from src.context.context_builder import build_musical_context


def test_build_musical_context_reuses_embeddings_and_serializes_predictions(monkeypatch):
    models = object()
    embeddings = np.ones((2, 1280))
    calls = []
    tempo_paths = []
    key_paths = []

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
    monkeypatch.setattr(
        "src.context.context_builder.analyze_tempo",
        lambda audio_path: tempo_paths.append(audio_path) or TempoPrediction(128.0, 0.91),
    )
    monkeypatch.setattr(
        "src.context.context_builder.analyze_key",
        lambda audio_path: key_paths.append(audio_path) or KeyPrediction("C", "major", 0.82),
    )

    context = build_musical_context("data/guitar_sample.mp3", models=models)

    assert calls == [(Path("data/guitar_sample.mp3"), models)]
    assert tempo_paths == ["data/guitar_sample.mp3"]
    assert key_paths == ["data/guitar_sample.mp3"]
    assert context["genre"] == [{"label": "rock", "confidence": 0.2607}]
    assert context["instruments"] == [{"label": "guitar", "confidence": 0.5935}]
    assert context["tempo"] == {"bpm": {"value": 128.0, "confidence": 0.91}}
    assert context["tonality"] == {
        "key": {"value": "C", "mode": "major", "confidence": 0.82}
    }
    assert context["audio"] == {"filename": "guitar_sample.mp3"}
    assert context["rhythm"] is None
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
    monkeypatch.setattr(
        "src.context.context_builder.analyze_tempo",
        lambda audio_path: TempoPrediction(128.0, 0.91),
    )
    monkeypatch.setattr(
        "src.context.context_builder.analyze_key",
        lambda audio_path: KeyPrediction("C", "major", 0.82),
    )

    context = build_musical_context("demo.wav")

    assert context["genre"] == []
    assert context["instruments"] == []