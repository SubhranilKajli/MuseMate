"""Multilabel instrument analysis using the pretrained MTG-Jamendo classifier."""

from __future__ import annotations

import numpy as np

from .model_runtime import AudioTaggingModels, LabelPrediction, predict_instrument_probabilities


def analyze_instruments(
    embeddings: np.ndarray,
    *,
    models: AudioTaggingModels,
    threshold: float = 0.35,
    max_labels: int = 10,
) -> list[LabelPrediction]:
    """Return ranked simultaneous instrument labels from the trained model."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0")
    if max_labels < 1:
        raise ValueError("max_labels must be at least 1")
    if np.asarray(embeddings).size == 0:
        raise ValueError("embeddings must not be empty")

    probabilities = predict_instrument_probabilities(embeddings, models=models)
    predictions = [
        LabelPrediction(label=label, confidence=confidence)
        for label, confidence in probabilities.items()
        if threshold <= confidence <= 1.0
    ]
    return sorted(predictions, key=lambda item: item.confidence, reverse=True)[:max_labels]
