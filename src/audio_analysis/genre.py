"""Genre analysis using the pretrained MTG-Jamendo classifier."""

from __future__ import annotations

import numpy as np

from .model_runtime import AudioTaggingModels, LabelPrediction, predict_genre_probabilities


def analyze_genre(
    embeddings: np.ndarray,
    *,
    models: AudioTaggingModels,
    threshold: float = 0.35,
    max_labels: int = 5,
) -> list[LabelPrediction]:
    """Return ranked genre labels produced by the trained MTG-Jamendo model."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0.0 and 1.0")
    if max_labels < 1:
        raise ValueError("max_labels must be at least 1")
    if np.asarray(embeddings).size == 0:
        raise ValueError("embeddings must not be empty")

    probabilities = predict_genre_probabilities(embeddings, models=models)
    ranked_predictions = sorted(
        (LabelPrediction(label=label, confidence=confidence) for label, confidence in probabilities.items()),
        key=lambda item: item.confidence,
        reverse=True,
    )
    threshold_predictions = [
        prediction for prediction in ranked_predictions if prediction.confidence >= threshold
    ]
    if threshold_predictions:
        return threshold_predictions[:max_labels]
    return ranked_predictions[:3]
