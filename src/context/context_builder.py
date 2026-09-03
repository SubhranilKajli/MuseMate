"""Build the shared musical context from available audio-analysis outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.audio_analysis import (
	AudioTaggingModels,
	analyze_genre,
	analyze_instruments,
	get_audio_tagging_models,
	prepare_tagging_embeddings,
)
from src.audio_analysis.key import analyze_key
from src.audio_analysis.tempo import analyze_tempo


MusicalContext = dict[str, Any]


def _serialize_predictions(predictions: list[Any]) -> list[dict[str, Any]]:
	"""Convert model prediction objects at the context boundary."""
	return [
		{"label": prediction.label, "confidence": float(prediction.confidence)}
		for prediction in predictions
	]


def build_musical_context(
	audio_path: str | Path,
	*,
	models: AudioTaggingModels | None = None,
	genre_threshold: float = 0.35,
	genre_max_labels: int = 5,
	instrument_threshold: float = 0.35,
	instrument_max_labels: int = 10,
) -> MusicalContext:
	"""Analyze one audio file and assemble its current MusicalContext.

	Genre and instrument classifiers consume the same Discogs-EffNet embeddings.
	Other context sections remain placeholders until their MIR modules expose
	analysis functions.
	"""
	path = Path(audio_path)
	tagging_models = models if models is not None else get_audio_tagging_models()
	embeddings = prepare_tagging_embeddings(path, models=tagging_models)
	tempo = analyze_tempo(audio_path)
	key = analyze_key(audio_path)

	return {
		"schema_version": "1.0",
		"audio": {"filename": path.name},
		"tempo": {
			"bpm": {
				"value": tempo.bpm,
				"confidence": tempo.confidence,
			}
		},
		"tonality": {
			"key": {
				"value": key.key,
				"mode": key.scale,
				"confidence": key.confidence,
			}
		},
		"rhythm": None,
		"structure": {"sections": []},
		"instruments": _serialize_predictions(
			analyze_instruments(
				embeddings,
				models=tagging_models,
				threshold=instrument_threshold,
				max_labels=instrument_max_labels,
			)
		),
		"genre": _serialize_predictions(
			analyze_genre(
				embeddings,
				models=tagging_models,
				threshold=genre_threshold,
				max_labels=genre_max_labels,
			)
		),
	}
