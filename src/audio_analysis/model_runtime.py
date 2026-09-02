"""Shared Essentia runtime for MuseMate's pretrained audio-tagging models.

This module owns audio preprocessing and model lifecycle.  Feature modules
receive reusable Discogs-EffNet embeddings and never load models themselves.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


MODEL_DIRECTORY_ENV = "MUSEMATE_MODEL_DIR"
SAMPLE_RATE = 16_000
MIN_AUDIO_SECONDS = 3.0
SUPPORTED_AUDIO_SUFFIXES = frozenset({".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"})

EMBEDDING_MODEL_FILENAME = "discogs-effnet-bs64-1.pb"
GENRE_MODEL_FILENAME = "mtg_jamendo_genre-discogs-effnet-1.pb"
INSTRUMENT_MODEL_FILENAME = "mtg_jamendo_instrument-discogs-effnet-1.pb"
GENRE_METADATA_FILENAME = "mtg_jamendo_genre-discogs-effnet-1.json"
INSTRUMENT_METADATA_FILENAME = "mtg_jamendo_instrument-discogs-effnet-1.json"
EMBEDDING_DIMENSIONS = 1280


class AudioTaggingError(RuntimeError):
    """Base exception for the audio-tagging runtime."""


class AudioInputError(AudioTaggingError):
    """Raised when an audio file cannot safely be analysed."""


class ModelConfigurationError(AudioTaggingError):
    """Raised when required pretrained model assets are unavailable or invalid."""


@dataclass(frozen=True)
class LabelPrediction:
    """One ranked multilabel prediction from a trained classifier."""

    label: str
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")
        if not isinstance(self.confidence, (float, int)) or isinstance(self.confidence, bool):
            raise TypeError("confidence must be a number")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")


@dataclass(frozen=True)
class AudioTaggingModels:
    """Cached Essentia model instances and their ordered output vocabularies."""

    embedding_model: Any
    genre_model: Any
    instrument_model: Any
    genre_labels: tuple[str, ...]
    instrument_labels: tuple[str, ...]


def _model_directory() -> Path:
    configured_path = os.getenv(MODEL_DIRECTORY_ENV)
    if not configured_path:
        raise ModelConfigurationError(
            f"{MODEL_DIRECTORY_ENV} is not set. Point it at the directory containing "
            "the downloaded Essentia .pb and .json model assets."
        )
    model_directory = Path(configured_path).expanduser()
    if not model_directory.is_dir():
        raise ModelConfigurationError(f"Model directory does not exist: {model_directory}")
    return model_directory


def _require_file(model_directory: Path, filename: str) -> Path:
    path = model_directory / filename
    if not path.is_file():
        raise ModelConfigurationError(f"Required model asset is missing: {path}")
    return path


def _read_labels(metadata_path: Path) -> tuple[str, ...]:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelConfigurationError(f"Cannot read model metadata: {metadata_path}") from exc

    for key in ("classes", "labels", "class_names"):
        labels = metadata.get(key)
        if isinstance(labels, list) and all(isinstance(label, str) for label in labels):
            return tuple(labels)
    raise ModelConfigurationError(
        f"Model metadata {metadata_path} has no supported label list "
        "('classes', 'labels', or 'class_names')."
    )


@lru_cache(maxsize=1)
def get_audio_tagging_models() -> AudioTaggingModels:
    """Load and cache Essentia models once for the Python process."""
    model_directory = _model_directory()
    embedding_path = _require_file(model_directory, EMBEDDING_MODEL_FILENAME)
    genre_path = _require_file(model_directory, GENRE_MODEL_FILENAME)
    instrument_path = _require_file(model_directory, INSTRUMENT_MODEL_FILENAME)
    genre_metadata_path = _require_file(model_directory, GENRE_METADATA_FILENAME)
    instrument_metadata_path = _require_file(model_directory, INSTRUMENT_METADATA_FILENAME)

    try:
        from essentia.standard import TensorflowPredict2D, TensorflowPredictEffnetDiscogs
    except ImportError as exc:
        raise ModelConfigurationError(
            "Essentia TensorFlow support is required. Install the project's dependencies "
            "in a supported Python environment before running inference."
        ) from exc

    try:
        return AudioTaggingModels(
            embedding_model=TensorflowPredictEffnetDiscogs(
                graphFilename=str(embedding_path), output="PartitionedCall:1"
            ),
            genre_model=TensorflowPredict2D(
                graphFilename=str(genre_path), output="model/Sigmoid"
            ),
            instrument_model=TensorflowPredict2D(
                graphFilename=str(instrument_path), output="model/Sigmoid"
            ),
            genre_labels=_read_labels(genre_metadata_path),
            instrument_labels=_read_labels(instrument_metadata_path),
        )
    except Exception as exc:  # Essentia raises implementation-specific errors.
        raise ModelConfigurationError("Unable to initialise the Essentia model graphs.") from exc


def load_audio_for_tagging(audio_path: str | Path) -> np.ndarray:
    """Decode supported audio to a non-empty mono 16 kHz floating-point waveform."""
    path = Path(audio_path)
    if not path.is_file():
        raise AudioInputError(f"Audio file does not exist or is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_SUFFIXES))
        raise AudioInputError(f"Unsupported audio format '{path.suffix}'. Supported formats: {supported}")

    try:
        from essentia.standard import MonoLoader
        audio = np.asarray(MonoLoader(filename=str(path), sampleRate=SAMPLE_RATE, resampleQuality=4)())
    except ImportError as exc:
        raise ModelConfigurationError("Essentia is required to decode audio for tagging.") from exc
    except Exception as exc:
        raise AudioInputError(f"Could not decode audio file: {path}") from exc

    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise AudioInputError(f"Audio file contains no valid mono samples: {path}")
    if audio.size < int(SAMPLE_RATE * MIN_AUDIO_SECONDS):
        raise AudioInputError(
            f"Audio is too short for reliable tagging ({audio.size / SAMPLE_RATE:.2f}s). "
            f"At least {MIN_AUDIO_SECONDS:.1f}s is required."
        )
    return audio.astype(np.float32, copy=False)


def extract_discogs_effnet_embeddings(
    audio: np.ndarray, *, models: AudioTaggingModels
) -> np.ndarray:
    """Extract reusable Discogs-EffNet embeddings from a preprocessed waveform."""
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim != 1 or waveform.size < int(SAMPLE_RATE * MIN_AUDIO_SECONDS):
        raise AudioInputError("audio must be a one-dimensional 16 kHz waveform of at least 3 seconds")
    if not np.isfinite(waveform).all():
        raise AudioInputError("audio contains non-finite samples")
    try:
        embeddings = np.asarray(models.embedding_model(waveform))
    except Exception as exc:
        raise AudioTaggingError("Discogs-EffNet embedding extraction failed.") from exc
    _validate_embeddings(embeddings)
    return embeddings


def prepare_tagging_embeddings(audio_path: str | Path, *, models: AudioTaggingModels) -> np.ndarray:
    """Load/preprocess one audio file and extract embeddings exactly once."""
    return extract_discogs_effnet_embeddings(load_audio_for_tagging(audio_path), models=models)


def _validate_embeddings(embeddings: np.ndarray) -> None:
    if embeddings.ndim != 2 or embeddings.shape[0] == 0 or embeddings.shape[1] != EMBEDDING_DIMENSIONS:
        raise AudioTaggingError(
            "Discogs-EffNet embeddings must have shape "
            f"(timestamps, {EMBEDDING_DIMENSIONS}); received {embeddings.shape}."
        )
    if not np.isfinite(embeddings).all():
        raise AudioTaggingError("Discogs-EffNet embeddings contain non-finite values.")


def _probabilities_for_head(embeddings: np.ndarray, model: Any, labels: tuple[str, ...], task: str) -> dict[str, float]:
    if not labels:
        raise ModelConfigurationError(f"{task} model has an empty label vocabulary")
    embedding_matrix = np.asarray(embeddings, dtype=np.float32)
    _validate_embeddings(embedding_matrix)
    try:
        frame_predictions = np.asarray(model(embedding_matrix), dtype=float)
    except Exception as exc:
        raise AudioTaggingError(f"{task} classifier inference failed.") from exc
    if frame_predictions.ndim != 2 or frame_predictions.shape[0] == 0:
        raise AudioTaggingError(
            f"{task} model must return a 2D matrix shaped "
            "(timestamps, labels); "
            f"received {frame_predictions.shape}."
        )
    if frame_predictions.shape[1] != len(labels):
        raise ModelConfigurationError(
            f"{task} model returned {frame_predictions.shape[1]} scores per timestamp "
            f"for {len(labels)} labels."
        )
    probabilities = np.mean(frame_predictions, axis=0)
    if not np.isfinite(probabilities).all():
        raise AudioTaggingError(f"{task} model returned non-finite probabilities.")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise AudioTaggingError(f"{task} model returned values outside the probability range.")
    return {label: float(score) for label, score in zip(labels, probabilities, strict=True)}


def predict_genre_probabilities(embeddings: np.ndarray, *, models: AudioTaggingModels) -> dict[str, float]:
    """Run the MTG-Jamendo genre head and return its raw probabilities."""
    return _probabilities_for_head(embeddings, models.genre_model, models.genre_labels, "Genre")


def predict_instrument_probabilities(embeddings: np.ndarray, *, models: AudioTaggingModels) -> dict[str, float]:
    """Run the MTG-Jamendo instrument head and return its raw probabilities."""
    return _probabilities_for_head(
        embeddings, models.instrument_model, models.instrument_labels, "Instrument"
    )
