"""Tempo analysis using Essentia's rhythm extractor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .model_runtime import AudioInputError, AudioTaggingError, ModelConfigurationError


@dataclass(frozen=True)
class TempoPrediction:
	"""Estimated tempo in beats per minute and optional confidence."""

	bpm: float
	confidence: float | None


def _audio_path(audio_path: str | Path) -> Path:
	path = Path(audio_path)
	if not path.is_file():
		raise AudioInputError(f"Audio file does not exist or is not a file: {path}")
	return path


def _as_finite_number(value: Any, name: str) -> float:
	if isinstance(value, bool) or not isinstance(value, (float, int, np.number)):
		raise AudioTaggingError(f"Essentia returned an invalid {name} value.")
	result = float(value)
	if not np.isfinite(result):
		raise AudioTaggingError(f"Essentia returned a non-finite {name} value.")
	return result


def analyze_tempo(audio_path: str | Path) -> TempoPrediction:
	"""Estimate the dominant tempo of an audio file with Essentia."""
	path = _audio_path(audio_path)
	try:
		from essentia.standard import MonoLoader, RhythmExtractor2013
	except ImportError as exc:
		raise ModelConfigurationError("Essentia is required for tempo analysis.") from exc

	try:
		audio = MonoLoader(filename=str(path))()
		result = RhythmExtractor2013(method="multifeature")(audio)
	except Exception as exc:  # Essentia raises implementation-specific errors.
		raise AudioTaggingError(f"Tempo analysis failed for audio file: {path}") from exc

	if not isinstance(result, (tuple, list)) or len(result) == 0:
		raise AudioTaggingError("Essentia returned no tempo estimate.")
	bpm = _as_finite_number(result[0], "BPM")
	if bpm <= 0.0:
		raise AudioTaggingError("Essentia returned a non-positive BPM estimate.")

	confidence = None
	if len(result) > 2 and result[2] is not None:
		confidence = _as_finite_number(result[2], "tempo confidence")
		if not 0.0 <= confidence <= 1.0:
			raise AudioTaggingError("Essentia returned a tempo confidence outside 0.0 to 1.0.")
	return TempoPrediction(bpm=bpm, confidence=confidence)
