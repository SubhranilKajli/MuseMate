"""Musical key analysis using Essentia's key extractor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .model_runtime import AudioInputError, AudioTaggingError, ModelConfigurationError


@dataclass(frozen=True)
class KeyPrediction:
	"""Estimated tonic, scale, and Essentia key strength."""

	key: str
	scale: str
	confidence: float


def analyze_key(audio_path: str | Path) -> KeyPrediction:
	"""Estimate the musical key and scale of an audio file with Essentia."""
	path = Path(audio_path)
	if not path.is_file():
		raise AudioInputError(f"Audio file does not exist or is not a file: {path}")

	try:
		from essentia.standard import KeyExtractor, MonoLoader
	except ImportError as exc:
		raise ModelConfigurationError("Essentia is required for key analysis.") from exc

	try:
		audio = MonoLoader(filename=str(path))()
		result = KeyExtractor()(audio)
	except Exception as exc:  # Essentia raises implementation-specific errors.
		raise AudioTaggingError(f"Key analysis failed for audio file: {path}") from exc

	if not isinstance(result, (tuple, list)) or len(result) < 3:
		raise AudioTaggingError("Essentia returned an incomplete key estimate.")
	key, scale, confidence = result[:3]
	if not isinstance(key, str) or not key.strip():
		raise AudioTaggingError("Essentia returned an empty key estimate.")
	if not isinstance(scale, str) or not scale.strip():
		raise AudioTaggingError("Essentia returned an empty scale estimate.")
	if isinstance(confidence, bool) or not isinstance(confidence, (float, int, np.number)):
		raise AudioTaggingError("Essentia returned an invalid key confidence.")
	confidence_value = float(confidence)
	if not np.isfinite(confidence_value) or not 0.0 <= confidence_value <= 1.0:
		raise AudioTaggingError("Essentia returned an invalid key confidence range.")
	return KeyPrediction(key=key.strip(), scale=scale.strip(), confidence=confidence_value)
