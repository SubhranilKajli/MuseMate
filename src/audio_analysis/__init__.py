"""Public APIs for MuseMate audio analysis."""

from .genre import analyze_genre
from .instruments import analyze_instruments
from .model_runtime import (
    AudioInputError,
    AudioTaggingError,
    AudioTaggingModels,
    LabelPrediction,
    ModelConfigurationError,
    get_audio_tagging_models,
    prepare_tagging_embeddings,
)

__all__ = [
    "AudioInputError",
    "AudioTaggingError",
    "AudioTaggingModels",
    "LabelPrediction",
    "ModelConfigurationError",
    "analyze_genre",
    "analyze_instruments",
    "get_audio_tagging_models",
    "prepare_tagging_embeddings",
]
