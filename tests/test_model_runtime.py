import numpy as np
import pytest

from src.audio_analysis.model_runtime import (
    AudioTaggingError,
    AudioTaggingModels,
    ModelConfigurationError,
    extract_discogs_effnet_embeddings,
    predict_genre_probabilities,
    predict_instrument_probabilities,
)


class FakeModel:
    def __init__(self, output):
        self.output = output

    def __call__(self, _input):
        return self.output


def make_models(*, embeddings=None, genre_predictions=None, instrument_predictions=None):
    return AudioTaggingModels(
        embedding_model=FakeModel(
            np.ones((2, 1280), dtype=np.float32) if embeddings is None else embeddings
        ),
        genre_model=FakeModel(
            np.array([[0.2, 0.8], [0.6, 0.4]])
            if genre_predictions is None
            else genre_predictions
        ),
        instrument_model=FakeModel(
            np.array([[0.9, 0.1], [0.5, 0.3]])
            if instrument_predictions is None
            else instrument_predictions
        ),
        genre_labels=("rock", "jazz"),
        instrument_labels=("drums", "piano"),
    )


def test_classifier_averages_multiple_timestamp_predictions():
    probabilities = predict_genre_probabilities(np.ones((2, 1280)), models=make_models())

    assert probabilities == {"rock": pytest.approx(0.4), "jazz": pytest.approx(0.6)}


def test_classifier_accepts_single_timestamp_predictions():
    models = make_models(instrument_predictions=np.array([[0.7, 0.2]]))

    assert predict_instrument_probabilities(np.ones((1, 1280)), models=models) == {
        "drums": 0.7,
        "piano": 0.2,
    }


@pytest.mark.parametrize(
    "predictions, expected_exception",
    [
        (np.array([0.2, 0.8]), AudioTaggingError),
        (np.ones((2, 3)), ModelConfigurationError),
        (np.empty((0, 2)), AudioTaggingError),
    ],
)
def test_classifier_rejects_unexpected_prediction_shapes(predictions, expected_exception):
    with pytest.raises(expected_exception):
        predict_genre_probabilities(
            np.ones((2, 1280)), models=make_models(genre_predictions=predictions)
        )


@pytest.mark.parametrize("embedding_shape", [(1280,), (2, 64), (0, 1280)])
def test_classifier_rejects_invalid_embedding_shape_before_inference(embedding_shape):
    with pytest.raises(AudioTaggingError, match="shape"):
        predict_genre_probabilities(
            np.ones(embedding_shape, dtype=np.float32), models=make_models()
        )


@pytest.mark.parametrize("embedding_shape", [(1280,), (2, 64), (0, 1280)])
def test_embedding_extraction_rejects_unexpected_embedding_shape(embedding_shape):
    models = make_models(embeddings=np.ones(embedding_shape, dtype=np.float32))

    with pytest.raises(AudioTaggingError, match="shape"):
        extract_discogs_effnet_embeddings(np.ones(48_000, dtype=np.float32), models=models)
