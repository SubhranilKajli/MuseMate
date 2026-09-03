import sys
import types

import pytest

from src.audio_analysis.key import KeyPrediction, analyze_key
from src.audio_analysis.model_runtime import AudioInputError, AudioTaggingError


def install_essentia(monkeypatch, *, key_result):
    class FakeLoader:
        def __init__(self, **_kwargs):
            pass

        def __call__(self):
            return [0.0, 0.1]

    class FakeKeyExtractor:
        def __call__(self, _audio):
            return key_result

    standard = types.ModuleType("essentia.standard")
    standard.MonoLoader = FakeLoader
    standard.KeyExtractor = FakeKeyExtractor
    essentia = types.ModuleType("essentia")
    essentia.standard = standard
    monkeypatch.setitem(sys.modules, "essentia", essentia)
    monkeypatch.setitem(sys.modules, "essentia.standard", standard)


def test_analyze_key_returns_key_scale_and_confidence(tmp_path, monkeypatch):
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"audio")
    install_essentia(monkeypatch, key_result=("C", "major", 0.82))

    assert analyze_key(audio_path) == KeyPrediction("C", "major", 0.82)


def test_analyze_key_rejects_missing_file(tmp_path):
    with pytest.raises(AudioInputError, match="does not exist"):
        analyze_key(tmp_path / "missing.wav")


@pytest.mark.parametrize("result", [("", "major", 0.8), ("C", "", 0.8), ("C", "major", 1.2)])
def test_analyze_key_rejects_invalid_results(tmp_path, monkeypatch, result):
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"audio")
    install_essentia(monkeypatch, key_result=result)

    with pytest.raises(AudioTaggingError):
        analyze_key(audio_path)