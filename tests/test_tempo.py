import sys
import types

import pytest

from src.audio_analysis.model_runtime import AudioInputError, AudioTaggingError
from src.audio_analysis.tempo import TempoPrediction, analyze_tempo


def install_essentia(monkeypatch, *, rhythm_result):
    class FakeLoader:
        def __init__(self, **_kwargs):
            pass

        def __call__(self):
            return [0.0, 0.1]

    class FakeRhythmExtractor:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, _audio):
            return rhythm_result

    standard = types.ModuleType("essentia.standard")
    standard.MonoLoader = FakeLoader
    standard.RhythmExtractor2013 = FakeRhythmExtractor
    essentia = types.ModuleType("essentia")
    essentia.standard = standard
    monkeypatch.setitem(sys.modules, "essentia", essentia)
    monkeypatch.setitem(sys.modules, "essentia.standard", standard)


def test_analyze_tempo_returns_bpm_and_confidence(tmp_path, monkeypatch):
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"audio")
    install_essentia(monkeypatch, rhythm_result=(128.0, [], 0.91, [], []))

    assert analyze_tempo(audio_path) == TempoPrediction(128.0, None)


def test_analyze_tempo_ignores_non_normalized_rhythm_confidence(tmp_path, monkeypatch):
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"audio")
    install_essentia(
        monkeypatch,
        rhythm_result=(149.6829071044922, [], 2.3106372356414795, [], []),
    )

    assert analyze_tempo(audio_path) == TempoPrediction(149.6829071044922, None)


def test_analyze_tempo_rejects_missing_file(tmp_path):
    with pytest.raises(AudioInputError, match="does not exist"):
        analyze_tempo(tmp_path / "missing.wav")


@pytest.mark.parametrize("result", [(0.0, [], 0.5), (float("nan"), [], 0.5), ()])
def test_analyze_tempo_rejects_invalid_results(tmp_path, monkeypatch, result):
    audio_path = tmp_path / "demo.wav"
    audio_path.write_bytes(b"audio")
    install_essentia(monkeypatch, rhythm_result=result)

    with pytest.raises(AudioTaggingError):
        analyze_tempo(audio_path)