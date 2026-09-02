# Genre and Instrument Integration

Genre and instrument analysis share one Discogs-EffNet embedding extraction.
Call the package-level APIs below once per audio file; do not call the
low-level loaders or classifier heads directly.

```python
from src.audio_analysis import (
    analyze_genre,
    analyze_instruments,
    get_audio_tagging_models,
    prepare_tagging_embeddings,
)

models = get_audio_tagging_models()
embeddings = prepare_tagging_embeddings(audio_path, models=models)

genre_predictions = analyze_genre(embeddings, models=models)
instrument_predictions = analyze_instruments(embeddings, models=models)
```

Both analyser functions return `list[LabelPrediction]`. Each prediction has a
`label` and the unchanged model `confidence` score.

`analyze_genre` returns threshold-qualified predictions (up to five by
default); if none meet its threshold, it returns the top three ranked genre
predictions. `analyze_instruments` is a multilabel thresholded result (up to
ten by default) and has no fallback.

This module deliberately does not assemble `MusicalContext`; future context
integration should consume these returned prediction lists alongside the other
audio analysis results.
