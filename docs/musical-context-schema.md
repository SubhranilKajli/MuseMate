# MusicalContext Schema

MusicalContext is the standardized representation of the musical
information extracted from an audio file.

## Initial Information

The initial schema will contain:

- Audio information
- Tempo
- Key / Tonality
- Time Signature
- Song Structure
- Instruments
- Genre

## Example

```json
{
  "schema_version": "1.0",

  "audio": {
    "filename": "demo.wav",
    "duration_seconds": 210.4,
    "sample_rate": 44100
  },

  "tempo": {
    "bpm": {
      "value": 120.0,
      "confidence": 0.94
    }
  },

  "tonality": {
    "key": {
      "value": "C",
      "mode": "minor",
      "confidence": 0.82
    }
  },

  "rhythm": {
    "time_signature": {
      "value": "4/4",
      "confidence": 0.88
    }
  },

  "structure": {
    "sections": []
  },

  "instruments": [],

  "genre": []
}