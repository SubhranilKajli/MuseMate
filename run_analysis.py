import argparse
import json
from pathlib import Path

from src.context.context_builder import build_musical_context


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze an audio file with MuseMate.")
    parser.add_argument(
        "audio_path",
        nargs="?",
        default="data/guitar_sample.mp3",
        help="audio file to analyze (default: data/guitar_sample.mp3)",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio_path)
    if not audio_path.is_file():
        print(f"Error: audio file does not exist: {audio_path}")
        return 1

    context = build_musical_context(audio_path)
    print(json.dumps(context, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())