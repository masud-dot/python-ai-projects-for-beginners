"""Write .txt and .srt transcripts for a folder of audio."""
import sys
from pathlib import Path

from common import (audio_files, load_model, to_srt, to_text,
                    transcribe)

folder = Path(sys.argv[1])
size = sys.argv[2] if len(sys.argv) > 2 else "base"
out = Path("transcripts")
out.mkdir(exist_ok=True)

files = audio_files(folder)
if not files:
    sys.exit(f"No audio files found in {folder}")

model = load_model(size)
for path in files:
    segments, info = transcribe(model, path)
    stem = out / path.stem
    stem.with_suffix(".txt").write_text(
        to_text(segments), encoding="utf-8")
    stem.with_suffix(".srt").write_text(
        to_srt(segments), encoding="utf-8")
    print(f"{path.name}: {len(segments)} segments, "
          f"language {info.language}")
