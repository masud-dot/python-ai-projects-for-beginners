"""Appendix A, Chapter 8, exercise 2: one sentence per line."""
import re
import sys
from pathlib import Path

sys.path.insert(0, "vendor/ch08")
from common import audio_files, load_model, transcribe  # noqa: E402


def sentences(segments):
    text = " ".join(seg.text.strip() for seg in segments)
    parts = re.split(r"(?<=[.?!])\s+", text.strip())
    return [p for p in parts if p]


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    size = sys.argv[2] if len(sys.argv) > 2 else "base"
    model = load_model(size)
    for path in audio_files(folder):
        segments, _ = transcribe(model, path)
        print(f"--- {path.name}")
        print("\n".join(sentences(segments)))
