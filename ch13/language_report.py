"""Appendix A, Chapter 8, exercise 1: language and confidence.

python language_report.py FOLDER [SIZE] [THRESHOLD]
"""
import sys
from pathlib import Path

sys.path.insert(0, "vendor/ch08")
from common import audio_files, load_model, transcribe  # noqa: E402

THRESHOLD = 0.80


def report(name, info, threshold=THRESHOLD):
    line = (f"{name}: language {info.language}, "
            f"probability {info.language_probability:.2f}")
    if info.language_probability < threshold:
        line += f"  WARNING: below {threshold:.2f}"
    return line


if __name__ == "__main__":
    folder = Path(sys.argv[1])
    size = sys.argv[2] if len(sys.argv) > 2 else "base"
    limit = float(sys.argv[3]) if len(sys.argv) > 3 else THRESHOLD
    model = load_model(size)
    for path in audio_files(folder):
        _, info = transcribe(model, path)
        print(report(path.name, info, limit))
