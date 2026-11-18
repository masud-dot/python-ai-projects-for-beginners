"""Transcribe one file and print timestamped segments."""
import sys
from pathlib import Path

from common import load_model

path = sys.argv[1]
size = sys.argv[2] if len(sys.argv) > 2 else "base"

model = load_model(size)
segments, info = model.transcribe(path, beam_size=5)
print(f"Language: {info.language} "
      f"({info.language_probability:.2f})")
texts = []
for seg in segments:
    text = seg.text.strip()
    texts.append(text)
    print(f"[{seg.start:6.1f} -> {seg.end:6.1f}] {text}")
out = Path(path).stem + "_" + size + ".txt"
Path(out).write_text(" ".join(texts) + "\n", encoding="utf-8")
print(f"Plain text saved to {out}")
