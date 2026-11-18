"""Time tiny, base and small on the same WAV clip."""
import sys
import time

from common import load_model, transcribe, wav_seconds

path = sys.argv[1]
sizes = sys.argv[2:] or ["tiny", "base", "small"]
clip = wav_seconds(path)
print(f"Clip length: {clip:.1f} s")
print(f"{'model':8}{'load s':>8}{'run s':>8}{'x real':>8}")
for size in sizes:
    t0 = time.perf_counter()
    model = load_model(size)
    t1 = time.perf_counter()
    segments, _ = transcribe(model, path)
    t2 = time.perf_counter()
    text = " ".join(s.text.strip() for s in segments)
    with open(f"timing_{size}.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"{size:8}{t1 - t0:8.1f}{t2 - t1:8.1f}"
          f"{clip / (t2 - t1):8.1f}")
