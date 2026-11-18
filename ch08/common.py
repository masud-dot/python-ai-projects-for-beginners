"""Shared helpers for Chapter 8.

faster_whisper is imported inside load_model so the text
helpers below work (and are tested) without the package.
"""
import wave
from pathlib import Path

AUDIO_TYPES = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


def load_model(size="base"):
    from faster_whisper import WhisperModel

    return WhisperModel(size, device="cpu", compute_type="int8")


def transcribe(model, path):
    segments, info = model.transcribe(str(path), beam_size=5)
    return list(segments), info


def wav_seconds(path):
    with wave.open(str(path), "rb") as f:
        return f.getnframes() / f.getframerate()


def stamp(seconds, sep=","):
    total = round(seconds * 1000)
    hours, total = divmod(total, 3_600_000)
    minutes, total = divmod(total, 60_000)
    secs, millis = divmod(total, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}{sep}{millis:03}"


def to_text(segments):
    lines = [seg.text.strip() for seg in segments]
    return "\n".join(lines) + "\n"


def to_srt(segments):
    blocks = []
    for n, seg in enumerate(segments, start=1):
        times = f"{stamp(seg.start)} --> {stamp(seg.end)}"
        blocks.append(f"{n}\n{times}\n{seg.text.strip()}\n")
    return "\n".join(blocks)


def audio_files(folder):
    paths = sorted(Path(folder).iterdir())
    return [p for p in paths if p.suffix.lower() in AUDIO_TYPES]
