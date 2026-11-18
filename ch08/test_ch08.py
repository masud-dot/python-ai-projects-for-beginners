# Chapter 8 tests. They run offline: the model is replaced by
# a fake, and WAV files are synthetic. The real transcription
# listings (first.py, timing.py) need the downloaded model
# and a recording, so they are not run here.
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import degrade  # noqa: E402
from wer import wer, words  # noqa: E402


class FakeModel:
    def transcribe(self, path, beam_size=5):
        segs = iter([
            NS(start=0.0, end=2.5, text=" Hello there."),
            NS(start=2.5, end=3661.25, text=" Bye."),
        ])
        info = NS(language="en", language_probability=0.99)
        return segs, info


def test_stamp():
    assert common.stamp(0) == "00:00:00,000"
    assert common.stamp(3661.25) == "01:01:01,250"
    assert common.stamp(1.5, ".") == "00:00:01.500"


def test_srt_and_text():
    segs, info = common.transcribe(FakeModel(), "x.wav")
    assert info.language == "en"
    srt = common.to_srt(segs)
    assert srt.startswith("1\n00:00:00,000 --> 00:00:02,500\n")
    assert "2\n00:00:02,500 --> 01:01:01,250\nBye." in srt
    assert common.to_text(segs) == "Hello there.\nBye.\n"


def test_audio_files(tmp_path):
    for name in ("a.wav", "b.MP3", "notes.txt"):
        (tmp_path / name).write_bytes(b"")
    names = [p.name for p in common.audio_files(tmp_path)]
    assert names == ["a.wav", "b.MP3"]


def test_words_and_wer():
    assert words("Don't STOP, now!") == ["dont", "stop", "now"]
    rate, counts = wer("the cat sat", "the cat sat")
    assert rate == 0 and counts == (0, 0, 0)
    rate, counts = wer("the cat sat down", "the hat sat")
    assert counts == (1, 1, 0)
    assert rate == pytest.approx(0.5)
    rate, counts = wer("a b", "a x b y")
    assert counts == (0, 0, 2)


def test_degrade_roundtrip(tmp_path):
    rate = 16000
    t = np.arange(rate) / rate
    tone = 8000 * np.sin(2 * np.pi * 440 * t)
    src = str(tmp_path / "in.wav")
    degrade.write(src, rate, tone)
    r, data = degrade.read(src)
    assert r == rate and len(data) == rate
    noisy = degrade.add_noise(data, 10)
    snr = 10 * np.log10(
        np.mean(data ** 2) / np.mean((noisy - data) ** 2))
    assert snr == pytest.approx(10, abs=0.3)
    padded = degrade.add_silence(data, rate, 2)
    assert len(padded) == 3 * rate
    assert not padded[:rate].any()
    assert common.wav_seconds(src) == 1.0
