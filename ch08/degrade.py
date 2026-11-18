"""Make a harder copy of a 16-bit mono WAV clip.

python degrade.py noise in.wav out.wav 10   (SNR in dB)
python degrade.py silence in.wav out.wav 30 (add seconds)
"""
import sys
import wave

import numpy as np


def read(path):
    with wave.open(path, "rb") as f:
        assert f.getsampwidth() == 2 and f.getnchannels() == 1
        rate = f.getframerate()
        raw = f.readframes(f.getnframes())
        data = np.frombuffer(raw, "<i2")
    return rate, data.astype(np.float64)


def write(path, rate, data):
    pcm = np.clip(data, -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm.tobytes())


def add_noise(data, snr_db, seed=0):
    power = np.mean(data ** 2)
    sigma = np.sqrt(power / 10 ** (snr_db / 10))
    rng = np.random.default_rng(seed)
    return data + rng.normal(0, sigma, data.shape)


def add_silence(data, rate, seconds):
    return np.concatenate([np.zeros(int(rate * seconds)), data])


if __name__ == "__main__":
    mode, src, dst, amount = sys.argv[1:5]
    rate, data = read(src)
    if mode == "noise":
        data = add_noise(data, float(amount))
    else:
        data = add_silence(data, rate, float(amount))
    write(dst, rate, data)
    print(f"Wrote {dst}")
