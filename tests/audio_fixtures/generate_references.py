"""Generate the synthetic reference clips used by the audio regression harness.

Run once (outputs are committed):

    python tests/audio_fixtures/generate_references.py

The synthesis is deliberately simple and fully deterministic (fixed seeds,
numpy only): a harmonically rich glottal source with a slowly moving pitch is
filtered through two formant resonators. It is speech-*like* (voiced, varying
pitch, vowel formants) which is enough for voice-cloning conditioners, and it
carries no third-party rights.
"""

from __future__ import annotations

import os

import numpy as np
import soundfile as sf

SR = 24000
SECONDS = 8.0

VOICES = {
    "ref_voice_a.wav": {
        "seed": 20261104,
        "f0": 118.0,
        "f0_drift": 14.0,
        "formants": ((720.0, 90.0), (1240.0, 110.0), (2600.0, 160.0)),
    },
    "ref_voice_b.wav": {
        "seed": 424242,
        "f0": 196.0,
        "f0_drift": 22.0,
        "formants": ((520.0, 70.0), (1850.0, 130.0), (2900.0, 180.0)),
    },
}


def _resonator(signal: np.ndarray, freq: float, bandwidth: float, sr: int) -> np.ndarray:
    """Single two-pole formant resonator (direct-form II)."""
    r = np.exp(-np.pi * bandwidth / sr)
    theta = 2.0 * np.pi * freq / sr
    a1 = -2.0 * r * np.cos(theta)
    a2 = r * r
    gain = (1.0 - r) * np.sqrt(1.0 - 2.0 * r * np.cos(2 * theta) + r * r)
    out = np.zeros_like(signal)
    y1 = y2 = 0.0
    for i, sample in enumerate(signal):
        y = gain * sample - a1 * y1 - a2 * y2
        out[i] = y
        y2, y1 = y1, y
    return out


def synthesize(cfg: dict) -> np.ndarray:
    rng = np.random.default_rng(cfg["seed"])
    n = int(SR * SECONDS)
    t = np.arange(n) / SR

    # Pitch contour: slow sinusoidal drift plus a small vibrato.
    f0 = cfg["f0"] + cfg["f0_drift"] * np.sin(2 * np.pi * 0.23 * t) + 2.0 * np.sin(2 * np.pi * 5.1 * t)
    phase = 2.0 * np.pi * np.cumsum(f0) / SR

    # Glottal-ish source: sum of harmonics with 1/k roll-off plus breath noise.
    source = np.zeros(n)
    for k in range(1, 24):
        source += np.sin(k * phase) / k
    source += 0.02 * rng.standard_normal(n)

    # Amplitude envelope with syllable-rate pauses so it is not a drone.
    envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 2.7 * t) ** 2
    envelope *= (np.sin(np.pi * np.clip(t / SECONDS, 0, 1)) ** 0.35)
    source *= envelope

    voiced = np.zeros(n)
    for freq, bandwidth in cfg["formants"]:
        voiced += _resonator(source, freq, bandwidth, SR)

    voiced /= np.max(np.abs(voiced)) or 1.0
    return (0.75 * voiced).astype(np.float32)


def main() -> None:
    here = os.path.dirname(__file__)
    for name, cfg in VOICES.items():
        path = os.path.join(here, name)
        sf.write(path, synthesize(cfg), SR, subtype="PCM_16")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
