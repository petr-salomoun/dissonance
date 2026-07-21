"""Slow-beating multi-tone synthesizer."""

from __future__ import annotations

import numpy as np


def render(params: dict, sr: int, n: int) -> np.ndarray:
    """Render close-frequency tones with harmonic content to create beating."""
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    base_hz = float(params.get("base_hz", 220.0))
    n_beaters = max(1, int(params.get("n_beaters", 3)))
    beat_rate_hz = float(np.clip(params.get("beat_rate_hz", 7.0), 1.0, 20.0))
    beat_jitter = float(np.clip(params.get("beat_jitter", 0.3), 0.0, 1.0))
    harmonics = max(1, int(params.get("harmonics", 3)))
    harmonic_rolloff = float(np.clip(params.get("harmonic_rolloff", 0.0), 0.0, 1.0))

    t = np.arange(n, dtype=np.float32) / np.float32(sr)
    nyq = 0.5 * float(sr) - 1.0
    rng = np.random.default_rng()

    y = np.zeros(n, dtype=np.float32)
    jitter = rng.uniform(-1.0, 1.0, size=n_beaters).astype(np.float32)

    for i in range(n_beaters):
        f0 = base_hz + i * beat_rate_hz * (1.0 + beat_jitter * float(jitter[i]))
        f0 = float(np.clip(f0, 20.0, nyq))
        for k in range(1, harmonics + 1):
            fk = f0 * k
            if fk >= nyq:
                break
            amp = harmonic_rolloff ** (k - 1)
            y += np.float32(amp) * np.sin(2.0 * np.pi * np.float32(fk) * t).astype(np.float32)

    return y.astype(np.float32)
