"""Temporal modulation helpers for non-stationary synthesis."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def _periodic_shape(phase: np.ndarray, shape: str) -> np.ndarray:
    shape = str(shape).lower().strip()
    if shape == "saw":
        out = 2.0 * (phase - np.floor(phase + 0.5))
    elif shape == "square":
        out = np.where(np.sin(2.0 * np.pi * phase) >= 0.0, 1.0, -1.0)
    elif shape == "sample_hold":
        out = np.zeros_like(phase, dtype=np.float64)
        prev_bin = -1
        hold = 0.0
        for i in range(phase.shape[0]):
            b = int(np.floor(phase[i]))
            if b != prev_bin:
                hold = np.sin(2.0 * np.pi * np.random.uniform(0.0, 1.0))
                prev_bin = b
            out[i] = hold
    else:
        out = np.sin(2.0 * np.pi * phase)
    return np.asarray(out, dtype=np.float64)


def stochastic_lfo(n, sr, rate_hz, chaos=0.0, shape="sine") -> np.ndarray:
    """Return stochastic LFO of shape (n,) in [-1, 1]."""
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    sr = int(sr)
    rate_hz = max(1e-4, float(rate_hz))
    chaos = float(np.clip(chaos, 0.0, 1.0))

    phase = np.cumsum(np.full(n, rate_hz / float(sr), dtype=np.float64))
    periodic = _periodic_shape(phase, shape)

    rng = np.random.default_rng()
    noise = rng.standard_normal(n).astype(np.float64)
    cutoff_hz = max(0.5, min(0.45 * sr, 2.0 * rate_hz + 0.5))
    sos = butter(2, cutoff_hz, btype="low", fs=sr, output="sos")
    bln = sosfiltfilt(sos, noise)
    bln -= np.mean(bln)
    mx = float(np.max(np.abs(bln)))
    if mx > 1e-8:
        bln /= mx

    out = (1.0 - chaos) * periodic + chaos * bln
    out = np.clip(out, -1.0, 1.0)
    return out.astype(np.float32)


def chaotic_map(n, r=3.9) -> np.ndarray:
    """Return logistic-map sequence of shape (n,) mapped to [-1, 1]."""
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    r = float(np.clip(r, 0.0, 4.0))
    x = np.empty(n, dtype=np.float64)
    x0 = 0.123456789
    for i in range(n):
        x0 = r * x0 * (1.0 - x0)
        x[i] = x0

    y = (2.0 * x) - 1.0
    return np.clip(y, -1.0, 1.0).astype(np.float32)
