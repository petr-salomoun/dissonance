"""Roughness-oriented tone-cluster synthesizer."""

from __future__ import annotations

import numpy as np

from dissonance.core.dsp.bark import bark_to_hz, hz_to_bark
from dissonance.core.dsp.envelopes import smooth_am_env


def render(params: dict, sr: int, n: int) -> np.ndarray:
    """Render a Bark-spread partial cluster with smooth AM roughness."""
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    carrier_hz = float(params.get("carrier_hz", 3000.0))
    n_partials = int(params.get("n_partials", 8))
    partial_spread_bark = float(params.get("partial_spread_bark", 0.25))
    am_rate_hz = float(params.get("am_rate_hz", 70.0))
    am_depth = float(params.get("am_depth", 0.9))
    gain_db = float(params.get("gain_db", -6.0))

    n_partials = max(1, n_partials)
    carrier_bark = float(hz_to_bark(carrier_hz))
    partials_bark = np.linspace(
        carrier_bark - partial_spread_bark,
        carrier_bark + partial_spread_bark,
        n_partials,
        dtype=np.float32,
    )
    partials_hz = np.asarray(bark_to_hz(partials_bark), dtype=np.float32)
    partials_hz = np.clip(partials_hz, 20.0, (sr * 0.5) - 1.0)

    t = np.arange(n, dtype=np.float32) / np.float32(sr)
    y = np.zeros(n, dtype=np.float32)
    for f_hz in partials_hz:
        y += np.sin(2.0 * np.pi * f_hz * t).astype(np.float32)
    y /= np.float32(n_partials)

    env = smooth_am_env(n=n, sr=sr, rate_hz=am_rate_hz, depth=am_depth)
    y *= env

    y *= np.float32(10.0 ** (gain_db / 20.0))
    peak = float(np.max(np.abs(y)))
    if peak > 0.0:
        y /= np.float32(peak)
    return y.astype(np.float32)
