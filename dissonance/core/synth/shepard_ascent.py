"""Shepard-Risset endless ascending glissando."""

from __future__ import annotations

import numpy as np


def render(params: dict, sr: int, n: int) -> np.ndarray:
    """Render a wrapped multi-voice Shepard ascent with Gaussian voice weighting."""
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    n_voices = max(1, int(params.get("n_voices", 8)))
    octave_spread = float(max(1.0, params.get("octave_spread", 4.0)))
    rise_rate_hz_per_s = float(max(1e-4, params.get("rise_rate_hz_per_s", 0.5)))
    roughness_am_hz = float(max(0.0, params.get("roughness_am_hz", 40.0)))

    sr_f = float(sr)
    t_s = np.arange(n, dtype=np.float64) / sr_f
    oct_center = octave_spread * 0.5
    sigma = octave_spread / 6.0
    base_hz = 55.0
    y = np.zeros(n, dtype=np.float64)

    for v in range(n_voices):
        offset = octave_spread * v / n_voices
        oct_pos = np.mod(offset + rise_rate_hz_per_s * t_s, octave_spread)
        freq = base_hz * (2.0 ** oct_pos)
        freq = np.clip(freq, 20.0, 0.5 * sr_f - 1.0)
        amp = np.exp(-0.5 * ((oct_pos - oct_center) / max(1e-6, sigma)) ** 2)
        phase = 2.0 * np.pi * np.cumsum(freq) / sr_f
        y += amp * np.sin(phase)

    y /= float(n_voices)
    if roughness_am_hz > 0.0:
        am = 1.0 + 0.4 * np.sin(2.0 * np.pi * roughness_am_hz * t_s)
        y *= am

    return y.astype(np.float32)
