"""Sub-bass throb with slowly accelerating beating."""

from __future__ import annotations

import numpy as np


def render(params: dict, sr: int, n: int) -> np.ndarray:
    """Render detuned low-frequency pair with drift and slow AM."""
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    center_hz = float(params.get("center_hz", 30.0))
    detune_hz_start = float(max(0.0, params.get("detune_hz_start", 0.5)))
    detune_hz_end = float(max(detune_hz_start, params.get("detune_hz_end", 3.0)))
    am_rate_hz = float(max(0.0, params.get("am_rate_hz", 0.3)))

    sr_f = float(sr)
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    detune = detune_hz_start + (detune_hz_end - detune_hz_start) * t
    f1 = np.clip(center_hz - 0.5 * detune, 15.0, 80.0)
    f2 = np.clip(center_hz + 0.5 * detune, 15.0, 80.0)

    p1 = 2.0 * np.pi * np.cumsum(f1) / sr_f
    p2 = 2.0 * np.pi * np.cumsum(f2) / sr_f
    y = 0.5 * np.sin(p1) + 0.5 * np.sin(p2)

    if am_rate_hz > 0.0:
        am = 0.7 + 0.3 * (0.5 * (1.0 + np.sin(2.0 * np.pi * am_rate_hz * np.arange(n, dtype=np.float64) / sr_f)))
        y *= am

    return y.astype(np.float32)
