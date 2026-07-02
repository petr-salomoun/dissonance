"""Signal rendering helpers."""

from __future__ import annotations

import numpy as np

from dissonance.core.dsp.envelopes import jittered_am_env, smooth_am_env


def render(
    duration_s: float = 1.0,
    sr: int = 48000,
    carrier_hz: float = 3000.0,
    mode: str = "jittered",
) -> np.ndarray:
    """Render a simple mono test signal as float32."""
    n = max(1, int(float(duration_s) * int(sr)))
    t = np.arange(n, dtype=np.float32) / np.float32(sr)

    carrier = np.sin(2.0 * np.pi * np.float32(carrier_hz) * t).astype(np.float32)
    noise = np.random.default_rng().standard_normal(n).astype(np.float32)

    if mode.lower() == "smooth":
        env = smooth_am_env(n, sr, rate_hz=70.0, depth=0.9)
    else:
        env = jittered_am_env(n, sr)

    y = (0.7 * carrier + 0.3 * noise) * env
    peak = float(np.max(np.abs(y)) + 1e-12)
    y = (0.95 * y / peak).astype(np.float32)
    return y
