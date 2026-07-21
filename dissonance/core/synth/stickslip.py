"""Stick-slip / screech-like synthesizer."""

from __future__ import annotations

import numpy as np

from dissonance.core.dsp.envelopes import jittered_am_env
from dissonance.core.dsp.filters import bandpass


def render(params: dict, sr: int, n: int) -> np.ndarray:
    """Render a jitter-excited resonant noise cluster."""
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    ioi_mean_ms = float(params.get("ioi_mean_ms", 4.0))
    ioi_jitter = float(params.get("ioi_jitter", 0.6))
    impulse_decay_ms = float(params.get("impulse_decay_ms", 1.5))
    resonance_hz = params.get("resonance_hz", [2400.0, 3300.0, 4100.0])

    if not isinstance(resonance_hz, (list, tuple, np.ndarray)):
        resonance_hz = [float(resonance_hz)]
    resonances = [float(r) for r in resonance_hz if float(r) > 0.0]
    if len(resonances) == 0:
        resonances = [2400.0, 3300.0, 4100.0]

    env = jittered_am_env(
        n=n,
        sr=sr,
        ioi_mean_ms=ioi_mean_ms,
        ioi_jitter=ioi_jitter,
        decay_ms=impulse_decay_ms,
    )

    rng = np.random.default_rng()
    noise = rng.standard_normal(n).astype(np.float32)
    noise = bandpass(noise, sr=sr, low_hz=80.0, high_hz=min(12000.0, sr * 0.5 - 100.0))

    y = np.zeros(n, dtype=np.float32)
    q = 8.0
    nyq_limit = sr * 0.5 - 1.0
    for f0 in resonances:
        bw = max(f0 / q, 10.0)
        low = max(20.0, f0 - 0.5 * bw)
        high = min(nyq_limit, f0 + 0.5 * bw)
        y += bandpass(noise, sr=sr, low_hz=low, high_hz=high)

    y *= env
    return y.astype(np.float32)
