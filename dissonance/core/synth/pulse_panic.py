"""Accelerating panic-pulse synthesizer."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def render(params: dict, sr: int, n: int) -> np.ndarray:
    """Render an accelerating impulse/burst rhythm with rough AM."""
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    start_rate_hz = float(max(1e-3, params.get("start_rate_hz", 1.0)))
    end_rate_hz = float(max(start_rate_hz, params.get("end_rate_hz", 8.0)))
    acceleration = str(params.get("acceleration", "linear")).lower().strip()
    roughness_am_hz = float(max(0.0, params.get("roughness_am_hz", 70.0)))
    burst_decay_ms = float(max(1e-3, params.get("burst_decay_ms", 8.0)))

    sr_f = float(sr)
    duration_s = n / sr_f
    times = []
    t = 0.0
    while t < duration_s:
        p = t / max(duration_s, 1e-6)
        if acceleration == "exponential":
            rate = start_rate_hz * ((end_rate_hz / start_rate_hz) ** p)
        else:
            rate = start_rate_hz + (end_rate_hz - start_rate_hz) * p
        rate = max(1e-3, rate)
        times.append(t)
        t += 1.0 / rate

    y = np.zeros(n, dtype=np.float64)
    decay_s = burst_decay_ms * 1e-3
    burst_len = max(8, int(0.08 * sr_f))
    bt = np.arange(burst_len, dtype=np.float64) / sr_f
    burst = np.exp(-bt / max(1e-6, decay_s)) * np.sin(2.0 * np.pi * 3000.0 * bt)
    noise = np.random.default_rng().standard_normal(burst_len)
    sos = butter(4, [2000.0, min(4000.0, sr_f * 0.5 - 10.0)], btype="bandpass", fs=sr, output="sos")
    burst += 0.6 * sosfiltfilt(sos, noise)

    for tt in times:
        i0 = int(tt * sr_f)
        if i0 >= n:
            break
        i1 = min(n, i0 + burst_len)
        y[i0:i1] += burst[: i1 - i0]

    if roughness_am_hz > 0.0:
        am = 1.0 + 0.35 * np.sin(2.0 * np.pi * roughness_am_hz * np.arange(n, dtype=np.float64) / sr_f)
        y *= am

    return y.astype(np.float32)
