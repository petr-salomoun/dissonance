"""Critical-band noise-shaped synthesizer."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def _band_lims(center_hz: float, bandwidth_hz: float, sr: int) -> tuple[float, float]:
    nyq = 0.5 * float(sr)
    low = max(20.0, center_hz - 0.5 * bandwidth_hz)
    high = min(nyq - 50.0, center_hz + 0.5 * bandwidth_hz)
    if high <= low:
        high = min(nyq - 10.0, low + 200.0)
        low = max(20.0, high - 100.0)
    return float(low), float(high)


def render(params: dict, sr: int, n: int) -> np.ndarray:
    """Render band-shaped harsh noise with sub-band gain variation and AM."""
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    center_hz = float(params.get("center_hz", 3150.0))
    bandwidth_hz = float(max(10.0, params.get("bandwidth_hz", 2000.0)))
    n_bands = max(1, int(params.get("n_bands", 4)))
    modulation_rate_hz = float(max(0.0, params.get("modulation_rate_hz", 70.0)))
    modulation_depth = float(np.clip(params.get("modulation_depth", 0.6), 0.0, 1.0))
    gain_db = float(params.get("gain_db", -9.0))

    rng = np.random.default_rng()
    white = rng.standard_normal(n).astype(np.float32)
    t = np.arange(n, dtype=np.float32) / np.float32(sr)

    low, high = _band_lims(center_hz=center_hz, bandwidth_hz=bandwidth_hz, sr=sr)
    edges = np.linspace(low, high, n_bands + 1, dtype=np.float64)

    y = np.zeros(n, dtype=np.float32)
    for i in range(n_bands):
        b0, b1 = float(edges[i]), float(edges[i + 1])
        if b1 <= b0 + 1.0:
            continue
        sos = butter(4, [b0, b1], btype="bandpass", fs=sr, output="sos")
        band = sosfiltfilt(sos, white).astype(np.float32)
        gain = np.float32(rng.uniform(0.7, 1.3))
        y += gain * band

    am = 1.0 + modulation_depth * np.sin(2.0 * np.pi * np.float32(modulation_rate_hz) * t)
    y *= am.astype(np.float32)

    y *= np.float32(10.0 ** (gain_db / 20.0))
    peak = float(np.max(np.abs(y)))
    if peak > 0.0:
        y /= np.float32(peak)
    return y.astype(np.float32)
