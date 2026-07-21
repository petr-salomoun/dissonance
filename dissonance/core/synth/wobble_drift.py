"""Two-tone detune drift synthesizer."""

from __future__ import annotations

import numpy as np


def render(params: dict, sr: int, n: int) -> np.ndarray:
    """Render harmonically rich pair drifting from near-unison to rough detune."""
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    base_hz = float(params.get("base_hz", 300.0))
    detune_start_hz = float(max(0.0, params.get("detune_start_hz", 0.5)))
    detune_end_hz = float(max(detune_start_hz, params.get("detune_end_hz", 12.0)))
    drift_shape = str(params.get("drift_shape", "linear")).lower().strip()
    n_harmonics = max(1, int(params.get("n_harmonics", 4)))

    sr_f = float(sr)
    p = np.linspace(0.0, 1.0, n, dtype=np.float64)
    if drift_shape == "exponential":
        detune = detune_start_hz * ((detune_end_hz / max(detune_start_hz, 1e-6)) ** p)
    else:
        detune = detune_start_hz + (detune_end_hz - detune_start_hz) * p

    f1 = np.clip(base_hz - 0.5 * detune, 20.0, sr_f * 0.5 - 1.0)
    f2 = np.clip(base_hz + 0.5 * detune, 20.0, sr_f * 0.5 - 1.0)

    y = np.zeros(n, dtype=np.float64)
    for k in range(1, n_harmonics + 1):
        amp = 1.0 / k
        ph1 = 2.0 * np.pi * np.cumsum(np.clip(k * f1, 20.0, sr_f * 0.5 - 1.0)) / sr_f
        ph2 = 2.0 * np.pi * np.cumsum(np.clip(k * f2, 20.0, sr_f * 0.5 - 1.0)) / sr_f
        y += amp * (np.sin(ph1) + np.sin(ph2))

    return y.astype(np.float32)
