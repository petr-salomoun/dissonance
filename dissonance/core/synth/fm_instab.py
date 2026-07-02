"""FM-instability / alarm-like synthesizer."""

from __future__ import annotations

import numpy as np

from dissonance.core.dsp.filters import bandpass


def render(params: dict, sr: int, n: int) -> np.ndarray:
    """Render a chaotic FM tone with low-frequency modulator perturbation."""
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    carrier_hz = float(params.get("carrier_hz", 3500.0))
    mod_rate_hz = float(params.get("mod_rate_hz", 12.0))
    mod_index = float(params.get("mod_index", 8.0))
    mod_chaos = float(np.clip(params.get("mod_chaos", 0.7), 0.0, 1.0))
    gain_db = float(params.get("gain_db", -9.0))

    t = np.arange(n, dtype=np.float32) / np.float32(sr)

    rng = np.random.default_rng()
    lfn = rng.standard_normal(n).astype(np.float32)
    lfn = bandpass(lfn, sr=sr, low_hz=0.1, high_hz=3.0)
    lfn -= np.mean(lfn, dtype=np.float32)
    lfn_std = float(np.std(lfn))
    if lfn_std > 1e-8:
        lfn /= np.float32(lfn_std)

    modulator_hz = mod_rate_hz + (mod_chaos * 0.5 * mod_rate_hz) * lfn
    modulator_hz = np.maximum(modulator_hz, 0.1).astype(np.float32)

    cumulative_mod_phase = 2.0 * np.pi * np.cumsum(modulator_hz, dtype=np.float64) / float(sr)
    y = np.sin(2.0 * np.pi * np.float32(carrier_hz) * t + np.float32(mod_index) * cumulative_mod_phase).astype(np.float32)
    y *= np.float32(10.0 ** (gain_db / 20.0))

    peak = float(np.max(np.abs(y)))
    if peak > 0.0:
        y /= np.float32(peak)
    return y.astype(np.float32)
