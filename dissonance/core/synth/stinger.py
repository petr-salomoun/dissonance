"""Jump-scare transient synthesizer."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def render(params: dict, sr: int, n: int) -> np.ndarray:
    """Render delayed sharp broadband transient with inharmonic add-ons."""
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    attack_s = float(max(1e-5, params.get("attack_s", 0.001)))
    silence_before_s = float(max(0.0, params.get("silence_before_s", 0.5)))
    burst_duration_s = float(max(1e-3, params.get("burst_duration_s", 0.3)))

    sr_f = float(sr)
    y = np.random.default_rng().standard_normal(n).astype(np.float64) * (10.0 ** (-40.0 / 20.0))

    start = min(n, int(silence_before_s * sr_f))
    dur = max(1, int(burst_duration_s * sr_f))
    end = min(n, start + dur)
    if end <= start:
        return y.astype(np.float32)

    seg_len = end - start
    bt = np.arange(seg_len, dtype=np.float64) / sr_f
    atk_n = max(1, int(attack_s * sr_f))
    env = np.exp(-bt / max(1e-4, burst_duration_s / 3.0))
    env[:atk_n] *= np.linspace(0.0, 1.0, atk_n, dtype=np.float64)

    noise = np.random.default_rng().standard_normal(seg_len)
    sos = butter(4, [500.0, min(12000.0, sr_f * 0.5 - 10.0)], btype="bandpass", fs=sr, output="sos")
    burst = sosfiltfilt(sos, noise)

    peak_f = 3000.0
    ratios = np.array([1.37, 2.11, 2.93, 3.77, 4.61], dtype=np.float64)
    p = np.arange(seg_len, dtype=np.float64) / sr_f
    inharm = np.zeros(seg_len, dtype=np.float64)
    for r in ratios[: np.random.default_rng().integers(3, 6)]:
        f = min(0.5 * sr_f - 1.0, peak_f * r)
        inharm += np.sin(2.0 * np.pi * f * p)
    if np.max(np.abs(inharm)) > 1e-8:
        inharm /= np.max(np.abs(inharm))

    y[start:end] += env * (0.8 * burst + 0.4 * inharm)
    return y.astype(np.float32)
