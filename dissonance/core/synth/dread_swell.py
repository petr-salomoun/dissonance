"""Slowly rising-tension tonal synthesizer."""

from __future__ import annotations

import numpy as np

from dissonance.core.dsp.bark import bark_to_hz, hz_to_bark


def render(params: dict, sr: int, n: int) -> np.ndarray:
    """Render a multi-partial pitch swell with optional roughness and loudness rise."""
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    start_hz = float(params.get("start_hz", 200.0))
    end_hz = float(params.get("end_hz", 1200.0))
    rise_shape = str(params.get("rise_shape", "linear")).lower().strip()
    roughness_rise = bool(params.get("roughness_rise", True))
    loudness_rise = bool(params.get("loudness_rise", True))
    n_partials = int(np.random.default_rng().integers(4, 7))

    sr_f = float(sr)
    t = np.linspace(0.0, 1.0, n, dtype=np.float64)
    if rise_shape == "exponential":
        ratio = max(1e-6, end_hz / max(start_hz, 1e-6))
        f0 = start_hz * (ratio**t)
    else:
        f0 = start_hz + (end_hz - start_hz) * t

    f0 = np.clip(f0, 20.0, sr_f * 0.5 - 1.0)
    base_bark = hz_to_bark(f0)
    offsets = np.linspace(-0.1, 0.1, n_partials, dtype=np.float64)

    y = np.zeros(n, dtype=np.float64)
    for i, off in enumerate(offsets, start=1):
        fi = bark_to_hz(base_bark + off)
        fi = np.clip(fi * i, 20.0, sr_f * 0.5 - 1.0)
        phase = 2.0 * np.pi * np.cumsum(fi) / sr_f
        amp = 1.0 / i
        y += amp * np.sin(phase)

    am_depth = np.linspace(0.1, 0.9, n, dtype=np.float64) if roughness_rise else np.full(n, 0.1, dtype=np.float64)
    am = (1.0 - am_depth) + am_depth * (0.5 * (1.0 + np.sin(2.0 * np.pi * 70.0 * np.arange(n) / sr_f)))
    y *= am

    if loudness_rise:
        y *= np.linspace(0.1, 1.0, n, dtype=np.float64)

    return y.astype(np.float32)
