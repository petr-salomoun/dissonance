"""Amplitude envelope generators for rough/smooth modulation."""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


def jittered_am_env(
    n: int,
    sr: int,
    ioi_mean_ms: float = 4.0,
    ioi_jitter: float = 0.6,
    decay_ms: float = 1.5,
) -> np.ndarray:
    """Generate a jittered impulse-train AM envelope with exponential decays."""
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    env = np.zeros(n, dtype=np.float32)
    mean_s = max(float(ioi_mean_ms) * 1e-3, 1e-5)
    sigma = max(float(ioi_jitter), 1e-3)
    mu = np.log(mean_s) - 0.5 * sigma * sigma

    rng = np.random.default_rng()
    t = 0.0
    while True:
        t += float(rng.lognormal(mean=mu, sigma=sigma))
        idx = int(t * sr)
        if idx >= n:
            break
        env[idx] = 1.0

    decay_s = max(float(decay_ms) * 1e-3, 1e-5)
    k_len = max(4, int(np.ceil(8.0 * decay_s * sr)))
    tt = np.arange(k_len, dtype=np.float32) / np.float32(sr)
    kernel = np.exp(-tt / np.float32(decay_s)).astype(np.float32)

    out = fftconvolve(env, kernel, mode="full")[:n].astype(np.float32)
    mx = float(np.max(out))
    if mx > 0.0:
        out /= mx
    return out.astype(np.float32)


def smooth_am_env(n: int, sr: int, rate_hz: float = 70.0, depth: float = 0.9) -> np.ndarray:
    """Generate sinusoidal AM envelope with controllable depth."""
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    d = float(np.clip(depth, 0.0, 1.0))
    t = np.arange(n, dtype=np.float32) / np.float32(sr)
    env = (1.0 - d) + d * (0.5 * (1.0 + np.sin(2.0 * np.pi * np.float32(rate_hz) * t)))
    return env.astype(np.float32)
