"""Nonlinear vocalization / biological fear-cue synthesizer."""

from __future__ import annotations

import numpy as np

from dissonance.core.synth.temporal_modulator import chaotic_map


def render(params: dict, sr: int, n: int) -> np.ndarray:
    """Render chaotic carrier with subharmonics, biphonation, and pitch jumps."""
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    carrier_hz = float(params.get("carrier_hz", 900.0))
    subharmonic_gain = float(np.clip(params.get("subharmonic_gain", 0.4), 0.0, 2.0))
    chaos_amount = float(np.clip(params.get("chaos_amount", 0.7), 0.0, 1.0))
    pitch_jump_rate_hz = float(max(1e-4, params.get("pitch_jump_rate_hz", 0.5)))
    biphonation_ratio = float(max(0.1, params.get("biphonation_ratio", 1.52)))
    biphonation_gain = float(np.clip(params.get("biphonation_gain", 0.3), 0.0, 2.0))

    sr_f = float(sr)
    nyq = 0.5 * sr_f - 1.0

    jump_interval = max(1, int(sr_f / pitch_jump_rate_hz))
    rng = np.random.default_rng()
    jumps = np.ones(n, dtype=np.float32)
    i = 0
    while i < n:
        factor = 1.0 + rng.choice([-1.0, 1.0]) * rng.uniform(0.2, 0.5)
        j_end = min(n, i + jump_interval)
        jumps[i:j_end] = np.float32(factor)
        i = j_end

    f0 = np.clip(carrier_hz * jumps, 60.0, nyq).astype(np.float32)
    c = chaotic_map(n=n, r=3.9).astype(np.float32)
    inst_f = np.clip(f0 * (1.0 + 0.35 * chaos_amount * c), 20.0, nyq).astype(np.float64)
    phase = 2.0 * np.pi * np.cumsum(inst_f) / sr_f

    main = np.sin(phase)
    sub2 = np.sin(0.5 * phase)
    sub3 = np.sin((1.0 / 3.0) * phase)
    bi = np.sin(phase * biphonation_ratio)

    y = (
        1.0 * main
        + subharmonic_gain * (0.6 * sub2 + 0.4 * sub3)
        + biphonation_gain * bi
    )
    y *= (0.8 + 0.2 * (c + 1.0) * 0.5)
    return y.astype(np.float32)
