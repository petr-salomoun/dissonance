"""Inharmonic/metallic partial synthesizer."""

from __future__ import annotations

import numpy as np


def render(params: dict, sr: int, n: int) -> np.ndarray:
    """Render inharmonic partials with stretched overtone frequencies."""
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    base_hz = float(params.get("base_hz", 200.0))
    n_partials = max(1, int(params.get("n_partials", 12)))
    inharmonicity_B = float(np.clip(params.get("inharmonicity_B", 0.04), 0.0, 0.3))
    random_detune = float(np.clip(params.get("random_detune", 0.3), 0.0, 1.0))
    decay_rate = float(params.get("decay_rate", 3.0))

    t = np.arange(n, dtype=np.float32) / np.float32(sr)
    nyq = 0.5 * float(sr) - 1.0
    rng = np.random.default_rng()

    y = np.zeros(n, dtype=np.float32)
    ks = np.arange(1, n_partials + 1, dtype=np.float32)
    amps = np.exp(-decay_rate * (ks - 1.0) / float(n_partials)).astype(np.float32)

    detune_span = 0.5 * inharmonicity_B * base_hz * random_detune
    detunes = rng.uniform(-detune_span, detune_span, size=n_partials).astype(np.float32)

    freqs = base_hz * ks * np.sqrt(1.0 + inharmonicity_B * (ks**2)) + detunes
    freqs = np.clip(freqs, 20.0, nyq).astype(np.float32)

    for amp, f_hz in zip(amps, freqs):
        y += amp * np.sin(2.0 * np.pi * f_hz * t).astype(np.float32)

    return y.astype(np.float32)
