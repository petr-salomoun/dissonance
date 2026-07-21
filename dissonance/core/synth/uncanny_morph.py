"""Harmonic-to-inharmonic timbral morph synthesizer."""

from __future__ import annotations

import numpy as np


def render(params: dict, sr: int, n: int) -> np.ndarray:
    """Render partial bank with time-varying inharmonicity and optional formant sweep."""
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)

    base_hz = float(params.get("base_hz", 200.0))
    n_partials = max(1, int(params.get("n_partials", 12)))
    inharmonicity_start = float(max(0.0, params.get("inharmonicity_start", 0.0)))
    inharmonicity_end = float(max(inharmonicity_start, params.get("inharmonicity_end", 0.3)))
    formant_sweep = bool(params.get("formant_sweep", True))

    sr_f = float(sr)
    p = np.linspace(0.0, 1.0, n, dtype=np.float64)
    b = inharmonicity_start + (inharmonicity_end - inharmonicity_start) * p
    nyq = 0.5 * sr_f - 1.0

    y = np.zeros(n, dtype=np.float64)
    formant = 800.0 + (3200.0 - 800.0) * p
    bw = 500.0

    for k in range(1, n_partials + 1):
        fk = base_hz * k * np.sqrt(1.0 + b * (k**2))
        fk = np.clip(fk, 20.0, nyq)
        phase = 2.0 * np.pi * np.cumsum(fk) / sr_f
        amp = 1.0 / k
        if formant_sweep:
            emph = np.exp(-0.5 * ((fk - formant) / bw) ** 2)
            amp_env = amp * (0.3 + 0.7 * emph)
        else:
            amp_env = np.full(n, amp, dtype=np.float64)
        y += amp_env * np.sin(phase)

    return y.astype(np.float32)
