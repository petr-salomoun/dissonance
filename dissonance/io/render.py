"""I/O rendering helpers for parameter-driven synthesis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from dissonance.core import mixer


def render_from_params(params: dict, out_path: str) -> np.ndarray:
    """Render a signal from params, write WAV to out_path, and return the audio."""
    duration_s = float(params.get("duration_s", 4.0))
    sample_rate = int(params.get("sample_rate", 48000))
    layers = params.get("layers", [])
    global_params = params.get("global", {})

    signal = mixer.mix(
        layers=layers,
        sr=sample_rate,
        duration_s=duration_s,
        global_params=global_params,
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), signal, sample_rate)
    return np.asarray(signal, dtype=np.float32)
