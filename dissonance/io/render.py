"""I/O rendering helpers for parameter-driven synthesis."""

from __future__ import annotations

import json
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

    sidecar = out.with_name(f"{out.stem}.params.json")
    sidecar_payload = {
        "layers": layers,
        "duration_s": duration_s,
        "sample_rate": sample_rate,
        "global": global_params,
    }
    if isinstance(params.get("_ab_candidate"), dict):
        sidecar_payload["_ab_candidate"] = params["_ab_candidate"]
    if isinstance(params.get("_sweep_meta"), dict):
        sidecar_payload["_sweep_meta"] = params["_sweep_meta"]
    with sidecar.open("w", encoding="utf-8") as f:
        json.dump(sidecar_payload, f, indent=2, sort_keys=True)

    return np.asarray(signal, dtype=np.float32)
