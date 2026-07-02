"""Layer mixer for synthesizer outputs."""

from __future__ import annotations

import numpy as np

from dissonance.core.dsp.filters import boost_2_4khz, highpass
from dissonance.core.synth import beating, fm_instab, inharmonic, noise_shaped, rough, stickslip


def mix(layers: list[dict], sr: int, duration_s: float, global_params: dict) -> np.ndarray:
    """Render and combine configured synthesis layers into one mono signal."""
    n = max(1, int(float(duration_s) * int(sr)))
    global_params = global_params or {}
    hump_2_4khz_db = float(global_params.get("hump_2_4khz_db", 9.0))
    highpass_hz = float(global_params.get("highpass_hz", 800.0))

    dispatch = {
        "rough": rough.render,
        "stickslip": stickslip.render,
        "fm_instab": fm_instab.render,
        "inharmonic": inharmonic.render,
        "beating": beating.render,
        "noise_shaped": noise_shaped.render,
    }

    y = np.zeros(n, dtype=np.float32)
    for layer in layers or []:
        synth_type = str(layer.get("type", "")).lower().strip()
        if synth_type not in dispatch:
            raise ValueError(f"Unknown layer type: {synth_type!r}")

        layer_params = {k: v for k, v in layer.items() if k != "type"}
        layer_sig = dispatch[synth_type](params=layer_params, sr=sr, n=n)
        if layer_sig.shape[0] != n:
            if layer_sig.shape[0] > n:
                layer_sig = layer_sig[:n]
            else:
                layer_sig = np.pad(layer_sig, (0, n - layer_sig.shape[0]))
        y += np.asarray(layer_sig, dtype=np.float32)

    y = highpass(y, sr=sr, cutoff_hz=highpass_hz)
    y = boost_2_4khz(y, sr=sr, gain_db=hump_2_4khz_db)

    peak = float(np.max(np.abs(y)))
    if peak > 0.0:
        y = 0.9 * y / peak
    return np.asarray(y, dtype=np.float32)
