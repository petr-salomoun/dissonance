"""Layer mixer for synthesizer outputs."""

from __future__ import annotations

import numpy as np

from dissonance.core.dsp.filters import boost_2_4khz, highpass
from dissonance.core.synth import (
    beating,
    doom_throb,
    dread_swell,
    fm_instab,
    inharmonic,
    noise_shaped,
    pulse_panic,
    rough,
    scream_chaos,
    shepard_ascent,
    stickslip,
    uncanny_morph,
    wobble_drift,
)


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
        "scream_chaos": scream_chaos.render,
        "dread_swell": dread_swell.render,
        "shepard_ascent": shepard_ascent.render,
        "pulse_panic": pulse_panic.render,
        "doom_throb": doom_throb.render,
        "wobble_drift": wobble_drift.render,
        "uncanny_morph": uncanny_morph.render,
    }
    rms_target = np.float32(10.0 ** (-18.0 / 20.0))

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
        layer_sig = np.asarray(layer_sig, dtype=np.float32)
        rms = float(np.sqrt(np.mean(layer_sig * layer_sig, dtype=np.float64)))
        if rms > 1e-8:
            layer_sig = layer_sig * (rms_target / np.float32(rms))
        gain_db = float(layer.get("gain_db", 0.0))
        layer_sig *= np.float32(10.0 ** (gain_db / 20.0))
        y += layer_sig

    y = highpass(y, sr=sr, cutoff_hz=highpass_hz)
    y = boost_2_4khz(y, sr=sr, gain_db=hump_2_4khz_db)

    peak = float(np.max(np.abs(y)))
    if peak > 0.0:
        y = 0.9 * y / peak
    return np.asarray(y, dtype=np.float32)
