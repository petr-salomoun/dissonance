"""Synth-structural feature extraction from layer metadata."""

from __future__ import annotations

import numpy as np


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def compute_synth_features(layers: list[dict], duration_s: float) -> dict[str, float]:
    """Compute normalized synth-structural features from layer params."""
    _ = float(duration_s)
    layer_list = list(layers or [])

    synth_types = {
        "scream_chaos",
        "dread_swell",
        "shepard_ascent",
        "pulse_panic",
        "doom_throb",
        "wobble_drift",
        "uncanny_morph",
    }

    present: dict[str, list[dict]] = {k: [] for k in synth_types}
    for layer in layer_list:
        if not isinstance(layer, dict):
            continue
        t = str(layer.get("type", ""))
        if t in present:
            present[t].append(layer)

    n_layers = max(0, len(layer_list))
    temporal_count = sum(1 for layer in layer_list if isinstance(layer, dict) and str(layer.get("type", "")) in synth_types)
    temporal_density = 0.0 if n_layers == 0 else _clip01(float(temporal_count) / float(n_layers))

    def _first(t: str) -> dict | None:
        items = present.get(t, [])
        return items[0] if items else None

    scream = _first("scream_chaos")
    dread = _first("dread_swell")
    shep = _first("shepard_ascent")
    pulse = _first("pulse_panic")
    doom = _first("doom_throb")
    wobble = _first("wobble_drift")
    uncanny = _first("uncanny_morph")

    return {
        "layer_scream_chaos": 1.0 if scream is not None else 0.0,
        "layer_dread_swell": 1.0 if dread is not None else 0.0,
        "layer_shepard_ascent": 1.0 if shep is not None else 0.0,
        "layer_pulse_panic": 1.0 if pulse is not None else 0.0,
        "layer_doom_throb": 1.0 if doom is not None else 0.0,
        "layer_wobble_drift": 1.0 if wobble is not None else 0.0,
        "layer_uncanny_morph": 1.0 if uncanny is not None else 0.0,
        "synth_temporal_density": temporal_density,
        "synth_n_layers": _clip01(float(n_layers) / 10.0),
        "scream_chaos_biphonation": _clip01(float((scream or {}).get("biphonation_ratio", 0.3)) / 1.0) if scream is not None else 0.0,
        "dread_swell_rise": _clip01(
            (float((dread or {}).get("end_hz", 400.0)) - float((dread or {}).get("start_hz", 100.0))) / 1000.0
        )
        if dread is not None
        else 0.0,
        "shepard_n_voices": _clip01(float((shep or {}).get("n_voices", 6)) / 16.0) if shep is not None else 0.0,
        "pulse_panic_rate": _clip01(float((pulse or {}).get("end_rate_hz", 8.0)) / 16.0) if pulse is not None else 0.0,
        "doom_throb_detune": _clip01(float((doom or {}).get("detune_hz_end", 3.0)) / 8.0) if doom is not None else 0.0,
        "wobble_detune_end": _clip01(float((wobble or {}).get("detune_end_hz", 20.0)) / 60.0) if wobble is not None else 0.0,
        "uncanny_inharmonicity": _clip01(float((uncanny or {}).get("inharmonicity_end", 0.2)) / 0.5) if uncanny is not None else 0.0,
    }
