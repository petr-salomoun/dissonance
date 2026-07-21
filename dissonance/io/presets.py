"""Built-in synthesis presets."""

from __future__ import annotations

import copy
import json
from pathlib import Path


PRESET_DEFAULT_GEN: dict = {
    "duration_s": 4.0,
    "sample_rate": 48000,
    "global": {
        "hump_2_4khz_db": 9.0,
        "highpass_hz": 800.0,
    },
    "layers": [
        {
            "type": "rough",
            "carrier_hz": 3000.0,
            "n_partials": 8,
            "partial_spread_bark": 0.25,
            "am_rate_hz": 70.0,
            "am_depth": 0.9,
            "gain_db": -6.0,
        },
        {
            "type": "stickslip",
            "ioi_mean_ms": 4.0,
            "ioi_jitter": 0.6,
            "impulse_decay_ms": 1.5,
            "resonance_hz": [2400.0, 3300.0, 4100.0],
            "gain_db": -3.0,
        },
        {
            "type": "fm_instab",
            "carrier_hz": 3500.0,
            "mod_rate_hz": 12.0,
            "mod_index": 8.0,
            "mod_chaos": 0.7,
            "gain_db": -9.0,
        },
    ],
}

PRESET_MY_GEN: dict = {
    "duration_s": 4.0,
    "sample_rate": 44100,
    "global": {
        "hump_2_4khz_db": 15,
        "highpass_hz": 1200,
    },
    "layers": [
        {
            "type": "rough",
            "carrier_hz": 2500,
            "n_partials": 12,
            "partial_spread_bark": 0.75,
            "am_rate_hz": 100,
            "am_depth": 0.7,
            "gain_db": -6.0,
        },
        {
            "type": "stickslip",
            "ioi_mean_ms": 6.0,
            "ioi_jitter": 0.5,
            "resonance_hz": [2400, 3300, 4100],
            "gain_db": -3.0,
        },
        {
            "type": "fm_instab",
            "carrier_hz": 2500,
            "mod_rate_hz": 8,
            "mod_index": 16,
            "mod_chaos": 0.5,
            "gain_db": -9.0,
        },
    ],
}

TEMPORAL_PRESET: dict = {
    "duration_s": 6.0,
    "sample_rate": 44100,
    "global": {
        "hump_2_4khz_db": 10.0,
        "highpass_hz": 40.0,
    },
    "layers": [
        {
            "type": "doom_throb",
            "center_hz": 32.0,
            "detune_hz_start": 0.4,
            "detune_hz_end": 2.8,
            "am_rate_hz": 0.28,
            "gain_db": -6.0,
        },
        {
            "type": "dread_swell",
            "start_hz": 180.0,
            "end_hz": 1400.0,
            "rise_shape": "exponential",
            "roughness_rise": True,
            "loudness_rise": True,
            "gain_db": -8.0,
        },
        {
            "type": "wobble_drift",
            "base_hz": 260.0,
            "detune_start_hz": 0.3,
            "detune_end_hz": 10.0,
            "drift_shape": "linear",
            "n_harmonics": 5,
            "gain_db": -10.0,
        },
        {
            "type": "uncanny_morph",
            "base_hz": 220.0,
            "n_partials": 10,
            "inharmonicity_start": 0.0,
            "inharmonicity_end": 0.22,
            "formant_sweep": True,
            "gain_db": -12.0,
        },
        {
            "type": "pulse_panic",
            "start_rate_hz": 1.0,
            "end_rate_hz": 7.0,
            "acceleration": "exponential",
            "roughness_am_hz": 65.0,
            "burst_decay_ms": 10.0,
            "gain_db": -9.0,
        },
    ],
}

HORROR_CINEMATIC_PRESET: dict = {
    "duration_s": 8.0,
    "sample_rate": 48000,
    "global": {
        "hump_2_4khz_db": 12.0,
        "highpass_hz": 30.0,
    },
    "layers": [
        {
            "type": "dread_swell",
            "start_hz": 140.0,
            "end_hz": 1800.0,
            "rise_shape": "exponential",
            "roughness_rise": True,
            "loudness_rise": True,
            "gain_db": -7.0,
        },
        {
            "type": "shepard_ascent",
            "n_voices": 10,
            "octave_spread": 5.0,
            "rise_rate_hz_per_s": 0.45,
            "roughness_am_hz": 36.0,
            "gain_db": -10.0,
        },
        {
            "type": "doom_throb",
            "center_hz": 28.0,
            "detune_hz_start": 0.3,
            "detune_hz_end": 2.2,
            "am_rate_hz": 0.22,
            "gain_db": -4.0,
        },
    ],
}

def get_preset(name: str) -> dict:
    """Return a deep-copied preset by name."""
    import os

    p = Path(os.fspath(name))
    if p.suffix.lower() == ".json" or (p.exists() and p.is_file()):
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    key = str(name).strip().lower()
    if key == "default_gen":
        return copy.deepcopy(PRESET_DEFAULT_GEN)
    if key == "my_gen":
        return copy.deepcopy(PRESET_MY_GEN)
    if key == "temporal_preset":
        return copy.deepcopy(TEMPORAL_PRESET)
    if key == "horror_cinematic_preset":
        return copy.deepcopy(HORROR_CINEMATIC_PRESET)
    # Fallback: attempt to load JSON file path by name handled above
    raise ValueError(f"Unknown preset name: {name!r}")
