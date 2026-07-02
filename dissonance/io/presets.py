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
    # Fallback: attempt to load JSON file path by name handled above
    raise ValueError(f"Unknown preset name: {name!r}")
