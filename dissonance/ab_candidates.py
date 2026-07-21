"""Deterministic A/B calibration candidate generation for temporal synth features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dissonance.io.render import render_from_params


@dataclass(frozen=True, slots=True)
class TemporalCalibrationSpec:
    layer_type: str
    param_key: str
    values: tuple[float, ...]
    gain_db: float
    base_params: dict[str, float | int | bool | str]


BASE_GLOBAL: dict[str, float] = {
    "hump_2_4khz_db": 9.0,
    "highpass_hz": 800.0,
}


BASE_LEGACY_LAYERS: list[dict[str, float | int | list[float] | str]] = [
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
]


TEMPORAL_CALIBRATION_SPECS: tuple[TemporalCalibrationSpec, ...] = (
    TemporalCalibrationSpec(
        layer_type="scream_chaos",
        param_key="biphonation_ratio",
        values=(1.31, 1.52, 1.83),
        gain_db=-8.0,
        base_params={
            "carrier_hz": 700.0,
            "subharmonic_gain": 0.4,
            "chaos_amount": 0.5,
            "pitch_jump_rate_hz": 0.5,
            "biphonation_ratio": 1.52,
            "biphonation_gain": 0.2,
        },
    ),
    TemporalCalibrationSpec(
        layer_type="dread_swell",
        param_key="end_hz",
        values=(900.0, 1400.0, 2200.0),
        gain_db=-8.0,
        base_params={
            "start_hz": 160.0,
            "end_hz": 1400.0,
            "rise_shape": "exponential",
            "roughness_rise": True,
            "loudness_rise": True,
        },
    ),
    TemporalCalibrationSpec(
        layer_type="shepard_ascent",
        param_key="n_voices",
        values=(6.0, 9.0, 12.0),
        gain_db=-10.0,
        base_params={
            "n_voices": 9,
            "octave_spread": 4.0,
            "rise_rate_hz_per_s": 0.5,
            "roughness_am_hz": 40.0,
        },
    ),
    TemporalCalibrationSpec(
        layer_type="pulse_panic",
        param_key="end_rate_hz",
        values=(5.0, 8.0, 12.0),
        gain_db=-9.0,
        base_params={
            "start_rate_hz": 1.0,
            "end_rate_hz": 8.0,
            "acceleration": "exponential",
            "roughness_am_hz": 70.0,
            "burst_decay_ms": 10.0,
        },
    ),
    TemporalCalibrationSpec(
        layer_type="doom_throb",
        param_key="detune_hz_end",
        values=(1.5, 3.0, 6.0),
        gain_db=-6.0,
        base_params={
            "center_hz": 30.0,
            "detune_hz_start": 0.5,
            "detune_hz_end": 3.0,
            "am_rate_hz": 0.3,
        },
    ),
    TemporalCalibrationSpec(
        layer_type="wobble_drift",
        param_key="detune_end_hz",
        values=(6.0, 12.0, 30.0),
        gain_db=-8.0,
        base_params={
            "base_hz": 200.0,
            "detune_start_hz": 0.5,
            "detune_end_hz": 12.0,
            "drift_shape": "linear",
            "n_harmonics": 4,
        },
    ),
    TemporalCalibrationSpec(
        layer_type="uncanny_morph",
        param_key="inharmonicity_end",
        values=(0.1, 0.25, 0.4),
        gain_db=-9.0,
        base_params={
            "base_hz": 200.0,
            "n_partials": 12,
            "inharmonicity_start": 0.02,
            "inharmonicity_end": 0.25,
            "formant_sweep": True,
        },
    ),
)


def build_ab_candidate_payloads(
    duration_s: float,
    sample_rate: int,
    seed: int,
    repeats: int,
) -> list[tuple[str, dict]]:
    """Build paired A/B calibration payloads with one temporal feature varied at a time."""
    rng = np.random.default_rng(int(seed))
    payloads: list[tuple[str, dict]] = []
    pair_idx = 1

    for rep in range(max(1, int(repeats))):
        spec_order = rng.permutation(len(TEMPORAL_CALIBRATION_SPECS))
        for spec_pos in spec_order:
            spec = TEMPORAL_CALIBRATION_SPECS[int(spec_pos)]
            value_order = rng.permutation(len(spec.values))
            for value_pos in value_order:
                value = float(spec.values[int(value_pos)])
                level_rank = int(value_pos) + 1
                stem = f"pair{pair_idx:04d}_{spec.layer_type}_{spec.param_key}_l{level_rank}_r{rep + 1}"

                base_payload = {
                    "duration_s": float(duration_s),
                    "sample_rate": int(sample_rate),
                    "global": dict(BASE_GLOBAL),
                    "layers": [dict(layer) for layer in BASE_LEGACY_LAYERS],
                    "_ab_candidate": {
                        "pair_id": pair_idx,
                        "role": "A_base",
                        "temporal_layer": spec.layer_type,
                        "param": spec.param_key,
                        "param_value": value,
                        "repeat": rep + 1,
                        "level_rank": level_rank,
                    },
                }
                variant_layer = {
                    "type": spec.layer_type,
                    "gain_db": float(spec.gain_db),
                    **dict(spec.base_params),
                    spec.param_key: value,
                }
                variant_payload = {
                    "duration_s": float(duration_s),
                    "sample_rate": int(sample_rate),
                    "global": dict(BASE_GLOBAL),
                    "layers": [dict(layer) for layer in BASE_LEGACY_LAYERS] + [variant_layer],
                    "_ab_candidate": {
                        "pair_id": pair_idx,
                        "role": "B_variant",
                        "temporal_layer": spec.layer_type,
                        "param": spec.param_key,
                        "param_value": value,
                        "repeat": rep + 1,
                        "level_rank": level_rank,
                    },
                }
                payloads.append((f"{stem}_A.wav", base_payload))
                payloads.append((f"{stem}_B.wav", variant_payload))
                pair_idx += 1
    return payloads


def generate_ab_candidates(
    out_dir: str | Path,
    duration_s: float = 2.0,
    sample_rate: int = 22050,
    seed: int = 42,
    repeats: int = 1,
) -> list[Path]:
    """Render A/B calibration WAVs (+ sidecars) that isolate temporal layer presence and key params."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    for filename, payload in build_ab_candidate_payloads(duration_s, sample_rate, seed, repeats):
        wav_path = out / filename
        render_from_params(payload, str(wav_path))
        rendered.append(wav_path)
    return rendered
