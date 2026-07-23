from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from dissonance.analysis.scorer import UnpleasantnessScorer
from dissonance.io.render import render_from_params


ROOT = Path(__file__).resolve().parents[1]
SHOWCASE_DIR = ROOT / "examples" / "showcase"
PRESETS_DIR = SHOWCASE_DIR / "presets"
AUDIO_DIR = SHOWCASE_DIR / "audio"
MANIFEST_PATH = SHOWCASE_DIR / "manifest.json"


def _legacy_layers_profile() -> list[dict]:
    return [
        {
            "type": "rough",
            "carrier_hz": 3000.0,
            "n_partials": 12,
            "partial_spread_bark": 0.45,
            "am_rate_hz": 80.0,
            "am_depth": 0.95,
            "gain_db": -7.0,
        },
        {
            "type": "stickslip",
            "ioi_mean_ms": 4.5,
            "ioi_jitter": 0.75,
            "impulse_decay_ms": 1.4,
            "resonance_hz": [2200.0, 3150.0, 4200.0],
            "gain_db": -4.5,
        },
        {
            "type": "fm_instab",
            "carrier_hz": 3400.0,
            "mod_rate_hz": 16.0,
            "mod_index": 12.0,
            "mod_chaos": 0.9,
            "gain_db": -10.0,
        },
        {
            "type": "inharmonic",
            "base_hz": 220.0,
            "n_partials": 16,
            "inharmonicity_B": 0.12,
            "random_detune": 0.7,
            "gain_db": -11.0,
        },
        {
            "type": "beating",
            "base_hz": 220.0,
            "n_beaters": 5,
            "beat_rate_hz": 10.0,
            "beat_jitter": 0.6,
            "gain_db": -8.0,
        },
        {
            "type": "noise_shaped",
            "center_hz": 3600.0,
            "bandwidth_hz": 2200.0,
            "modulation_rate_hz": 70.0,
            "modulation_depth": 0.85,
            "gain_db": -12.0,
        },
    ]


def _base_global() -> dict:
    return {
        "hump_2_4khz_db": 8.0,
        "highpass_hz": 80.0,
    }


def _component_presets() -> dict[str, dict]:
    return {
        "component_temporal_01_scream_chaos": {
            "duration_s": 2.4,
            "sample_rate": 22050,
            "global": {"hump_2_4khz_db": 7.0, "highpass_hz": 120.0},
            "layers": [
                {
                    "type": "scream_chaos",
                    "carrier_hz": 900.0,
                    "subharmonic_gain": 0.55,
                    "chaos_amount": 0.85,
                    "pitch_jump_rate_hz": 0.9,
                    "biphonation_ratio": 1.67,
                    "biphonation_gain": 0.35,
                    "gain_db": -9.0,
                }
            ],
        },
        "component_temporal_02_dread_swell": {
            "duration_s": 2.8,
            "sample_rate": 22050,
            "global": {"hump_2_4khz_db": 8.0, "highpass_hz": 60.0},
            "layers": [
                {
                    "type": "dread_swell",
                    "start_hz": 140.0,
                    "end_hz": 1800.0,
                    "rise_shape": "exponential",
                    "roughness_rise": True,
                    "loudness_rise": True,
                    "gain_db": -10.0,
                }
            ],
        },
        "component_temporal_03_shepard_ascent": {
            "duration_s": 2.6,
            "sample_rate": 22050,
            "global": {"hump_2_4khz_db": 7.0, "highpass_hz": 90.0},
            "layers": [
                {
                    "type": "shepard_ascent",
                    "n_voices": 10,
                    "octave_spread": 4.5,
                    "rise_rate_hz_per_s": 0.65,
                    "roughness_am_hz": 42.0,
                    "gain_db": -11.0,
                }
            ],
        },
        "component_temporal_04_pulse_panic": {
            "duration_s": 2.4,
            "sample_rate": 22050,
            "global": {"hump_2_4khz_db": 8.0, "highpass_hz": 120.0},
            "layers": [
                {
                    "type": "pulse_panic",
                    "start_rate_hz": 0.8,
                    "end_rate_hz": 12.0,
                    "acceleration": "exponential",
                    "roughness_am_hz": 75.0,
                    "burst_decay_ms": 8.0,
                    "gain_db": -9.0,
                }
            ],
        },
        "component_temporal_05_doom_throb": {
            "duration_s": 2.8,
            "sample_rate": 22050,
            "global": {"hump_2_4khz_db": 5.0, "highpass_hz": 20.0},
            "layers": [
                {
                    "type": "doom_throb",
                    "center_hz": 32.0,
                    "detune_hz_start": 0.3,
                    "detune_hz_end": 5.2,
                    "am_rate_hz": 0.28,
                    "gain_db": -8.0,
                }
            ],
        },
        "component_temporal_06_wobble_drift": {
            "duration_s": 2.5,
            "sample_rate": 22050,
            "global": {"hump_2_4khz_db": 7.0, "highpass_hz": 80.0},
            "layers": [
                {
                    "type": "wobble_drift",
                    "base_hz": 260.0,
                    "detune_start_hz": 0.2,
                    "detune_end_hz": 24.0,
                    "drift_shape": "exponential",
                    "n_harmonics": 6,
                    "gain_db": -9.0,
                }
            ],
        },
        "component_temporal_07_uncanny_morph": {
            "duration_s": 2.6,
            "sample_rate": 22050,
            "global": {"hump_2_4khz_db": 7.5, "highpass_hz": 70.0},
            "layers": [
                {
                    "type": "uncanny_morph",
                    "base_hz": 210.0,
                    "n_partials": 14,
                    "inharmonicity_start": 0.01,
                    "inharmonicity_end": 0.38,
                    "formant_sweep": True,
                    "gain_db": -10.0,
                }
            ],
        },
    }


def _composite_presets() -> dict[str, dict]:
    return {
        "composite_01_predator_clock": {
            "duration_s": 2.6,
            "sample_rate": 22050,
            "global": _base_global(),
            "layers": _legacy_layers_profile()
            + [
                {
                    "type": "scream_chaos",
                    "carrier_hz": 720.0,
                    "subharmonic_gain": 0.45,
                    "chaos_amount": 0.72,
                    "pitch_jump_rate_hz": 0.7,
                    "biphonation_ratio": 1.52,
                    "biphonation_gain": 0.25,
                    "gain_db": -10.0,
                },
                {
                    "type": "pulse_panic",
                    "start_rate_hz": 0.7,
                    "end_rate_hz": 10.0,
                    "acceleration": "exponential",
                    "roughness_am_hz": 65.0,
                    "burst_decay_ms": 10.0,
                    "gain_db": -10.0,
                },
                {
                    "type": "uncanny_morph",
                    "base_hz": 180.0,
                    "n_partials": 12,
                    "inharmonicity_start": 0.02,
                    "inharmonicity_end": 0.28,
                    "formant_sweep": True,
                    "gain_db": -11.0,
                },
            ],
        },
        "composite_02_escalating_machine": {
            "duration_s": 2.8,
            "sample_rate": 22050,
            "global": {"hump_2_4khz_db": 9.0, "highpass_hz": 45.0},
            "layers": _legacy_layers_profile()
            + [
                {
                    "type": "dread_swell",
                    "start_hz": 120.0,
                    "end_hz": 1900.0,
                    "rise_shape": "exponential",
                    "roughness_rise": True,
                    "loudness_rise": True,
                    "gain_db": -10.0,
                },
                {
                    "type": "shepard_ascent",
                    "n_voices": 12,
                    "octave_spread": 5.0,
                    "rise_rate_hz_per_s": 0.6,
                    "roughness_am_hz": 36.0,
                    "gain_db": -11.0,
                },
                {
                    "type": "doom_throb",
                    "center_hz": 28.0,
                    "detune_hz_start": 0.2,
                    "detune_hz_end": 4.6,
                    "am_rate_hz": 0.22,
                    "gain_db": -9.0,
                },
                {
                    "type": "wobble_drift",
                    "base_hz": 220.0,
                    "detune_start_hz": 0.3,
                    "detune_end_hz": 20.0,
                    "drift_shape": "linear",
                    "n_harmonics": 4,
                    "gain_db": -10.0,
                },
            ],
        },
    }


def _ab_candidate_presets(composites: dict[str, dict]) -> dict[str, dict]:
    return {
        "ab_pair01_A_legacy_control": {
            "duration_s": 2.6,
            "sample_rate": 22050,
            "global": _base_global(),
            "layers": _legacy_layers_profile(),
            "_ab_candidate": {
                "pair_id": 1,
                "role": "A_control",
                "contrast": "legacy_vs_legacy_plus_all_temporal",
            },
        },
        "ab_pair01_B_legacy_plus_all_temporal": {
            "duration_s": 2.6,
            "sample_rate": 22050,
            "global": _base_global(),
            "layers": _legacy_layers_profile()
            + [
                {
                    "type": "scream_chaos",
                    "carrier_hz": 700.0,
                    "subharmonic_gain": 0.5,
                    "chaos_amount": 0.7,
                    "pitch_jump_rate_hz": 0.6,
                    "biphonation_ratio": 1.52,
                    "biphonation_gain": 0.25,
                    "gain_db": -10.0,
                },
                {
                    "type": "dread_swell",
                    "start_hz": 140.0,
                    "end_hz": 1700.0,
                    "rise_shape": "exponential",
                    "roughness_rise": True,
                    "loudness_rise": True,
                    "gain_db": -10.0,
                },
                {
                    "type": "shepard_ascent",
                    "n_voices": 10,
                    "octave_spread": 4.5,
                    "rise_rate_hz_per_s": 0.55,
                    "roughness_am_hz": 30.0,
                    "gain_db": -11.0,
                },
                {
                    "type": "pulse_panic",
                    "start_rate_hz": 1.0,
                    "end_rate_hz": 9.0,
                    "acceleration": "exponential",
                    "roughness_am_hz": 70.0,
                    "burst_decay_ms": 10.0,
                    "gain_db": -10.0,
                },
                {
                    "type": "doom_throb",
                    "center_hz": 30.0,
                    "detune_hz_start": 0.4,
                    "detune_hz_end": 4.2,
                    "am_rate_hz": 0.2,
                    "gain_db": -9.5,
                },
                {
                    "type": "wobble_drift",
                    "base_hz": 200.0,
                    "detune_start_hz": 0.4,
                    "detune_end_hz": 18.0,
                    "drift_shape": "linear",
                    "n_harmonics": 4,
                    "gain_db": -10.0,
                },
                {
                    "type": "uncanny_morph",
                    "base_hz": 190.0,
                    "n_partials": 12,
                    "inharmonicity_start": 0.01,
                    "inharmonicity_end": 0.3,
                    "formant_sweep": True,
                    "gain_db": -11.0,
                },
            ],
            "_ab_candidate": {
                "pair_id": 1,
                "role": "B_variant",
                "contrast": "legacy_vs_legacy_plus_all_temporal",
            },
        },
        "ab_pair02_A_predator_clock": {
            **composites["composite_01_predator_clock"],
            "_ab_candidate": {
                "pair_id": 2,
                "role": "A_variant",
                "contrast": "composite_01_vs_composite_02",
            },
        },
        "ab_pair02_B_escalating_machine": {
            **composites["composite_02_escalating_machine"],
            "_ab_candidate": {
                "pair_id": 2,
                "role": "B_variant",
                "contrast": "composite_01_vs_composite_02",
            },
        },
    }


def _build_presets() -> dict[str, dict]:
    components = _component_presets()
    composites = _composite_presets()
    ab_set = _ab_candidate_presets(composites)
    return {**components, **composites, **ab_set}


def _collect_temporal_types(layers: list[dict]) -> list[str]:
    temporal_types = {
        "scream_chaos",
        "dread_swell",
        "shepard_ascent",
        "pulse_panic",
        "doom_throb",
        "wobble_drift",
        "uncanny_morph",
    }
    return [str(layer.get("type", "")) for layer in layers if str(layer.get("type", "")) in temporal_types]


def _collect_legacy_types(layers: list[dict]) -> list[str]:
    legacy_types = {"rough", "stickslip", "fm_instab", "inharmonic", "beating", "noise_shaped"}
    return [str(layer.get("type", "")) for layer in layers if str(layer.get("type", "")) in legacy_types]


def _load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    x = np.asarray(data, dtype=np.float32)
    if x.ndim == 2:
        x = np.mean(x, axis=1, dtype=np.float32)
    return x, int(sr)


def generate_showcase() -> dict:
    presets = _build_presets()
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    scorer = UnpleasantnessScorer()
    manifest_entries: list[dict] = []

    for name in sorted(presets.keys()):
        params = presets[name]
        preset_path = PRESETS_DIR / f"{name}.json"
        wav_path = AUDIO_DIR / f"{name}.wav"

        with preset_path.open("w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, sort_keys=True)

        render_from_params(params, str(wav_path))
        sidecar_path = wav_path.with_name(f"{wav_path.stem}.params.json")

        if not sidecar_path.exists():
            raise RuntimeError(f"Missing sidecar for {wav_path.name}")

        with sidecar_path.open("r", encoding="utf-8") as f:
            sidecar = json.load(f)

        sidecar_layers = sidecar.get("layers", []) if isinstance(sidecar, dict) else []
        temporal_layers = _collect_temporal_types(sidecar_layers)
        legacy_layers = _collect_legacy_types(sidecar_layers)

        if name.startswith("component_temporal_") and len(temporal_layers) != 1:
            raise RuntimeError(f"Expected exactly one temporal layer in {name}, got {temporal_layers}")
        if name.startswith("composite_") and (len(temporal_layers) < 2 or len(legacy_layers) < 3):
            raise RuntimeError(f"Composite verification failed for {name}")
        if name.startswith("ab_pair") and "_ab_candidate" not in sidecar:
            raise RuntimeError(f"A/B sidecar metadata missing for {name}")

        x, sr = _load_wav_mono(wav_path)
        score, features = scorer.score(
            x,
            sr,
            layers=sidecar_layers,
            duration_s=float(sidecar.get("duration_s", params.get("duration_s", 2.5))),
        )

        category = "component" if name.startswith("component_") else "composite" if name.startswith("composite_") else "ab_candidate"
        entry = {
            "id": name,
            "category": category,
            "preset_path": str(preset_path.relative_to(ROOT)),
            "wav_path": str(wav_path.relative_to(ROOT)),
            "sidecar_path": str(sidecar_path.relative_to(ROOT)),
            "duration_s": float(sidecar.get("duration_s", params.get("duration_s", 0.0))),
            "sample_rate": int(sidecar.get("sample_rate", params.get("sample_rate", 0))),
            "legacy_layers": legacy_layers,
            "temporal_layers": temporal_layers,
            "score": float(score),
            "features": {k: float(v) for k, v in sorted(features.items())},
        }
        if isinstance(sidecar.get("_ab_candidate"), dict):
            entry["ab"] = sidecar["_ab_candidate"]
        manifest_entries.append(entry)

    manifest = {
        "generator": "scripts/generate_showcase_examples.py",
        "render_pipeline": "dissonance.io.render.render_from_params",
        "scorer": "dissonance.analysis.scorer.UnpleasantnessScorer",
        "entries": manifest_entries,
    }
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


if __name__ == "__main__":
    result = generate_showcase()
    print(f"Generated {len(result['entries'])} curated showcase examples.")
    print(f"Manifest: {MANIFEST_PATH.relative_to(ROOT)}")
