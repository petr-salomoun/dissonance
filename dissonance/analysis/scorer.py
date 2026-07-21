"""Heuristic unpleasantness scoring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dissonance.analysis.features import compute_features
from dissonance.analysis.synth_features import compute_synth_features


class UnpleasantnessScorer:
    """Weighted feature scorer producing an unpleasantness score in [0, 1]."""

    SYNTH_FEATURE_NAMES: list[str] = [
        "layer_scream_chaos",
        "layer_dread_swell",
        "layer_shepard_ascent",
        "layer_pulse_panic",
        "layer_doom_throb",
        "layer_wobble_drift",
        "layer_uncanny_morph",
        "synth_temporal_density",
        "synth_n_layers",
        "scream_chaos_biphonation",
        "dread_swell_rise",
        "shepard_n_voices",
        "pulse_panic_rate",
        "doom_throb_detune",
        "wobble_detune_end",
        "uncanny_inharmonicity",
    ]

    DEFAULT_WEIGHTS: dict[str, float] = {
        "roughness": 0.30,
        "sharpness": 0.20,
        "dissonance": 0.15,
        "crest_factor": 0.15,
        "band_energy_2_4khz": 0.10,
        "am_energy_70hz": 0.10,
        "roughness_x_sharpness": 0.10,
    }
    DEFAULT_SYNTH_WEIGHTS: dict[str, float] = {k: 0.0 for k in SYNTH_FEATURE_NAMES}

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        preference_model: dict[str, object] | None = None,
    ) -> None:
        merged = dict(self.DEFAULT_WEIGHTS)
        merged.update(self.DEFAULT_SYNTH_WEIGHTS)
        if weights:
            for key, value in weights.items():
                if key in merged:
                    merged[key] = float(value)
        self.weights = merged
        self.preference_model = preference_model if isinstance(preference_model, dict) else None

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "UnpleasantnessScorer":
        model = payload.get("preference_model") if isinstance(payload, dict) else None
        raw_weights = payload.get("weights") if isinstance(payload, dict) else None
        if isinstance(raw_weights, dict):
            weights = {str(k): float(v) for k, v in raw_weights.items() if str(k) in (cls.DEFAULT_WEIGHTS | cls.DEFAULT_SYNTH_WEIGHTS)}
        else:
            weights = None
        return cls(weights=weights, preference_model=model if isinstance(model, dict) else None)

    @classmethod
    def from_weights_file(cls, path: str) -> "UnpleasantnessScorer":
        """Load a scorer calibrated from an AB test weights file."""
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid weights JSON at {path}: expected object mapping feature->weight")
        if "weights" in payload or "preference_model" in payload:
            return cls.from_payload(payload)
        return cls(weights={str(k): float(v) for k, v in payload.items()})

    def _score_preference_model(self, features: dict[str, float]) -> float | None:
        model = self.preference_model
        if not isinstance(model, dict):
            return None
        feature_names = model.get("feature_names")
        identified = model.get("identified_features")
        means = model.get("means")
        scales = model.get("scales")
        beta_std = model.get("beta_std")
        intercept = float(model.get("intercept", 0.0))
        if not all(isinstance(x, dict) for x in (means, scales, beta_std)):
            return None
        if not isinstance(feature_names, list):
            return None
        identified_set: set[str] | None = None
        if isinstance(identified, list):
            identified_set = {str(x) for x in identified if isinstance(x, str)}
        lin = intercept
        for name in feature_names:
            if not isinstance(name, str):
                continue
            if identified_set is not None and name not in identified_set:
                continue
            mu = float(means.get(name, 0.0))
            scale = max(float(scales.get(name, 1.0)), 1e-3)
            beta = float(beta_std.get(name, 0.0))
            x = float(features.get(name, 0.0))
            lin += beta * ((x - mu) / scale)
        lin = float(np.clip(lin, -12.0, 12.0))
        return float(1.0 / (1.0 + np.exp(-lin)))

    def score(
        self,
        signal: np.ndarray,
        sr: int,
        layers: list[dict] | None = None,
        duration_s: float = 5.0,
    ) -> tuple[float, dict[str, float]]:
        """Compute unpleasantness score and return score with feature breakdown."""
        acoustic_feats = compute_features(signal, sr)
        synth_feats = dict(self.DEFAULT_SYNTH_WEIGHTS)
        if layers is not None:
            synth_feats.update(compute_synth_features(layers, duration_s))

        rough = float(acoustic_feats["roughness"])
        sharp = float(acoustic_feats["sharpness"])
        diss = float(acoustic_feats["dissonance"])
        crest = float(acoustic_feats["crest_factor"])
        band = float(acoustic_feats["band_energy_2_4khz"])
        am70 = float(acoustic_feats["am_energy_70hz"])

        w = self.weights
        acoustic_score = (
            w["roughness"] * rough
            + w["sharpness"] * sharp
            + w["dissonance"] * diss
            + w["crest_factor"] * crest
            + w["band_energy_2_4khz"] * band
            + w["am_energy_70hz"] * am70
            + w["roughness_x_sharpness"] * rough * sharp
        )
        s_synth = float(sum(w[k] * float(synth_feats[k]) for k in self.SYNTH_FEATURE_NAMES))
        all_features = {**acoustic_feats, **synth_feats}
        model_score = self._score_preference_model(all_features)
        s_total = float(np.clip(acoustic_score + s_synth, 0.0, 1.0))
        if model_score is not None:
            model = self.preference_model if isinstance(self.preference_model, dict) else {}
            n_identified = len(model.get("identified_features", [])) if isinstance(model.get("identified_features"), list) else len(model.get("active_features", [])) if isinstance(model.get("active_features"), list) else 0
            n_features = len(model.get("feature_names", [])) if isinstance(model.get("feature_names"), list) else len(self.DEFAULT_WEIGHTS | self.DEFAULT_SYNTH_WEIGHTS)
            evidence = float(model.get("n_directional", 0)) if isinstance(model, dict) else 0.0
            coverage = 0.0 if n_features <= 0 else float(np.clip(n_identified / float(n_features), 0.0, 1.0))
            confidence = float(np.clip(evidence / (evidence + 24.0), 0.0, 1.0))
            blend = float(np.clip(0.65 * coverage * confidence, 0.0, 0.65))
            s_total = float(np.clip((1.0 - blend) * s_total + blend * float(model_score), 0.0, 1.0))
        return s_total, all_features
