"""Heuristic unpleasantness scoring."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dissonance.analysis.features import compute_features


class UnpleasantnessScorer:
    """Weighted feature scorer producing an unpleasantness score in [0, 1]."""

    DEFAULT_WEIGHTS: dict[str, float] = {
        "roughness": 0.30,
        "sharpness": 0.20,
        "dissonance": 0.15,
        "crest_factor": 0.15,
        "band_energy_2_4khz": 0.10,
        "am_energy_70hz": 0.10,
        "roughness_x_sharpness": 0.10,
    }

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        merged = dict(self.DEFAULT_WEIGHTS)
        if weights:
            for key, value in weights.items():
                if key in merged:
                    merged[key] = float(value)
        self.weights = merged

    @classmethod
    def from_weights_file(cls, path: str) -> "UnpleasantnessScorer":
        """Load a scorer calibrated from an AB test weights file."""
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid weights JSON at {path}: expected object mapping feature->weight")
        return cls(weights={str(k): float(v) for k, v in payload.items()})

    def score(self, signal: np.ndarray, sr: int) -> tuple[float, dict[str, float]]:
        """Compute unpleasantness score and return score with feature breakdown."""
        f = compute_features(signal, sr)

        rough = float(f["roughness"])
        sharp = float(f["sharpness"])
        diss = float(f["dissonance"])
        crest = float(f["crest_factor"])
        band = float(f["band_energy_2_4khz"])
        am70 = float(f["am_energy_70hz"])

        w = self.weights
        s = (
            w["roughness"] * rough
            + w["sharpness"] * sharp
            + w["dissonance"] * diss
            + w["crest_factor"] * crest
            + w["band_energy_2_4khz"] * band
            + w["am_energy_70hz"] * am70
            + w["roughness_x_sharpness"] * rough * sharp
        )
        s = float(np.clip(s, 0.0, 1.0))
        return s, f
