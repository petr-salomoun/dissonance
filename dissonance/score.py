"""Top-level scoring convenience wrapper."""

from __future__ import annotations

import numpy as np

from dissonance.analysis.scorer import UnpleasantnessScorer


def score(signal: np.ndarray, sr: int) -> tuple[float, dict[str, float]]:
    """Score a signal for unpleasantness and return score + feature breakdown."""
    return UnpleasantnessScorer().score(signal, sr)
