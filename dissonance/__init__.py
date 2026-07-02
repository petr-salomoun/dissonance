"""Top-level package exports."""

from dissonance.analysis.features import compute_features
from dissonance.analysis.scorer import UnpleasantnessScorer
from dissonance.io.render import render_from_params as render


def score(signal, sr):
    return UnpleasantnessScorer().score(signal, sr)


__all__ = ["render", "score", "compute_features", "UnpleasantnessScorer"]
