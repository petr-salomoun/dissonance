"""Audio feature extraction for unpleasantness heuristics."""

from __future__ import annotations

import math
from typing import Any

import librosa
import numpy as np
from scipy.signal import hilbert, find_peaks


def _mono_float32(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=-1, dtype=np.float32)
    return x.astype(np.float32)


def _band_energy_fraction(x: np.ndarray, sr: int, low_hz: float, high_hz: float) -> float:
    spec = np.fft.rfft(x.astype(np.float64))
    p = (np.abs(spec) ** 2).astype(np.float64)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / float(sr))
    total = float(np.sum(p) + 1e-12)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    return float(np.sum(p[mask]) / total)


def _am_energy_around(x: np.ndarray, sr: int, center_hz: float, width_hz: float) -> float:
    env = np.abs(hilbert(x.astype(np.float64))).astype(np.float64)
    env = env - np.mean(env)
    m = np.fft.rfft(env)
    p = np.abs(m) ** 2
    f = np.fft.rfftfreq(env.size, d=1.0 / float(sr))
    total = float(np.sum(p) + 1e-12)
    band = (f >= (center_hz - width_hz)) & (f <= (center_hz + width_hz))
    return float(np.sum(p[band]) / total)


def _plomp_levelt_dissonance(x: np.ndarray, sr: int, max_peaks: int = 20) -> float:
    spec = np.fft.rfft(x.astype(np.float64) * np.hanning(x.size))
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / float(sr))
    peaks, _ = find_peaks(mag)
    if peaks.size < 2:
        return 0.0

    order = np.argsort(mag[peaks])[::-1]
    peaks = peaks[order[:max_peaks]]
    f = freqs[peaks]
    a = mag[peaks] / (np.max(mag[peaks]) + 1e-12)

    dsum = 0.0
    for i in range(len(f)):
        for j in range(i + 1, len(f)):
            f1, f2 = min(f[i], f[j]), max(f[i], f[j])
            aij = float(a[i] * a[j])
            s = 0.24 / (0.021 * f1 + 19.0)
            xij = (f2 - f1) * s
            d = aij * (math.exp(-3.5 * xij) - math.exp(-5.75 * xij))
            if d > 0.0:
                dsum += d

    return float(np.clip(dsum / 5.0, 0.0, 1.0))


def _roughness_with_mosqito(signal: np.ndarray, sr: int) -> float | None:
    try:
        from mosqito.sq_metrics.roughness.roughness_dw import roughness_dw  # type: ignore

        time, values = roughness_dw(signal.astype(np.float64), int(sr), overlap=0)
        _ = time
        if np.size(values) == 0:
            return None
        val = float(np.nanmean(values))
        return float(np.clip(val / 1.0, 0.0, 1.0))
    except Exception:
        return None


def compute_features(signal: np.ndarray, sr: int) -> dict[str, float]:
    """Compute heuristic unpleasantness-related audio features."""
    x = _mono_float32(signal)
    if x.size == 0:
        return {
            "roughness": 0.0,
            "sharpness": 0.0,
            "dissonance": 0.0,
            "crest_factor": 0.0,
            "band_energy_2_4khz": 0.0,
            "spectral_centroid": 0.0,
            "spectral_flatness": 0.0,
            "am_energy_70hz": 0.0,
        }

    am70 = _am_energy_around(x, sr, center_hz=70.0, width_hz=20.0)
    rough = _roughness_with_mosqito(x, sr)
    if rough is None:
        rough = float(np.clip(am70 * 8.0, 0.0, 1.0))

    band_2_4 = _band_energy_fraction(x, sr, 2000.0, 4000.0)
    sharpness = float(np.clip(band_2_4 / 0.35, 0.0, 1.0))

    diss = _plomp_levelt_dissonance(x, sr)

    peak = float(np.max(np.abs(x)) + 1e-12)
    rms = float(np.sqrt(np.mean(np.square(x.astype(np.float64))) + 1e-12))
    crest = peak / rms
    crest_norm = float(np.clip(np.log10(crest) / np.log10(20.0), 0.0, 1.0))

    centroid = float(
        np.mean(librosa.feature.spectral_centroid(y=x.astype(np.float64), sr=int(sr)))
    )
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=x.astype(np.float64))))

    return {
        "roughness": float(np.clip(rough, 0.0, 1.0)),
        "sharpness": float(np.clip(sharpness, 0.0, 1.0)),
        "dissonance": float(np.clip(diss, 0.0, 1.0)),
        "crest_factor": crest_norm,
        "band_energy_2_4khz": float(np.clip(band_2_4, 0.0, 1.0)),
        "spectral_centroid": centroid,
        "spectral_flatness": float(np.clip(flatness, 0.0, 1.0)),
        "am_energy_70hz": float(np.clip(am70, 0.0, 1.0)),
    }
