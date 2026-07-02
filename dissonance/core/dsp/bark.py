"""Bark scale and critical-band helper utilities."""

from __future__ import annotations

import numpy as np


def hz_to_bark(f: float | np.ndarray) -> float | np.ndarray:
    """Convert frequency in Hz to Bark using Traunmüller's formula."""
    f_arr = np.asarray(f, dtype=np.float64)
    z = (26.81 * f_arr) / (1960.0 + f_arr) - 0.53
    z = np.where(z < 2.0, z + 0.15 * (2.0 - z), z)
    z = np.where(z > 20.1, z + 0.22 * (z - 20.1), z)
    z = z.astype(np.float32)
    return float(z) if np.isscalar(f) else z


def bark_to_hz(b: float | np.ndarray) -> float | np.ndarray:
    """Convert Bark to frequency in Hz (inverse of Traunmüller approximation)."""
    z = np.asarray(b, dtype=np.float64)
    z0 = np.where(z < 2.0, (z - 0.3) / 0.85, z)
    z0 = np.where(z > 20.1, (z + 4.422) / 1.22, z0)
    f = 1960.0 * (z0 + 0.53) / (26.28 - z0)
    f = np.maximum(f, 0.0).astype(np.float32)
    return float(f) if np.isscalar(b) else f


def critical_bandwidth_hz(f: float | np.ndarray) -> float | np.ndarray:
    """Approximate critical bandwidth (Hz) at center frequency f (Hz)."""
    f_arr = np.asarray(f, dtype=np.float64)
    cbw = 25.0 + 75.0 * np.power(1.0 + 1.4 * np.power(f_arr / 1000.0, 2.0), 0.69)
    cbw = cbw.astype(np.float32)
    return float(cbw) if np.isscalar(f) else cbw


CRITICAL_BAND_EDGES = bark_to_hz(np.arange(0, 25, dtype=np.float32)).tolist()
