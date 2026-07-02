"""Small filter/EQ utility functions."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter


def _as_float32(signal: np.ndarray) -> np.ndarray:
    return np.asarray(signal, dtype=np.float32)


def _safe_cast_float32(arr: np.ndarray) -> np.ndarray:
    """Safely cast a numeric array to float32: replace NaN/Inf, clip extremes."""
    a = np.asarray(arr)
    if a.dtype != np.float64:
        a = a.astype(np.float64)
    # replace NaN/Inf with 0 and clip to a bounded range before casting
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    a = np.clip(a, -1e6, 1e6)
    return a.astype(np.float32)


def peaking_eq(signal: np.ndarray, sr: int, center_hz: float, gain_db: float, q: float) -> np.ndarray:
    """Apply a peaking EQ biquad (RBJ cookbook form)."""
    # keep input in float64 for filtering precision/stability
    x = np.asarray(signal, dtype=np.float64)
    omega = 2.0 * np.pi * float(center_hz) / float(sr)
    alpha = np.sin(omega) / (2.0 * float(q))
    a = 10.0 ** (float(gain_db) / 40.0)

    b0 = 1.0 + alpha * a
    b1 = -2.0 * np.cos(omega)
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * np.cos(omega)
    a2 = 1.0 - alpha / a

    b = np.array([b0, b1, b2], dtype=np.float64) / a0
    aa = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    # perform filtering in float64
    y = lfilter(b, aa, x)
    return _safe_cast_float32(y)


def bandpass(signal: np.ndarray, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    """Apply a 4th-order Butterworth band-pass filter."""
    # ensure float64 input for filtering
    x = np.asarray(signal, dtype=np.float64)
    nyq = 0.5 * float(sr)
    low = max(float(low_hz) / nyq, 1e-6)
    high = min(float(high_hz) / nyq, 0.999999)
    if high <= low:
        return np.zeros_like(x, dtype=np.float32)
    b, a = butter(4, [low, high], btype="band")
    y = lfilter(b, a, x)
    return _safe_cast_float32(y)


def highpass(signal: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    """Apply a 4th-order Butterworth high-pass filter."""
    x = np.asarray(signal, dtype=np.float64)
    nyq = 0.5 * float(sr)
    cutoff = min(max(float(cutoff_hz) / nyq, 1e-6), 0.999999)
    b, a = butter(4, cutoff, btype="high")
    y = lfilter(b, a, x)
    return _safe_cast_float32(y)


def boost_2_4khz(signal: np.ndarray, sr: int, gain_db: float = 9.0) -> np.ndarray:
    """Apply the characteristic 2–4 kHz emphasis hump."""
    return peaking_eq(signal=signal, sr=sr, center_hz=3000.0, gain_db=gain_db, q=2.0)
