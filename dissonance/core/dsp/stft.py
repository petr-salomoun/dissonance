"""STFT and simple phase-vocoder utilities."""

from __future__ import annotations

import numpy as np
import librosa


def stft(signal: np.ndarray, n_fft: int = 2048, hop: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute STFT and return magnitudes, phases, and frequency bins."""
    x = np.asarray(signal, dtype=np.float32)
    spec = librosa.stft(x, n_fft=int(n_fft), hop_length=int(hop), center=True)
    mags = np.abs(spec).astype(np.float32)
    phases = np.angle(spec).astype(np.float32)
    freqs = np.fft.rfftfreq(int(n_fft), d=1.0).astype(np.float32)
    return mags, phases, freqs


def istft(mags: np.ndarray, phases: np.ndarray, hop: int = 512) -> np.ndarray:
    """Reconstruct a time-domain signal from magnitude and phase STFT parts."""
    m = np.asarray(mags, dtype=np.float32)
    p = np.asarray(phases, dtype=np.float32)
    spec = m * np.exp(1j * p)
    y = librosa.istft(spec, hop_length=int(hop), center=True)
    return np.asarray(y, dtype=np.float32)


def phase_vocoder_stretch(signal: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """Simple time-stretch wrapper using librosa phase vocoder."""
    x = np.asarray(signal, dtype=np.float32)
    r = max(float(rate), 1e-3)
    y = librosa.effects.time_stretch(x, rate=r)
    return np.asarray(y, dtype=np.float32)
