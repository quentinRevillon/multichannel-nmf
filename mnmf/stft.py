"""Multichannel STFT and inverse STFT (Hann window, 50 % overlap).

Usage:
    from mnmf.stft import stft, istft
    X = stft(x, n_fft=2048)                       # x: (I, N) -> X: (F, T, I)
    y = istft(X, n_fft=2048, length=x.shape[1])   # X: (F, T, I) -> y: (I, N)
"""
import numpy as np
from scipy import signal


def stft(x, n_fft=2048):
    """Compute the STFT of each channel.

    Args:
        x: real array (I, N), one row per microphone.
        n_fft: window length in samples; the hop is n_fft // 2.

    Returns:
        Complex array (F, T, I) with F = n_fft // 2 + 1.
    """
    _, _, X = signal.stft(x, nperseg=n_fft, noverlap=n_fft // 2, window="hann")
    return np.moveaxis(X, 0, -1)


def istft(X, n_fft=2048, length=None):
    """Invert `stft` channel by channel.

    Args:
        X: complex array (F, T, I).
        n_fft: window length used by `stft`.
        length: number of output samples to keep (None keeps all).

    Returns:
        Real array (I, N).
    """
    _, x = signal.istft(np.moveaxis(X, -1, 0), nperseg=n_fft, noverlap=n_fft // 2, window="hann")
    return x[:, :length]
