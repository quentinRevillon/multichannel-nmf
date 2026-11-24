"""Simulation of reverberant (convolutive) multichannel mixtures.

Each source j is convolved with one room impulse response (RIR) per
microphone i; the mixture is the sum of the resulting spatial images.

Usage:
    from mnmf.mixing import load_air_rir, spatial_images
    h = load_air_rir("data/AIR_1_4/air_binaural_aula_carolina_1_1_3_45_3.mat", sr=44100)
    images = spatial_images([vocals, guitar], [[h11, h21], [h12, h22]])  # (J, I, N)
    mixture = images.sum(axis=0)                                         # (I, N)
"""
import numpy as np
from scipy.io import loadmat
from scipy.signal import fftconvolve, resample_poly


def load_air_rir(path, sr):
    """Load a RIR from an Aachen Impulse Response (AIR) .mat file.

    Args:
        path: path to an AIR .mat file (fields `h_air` and `air_info.fs`).
        sr: target sampling rate in Hz; the RIR is resampled if needed.

    Returns:
        1-D float array, normalised to unit energy.
    """
    mat = loadmat(path, squeeze_me=True)
    h = resample_poly(mat["h_air"], sr, int(mat["air_info"]["fs"]))
    return h / np.linalg.norm(h)


def spatial_image(source, rirs):
    """Convolve one mono source with one RIR per microphone.

    Args:
        source: 1-D array (N,).
        rirs: list of I 1-D arrays, one RIR per microphone.

    Returns:
        Array (I, N), truncated to the source length.
    """
    return np.stack([fftconvolve(source, h)[: len(source)] for h in rirs])


def spatial_images(sources, rirs):
    """Compute the spatial image of every source.

    Args:
        sources: list of J 1-D arrays of equal length N.
        rirs: rirs[j][i] is the RIR from source j to microphone i.

    Returns:
        Array (J, I, N); the mixture is its sum over axis 0.
    """
    return np.stack([spatial_image(s, r) for s, r in zip(sources, rirs)])
