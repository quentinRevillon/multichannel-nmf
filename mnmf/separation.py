"""Source separation by multichannel Wiener filtering of the fitted model.

The spatial image of source j is estimated as y_j(f, t) = a_j(f) E[s_j(f, t) | x].

Usage:
    from mnmf.separation import separate
    images = separate(mixture, n_sources=2, n_components=4, n_iter=100)  # (I, N) -> (J, I, N)
"""
import numpy as np

from mnmf.em import component_variances, fit, mixture_covariance, source_variances, wiener_gain
from mnmf.stft import istft, stft


def source_images_stft(X, params, labels):
    """Estimate the STFT of each source spatial image.

    Args:
        X: mixture STFT (F, T, I).
        params: fitted parameters from `mnmf.em.fit`.
        labels: source index of each component (K,).

    Returns:
        Complex array (J, F, T, I).
    """
    Vs = source_variances(component_variances(params["W"], params["H"]), labels)
    Sx_inv = np.linalg.inv(mixture_covariance(params["A"], Vs, params["sigma_b"]))
    s = np.einsum("ftji,fti->ftj", wiener_gain(params["A"], Vs, Sx_inv), X)
    return np.einsum("fij,ftj->jfti", params["A"], s)


def separate(mixture, n_sources, n_components, n_iter=100, n_fft=2048, seed=0):
    """Separate a multichannel mixture into source spatial images.

    Args:
        mixture: real array (I, N).
        n_sources: number of sources J.
        n_components: number of NMF components per source.
        n_iter: number of EM iterations.
        n_fft: STFT window length in samples.
        seed: seed of the random initialisation.

    Returns:
        Real array (J, I, N), in arbitrary source order.
    """
    X = stft(mixture, n_fft)
    params, labels, _ = fit(X, n_sources, n_components, n_iter, seed)
    Y = source_images_stft(X, params, labels)
    return np.stack([istft(Yj, n_fft, mixture.shape[1]) for Yj in Y])
