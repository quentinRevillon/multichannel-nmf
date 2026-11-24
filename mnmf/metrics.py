"""Signal-to-distortion ratio between reference and estimated spatial images.

Uses the plain energy ratio 10 log10(||y||^2 / ||y - y_hat||^2), without the
BSS Eval projections, and resolves the source permutation by exhaustive search.

Usage:
    from mnmf.metrics import best_permutation_sdr
    sdrs, order = best_permutation_sdr(images, estimates)   # both (J, I, N)
"""
from itertools import permutations

import numpy as np


def sdr(reference, estimate):
    """Return the SDR in dB of one estimated image against its reference."""
    return 10 * np.log10(np.sum(reference ** 2) / np.sum((reference - estimate) ** 2))


def best_permutation_sdr(references, estimates):
    """Match estimates to references by maximising the mean SDR.

    Args:
        references: array (J, I, N).
        estimates: array (J, I, N) in arbitrary order.

    Returns:
        (sdrs, order): per-source SDR in dB and the estimate index for each reference.
    """
    scores = {p: [sdr(references[j], estimates[p[j]]) for j in range(len(references))]
              for p in permutations(range(len(references)))}
    order = max(scores, key=lambda p: np.mean(scores[p]))
    return np.array(scores[order]), list(order)
