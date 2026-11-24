"""Multichannel NMF for convolutive audio source separation (Ozerov & Fevotte, 2010).

Modules:
    mnmf.stft        multichannel STFT / inverse STFT
    mnmf.mixing      simulation of reverberant mixtures with measured RIRs
    mnmf.em          EM estimation of the model parameters
    mnmf.separation  multichannel Wiener filtering and end-to-end `separate`
    mnmf.metrics     signal-to-distortion ratio of spatial images

Usage:
    from mnmf.separation import separate
    images = separate(mixture, n_sources=2, n_components=4)
"""
