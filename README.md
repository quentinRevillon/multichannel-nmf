# multichannel-nmf

Implementation of the EM algorithm of Ozerov & Févotte (2010) for multichannel NMF in convolutive mixtures, applied to audio source separation. Course project, Télécom Paris (2nd year, TSIA track), course of Roland Badeau, June 2025.

<img width="100%" src="figures/spectrograms.png" />

## Results

Single run of [`notebooks/02_demo.ipynb`](notebooks/02_demo.ipynb): 15 s stereo mixture at 44.1 kHz, 2 sources, 4 components per source, 100 EM iterations. The baseline uses the mixture as the estimate of each source.

| Source | SDR, mixture (dB) | SDR, estimate (dB) |
|---|---|---|
| Vocals | 2.37 | 3.34 |
| Accompaniment | -2.37 | -0.52 |

The gain is small and the estimated spectrograms stay close to the mixture.

## Quick start

```bash
pip install -r requirements.txt
python prepare_stems.py --musdb data/musdb18 --out data
OPENBLAS_NUM_THREADS=1 python separate.py data/mixture.wav --n-sources 2 --n-components 4 --n-iter 100 --out outputs
```

The demo needs MUSDB18 in `data/musdb18/` and the binaural Aula Carolina impulse responses of the AIR database v1.4 in `data/AIR_1_4/`. The demo notebook writes `data/mixture.wav`.

## Method

- Mixture model per frequency bin: `x(f, t) = A(f) s(f, t) + b(f, t)`. Each source is a sum of Gaussian components whose variances follow an NMF `W[f, k] H[k, t]`.
- EM (Section IV of the paper) estimates `A`, `W`, `H` and the noise variances in closed form.
- Sources are recovered by multichannel Wiener filtering.
- Data: voice and accompaniment stems from MUSDB18, convolved with binaural room impulse responses to simulate a two-microphone reverberant recording.
- The submitted [course notebook](notebooks/01_course_implementation.ipynb) did not separate the sources. The `mnmf` package is a later vectorised rewrite that fixes the second-order statistics of the E-step.

A reading note on the paper (in French) is in [`docs/reading_note_fr.pdf`](docs/reading_note_fr.pdf).

## Structure

```
mnmf/        STFT, mixing simulation, EM, Wiener filtering, SDR
notebooks/   course notebook (French) and demo
docs/        reading note on the paper (French)
separate.py  command-line entry point
```

## References

1. A. Ozerov and C. Févotte, "Multichannel nonnegative matrix factorization in convolutive mixtures for audio source separation," *IEEE Trans. Audio, Speech, Lang. Process.*, vol. 18, no. 3, pp. 550-563, 2010.
2. M. Jeub, M. Schäfer and P. Vary, "A binaural room impulse response database for the evaluation of dereverberation algorithms," *Proc. DSP*, 2009.
3. Z. Rafii, A. Liutkus, F.-R. Stöter, S. I. Mimilakis and R. Bittner, "The MUSDB18 corpus for music separation," 2017.

## Author

Quentin Revillon, Télécom Paris. Course supervised by Roland Badeau.
