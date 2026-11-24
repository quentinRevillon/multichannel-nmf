"""Separate a multichannel WAV mixture into source spatial images.

Usage:
    python separate.py data/mixture.wav --n-sources 2 --n-components 4 --n-iter 100 --out outputs

Writes one multichannel WAV per estimated source: outputs/source0.wav, outputs/source1.wav, ...
"""
import argparse
from pathlib import Path

import soundfile as sf

from mnmf.separation import separate


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mixture", type=Path, help="multichannel WAV file (at least 2 channels)")
    parser.add_argument("--n-sources", type=int, default=2, help="number of sources J")
    parser.add_argument("--n-components", type=int, default=4, help="NMF components per source")
    parser.add_argument("--n-iter", type=int, default=100, help="EM iterations")
    parser.add_argument("--n-fft", type=int, default=2048, help="STFT window length in samples")
    parser.add_argument("--seed", type=int, default=0, help="seed of the random initialisation")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="output directory")
    return parser.parse_args()


def main():
    """Run the separation and write the estimated spatial images."""
    args = parse_args()
    mixture, sr = sf.read(args.mixture)
    images = separate(mixture.T, args.n_sources, args.n_components, args.n_iter, args.n_fft, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for j, image in enumerate(images):
        sf.write(args.out / f"source{j}.wav", image.T, sr)


if __name__ == "__main__":
    main()
