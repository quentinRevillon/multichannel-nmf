"""Extract the two mono source excerpts used by the demo from MUSDB18.

Usage:
    python prepare_stems.py --musdb data/musdb18 --out data

Writes `vocals_mono.wav` and `guitar_mono.wav` to the output directory: 15 s excerpts,
starting at 10 s, of the `vocals` and `other` stems of the first MUSDB18 training track,
averaged over the two channels. Requires `pip install musdb` (and ffmpeg to decode the
`.stem.mp4` files).
"""
import argparse
from pathlib import Path

import musdb
import soundfile as sf

START_S = 10
DURATION_S = 15
STEMS = {"vocals_mono": "vocals", "guitar_mono": "other"}


def mono_excerpt(audio, sr):
    """Average a (N, 2) stem over channels and keep DURATION_S seconds from START_S."""
    return audio.mean(axis=1)[START_S * sr:(START_S + DURATION_S) * sr]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--musdb", type=Path, default=Path("data/musdb18"), help="MUSDB18 root (train/, test/)")
    parser.add_argument("--out", type=Path, default=Path("data"), help="output directory")
    return parser.parse_args()


def main():
    """Write the mono excerpts of the first training track."""
    args = parse_args()
    track = musdb.DB(root=args.musdb, subsets="train").tracks[0]
    args.out.mkdir(parents=True, exist_ok=True)
    for name, stem in STEMS.items():
        sf.write(args.out / f"{name}.wav", mono_excerpt(track.targets[stem].audio, track.rate), track.rate)


if __name__ == "__main__":
    main()
