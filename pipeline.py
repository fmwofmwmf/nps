"""
pipeline.py  —  data collection → training, front to back.

Usage:
    python pipeline.py                          # uses Args.py defaults
    python pipeline.py --config_file configs/bar_walker_lots.json
    python pipeline.py --skip_collect          # jump straight to training

The SAMPLES list (torques + step counts for data collection) is still edited
directly in bench_playback.py, same as before.
"""

import argparse

from get_args import get_args
from bench_playback import main as collect
from train_gradient import main as train


def pipeline_args():
    parser = argparse.ArgumentParser(parents=[], add_help=True)
    parser.add_argument("--skip_collect", action="store_true",
                        help="Skip data collection (dataset must already exist in precompute/)")
    # Forward all other flags to get_args via parse_known_args
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    pargs = pipeline_args()
    args  = get_args()          # resolves config_file, experiment_id, etc.

    if not pargs.skip_collect:
        print("=" * 60)
        print("STAGE 1 — Data collection (bench_playback)")
        print("=" * 60)
        collect(args, pipeline_mode=True)   # no Polyscope, no input() prompt

    print("=" * 60)
    print("STAGE 2 — Training (train_gradient)")
    print("=" * 60)
    train(args)
