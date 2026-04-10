"""
Shared argument resolution.

Priority (highest first):
  1. CLI flags   (--config_file ...)
  2. Args.py     (edited by hand, serves as defaults)

Usage in any script:
    from get_args import get_args
    args = get_args()
"""

import argparse
from Args import Args


def get_args():
    defaults = Args()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config_file",     default=defaults.config_file)
    parser.add_argument("--experiment_id",   default=defaults.experiment_id)
    parser.add_argument("--integrator_name", default=defaults.integrator_name)
    parser.add_argument("--subspace_model",  default=defaults.subspace_model)
    parser.add_argument(
        "--use_subspace",
        default=defaults.use_subspace,
        type=lambda x: x.lower() not in ("false", "0", "no"),
    )
    # parse_known_args so extra flags (e.g. from Polyscope) don't crash things
    args, _ = parser.parse_known_args()
    return args