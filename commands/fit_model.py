"""Fit the riding model to the dataset and save its posterior.

Run gen_dataset first. Samples the model with PyMC and writes the trace to
processed_data/trace.nc for later commands to read without refitting.
"""

import argparse
import sys

import polars as pl
import pymc as pm

from commands.gen_dataset import OUTPUT_PARQUET as MODEL_INPUT_PARQUET
from commands.gen_dataset import TOPLINES_CSV
from paths import PROCESSED_DATA_DIR

OUTPUT_NETCDF = PROCESSED_DATA_DIR / "trace.nc"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--draws", type=int, default=1000, help="posterior draws per chain (default: %(default)s)")
    parser.add_argument("--tune", type=int, default=1000, help="tuning steps per chain (default: %(default)s)")
    parser.add_argument("--chains", type=int, default=4, help="number of chains (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=None, help="random seed, for reproducible fits")


def build_model(dataset: pl.DataFrame, toplines: pl.DataFrame) -> pm.Model:
    """Build the logit swing model: each riding's result is its previous result
    shifted on the logit scale by the province-wide swing, plus riding noise."""
    raise NotImplementedError("model not written yet")


def run(args: argparse.Namespace) -> int:
    missing = [p for p in (MODEL_INPUT_PARQUET, TOPLINES_CSV) if not p.exists()]
    if missing:
        print(f"missing {', '.join(map(str, missing))} (run gen_dataset first?)", file=sys.stderr)
        return 1

    dataset = pl.read_parquet(MODEL_INPUT_PARQUET)
    toplines = pl.read_csv(TOPLINES_CSV)

    with build_model(dataset, toplines):
        trace = pm.sample(draws=args.draws, tune=args.tune, chains=args.chains, random_seed=args.seed)

    trace.to_netcdf(OUTPUT_NETCDF)
    print(f"wrote trace to {OUTPUT_NETCDF}")

    return 0
