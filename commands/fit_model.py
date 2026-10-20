"""Fit the riding model to the dataset and save its posterior.

Run gen_dataset first. Samples the model with PyMC and writes the trace to
processed_data/trace.nc for later commands to read without refitting.
"""

import argparse
import sys

import numpy as np
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

    parties = ["green", "right", "other"]

    # right is lib + con
    coords = {
        "party": parties,
        "party_": parties,
        "year": dataset["year"].unique().sort().to_list(),  
        "riding": dataset["riding"].unique().sort().to_list(),
        # the index of each row in the dataset
        "obs": range(dataset.height)
    }

    logit = dataset.select([f'logit_{p}' for p in parties]).to_numpy()
    prev_logit = dataset.select([f'prev_logit_{p}' for p in parties]).to_numpy()
    topline_logit = dataset.select([f'topline_logit_{p}' for p in parties]).to_numpy()
    prev_topline_logit = dataset.select([f'prev_topline_logit_{p}' for p in parties]).to_numpy()
    topline_swing = topline_logit - prev_topline_logit

    # one row for each observation, each column is a boolean: "do we have NaNs for this party"?
    valid = ~np.isnan(logit) & ~np.isnan(prev_logit) & ~np.isnan(topline_swing)

    with pm.Model(coords=coords) as model:
        # covariance matrix from party -> party
        chol, corr, sigma = pm.LKJCholeskyCov(
            "chol",
            n=len(parties),
            eta=2.0,
            sd_dist=pm.HalfNormal.dist(0.5),
            compute_corr=True
        )

        # Build parameters to pass to our multivariate normal distribution
        pm.Deterministic("sigma", sigma, dims="party")
        pm.Deterministic("corr", corr, dims=("party", "party_"))

        cov = chol @ chol.T

        predicted_logit = prev_logit + topline_swing

        # Some rows have nulls (Green for example). We have to split these up and handle them with separate distributions.

        # First, grab all the unique patterns of party validity.
        # pattern_index maps each row in the dataset to the index of the pattern in the unique set`party_patterns`.
        party_patterns, pattern_index = np.unique(valid, axis=0, return_inverse=True)

        # degrees of freedom for the student T - let pymc estimate it.
        nu = pm.Gamma("nu", alpha=2, beta=0.1)

        # For each pattern
        for i, pattern in enumerate(party_patterns):
            # Get only the valid parties
            valid_party_indexes = np.flatnonzero(pattern)
            if valid_party_indexes.size == 0:
                continue

            # Extract the indices of the matching rows with this pattern
            matching_rows = np.flatnonzero(pattern_index == i)

            # Generate some kind of nice user_readable tag for this pattern
            tag = "".join(pattern.astype(int).astype(str))

            # And then declare our multivariate distribution for this grouping. Let's use StudentT because Nate Silver uses it.
            pm.MvStudentT(
                f'logit_{tag}',
                nu=nu,
                # extract out only the needed fields by party index from the predicted logit, covariance matrix, and observed logit.
                mu=predicted_logit[np.ix_(matching_rows, valid_party_indexes)],
                scale=cov[valid_party_indexes][:, valid_party_indexes],
                observed=logit[np.ix_(matching_rows, valid_party_indexes)]
            )

    return model


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
