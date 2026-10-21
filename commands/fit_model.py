"""Fit the riding model to the dataset and save its posterior.

Run gen_dataset first. Samples the model with PyMC and writes the trace to
processed_data/trace.nc for later commands to read without refitting.
"""

import argparse
import sys

import numpy as np
import polars as pl
import pymc as pm
from scipy.special import logsumexp
import xarray as xr

from commands.gen_dataset import OUTPUT_PARQUET as MODEL_INPUT_PARQUET
from commands.gen_dataset import TOPLINES_CSV
from paths import PROCESSED_DATA_DIR

OUTPUT_NETCDF = PROCESSED_DATA_DIR / "trace.nc"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--draws",
        type=int,
        default=1000,
        help="posterior draws per chain (default: %(default)s)",
    )
    parser.add_argument(
        "--tune",
        type=int,
        default=1000,
        help="tuning steps per chain (default: %(default)s)",
    )
    parser.add_argument(
        "--chains", type=int, default=4, help="number of chains (default: %(default)s)"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="random seed, for reproducible fits"
    )
    parser.add_argument(
        "--holdout", type=int, default=None, help="holdout year to evaluate against"
    )


MODEL_PARTIES = ["green", "right", "other"]


def build_model(dataset: pl.DataFrame) -> pm.Model:
    """Build the logit swing model: each riding's result is its previous result
    shifted on the logit scale by the province-wide swing, plus riding noise."""

    # right is lib + con
    coords = {
        "party": MODEL_PARTIES,
        "party_": MODEL_PARTIES,
        "year": dataset["year"].unique().sort().to_list(),
        "riding": dataset["riding"].unique().sort().to_list(),
        # the index of each row in the dataset
        "obs": range(dataset.height),
    }

    logit = dataset.select([f"logit_{p}" for p in MODEL_PARTIES]).to_numpy()
    prev_logit = dataset.select([f"prev_logit_{p}" for p in MODEL_PARTIES]).to_numpy()
    topline_logit = dataset.select(
        [f"topline_logit_{p}" for p in MODEL_PARTIES]
    ).to_numpy()
    prev_topline_logit = dataset.select(
        [f"prev_topline_logit_{p}" for p in MODEL_PARTIES]
    ).to_numpy()
    topline_swing = topline_logit - prev_topline_logit

    # one row for each observation, each column is a boolean: "do we have NaNs for this party"?
    valid = ~np.isnan(logit) & ~np.isnan(prev_logit) & ~np.isnan(topline_swing)

    with pm.Model(coords=coords) as model:
        # covariance matrix from party -> party
        chol, corr, sigma = pm.LKJCholeskyCov(
            "chol",
            n=len(MODEL_PARTIES),
            eta=2.0,
            sd_dist=pm.HalfNormal.dist(0.5),
            compute_corr=True,
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
            var_name = f"logit_{tag}"

            model.add_coord(
                f"riding_{tag}", dataset["riding"].gather(matching_rows).to_list()
            )
            model.add_coord(
                f"party_{tag}", [MODEL_PARTIES[j] for j in valid_party_indexes]
            )

            # And then declare our multivariate distribution for this grouping. Let's use StudentT because Nate Silver uses it.
            pm.MvStudentT(
                var_name,
                nu=nu,
                # extract out only the needed fields by party index from the predicted logit, covariance matrix, and observed logit.
                mu=predicted_logit[np.ix_(matching_rows, valid_party_indexes)],
                scale=cov[valid_party_indexes][:, valid_party_indexes],
                observed=logit[np.ix_(matching_rows, valid_party_indexes)],
                dims=(f"riding_{tag}", f"party_{tag}"),
            )

    return model


def run(args: argparse.Namespace) -> int:
    missing = [p for p in (MODEL_INPUT_PARQUET,) if not p.exists()]
    if missing:
        print(
            f"missing {', '.join(map(str, missing))} (run gen_dataset first?)",
            file=sys.stderr,
        )
        return 1

    dataset = pl.read_parquet(MODEL_INPUT_PARQUET)

    train = dataset
    if args.holdout:
        train = train.filter(pl.col("year") != args.holdout)

    train_model = build_model(train)
    with train_model:
        trace = pm.sample(
            draws=args.draws, tune=args.tune, chains=args.chains, random_seed=args.seed
        )

    if args.holdout:
        test = dataset.filter(pl.col("year") == args.holdout)
        test_model = build_model(test)
        with test_model:
            ppc = pm.sample_posterior_predictive(
                trace, model=test_model, predictions=True
            )

        preds = ppc["predictions"]
        # one table for each unique pattern. (chains, draw, parties_in_pattern, riding)
        # parties_in_pattern can vary (if only two parties are running its two)
        parts = [
            preds[var].rename({f"riding_{tag}": "riding", f"party_{tag}": "party"})
            for var in preds.data_vars
            for tag in [var.removeprefix("logit_")]
        ]

        # glue together all our pieces along the riding column, essentially all our pieces into one big frame.
        # join = "outer" fills in missing parties with NaN.
        pred = xr.concat(parts, dim="riding", join="outer")
        # reindex the parties to a fixed order across the whole frame.
        pred = pred.reindex(party=MODEL_PARTIES)

        # softmax each draw into vote shares. logits are against the NDP, which adds
        # exp(0) = 1 to the denominator; a party that didn't run (NaN) gets 0 votes
        e = np.exp(pred).fillna(0)
        denom = 1 + e.sum("party")
        shares = xr.concat(
            [e / denom, (1 / denom).expand_dims(party=["ndp"])],
            dim="party",
        )
        # concat leaves party as an object-dtype coord, which arrow can't convert
        shares = shares.assign_coords(party=shares.party.astype(str))

        shares = shares.stack(sample=("chain", "draw")).reset_index("sample")
        shares.name = "share"
        shares_df = pl.DataFrame(shares)

        summary_cols = {
            "mean": pl.col("share").mean(),
            "p5": pl.col("share").quantile(0.05),
            "p25": pl.col("share").quantile(0.25),
            "p75": pl.col("share").quantile(0.75),
            "p95": pl.col("share").quantile(0.95),
        }

        results = (
            shares_df.group_by("riding", "party")
            .agg(**summary_cols)
            .pivot("party", values=summary_cols.keys())
            .join(test, on="riding")
        )

        metric_parties = MODEL_PARTIES + ["ndp"]

        metrics = results.select(
            pl.col("riding"),
            *[pl.col(f"{stat}_{p}") for stat in summary_cols for p in metric_parties],
            *[
                (pl.col(f"votes_{p}") / pl.col("total")).alias(f"actual_{p}")
                for p in metric_parties
            ],
        )

        # a party that didn't run has a null actual share, so its metrics are null
        # too and drop out of the averages below
        metrics = metrics.with_columns(
            *[
                (pl.col(f"mean_{p}") - pl.col(f"actual_{p}"))
                .abs()
                .alias(f"abs_error_{p}")
                for p in metric_parties
            ],
            *[
                pl.col(f"actual_{p}")
                .is_between(pl.col(f"p5_{p}"), pl.col(f"p95_{p}"))
                .alias(f"in_90_{p}")
                for p in metric_parties
            ],
            *[
                pl.col(f"actual_{p}")
                .is_between(pl.col(f"p25_{p}"), pl.col(f"p75_{p}"))
                .alias(f"in_50_{p}")
                for p in metric_parties
            ],
        )

        print(
            metrics.select(
                *[
                    pl.col(f"{m}_{p}").mean()
                    for m in ("abs_error", "in_90", "in_50")
                    for p in metric_parties
                ]
            )
        )

    trace.to_netcdf(OUTPUT_NETCDF)
    print(f"wrote trace to {OUTPUT_NETCDF}")

    return 0
