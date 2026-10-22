"""Fit the riding model to the dataset and save its posterior.

Run gen_dataset, gen_demog and gen_candidates first. Samples the model with PyMC and writes the trace to
processed_data/trace.nc (or the --output name in processed_data) for later
commands to read without refitting. With --holdout, that year is left out of the fit and recorded in the
trace for backtest to evaluate against.
"""

import argparse
import sys

import arviz as az
import numpy as np
import polars as pl
import pymc as pm
import xarray as xr

from commands.gen_candidates import RIDING_CANDIDATES_PARQUET
from commands.gen_dataset import OUTPUT_PARQUET as MODEL_INPUT_PARQUET
from commands.gen_dataset import as_previous
from commands.gen_demog import OUTPUT_PARQUET as DEMOG_PARQUET
from paths import processed_path

DEFAULT_TRACE = "trace.nc"


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
        "--holdout", type=int, default=None, help="year to leave out of the fit"
    )
    parser.add_argument(
        "--output",
        type=processed_path,
        default=DEFAULT_TRACE,
        help="trace file name, inside processed_data (default: %(default)s)",
    )


MODEL_PARTIES = ["green", "right", "other"]

# standardized census rates from gen_demog, each with a fixed effect per party
DEMOG_COLS = ["postsecondary_z", "visible_minority_z"]

# each party's incumbency from gen_candidates, this election and the last:
# incumbents running for the party they won for, and those who switched
INCUMBENCY_KINDS = ["incumbent", "switched"]
INCUMBENT_COLS = [
    f"{kind}_{p}" for kind in INCUMBENCY_KINDS for p in MODEL_PARTIES + ["ndp"]
]
PREV_INCUMBENT_COLS = [f"prev_{c}" for c in INCUMBENT_COLS]


def load_dataset() -> pl.DataFrame | None:
    """Read the model input with each riding's demographics and incumbency
    joined on, or print what's missing and return None."""
    missing = [
        p
        for p in (MODEL_INPUT_PARQUET, DEMOG_PARQUET, RIDING_CANDIDATES_PARQUET)
        if not p.exists()
    ]
    if missing:
        print(
            f"missing {', '.join(map(str, missing))} (run gen_dataset, gen_demog and gen_candidates first?)",
            file=sys.stderr,
        )
        return None

    demog = pl.read_parquet(DEMOG_PARQUET).select("year", "riding", *DEMOG_COLS)
    incumbency = pl.read_parquet(RIDING_CANDIDATES_PARQUET).select(
        "year", "riding", *INCUMBENT_COLS
    )
    years = incumbency["year"].unique().sort().to_list()
    prev_incumbency = as_previous(incumbency, dict(zip(years, years[1:]))).rename(
        dict(zip(INCUMBENT_COLS, PREV_INCUMBENT_COLS))
    )
    return (
        pl.read_parquet(MODEL_INPUT_PARQUET)
        .join(demog, on=["year", "riding"], how="left", validate="1:1")
        .join(incumbency, on=["year", "riding"], how="left", validate="1:1")
        .join(prev_incumbency, on=["year", "riding"], how="left", validate="1:1")
    )


def build_model(dataset: pl.DataFrame) -> pm.Model:
    """Build the logit swing model: each riding's result is its previous result
    shifted on the logit scale by the province-wide swing, a demographic
    effect and the change in incumbency, plus riding noise."""

    # right is lib + con
    coords = {
        "party": MODEL_PARTIES,
        "party_": MODEL_PARTIES,
        "year": dataset["year"].unique().sort().to_list(),
        "riding": dataset["riding"].unique().sort().to_list(),
        "demog": DEMOG_COLS,
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
    demog = dataset.select(DEMOG_COLS).to_numpy()

    # logits are against the NDP, so an NDP incumbent counts against every
    # other party. the previous result already has last election's incumbency
    # in it, so only the change since then moves the swing
    def incumbency_swing(kind: str) -> np.ndarray:
        now, prev = (
            dataset.select(
                pl.col(f"{prefix}{kind}_{p}") - pl.col(f"{prefix}{kind}_ndp")
                for p in MODEL_PARTIES
            ).to_numpy()
            for prefix in ("", "prev_")
        )
        return now - prev

    stayed_swing = incumbency_swing("incumbent")
    switched_swing = incumbency_swing("switched")

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

        # how much each party's swing moves per standard deviation of each
        # demographic, the same in every election
        beta = pm.Normal("beta", mu=0, sigma=0.25, dims=("party", "demog"))

        # how much an incumbent on the ballot moves their party's logit, the
        # same for every party and election. one who switched parties may take
        # more or less of their vote with them
        incumbency = pm.Normal("incumbency", mu=0, sigma=0.25)
        switched_incumbency = pm.Normal("switched_incumbency", mu=0, sigma=0.25)

        predicted_logit = (
            prev_logit
            + topline_swing
            + demog @ beta.T
            + incumbency * stayed_swing
            + switched_incumbency * switched_swing
        )

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


def predict_shares(
    trace: az.InferenceData, dataset: pl.DataFrame, seed: int | None = None
) -> xr.DataArray:
    """Draw each riding's vote shares in the dataset from the trace's posterior,
    as a (chain, draw, riding, party) array with the NDP included. A party that
    didn't run gets a share of 0."""
    model = build_model(dataset)
    with model:
        ppc = pm.sample_posterior_predictive(
            trace, model=model, predictions=True, random_seed=seed
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
    return shares.assign_coords(party=shares.party.astype(str))


def run(args: argparse.Namespace) -> int:
    dataset = load_dataset()
    if dataset is None:
        return 1

    train = dataset
    if args.holdout:
        train = train.filter(pl.col("year") != args.holdout)

    train_model = build_model(train)
    with train_model:
        trace = pm.sample(
            draws=args.draws, tune=args.tune, chains=args.chains, random_seed=args.seed
        )

    if args.holdout:
        # recorded so backtest knows which year this trace never saw
        trace.attrs["holdout"] = args.holdout

    args.output.parent.mkdir(exist_ok=True)
    trace.to_netcdf(args.output)
    print(f"wrote trace to {args.output}")

    return 0
