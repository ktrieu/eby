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
from paths import RAW_DATA_DIR, processed_path

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

# each party's incumbency from gen_candidates, this election and the last,
# and incumbents who won for a party but now run as other
INCUMBENT_COLS = [f"incumbent_{p}" for p in MODEL_PARTIES + ["ndp"]] + [
    "switched_other"
]
PREV_INCUMBENT_COLS = [f"prev_{c}" for c in INCUMBENT_COLS]
# already a change since the last election, so it has no prev_ column
LEFT_OTHER_COL = "left_other"

# the region each 2023 riding is in, for regional swings
REGIONS_CSV = RAW_DATA_DIR / "regions.csv"


def load_dataset() -> pl.DataFrame | None:
    """Read the model input with each riding's demographics and incumbency
    joined on, or print what's missing and return None."""
    missing = [
        p
        for p in (
            MODEL_INPUT_PARQUET,
            DEMOG_PARQUET,
            RIDING_CANDIDATES_PARQUET,
            REGIONS_CSV,
        )
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
        "year", "riding", *INCUMBENT_COLS, LEFT_OTHER_COL
    )
    years = incumbency["year"].unique().sort().to_list()
    prev_incumbency = (
        as_previous(incumbency, dict(zip(years, years[1:])))
        .drop(LEFT_OTHER_COL)
        .rename(dict(zip(INCUMBENT_COLS, PREV_INCUMBENT_COLS)))
    )
    return (
        pl.read_parquet(MODEL_INPUT_PARQUET)
        .join(demog, on=["year", "riding"], how="left", validate="1:1")
        .join(incumbency, on=["year", "riding"], how="left", validate="1:1")
        .join(prev_incumbency, on=["year", "riding"], how="left", validate="1:1")
        .join(pl.read_csv(REGIONS_CSV), on="riding", how="left", validate="m:1")
        # a party with nothing to swing from (it didn't run last time, or not
        # across enough of the riding) starts from the province-wide logit, as
        # if the riding were average. predict does the same for 2026
        .with_columns(
            pl.coalesce(f"prev_logit_{p}", f"prev_topline_logit_{p}").alias(
                f"prev_logit_{p}"
            )
            for p in MODEL_PARTIES
        )
    )


def build_model(dataset: pl.DataFrame) -> pm.Model:
    """Build the logit swing model: each riding's result is its previous result
    shifted on the logit scale by the province-wide swing, its region's swing
    that election, a demographic effect and the change in incumbency, plus
    riding noise."""

    years = dataset["year"].unique().sort().to_list()
    regions = pl.read_csv(REGIONS_CSV)["region"].unique().sort().to_list()

    # right is lib + con
    coords = {
        "party": MODEL_PARTIES,
        "party_": MODEL_PARTIES,
        "year": years,
        "region": regions,
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
    incumbent, prev_incumbent = (
        dataset.select(
            pl.col(f"{prefix}incumbent_{p}") - pl.col(f"{prefix}incumbent_ndp")
            for p in MODEL_PARTIES
        ).to_numpy()
        for prefix in ("", "prev_")
    )
    incumbency_swing = incumbent - prev_incumbent

    # an MLA who left their party to run as other only moves other's logit. a
    # strong other candidate who now runs for a party takes the same kind of
    # personal vote off the other line
    switched_swing = np.zeros_like(incumbency_swing)
    switched_swing[:, MODEL_PARTIES.index("other")] = (
        dataset["switched_other"]
        - dataset["prev_switched_other"]
        - dataset[LEFT_OTHER_COL]
    ).to_numpy()

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
        # same for every party and election
        incumbency = pm.Normal("incumbency", mu=0, sigma=0.25)
        # an MLA running as other after leaving their party takes a personal
        # vote with them, from a party that usually gets a few percent, so
        # this can be several times larger on the logit scale
        switched_other = pm.Normal("switched_other", mu=0, sigma=2)

        # each region swings a little differently from the province in each
        # election (Surrey and the Island went much further right in 2024), so
        # these are fresh every election, pooled by how far regions usually
        # stray. an election the trace never saw draws new ones
        region_sigma = pm.HalfNormal("region_sigma", sigma=0.3, dims="party")
        region_z = pm.Normal("region_z", mu=0, sigma=1, dims=("year", "region", "party"))
        region_swing = region_z * region_sigma
        year_index = np.searchsorted(years, dataset["year"].to_numpy())
        region_index = np.searchsorted(regions, dataset["region"].to_numpy())

        predicted_logit = (
            prev_logit
            + topline_swing
            + region_swing[year_index, region_index]
            + demog @ beta.T
            + incumbency * incumbency_swing
            + switched_other * switched_swing
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
        # the regional swings for these elections weren't in the fit, so draw
        # them fresh from region_sigma. pymc would pass the trace's (the
        # training elections') straight through, so drop them first
        ppc = pm.sample_posterior_predictive(
            trace.posterior.to_dataset().drop_vars("region_z", errors="ignore"),
            model=model,
            var_names=["region_z", *(v.name for v in model.observed_RVs)],
            predictions=True,
            random_seed=seed,
        )

    preds = ppc["predictions"]
    # one table for each unique pattern. (chains, draw, parties_in_pattern, riding)
    # parties_in_pattern can vary (if only two parties are running its two)
    parts = [
        preds[var].rename({f"riding_{tag}": "riding", f"party_{tag}": "party"})
        for var in preds.data_vars
        if var.startswith("logit_")
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
