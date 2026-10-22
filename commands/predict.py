"""Predict the 2026 election from a fitted trace and a provincial topline.

Run fit_model first, without --holdout so the fit has seen 2024. Swings each
riding from its 2024 result by the change from the 2024 topline to the one
given (--ndp, --right, --green, --other, in percent; only their ratios matter,
so they needn't add to 100), using the 2024 demographics.

Who runs where comes from raw_data/candidates_2026.csv, a yes/no per riding
for each party on the ballot. The Conservatives are "right", and CentreBC,
OneBC and everyone else are "other". A party running where it had no 2024
logit starts from the 2024 provincial logit instead, as if the riding were
average. Ridings without an NDP candidate can't be modeled, since every logit
is against the NDP; they go to the winner set in FIXED_WINNERS.

For each posterior draw, takes every riding's vote shares and its plurality
winner, and counts each party's seats. Writes the p5/p25/p50/p75/p95 of each
riding's shares, plus each party's chance of winning it (win_<party>, the
share of draws it won), to processed_data/prediction.parquet (or the --output
name), and the same quantiles of each party's seat count to
processed_data/prediction_seats.parquet (or the --seats-output name).
"""

import argparse
import math
import sys

import arviz as az
import polars as pl
import xarray as xr

from commands.fit_model import (
    DEFAULT_TRACE,
    DEMOG_COLS,
    MODEL_PARTIES,
    load_dataset,
    predict_shares,
)
from paths import RAW_DATA_DIR, processed_path

PREV_YEAR = 2024
YEAR = 2026

CANDIDATES_CSV = RAW_DATA_DIR / "candidates_2026.csv"

# the model party each column of the candidates file runs under
CANDIDATE_PARTIES = {
    "ndp": "ndp",
    "conservative": "right",
    "centrebc": "other",
    "green": "green",
    "onebc": "other",
    "other": "other",
}

# ridings with no NDP candidate, and who takes them. Peace River North is the
# incumbent Jordan Kealy (independent, elected as a Conservative) against the
# Conservatives; we call it for Kealy
FIXED_WINNERS = {"Peace River North": "other"}

DEFAULT_OUTPUT = "prediction.parquet"
DEFAULT_SEATS_OUTPUT = "prediction_seats.parquet"

TOPLINE_PARTIES = MODEL_PARTIES + ["ndp"]

QUANTILES = {"p5": 0.05, "p25": 0.25, "p50": 0.5, "p75": 0.75, "p95": 0.95}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "trace",
        type=processed_path,
        nargs="?",
        default=DEFAULT_TRACE,
        help="trace file name in processed_data, written by fit_model"
        " (default: %(default)s)",
    )
    for party in TOPLINE_PARTIES:
        parser.add_argument(
            f"--{party}",
            type=float,
            required=True,
            help=f"{YEAR} provincial vote share for {party}, in percent",
        )
    parser.add_argument(
        "--seed", type=int, default=None, help="random seed, for reproducible draws"
    )
    parser.add_argument(
        "--output",
        type=processed_path,
        default=DEFAULT_OUTPUT,
        help="riding shares file name, inside processed_data (default: %(default)s)",
    )
    parser.add_argument(
        "--seats-output",
        type=processed_path,
        default=DEFAULT_SEATS_OUTPUT,
        help="seat counts file name, inside processed_data (default: %(default)s)",
    )


def load_candidates() -> pl.DataFrame:
    """Read which model parties run in each riding, as running_<party> booleans."""
    candidates = pl.read_csv(CANDIDATES_CSV).with_columns(
        pl.col(c).replace_strict({"yes": True, "no": False})
        for c in CANDIDATE_PARTIES
    )
    return candidates.select(
        "riding",
        *[
            pl.any_horizontal(
                c for c, party in CANDIDATE_PARTIES.items() if party == p
            ).alias(f"running_{p}")
            for p in TOPLINE_PARTIES
        ],
    )


def build_frame(
    dataset: pl.DataFrame, candidates: pl.DataFrame, topline: dict[str, float]
) -> pl.DataFrame:
    """Turn the 2024 rows into 2026 rows: 2024's results become the previous
    ones, and the topline is the one given."""
    return (
        dataset.filter(pl.col("year") == PREV_YEAR)
        .join(candidates, on="riding", validate="1:1")
        .filter(pl.col("running_ndp"))
        .select(
            pl.lit(YEAR).alias("year"),
            "riding",
            # a party not running has no logit. one running without a 2024
            # logit starts from the province-wide one
            *[
                pl.when(pl.col(f"running_{p}"))
                .then(pl.coalesce(f"logit_{p}", f"topline_logit_{p}"))
                .alias(f"prev_logit_{p}")
                for p in MODEL_PARTIES
            ],
            *[
                pl.col(f"topline_logit_{p}").alias(f"prev_topline_logit_{p}")
                for p in MODEL_PARTIES
            ],
            *[
                pl.lit(math.log(topline[p] / topline["ndp"])).alias(
                    f"topline_logit_{p}"
                )
                for p in MODEL_PARTIES
            ],
            # there's no 2026 result, but build_model reads which parties ran from
            # the observed logits. the values are placeholders that posterior
            # predictive sampling ignores
            *[
                pl.when(pl.col(f"running_{p}")).then(0.0).alias(f"logit_{p}")
                for p in MODEL_PARTIES
            ],
            *DEMOG_COLS,
        )
    )


def summarize(frame: pl.DataFrame, value: str, by: list[str]) -> pl.DataFrame:
    """Take the quantiles of value across draws, one column per stat."""
    return frame.group_by(*by, "party").agg(
        **{stat: pl.col(value).quantile(q) for stat, q in QUANTILES.items()}
    )


def run(args: argparse.Namespace) -> int:
    if not args.trace.exists():
        print(f"missing {args.trace} (run fit_model first?)", file=sys.stderr)
        return 1

    topline = {p: getattr(args, p) for p in TOPLINE_PARTIES}
    nonpositive = [p for p, share in topline.items() if share <= 0]
    if nonpositive:
        print(
            f"topline shares must be positive: {', '.join(nonpositive)}",
            file=sys.stderr,
        )
        return 1

    trace = az.from_netcdf(args.trace)
    holdout = trace.attrs.get("holdout")
    if holdout is not None:
        print(
            f"warning: {args.trace} was fit without {holdout}", file=sys.stderr
        )

    dataset = load_dataset()
    if dataset is None:
        return 1

    if not CANDIDATES_CSV.exists():
        print(f"missing {CANDIDATES_CSV}", file=sys.stderr)
        return 1
    candidates = load_candidates()

    ridings_2024 = set(dataset.filter(pl.col("year") == PREV_YEAR)["riding"])
    mismatched = ridings_2024 ^ set(candidates["riding"])
    if mismatched:
        print(
            f"{CANDIDATES_CSV} and the {PREV_YEAR} ridings differ:"
            f" {', '.join(sorted(mismatched))}",
            file=sys.stderr,
        )
        return 1
    unset = set(candidates.filter(~pl.col("running_ndp"))["riding"]) ^ set(
        FIXED_WINNERS
    )
    if unset:
        print(
            "FIXED_WINNERS should be exactly the ridings without an NDP"
            f" candidate, but differs on: {', '.join(sorted(unset))}",
            file=sys.stderr,
        )
        return 1

    shares = predict_shares(
        trace, build_frame(dataset, candidates, topline), seed=args.seed
    )

    # each riding goes to the party with the largest share in that draw. every
    # party gets a count in every draw, even when it's 0, so the quantiles
    # aren't taken over only the draws it won something
    won = shares == shares.max("party")
    fixed = xr.DataArray(
        [list(FIXED_WINNERS.values()).count(p) for p in shares.party.values],
        coords={"party": shares.party},
    )
    seats = won.sum("riding") + fixed
    win = won.mean(("chain", "draw"))
    win.name = "win"

    shares = shares.stack(sample=("chain", "draw")).reset_index("sample")
    shares.name = "share"
    seats = seats.stack(sample=("chain", "draw")).reset_index("sample")
    seats.name = "seats"

    # a single value column pivots to bare party names, so prefix them
    win_chances = (
        pl.DataFrame(win)
        .pivot("party", index="riding", values="win")
        .rename({p: f"win_{p}" for p in TOPLINE_PARTIES})
    )

    # fixed ridings have no shares, and a certain winner
    fixed_ridings = pl.DataFrame(
        [
            {"riding": riding, **{f"win_{p}": float(p == winner) for p in TOPLINE_PARTIES}}
            for riding, winner in FIXED_WINNERS.items()
        ]
    )

    ridings = (
        summarize(pl.DataFrame(shares), "share", ["riding"])
        .pivot("party", index="riding", values=list(QUANTILES))
        .join(win_chances, on="riding", validate="1:1")
    )
    ridings = (
        pl.concat([ridings, fixed_ridings], how="diagonal_relaxed")
        .select(
            "riding",
            *[pl.col(f"{stat}_{p}") for stat in QUANTILES for p in TOPLINE_PARTIES],
            *[pl.col(f"win_{p}") for p in TOPLINE_PARTIES],
        )
        .sort("riding")
    )
    seat_counts = summarize(pl.DataFrame(seats), "seats", []).sort("party")

    print(seat_counts)

    args.output.parent.mkdir(exist_ok=True)
    ridings.write_parquet(args.output)
    print(f"wrote {ridings.height} rows to {args.output}")
    seat_counts.write_parquet(args.seats_output)
    print(f"wrote seat counts to {args.seats_output}")

    return 0
