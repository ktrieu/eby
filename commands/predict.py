"""Predict the 2026 election from a fitted trace and the 2026 polls.

Run fit_model first, without --holdout so the fit has seen 2024. Swings each
riding from its 2024 result by the change from the 2024 topline to the 2026
poll average plus a drawn polling error, and by its region's swing in the
regional polls plus a drawn regional error (gen_polls makes both averages),
using the 2024 demographics, and moves it by the change from the 2024
incumbents to the sitting MLAs running in 2026 (and whether each runs for the
party they were elected for).

Run gen_candidates first too: who runs where comes from its 2026 rows. The
Conservatives are "right", and CentreBC, OneBC and everyone else are
"other". A party running where it had no 2024
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
import sys

import arviz as az
import polars as pl
import xarray as xr

from commands.fit_model import (
    DEFAULT_TRACE,
    DEMOG_COLS,
    INCUMBENT_COLS,
    LEFT_OTHER_COL,
    MODEL_PARTIES,
    PREV_INCUMBENT_COLS,
    load_dataset,
    load_toplines,
    predict_shares,
    with_region_poll_swings,
)
from commands.gen_candidates import RIDING_CANDIDATES_PARQUET
from paths import processed_path

PREV_YEAR = 2024
YEAR = 2026

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


def build_frame(dataset: pl.DataFrame, candidates: pl.DataFrame) -> pl.DataFrame:
    """Turn the 2024 rows into 2026 rows: 2024's results and incumbency become
    the previous ones."""
    frame = (
        dataset.filter(pl.col("year") == PREV_YEAR)
        .drop(*PREV_INCUMBENT_COLS, LEFT_OTHER_COL)
        .rename(dict(zip(INCUMBENT_COLS, PREV_INCUMBENT_COLS)))
        .join(candidates, on="riding", validate="1:1")
        .filter(pl.col("running_ndp"))
        .select(
            pl.lit(YEAR).alias("year"),
            "riding",
            "region",
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
            # there's no 2026 result, but build_model reads which parties ran from
            # the observed logits. the values are placeholders that posterior
            # predictive sampling ignores
            *[
                pl.when(pl.col(f"running_{p}")).then(0.0).alias(f"logit_{p}")
                for p in MODEL_PARTIES
            ],
            *DEMOG_COLS,
            *INCUMBENT_COLS,
            *PREV_INCUMBENT_COLS,
            LEFT_OTHER_COL,
        )
    )
    return with_region_poll_swings(frame)


def summarize(frame: pl.DataFrame, value: str, by: list[str]) -> pl.DataFrame:
    """Take the quantiles of value across draws, one column per stat."""
    return frame.group_by(*by, "party").agg(
        **{stat: pl.col(value).quantile(q) for stat, q in QUANTILES.items()}
    )


def run(args: argparse.Namespace) -> int:
    if not args.trace.exists():
        print(f"missing {args.trace} (run fit_model first?)", file=sys.stderr)
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

    if not RIDING_CANDIDATES_PARQUET.exists():
        print(
            f"missing {RIDING_CANDIDATES_PARQUET} (run gen_candidates first?)",
            file=sys.stderr,
        )
        return 1
    candidates = pl.read_parquet(RIDING_CANDIDATES_PARQUET).filter(
        pl.col("year") == YEAR
    ).drop("year")
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

    toplines = load_toplines()
    if YEAR not in toplines["year"]:
        print(f"no {YEAR} polls (run gen_polls first?)", file=sys.stderr)
        return 1

    shares = predict_shares(
        trace, build_frame(dataset, candidates), toplines, seed=args.seed
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
