"""Backtest a held-out fit against the year it never saw.

Run fit_model with --holdout first. Predicts each riding in the held-out year
from the trace's posterior and that year's polls (not its actual topline),
compares the predicted vote shares to the actual results, and writes
per-riding metrics (summary stats, actual share, signed
error, 50/90% interval coverage, predicted and actual swing from the previous
election in percentage points, predicted and actual winner) to
processed_data/backtest.parquet (or the --output name in processed_data).
"""

import argparse
import sys

import arviz as az
import polars as pl

from commands.fit_model import MODEL_PARTIES, load_dataset, load_toplines, predict_shares
from commands.fit_model import DEFAULT_TRACE
from paths import processed_path

DEFAULT_OUTPUT = "backtest.parquet"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "trace",
        type=processed_path,
        nargs="?",
        default=DEFAULT_TRACE,
        help="trace file name in processed_data, written by fit_model --holdout"
        " (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=processed_path,
        default=DEFAULT_OUTPUT,
        help="metrics file name, inside processed_data (default: %(default)s)",
    )


def run(args: argparse.Namespace) -> int:
    if not args.trace.exists():
        print(f"missing {args.trace} (run fit_model first?)", file=sys.stderr)
        return 1

    trace = az.from_netcdf(args.trace)
    holdout = trace.attrs.get("holdout")
    if holdout is None:
        print(
            f"{args.trace} has no holdout year (fit with fit_model --holdout)",
            file=sys.stderr,
        )
        return 1

    dataset = load_dataset()
    if dataset is None:
        return 1

    test = dataset.filter(pl.col("year") == holdout)
    shares = predict_shares(trace, test, load_toplines())

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
        *[
            (pl.col(f"prev_votes_{p}") / pl.col("prev_total")).alias(f"prev_{p}")
            for p in metric_parties
        ],
    )

    # the model swings on the logit scale, so the same provincial swing moves
    # each riding by a different number of percentage points depending on
    # where it started
    metrics = metrics.with_columns(
        *[
            ((pl.col(f"mean_{p}") - pl.col(f"prev_{p}")) * 100).alias(f"swing_{p}")
            for p in metric_parties
        ],
        *[
            ((pl.col(f"actual_{p}") - pl.col(f"prev_{p}")) * 100).alias(
                f"actual_swing_{p}"
            )
            for p in metric_parties
        ],
    )

    # the winner is whoever has the largest share; a party with a null actual
    # share (didn't run) can't win
    def winner(prefix: str) -> pl.Expr:
        top = pl.max_horizontal(*[pl.col(f"{prefix}_{p}") for p in metric_parties])
        return pl.coalesce(
            pl.when(pl.col(f"{prefix}_{p}") == top).then(pl.lit(p))
            for p in metric_parties
        )

    metrics = metrics.with_columns(
        winner("mean").alias("elected"), winner("actual").alias("actual_elected")
    )

    # a party that didn't run has a null actual share, so its metrics are null
    # too and drop out of the averages below
    metrics = metrics.with_columns(
        *[
            (pl.col(f"mean_{p}") - pl.col(f"actual_{p}")).alias(f"error_{p}")
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
                for m in ("error", "in_90", "in_50")
                for p in metric_parties
            ]
        )
    )
    seats = (
        metrics["elected"]
        .value_counts()
        .join(
            metrics["actual_elected"]
            .value_counts()
            .rename({"actual_elected": "elected"}),
            on="elected",
            how="full",
            coalesce=True,
            suffix="_actual",
        )
        .rename({"elected": "party"})
        .fill_null(0)
        .sort("party")
    )
    print(seats)

    args.output.parent.mkdir(exist_ok=True)
    metrics.write_parquet(args.output)
    print(f"wrote {metrics.height} rows to {args.output}")

    return 0
