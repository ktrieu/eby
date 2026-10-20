"""Build the model's input from the riding results.

Takes no arguments. Run process_edata first. Writes one row per (election,
2023 riding, party) with the party's riding result in votes, next to the
riding's total, for both that election and the one before it. The first
election has nothing before it, so it only appears as a previous election.
Provincial toplines aren't included; read them from raw_data/toplines.csv.

A party only gets a row where it ran a candidate across the riding in both
elections, so every row has a real result to swing from and to.
"""

import argparse
import sys

import polars as pl

from commands.process_edata import OUTPUT_PARQUET as RIDING_RESULTS_PARQUET
from paths import PROCESSED_DATA_DIR, RAW_DATA_DIR

TOPLINES_CSV = RAW_DATA_DIR / "toplines.csv"

OUTPUT_PARQUET = PROCESSED_DATA_DIR / "model_input.parquet"

# Affiliations in the results grouped into the toplines' party columns;
# everyone else (minor parties, independents, no affiliation) is "other".
PARTIES: dict[str, str] = {
    "BC NDP": "ndp",
    "BC Liberal Party": "liberal",
    "BC Green Party": "green",
    "Conservative": "conservative",
    "Conservative Party": "conservative",
}
PARTY_ORDER = ["ndp", "liberal", "green", "conservative", "other"]

# Parties folded into another when looking back from an election. The
# Liberals (as BC United) dropped out of 2024 and backed the Conservatives, so
# 2024's Conservatives swing from the 2020 Liberal + Conservative vote.
MERGES: dict[int, dict[str, str]] = {
    2024: {"liberal": "conservative"},
}

# Share of a 2023 riding's votes that must come from old ridings where the
# party ran a candidate. Below this the result mostly reflects where the party
# didn't run, and would read as a huge swing.
MIN_COVERAGE = 0.9


def add_arguments(parser: argparse.ArgumentParser) -> None:
    pass


def as_previous(frame: pl.DataFrame, next_years: dict[int, int]) -> pl.DataFrame:
    """Relabel each election as the one after it, merging parties per MERGES."""
    frame = frame.with_columns(pl.col("year").replace_strict(next_years, default=None)).drop_nulls("year")
    for year, merges in MERGES.items():
        frame = frame.with_columns(
            pl.when(pl.col("year") == year).then(pl.col("party").replace(merges)).otherwise("party").alias("party")
        )
    return frame


def tally_results(results: pl.DataFrame) -> pl.DataFrame:
    """Sum each party's votes per 2023 riding, with how much of it they contested."""
    # Each piece is the part of an old riding inside a 2023 riding; a party
    # contested it if it ran a candidate in that old riding.
    pieces = results.group_by("year", "riding", "riding_2024").agg(
        pl.col("votes").sum().alias("piece_votes"), pl.col("party").unique()
    )
    contested = (
        pieces.explode("party")
        .group_by("year", "riding_2024", "party")
        .agg(pl.col("piece_votes").sum().alias("contested"))
    )
    return (
        results.group_by("year", "riding_2024", "party")
        .agg(pl.col("votes").sum().alias("result"))
        .join(contested, on=["year", "riding_2024", "party"])
        .with_columns(pl.col("result").sum().over("year", "riding_2024").alias("result_total"))
        .with_columns((pl.col("contested") / pl.col("result_total")).alias("coverage"))
        .drop("contested")
        .rename({"riding_2024": "riding"})
    )


def run(args: argparse.Namespace) -> int:
    missing = [p for p in (RIDING_RESULTS_PARQUET, TOPLINES_CSV) if not p.exists()]
    if missing:
        print(f"missing {', '.join(map(str, missing))} (run process_edata first?)", file=sys.stderr)
        return 1

    results = (
        pl.read_parquet(RIDING_RESULTS_PARQUET)
        .select("year", "riding", "riding_2024", "party", "votes")
        .with_columns(pl.col("party").replace_strict(PARTIES, default="other"))
    )

    # The model reads toplines straight from the CSV, so make sure they're the
    # same votes the ridings add up to.
    totals = results.group_by("year").agg(pl.col("votes").sum()).join(
        pl.read_csv(TOPLINES_CSV).select("year", "total"), on="year", how="left"
    )
    mismatched = totals.filter((pl.col("votes") - pl.col("total")).abs().fill_null(float("inf")) > 0.5)
    if mismatched.height:
        print(f"toplines don't match the results for {', '.join(map(str, mismatched['year']))}", file=sys.stderr)
        return 1

    years = sorted(results["year"].unique())
    current = tally_results(results)
    previous = tally_results(as_previous(results, dict(zip(years, years[1:])))).rename(
        lambda c: c if c in ("year", "riding", "party") else f"prev_{c}"
    )

    dataset = (
        current.join(previous, on=["year", "riding", "party"])
        .filter(pl.col("coverage") >= MIN_COVERAGE, pl.col("prev_coverage") >= MIN_COVERAGE)
        .select(
            "year",
            "riding",
            pl.col("party").cast(pl.Enum(PARTY_ORDER)),
            "prev_result",
            "prev_result_total",
            "result",
            "result_total",
        )
        .sort("year", "riding", "party")
    )

    OUTPUT_PARQUET.parent.mkdir(exist_ok=True)
    dataset.write_parquet(OUTPUT_PARQUET)
    print(f"wrote {dataset.height} rows to {OUTPUT_PARQUET}")

    return 0
