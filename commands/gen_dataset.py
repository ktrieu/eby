"""Build the model's input from the riding results and provincial toplines.

Takes no arguments. Run process_edata first. Writes one row per (election,
2023 riding), with the pieces of old ridings inside it added together. Each row
has the riding's and the province's results for that election and the one
before it (prev_*). The first election has nothing before it, so it only
appears as a previous election.

Results are given as raw votes (total, votes_<party>) and as each party's logit
against the NDP (logit_<party> = log(party / NDP)); the NDP is the reference,
so it has no logit. A riding's logit is null where the party got no votes or
didn't run a candidate across enough of the riding to swing from.
"""

import argparse
import sys

import polars as pl

from commands.process_edata import OUTPUT_PARQUET as RIDING_RESULTS_PARQUET
from paths import PROCESSED_DATA_DIR, RAW_DATA_DIR

TOPLINES_CSV = RAW_DATA_DIR / "toplines.csv"

OUTPUT_PARQUET = PROCESSED_DATA_DIR / "model_input.parquet"

# Affiliations in the results grouped into parties; everyone else (minor
# parties, independents, no affiliation) is "other". The Liberals and
# Conservatives are one right-wing block: the Liberals (as BC United) dropped
# out of 2024 and backed the Conservatives, so treating them as one party
# gives every election a right-wing vote to swing from and to.
PARTIES: dict[str, str] = {
    "BC NDP": "ndp",
    "BC Liberal Party": "right",
    "BC Green Party": "green",
    "Conservative": "right",
    "Conservative Party": "right",
}
# The same grouping for the toplines' party columns.
TOPLINE_PARTIES: dict[str, str] = {
    "ndp": "ndp",
    "liberal": "right",
    "green": "green",
    "conservative": "right",
    "other": "other",
}
PARTY_ORDER = ["ndp", "right", "green", "other"]

# The party every logit is taken against. It ran in every riding in every
# election, so it's never missing from the denominator.
REFERENCE_PARTY = "ndp"
LOGIT_PARTIES = [p for p in PARTY_ORDER if p != REFERENCE_PARTY]

# Share of a 2023 riding's votes that must come from old ridings where the
# party ran a candidate. Below this the result mostly reflects where the party
# didn't run, and would read as a huge swing.
MIN_COVERAGE = 0.9


def add_arguments(parser: argparse.ArgumentParser) -> None:
    pass


def as_previous(frame: pl.DataFrame, next_years: dict[int, int]) -> pl.DataFrame:
    """Relabel each election as the one after it."""
    return frame.with_columns(pl.col("year").replace_strict(next_years, default=None)).drop_nulls("year")


def tally_results(results: pl.DataFrame) -> pl.DataFrame:
    """Sum each party's votes per 2023 riding, marking whether they contested enough of it."""
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
        .agg(pl.col("votes").sum())
        .join(contested, on=["year", "riding_2024", "party"])
        .with_columns(
            (pl.col("contested") / pl.col("votes").sum().over("year", "riding_2024") >= MIN_COVERAGE).alias("usable")
        )
        .select("year", pl.col("riding_2024").alias("riding"), "party", "votes", "usable")
    )


def widen(frame: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Turn (party, votes, usable) rows into a total, plus votes and a logit per party."""

    def votes(party: str) -> pl.Expr:
        return pl.col("votes").filter(pl.col("party") == party).sum()

    def logit(party: str) -> pl.Expr:
        usable = pl.col("usable").filter(pl.col("party") == party).first()
        return pl.when(usable & (votes(party) > 0)).then((votes(party) / votes(REFERENCE_PARTY)).log())

    return frame.group_by(by).agg(
        pl.col("votes").sum().alias("total"),
        *(votes(p).alias(f"votes_{p}") for p in PARTY_ORDER),
        *(logit(p).alias(f"logit_{p}") for p in LOGIT_PARTIES),
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
    toplines = (
        pl.read_csv(TOPLINES_CSV)
        .unpivot(list(TOPLINE_PARTIES), index="year", variable_name="party", value_name="votes")
        .with_columns(pl.col("party").replace_strict(TOPLINE_PARTIES))
        .group_by("year", "party")
        .agg(pl.col("votes").sum().cast(pl.Float64), usable=pl.lit(True))
    )

    # Make sure the toplines are the same votes the ridings add up to.
    totals = (
        results.group_by("year", "party")
        .agg(pl.col("votes").sum())
        .join(toplines, on=["year", "party"], how="full", coalesce=True, suffix="_topline")
        .fill_null(0)
    )
    mismatched = totals.filter((pl.col("votes") - pl.col("votes_topline")).abs() > 0.5)
    if mismatched.height:
        print(f"toplines don't match the results for {', '.join(map(str, mismatched['year'].unique()))}", file=sys.stderr)
        return 1

    years = sorted(results["year"].unique())
    next_years = dict(zip(years, years[1:]))

    def prefixed(frame: pl.DataFrame, prefix: str) -> pl.DataFrame:
        return frame.rename(lambda c: c if c in ("year", "riding") else f"{prefix}{c}")

    dataset = (
        prefixed(widen(tally_results(as_previous(results, next_years)), ["year", "riding"]), "prev_")
        .join(widen(tally_results(results), ["year", "riding"]), on=["year", "riding"])
        .join(prefixed(widen(as_previous(toplines, next_years), ["year"]), "prev_topline_"), on="year")
        .join(prefixed(widen(toplines, ["year"]), "topline_"), on="year")
        .sort("year", "riding")
    )

    OUTPUT_PARQUET.parent.mkdir(exist_ok=True)
    dataset.write_parquet(OUTPUT_PARQUET)
    print(f"wrote {dataset.height} rows to {OUTPUT_PARQUET}")

    return 0
