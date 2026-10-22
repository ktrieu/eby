"""Aggregate census education and visible minority rates onto the 2023 ridings.

Takes no arguments. Run weight_dissems first. Writes one row per general
election and 2023 riding, using the census nearest each election that we have
dissemination area data for. Each rate also comes standardized (<part>_z)
across that election's ridings.
"""

import argparse
import sys

import polars as pl

from commands.process_edata import GENERAL_ELECTIONS
from paths import PROCESSED_DATA_DIR, RAW_DATA_DIR

DISSEM_WEIGHTS_PARQUET = PROCESSED_DATA_DIR / "dissem_weights.parquet"

CENSUS_CSVS: dict[int, str] = {
    2016: "census_2016/census_2016_da.csv",
    2021: "census_2021/census_2021_da.csv",
}

# The census used for each election. We have no 2011 census data, so every
# election before 2024 uses 2016.
CENSUS_YEARS: dict[int, int] = {year: 2021 if year >= 2024 else 2016 for year in GENERAL_ELECTIONS.values()}

# Each characteristic and the total its rate is out of.
PARTS = {"postsecondary": "pop_25_64", "visible_minority": "pop_private_hh"}

OUTPUT_PARQUET = PROCESSED_DATA_DIR / "riding_demog.parquet"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    pass


def aggregate(year: int, weights: pl.DataFrame) -> pl.DataFrame:
    census = pl.read_csv(RAW_DATA_DIR / CENSUS_CSVS[year], schema_overrides={"dauid": pl.String})
    pieces = weights.filter(pl.col("year") == year).join(census, on="dauid")

    # Split dissemination areas count toward each riding by their share of the
    # area's land. Areas with suppressed long-form data are left out of both
    # the count and its total.
    sums = []
    for part, total in PARTS.items():
        known = pl.col(f"{part}_count").is_not_null() & pl.col(f"{total}_count").is_not_null()
        sums += [
            (pl.col(f"{c}_count") * pl.col("weight")).filter(known).sum().alias(c) for c in (total, part)
        ]

    return (
        pieces.group_by("riding")
        .agg(sums)
        .with_columns((100 * pl.col(part) / pl.col(total)).alias(f"{part}_rate") for part, total in PARTS.items())
        .with_columns(pl.lit(year).alias("census_year"))
    )


def run(args: argparse.Namespace) -> int:
    paths = [DISSEM_WEIGHTS_PARQUET, *(RAW_DATA_DIR / f for f in CENSUS_CSVS.values())]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"missing {', '.join(map(str, missing))} (run weight_dissems first?)", file=sys.stderr)
        return 1

    weights = pl.read_parquet(DISSEM_WEIGHTS_PARQUET)
    by_census = pl.concat(aggregate(year, weights) for year in CENSUS_CSVS)

    elections = pl.DataFrame(
        {"year": list(CENSUS_YEARS), "census_year": list(CENSUS_YEARS.values())},
        schema={"year": pl.Int64, "census_year": pl.Int32},
    )

    # Standardize each rate across the ridings in an election, so it says how
    # a riding differs from the rest of the province that year.
    demog = (
        elections.join(by_census, on="census_year")
        .with_columns(
            ((pl.col(f"{part}_rate") - pl.col(f"{part}_rate").mean().over("year"))
             / pl.col(f"{part}_rate").std().over("year")).alias(f"{part}_z")
            for part in PARTS
        )
        .select(
            "year",
            "riding",
            "census_year",
            *(c for part, total in PARTS.items() for c in (total, part, f"{part}_rate", f"{part}_z")),
        )
        .sort("year", "riding")
    )

    ridings = weights["riding"].n_unique()
    counts = demog.group_by("year").len().filter(pl.col("len") != ridings)
    if counts.height:
        print(f"expected {ridings} ridings per election, got {counts.rows()}", file=sys.stderr)
        return 1

    OUTPUT_PARQUET.parent.mkdir(exist_ok=True)
    demog.write_parquet(OUTPUT_PARQUET)
    print(f"wrote {demog.height} rows to {OUTPUT_PARQUET}")

    return 0
