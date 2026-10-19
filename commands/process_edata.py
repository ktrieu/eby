"""Process the raw elections CSVs in raw_data/ onto the 2023 ridings.

Takes no arguments. Run weight_ridings first. Writes every general election
with each candidate's votes split across the 2023 ridings.
"""

import argparse
import codecs
import sys

import polars as pl

from paths import PROCESSED_DATA_DIR, RAW_DATA_DIR

VP_RESULTS_CSV = "vp_results.csv"
VA_RESULTS_CSV = "va_results.csv"

VA_WEIGHTS_PARQUET = PROCESSED_DATA_DIR / "va_weights.parquet"
RIDING_WEIGHTS_PARQUET = PROCESSED_DATA_DIR / "riding_weights.parquet"

OUTPUT_PARQUET = PROCESSED_DATA_DIR / "riding_results.parquet"

# Both files start with a UTF-8 BOM, but only vp_results is actually UTF-8;
# va_results' body is cp1252 (e.g. "Fran\xe7oise").
ENCODINGS: dict[str, str] = {
    VP_RESULTS_CSV: "utf-8",
    VA_RESULTS_CSV: "cp1252",
}

# Only general elections with voting area shapefiles are exported;
# by-elections are dropped.
GENERAL_ELECTIONS: dict[str, int] = {
    "2013 General Election": 2013,
    "2017 General Election": 2017,
    "2020 General Election": 2020,
    "2024 Provincial General Election": 2024,
}

MAP_VERSIONS: dict[int, str] = {
    2013: "2008",
    2017: "2015",
    2020: "2015",
    2024: "2023",
}

# The weights year (from weight_ridings) holding each election's voting areas.
# 2024 was held on the 2023 ridings and isn't redistributed.
WEIGHT_YEARS: dict[int, int] = {
    2013: 2013,
    2017: 2017,
    2020: 2017,
}

RESULT_COLUMNS = [
    "EVENT_NAME",
    "ED_NAME",
    "EDVA_CODE",
    "CANDIDATE",
    "AFFILIATION",
    "ELECTED",
    "VOTE_CATEGORY",
    "VOTES_CONSIDERED",
]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    pass


def read_frame(filename: str) -> pl.DataFrame:
    data = (RAW_DATA_DIR / filename).read_bytes().removeprefix(codecs.BOM_UTF8)
    # polars only reads UTF-8, so decode to text and re-encode first.
    return pl.read_csv(data.decode(ENCODINGS[filename]).encode())


def redistribute(results: pl.DataFrame) -> pl.DataFrame:
    va_weights = pl.read_parquet(VA_WEIGHTS_PARQUET).select(
        pl.col("year").cast(pl.Int64).alias("WEIGHT_YEAR"),
        pl.col("edva_code").alias("EDVA_CODE"),
        pl.col("riding").alias("RIDING_2024"),
        pl.col("weight").alias("WEIGHT"),
    )
    riding_weights = pl.read_parquet(RIDING_WEIGHTS_PARQUET).select(
        pl.col("year").cast(pl.Int64).alias("WEIGHT_YEAR"),
        pl.col("old_riding").alias("ED_NAME"),
        pl.col("riding").alias("RIDING_2024"),
        pl.col("weight").alias("WEIGHT"),
    )

    past = results.filter(pl.col("YEAR").is_in(list(WEIGHT_YEARS))).with_columns(
        pl.col("YEAR").replace_strict(WEIGHT_YEARS).alias("WEIGHT_YEAR")
    )
    by_va = past.join(va_weights, on=["WEIGHT_YEAR", "EDVA_CODE"])
    # Votes with no voting area (advance, absentee, mail) or an area missing
    # from the shapefiles follow the riding's election-day split instead.
    by_riding = past.join(va_weights, on=["WEIGHT_YEAR", "EDVA_CODE"], how="anti").join(
        riding_weights, on=["WEIGHT_YEAR", "ED_NAME"]
    )
    current = results.filter(~pl.col("YEAR").is_in(list(WEIGHT_YEARS))).with_columns(
        pl.col("ED_NAME").alias("RIDING_2024"), pl.lit(1.0).alias("WEIGHT")
    )

    columns = [*results.columns, "RIDING_2024", "WEIGHT"]
    return pl.concat(frame.select(columns) for frame in (by_va, by_riding, current)).with_columns(
        (pl.col("VOTES_CONSIDERED") * pl.col("WEIGHT")).alias("VOTES_CONSIDERED")
    )


def run(args: argparse.Namespace) -> int:
    paths = [
        RAW_DATA_DIR / VP_RESULTS_CSV,
        RAW_DATA_DIR / VA_RESULTS_CSV,
        VA_WEIGHTS_PARQUET,
        RIDING_WEIGHTS_PARQUET,
    ]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"missing {', '.join(map(str, missing))} (run weight_ridings first?)", file=sys.stderr)
        return 1

    # va_results covers 2005-2020 and vp_results 2022 onward, so rows don't overlap.
    # Rows merged into another area ("combined") report 0 votes themselves, so
    # summing every row doesn't double count.
    results = (
        pl.concat(read_frame(f).select(RESULT_COLUMNS) for f in (VA_RESULTS_CSV, VP_RESULTS_CSV))
        # Rejected ballots are reported as rows with no candidate.
        .filter(pl.col("VOTE_CATEGORY") == "Valid")
        .filter(pl.col("EVENT_NAME").is_in(list(GENERAL_ELECTIONS)))
        .with_columns(pl.col("EVENT_NAME").replace_strict(GENERAL_ELECTIONS).alias("YEAR"))
        .with_columns(
            pl.col("YEAR").replace_strict(MAP_VERSIONS).alias("MAP_VERSION"),
            # Some rows leave ELECTED blank (e.g. 2022 Surrey South), but every
            # candidate has it set on their other rows. Settle it before the
            # rows are split up.
            (pl.col("ELECTED") == "Y").any().over("YEAR", "ED_NAME", "CANDIDATE").alias("ELECTED"),
        )
    )
    total = results["VOTES_CONSIDERED"].sum()

    results = (
        redistribute(results)
        .group_by("YEAR", "MAP_VERSION", "ED_NAME", "RIDING_2024", "CANDIDATE", "AFFILIATION")
        .agg(pl.col("VOTES_CONSIDERED").sum().alias("VOTES"), pl.col("ELECTED").first())
        .sort("YEAR", "RIDING_2024", "ED_NAME", "VOTES", descending=[False, False, False, True])
        .rename(str.lower)
        .rename({"ed_name": "riding", "affiliation": "party"})
    )

    # Every set of weights sums to 1, so redistributing shouldn't gain or lose votes.
    lost = total - results["votes"].sum()
    if abs(lost) > 0.5:
        print(f"redistribution lost {lost:.1f} votes", file=sys.stderr)
        return 1

    OUTPUT_PARQUET.parent.mkdir(exist_ok=True)
    results.write_parquet(OUTPUT_PARQUET)
    print(f"wrote {results.height} rows to {OUTPUT_PARQUET}")

    return 0
