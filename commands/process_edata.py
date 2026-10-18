"""Process the raw elections CSVs in raw_data/.

Takes no arguments.
"""

import argparse
import codecs
import sys

import polars as pl

from paths import PROCESSED_DATA_DIR, RAW_DATA_DIR

VP_RESULTS_CSV = "vp_results.csv"
VA_RESULTS_CSV = "va_results.csv"

OUTPUT_PARQUET = PROCESSED_DATA_DIR / "riding_results.parquet"

# Both files start with a UTF-8 BOM, but only vp_results is actually UTF-8;
# va_results' body is cp1252 (e.g. "Fran\xe7oise").
ENCODINGS: dict[str, str] = {
    VP_RESULTS_CSV: "utf-8",
    VA_RESULTS_CSV: "cp1252",
}

# Only general elections are exported; by-elections are dropped.
GENERAL_ELECTIONS: dict[str, int] = {
    "2005 General Election": 2005,
    "2009 General Election": 2009,
    "2013 General Election": 2013,
    "2017 General Election": 2017,
    "2020 General Election": 2020,
    "2024 Provincial General Election": 2024,
}

MAP_VERSIONS: dict[int, str] = {
    2005: "1999",
    2009: "2008",
    2013: "2008",
    2017: "2015",
    2020: "2015",
    2024: "2023",
}

RESULT_COLUMNS = [
    "EVENT_NAME",
    "ED_NAME",
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


def run(args: argparse.Namespace) -> int:
    missing = [f for f in (VP_RESULTS_CSV, VA_RESULTS_CSV) if not (RAW_DATA_DIR / f).exists()]
    if missing:
        print(f"missing {', '.join(missing)} in {RAW_DATA_DIR}", file=sys.stderr)
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
        .with_columns(pl.col("YEAR").replace_strict(MAP_VERSIONS).alias("MAP_VERSION"))
        .group_by("YEAR", "MAP_VERSION", "ED_NAME", "CANDIDATE", "AFFILIATION")
        .agg(
            pl.col("VOTES_CONSIDERED").sum().alias("VOTES"),
            # Some rows leave ELECTED blank (e.g. 2022 Surrey South), but every
            # candidate has it set on their other rows.
            (pl.col("ELECTED") == "Y").any(),
        )
        .sort("YEAR", "ED_NAME", "VOTES", descending=[False, False, True])
        .rename(str.lower)
        .rename({"ed_name": "riding", "affiliation": "party"})
    )

    OUTPUT_PARQUET.parent.mkdir(exist_ok=True)
    results.write_parquet(OUTPUT_PARQUET)
    print(f"wrote {results.height} rows to {OUTPUT_PARQUET}")

    return 0
