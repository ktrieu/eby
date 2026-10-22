"""Extract the 2021 census characteristics we use from the full BC Census Profile.

The Census Profile download (98-401-X2021006, British Columbia) is a 3.6 GB
long-format CSV with ~2,600 characteristics per geography. This keeps only the
dissemination area rows for a few characteristics and pivots them to one row
per dissemination area.

Usage: uv run python scripts/filter_census_2021.py <98-401-X2021006_English_CSV_data_BritishColumbia.csv>
"""

import sys
from pathlib import Path

import polars as pl

OUTPUT_CSV = Path(__file__).resolve().parent.parent / "raw_data" / "census_2021" / "census_2021_da.csv"

# Census Profile CHARACTERISTIC_ID -> column prefix.
CHARACTERISTICS: dict[int, str] = {
    2014: "pop_25_64",  # Total - Highest certificate, diploma or degree, aged 25 to 64
    2017: "postsecondary",  # Postsecondary certificate, diploma or degree
    1683: "pop_private_hh",  # Total - Visible minority, population in private households
    1684: "visible_minority",  # Total visible minority population
}


def main(src: str) -> int:
    long = (
        pl.scan_csv(src, encoding="utf8-lossy", infer_schema=False)
        .select(
            pl.col("DGUID").alias("dguid"),
            pl.col("ALT_GEO_CODE").alias("dauid"),
            pl.col("GEO_LEVEL"),
            pl.col("CHARACTERISTIC_ID").cast(pl.Int32).alias("char_id"),
            # Suppressed values are symbols like "x" or "..", which become null.
            pl.col("C1_COUNT_TOTAL").cast(pl.Float64, strict=False).alias("count"),
            pl.col("C10_RATE_TOTAL").cast(pl.Float64, strict=False).alias("rate"),
        )
        .filter(
            (pl.col("GEO_LEVEL") == "Dissemination area")
            & pl.col("char_id").is_in(list(CHARACTERISTICS))
        )
        .collect(engine="streaming")
    )

    wide = long.with_columns(pl.col("char_id").replace_strict(CHARACTERISTICS).alias("var")).pivot(
        on="var", index=["dguid", "dauid"], values=["count", "rate"]
    )
    # pivot names columns "count_<var>"; flip to "<var>_count".
    wide = wide.rename(
        {c: f"{c.split('_', 1)[1]}_{c.split('_', 1)[0]}" for c in wide.columns if c.startswith(("count_", "rate_"))}
    ).select(
        "dguid",
        "dauid",
        "pop_25_64_count",
        "postsecondary_count",
        "postsecondary_rate",
        "pop_private_hh_count",
        "visible_minority_count",
        "visible_minority_rate",
    ).sort("dauid")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    wide.write_csv(OUTPUT_CSV)
    suppressed = wide["postsecondary_count"].null_count()
    print(f"{len(wide)} dissemination areas, {suppressed} with suppressed long-form data")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
