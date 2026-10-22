"""Extract the census characteristics we use from the full BC Census Profile.

The Census Profile downloads (98-401-X2016044 and 98-401-X2021006, British
Columbia) are multi-GB long-format CSVs with thousands of characteristics per
geography. This keeps only the dissemination area rows for a few
characteristics and pivots them to one row per dissemination area.

Usage: uv run python scripts/filter_census.py <year> <Census Profile CSV>
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl

RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "raw_data"


@dataclass(frozen=True)
class Profile:
    geo_level_column: str
    da_level: str
    dauid_column: str
    id_column: str
    count_column: str
    # Published percentages; None means compute them from the counts.
    rate_column: str | None
    # CHARACTERISTIC_ID of each total and the part of it we keep.
    pop_25_64: int  # Total - Highest certificate, diploma or degree, aged 25 to 64
    postsecondary: int  # Postsecondary certificate, diploma or degree
    pop_private_hh: int  # Total - Visible minority, population in private households
    visible_minority: int  # Total visible minority population


PROFILES: dict[int, Profile] = {
    2016: Profile(
        geo_level_column="GEO_LEVEL",
        da_level="4",
        dauid_column="GEO_CODE (POR)",
        id_column="Member ID: Profile of Dissemination Areas (2247)",
        count_column="Dim: Sex (3): Member ID: [1]: Total - Sex",
        rate_column=None,
        pop_25_64=1698,
        postsecondary=1701,
        pop_private_hh=1323,
        visible_minority=1324,
    ),
    2021: Profile(
        geo_level_column="GEO_LEVEL",
        da_level="Dissemination area",
        dauid_column="ALT_GEO_CODE",
        id_column="CHARACTERISTIC_ID",
        count_column="C1_COUNT_TOTAL",
        rate_column="C10_RATE_TOTAL",
        pop_25_64=2014,
        postsecondary=2017,
        pop_private_hh=1683,
        visible_minority=1684,
    ),
}

# Each kept characteristic and the total its rate is out of.
PARTS = {"postsecondary": "pop_25_64", "visible_minority": "pop_private_hh"}


def main(year: int, src: str) -> int:
    profile = PROFILES[year]
    ids = {getattr(profile, name): name for name in [*PARTS, *PARTS.values()]}

    long = (
        pl.scan_csv(src, encoding="utf8-lossy", infer_schema=False)
        .filter(pl.col(profile.geo_level_column) == profile.da_level)
        .select(
            pl.col(profile.dauid_column).alias("dauid"),
            pl.col(profile.id_column).cast(pl.Int32).alias("char_id"),
            # Suppressed values are symbols like "x" or "..", which become null.
            pl.col(profile.count_column).cast(pl.Float64, strict=False).alias("count"),
            (
                pl.col(profile.rate_column).cast(pl.Float64, strict=False)
                if profile.rate_column
                else pl.lit(None, pl.Float64)
            ).alias("rate"),
        )
        .filter(pl.col("char_id").is_in(list(ids)))
        .collect(engine="streaming")
    )

    wide = long.with_columns(pl.col("char_id").replace_strict(ids).alias("var")).pivot(
        on="var", index="dauid", values=["count", "rate"]
    )
    # pivot names columns "count_<var>"; flip to "<var>_count".
    wide = wide.rename(
        {c: f"{c.split('_', 1)[1]}_{c.split('_', 1)[0]}" for c in wide.columns if c.startswith(("count_", "rate_"))}
    )
    if profile.rate_column is None:
        wide = wide.with_columns(
            (100 * pl.col(f"{part}_count") / pl.col(f"{total}_count"))
            .round(1)
            .fill_nan(None)
            .alias(f"{part}_rate")
            for part, total in PARTS.items()
        )

    wide = wide.select(
        "dauid",
        *(c for part, total in PARTS.items() for c in (f"{total}_count", f"{part}_count", f"{part}_rate")),
    ).sort("dauid")

    out = RAW_DATA_DIR / f"census_{year}" / f"census_{year}_da.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    wide.write_csv(out)
    suppressed = wide["postsecondary_count"].null_count()
    print(f"{year}: {len(wide)} dissemination areas, {suppressed} with suppressed long-form data")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3 or int(sys.argv[1]) not in PROFILES:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    sys.exit(main(int(sys.argv[1]), sys.argv[2]))
