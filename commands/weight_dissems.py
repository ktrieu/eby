"""Compute the share of each census dissemination area's land that falls in each 2023 riding.

Takes no arguments. Writes one row per (census year, dissemination area,
2023 riding) piece. Dissemination area IDs are reused across censuses with
different boundaries, so they're only unique within a year.
"""

import argparse
import sys

import geopandas as gpd
import pandas as pd
import polars as pl

from commands.weight_ridings import MIN_WEIGHT, RIDINGS_SHAPEFILE, read_shapefile
from paths import PROCESSED_DATA_DIR, RAW_DATA_DIR

# Dissemination area boundaries from each census.
DISSEM_SHAPEFILES: dict[int, str] = {
    2011: "dissem_2011/dissem_2011.shp",
    2016: "dissem_2016/dissem_2016.shp",
    2021: "dissem_2021/dissem_2021.shp",
}

OUTPUT_PARQUET = PROCESSED_DATA_DIR / "dissem_weights.parquet"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    pass


def weigh(year: int, dissems: gpd.GeoDataFrame, ridings: gpd.GeoDataFrame) -> pl.DataFrame:
    dissems = dissems.to_crs(ridings.crs)
    dissems["DA_AREA"] = dissems.area
    pieces = gpd.overlay(dissems, ridings, how="intersection", keep_geom_type=True)

    return (
        pl.from_pandas(
            pd.DataFrame(
                {
                    "dauid": pieces["DAUID"],
                    "riding": pieces["ED_NAME"],
                    "weight": pieces.area / pieces["DA_AREA"],
                }
            )
        )
        .group_by("dauid", "riding")
        .agg(pl.col("weight").sum())
        .filter(pl.col("weight") >= MIN_WEIGHT)
        # Dropped slivers and coastline mismatches leave shares a little short of 1.
        .with_columns(pl.col("weight") / pl.col("weight").sum().over("dauid"))
        .with_columns(pl.lit(year).alias("year"))
        .select("year", "dauid", "riding", "weight")
    )


def run(args: argparse.Namespace) -> int:
    paths = [RIDINGS_SHAPEFILE, *(RAW_DATA_DIR / f for f in DISSEM_SHAPEFILES.values())]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"missing {', '.join(map(str, missing))}", file=sys.stderr)
        return 1

    ridings = read_shapefile(RIDINGS_SHAPEFILE, "ED_NAME")

    frames = []
    for year, filename in DISSEM_SHAPEFILES.items():
        dissems = read_shapefile(RAW_DATA_DIR / filename, "DAUID")
        weights = weigh(year, dissems, ridings)

        unmatched = set(dissems["DAUID"]) - set(weights["dauid"])
        if unmatched:
            print(f"{year}: no riding overlaps {', '.join(sorted(unmatched))}", file=sys.stderr)
            return 1

        split = weights.filter(pl.col("dauid").is_duplicated())["dauid"].n_unique()
        print(f"{year}: {len(dissems)} dissemination areas, {split} split between ridings")
        frames.append(weights)

    weights = pl.concat(frames).sort("year", "dauid", "weight", descending=[False, False, True])

    OUTPUT_PARQUET.parent.mkdir(exist_ok=True)
    weights.write_parquet(OUTPUT_PARQUET)
    print(f"wrote {weights.height} rows to {OUTPUT_PARQUET}")

    return 0
