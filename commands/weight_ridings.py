"""Compute the share of each voting area's land that falls in each 2023 riding.

Areas drawn as points (apartment buildings, care homes) are split by how many
of their points fall in each riding instead.

Takes no arguments. Writes one row per (year, voting area, 2023 riding) piece,
and one row per (year, riding, 2023 riding) pair weighting each riding's
voting areas by their election-day votes, for votes cast outside a voting area.
"""

import argparse
import sys

import geopandas as gpd
import pandas as pd
import polars as pl

from commands.process_edata import GENERAL_ELECTIONS, VA_RESULTS_CSV, read_frame
from paths import PROCESSED_DATA_DIR, RAW_DATA_DIR

RIDINGS_SHAPEFILE = RAW_DATA_DIR / "ridings_2023" / "EBC_ED_23_polygon.shp"

# Voting area boundaries used in each general election.
VA_SHAPEFILES: dict[int, str] = {
    2013: "va_2012/VA_BS08_polygon.shp",
    2017: "va_2017/VOT_AREA_polygon.shp",
}

# Voting areas that are a set of buildings rather than a polygon: point
# voting areas (apartment buildings) and site-based "S" areas (care homes,
# hospitals).
VA_SITE_SHAPEFILES: dict[int, str] = {
    2013: "va_2012_site/SBPVA_BS08_point.shp",
    2017: "va_2017_site/SVAS_2017_point.shp",
}

# Pieces smaller than this share of their voting area are boundary slivers
# (the two layers trace coastlines and roads slightly differently), not real
# overlaps.
MIN_WEIGHT = 0.01

OUTPUT_PARQUET = PROCESSED_DATA_DIR / "va_weights.parquet"
RIDING_OUTPUT_PARQUET = PROCESSED_DATA_DIR / "riding_weights.parquet"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    pass


def read_shapefile(path, column: str) -> gpd.GeoDataFrame:
    frame = gpd.read_file(path, columns=[column], use_arrow=True)
    frame.geometry = frame.make_valid()
    # Some areas are stored as several polygons (e.g. WSS004 in 2013).
    return frame.dissolve(by=column, as_index=False)


def weigh(year: int, vas: gpd.GeoDataFrame, ridings: gpd.GeoDataFrame) -> pl.DataFrame:
    vas = vas.to_crs(ridings.crs)
    vas["VA_AREA"] = vas.area
    pieces = gpd.overlay(vas, ridings, how="intersection", keep_geom_type=True)

    return (
        pl.from_pandas(
            pd.DataFrame(
                {
                    "edva_code": pieces["EDVA_CODE"],
                    "riding": pieces["ED_NAME"],
                    "weight": pieces.area / pieces["VA_AREA"],
                }
            )
        )
        .group_by("edva_code", "riding")
        .agg(pl.col("weight").sum())
        .filter(pl.col("weight") >= MIN_WEIGHT)
        # Dropped slivers and coastline mismatches leave shares a little short of 1.
        .with_columns(pl.col("weight") / pl.col("weight").sum().over("edva_code"))
        .with_columns(pl.lit(year).alias("year"))
        .select("year", "edva_code", "riding", "weight")
    )


def weigh_sites(year: int, sites: gpd.GeoDataFrame, ridings: gpd.GeoDataFrame) -> pl.DataFrame:
    # Nearest rather than within, so points just off the riding file's
    # coastline still land in a riding.
    joined = gpd.sjoin_nearest(sites.to_crs(ridings.crs), ridings, distance_col="DISTANCE")
    offshore = (joined["DISTANCE"] > 0).sum()
    if offshore:
        print(f"{year}: {offshore} site points outside every riding, using the nearest")

    return (
        pl.from_pandas(pd.DataFrame({"edva_code": joined["EDVA_CODE"], "riding": joined["ED_NAME"]}))
        .group_by("edva_code", "riding")
        .agg(pl.len().alias("weight"))
        .with_columns(pl.col("weight") / pl.col("weight").sum().over("edva_code"))
        .with_columns(pl.lit(year).alias("year"))
        .select("year", "edva_code", "riding", "weight")
    )


def weigh_ridings(va_weights: pl.DataFrame, results: pl.DataFrame) -> pl.DataFrame:
    # Votes with no voting area (advance, absentee) or with an area missing
    # from the shapefiles are excluded by the join.
    va_votes = (
        results.filter(pl.col("VOTING_OPPORTUNITY") == "General voting")
        .filter(pl.col("VOTE_CATEGORY") == "Valid")
        .group_by("YEAR", "ED_NAME", "EDVA_CODE")
        .agg(pl.col("VOTES_CONSIDERED").sum())
        .rename({"YEAR": "year", "ED_NAME": "old_riding", "EDVA_CODE": "edva_code"})
    )

    return (
        va_weights.join(va_votes, on=["year", "edva_code"])
        .group_by("year", "old_riding", "riding")
        .agg((pl.col("weight") * pl.col("VOTES_CONSIDERED")).sum())
        .with_columns(pl.col("weight") / pl.col("weight").sum().over("year", "old_riding"))
        .filter(pl.col("weight") > 0)
        .sort("year", "old_riding", "weight", descending=[False, False, True])
    )


def run(args: argparse.Namespace) -> int:
    paths = [
        RIDINGS_SHAPEFILE,
        RAW_DATA_DIR / VA_RESULTS_CSV,
        *(RAW_DATA_DIR / f for f in VA_SHAPEFILES.values()),
        *(RAW_DATA_DIR / f for f in VA_SITE_SHAPEFILES.values()),
    ]
    missing = [p for p in paths if not p.exists()]
    if missing:
        print(f"missing {', '.join(map(str, missing))}", file=sys.stderr)
        return 1

    ridings = read_shapefile(RIDINGS_SHAPEFILE, "ED_NAME")

    frames = []
    for year, filename in VA_SHAPEFILES.items():
        vas = read_shapefile(RAW_DATA_DIR / filename, "EDVA_CODE")
        weights = weigh(year, vas, ridings)

        unmatched = set(vas["EDVA_CODE"]) - set(weights["edva_code"])
        if unmatched:
            print(f"{year}: no riding overlaps {', '.join(sorted(unmatched))}", file=sys.stderr)
            return 1

        split = weights.filter(pl.col("edva_code").is_duplicated())["edva_code"].n_unique()
        print(f"{year}: {len(vas)} voting areas, {split} split between ridings")
        frames.append(weights)

        if year in VA_SITE_SHAPEFILES:
            sites = gpd.read_file(RAW_DATA_DIR / VA_SITE_SHAPEFILES[year], columns=["EDVA_CODE"], use_arrow=True)
            site_weights = weigh_sites(year, sites, ridings)
            split = site_weights.filter(pl.col("edva_code").is_duplicated())["edva_code"].n_unique()
            print(f"{year}: {site_weights['edva_code'].n_unique()} site voting areas, {split} split between ridings")
            frames.append(site_weights)

    weights = pl.concat(frames).sort("year", "edva_code", "weight", descending=[False, False, True])

    OUTPUT_PARQUET.parent.mkdir(exist_ok=True)
    weights.write_parquet(OUTPUT_PARQUET)
    print(f"wrote {weights.height} rows to {OUTPUT_PARQUET}")

    results = (
        read_frame(VA_RESULTS_CSV)
        .filter(pl.col("EVENT_NAME").is_in(list(GENERAL_ELECTIONS)))
        .with_columns(pl.col("EVENT_NAME").replace_strict(GENERAL_ELECTIONS).alias("YEAR"))
    )
    riding_weights = weigh_ridings(weights, results)

    unmatched = (
        results.filter(pl.col("YEAR").is_in(list(VA_SHAPEFILES)))
        .select("YEAR", "ED_NAME")
        .unique()
        .join(riding_weights, left_on=["YEAR", "ED_NAME"], right_on=["year", "old_riding"], how="anti")
    )
    if unmatched.height:
        print(f"no election-day votes mapped for {unmatched.rows()}", file=sys.stderr)
        return 1

    riding_weights.write_parquet(RIDING_OUTPUT_PARQUET)
    print(f"wrote {riding_weights.height} rows to {RIDING_OUTPUT_PARQUET}")

    return 0
