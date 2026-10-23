"""Average the last few days of polls before each election, province-wide and per region.

Takes no arguments. Run gen_dataset first. The average takes every poll
released in the WINDOW_DAYS days up to the last poll of that election, which
for past elections is the eve of the vote. A party a poll didn't report is
left out of that party's average, rather than counted as 0.

Writes each election's provincial average (each party's share, in percent) to
processed_data/poll_averages.parquet.

Writes each region's swing against the province in the regional polls to
processed_data/region_poll_swings.parquet, as region_poll_swing_<party> on the
logit scale against the NDP, for every election with a previous one in the
results. Pollsters report their own, coarser regions, so a poll is compared to
its own region's previous result (built from ours by region_weights.csv: how
much of the pollster's region lies in each of ours), and that swing is spread
over our regions inside it by the same weights. Comparing a pollster's level
to each of our regions' previous results instead would read every difference
between, say, Surrey and Burnaby as a swing back to the Metro Vancouver
average. The province's swing (each region's, weighted by its previous votes)
is taken out, which leaves only how the regions differ.
"""

import argparse
import datetime
import sys

import polars as pl

from commands.gen_dataset import OUTPUT_PARQUET as MODEL_INPUT_PARQUET
from paths import PROCESSED_DATA_DIR, RAW_DATA_DIR

PROV_POLLS_CSV = RAW_DATA_DIR / "prov_polls.csv"
REGIONAL_POLLS_CSV = RAW_DATA_DIR / "regional_polls.csv"
REGION_WEIGHTS_CSV = RAW_DATA_DIR / "region_weights.csv"
REGIONS_CSV = RAW_DATA_DIR / "regions.csv"

OUTPUT_PARQUET = PROCESSED_DATA_DIR / "poll_averages.parquet"
REGION_SWINGS_PARQUET = PROCESSED_DATA_DIR / "region_poll_swings.parquet"

POLL_PARTIES = ["ndp", "green", "right", "other"]
# every logit is against the NDP, as in gen_dataset
LOGIT_PARTIES = [p for p in POLL_PARTIES if p != "ndp"]

WINDOW_DAYS = 5


def add_arguments(parser: argparse.ArgumentParser) -> None:
    pass


def read_csv(path) -> pl.DataFrame:
    return pl.read_csv(path, encoding="utf8-lossy", try_parse_dates=True)


def in_window(polls: pl.DataFrame, last_dates: pl.DataFrame) -> pl.DataFrame:
    """Keep the polls released in the window up to their election's last poll."""
    return polls.join(last_dates, on="year").filter(
        pl.col("date") > pl.col("last_date") - datetime.timedelta(days=WINDOW_DAYS)
    )


def logit(party: str, prefix: str = "") -> pl.Expr:
    """party's logit against the NDP, null where either has no share."""
    share, ndp = pl.col(f"{prefix}{party}"), pl.col(f"{prefix}ndp")
    return pl.when((share > 0) & (ndp > 0)).then((share / ndp).log())


def weighted_mean(value: str, weight: str) -> pl.Expr:
    """The average of value by weight, over the rows that have it."""
    known = pl.col(value).is_not_null()
    return (pl.col(value) * pl.col(weight)).filter(known).sum() / pl.col(weight).filter(known).sum()


def previous_results(poll_years: list[int]) -> pl.DataFrame:
    """Each region's votes in the election before each poll year: the dataset's
    previous results for its own years, and the latest results for any poll
    year after them."""
    dataset = pl.read_parquet(MODEL_INPUT_PARQUET)
    latest = dataset["year"].max()
    prev = [
        dataset.select("year", "riding", pl.col("prev_total").alias("total"), *(pl.col(f"prev_votes_{p}").alias(p) for p in POLL_PARTIES))
    ] + [
        dataset.filter(pl.col("year") == latest).select(
            pl.lit(year, dtype=dataset["year"].dtype).alias("year"), "riding", "total", *(pl.col(f"votes_{p}").alias(p) for p in POLL_PARTIES)
        )
        for year in poll_years
        if year > latest
    ]
    return (
        pl.concat(prev)
        .join(read_csv(REGIONS_CSV), on="riding", validate="m:1")
        .group_by("year", "region")
        .agg(pl.col("total", *POLL_PARTIES).sum())
    )


def region_swings(regional_polls: pl.DataFrame, weights: pl.DataFrame, prev: pl.DataFrame) -> pl.DataFrame:
    """Each of our regions' poll swing against the province (see the module
    docstring)."""
    # each pollster region's previous shares, from ours by how much of it
    # lies in each
    prev_shares = prev.with_columns(pl.col(p) / pl.col("total") for p in POLL_PARTIES)
    pollster_prev = (
        weights.join(prev_shares, left_on=["year", "our_region"], right_on=["year", "region"])
        .group_by("pollster", "year", "region")
        .agg((pl.col(p) * pl.col("weight")).sum() / pl.col("weight").sum() for p in POLL_PARTIES)
    )

    # each poll's swing in its own region, spread over ours
    swings = (
        regional_polls.join(
            pollster_prev.rename({p: f"prev_{p}" for p in POLL_PARTIES}), on=["pollster", "year", "region"]
        )
        .select(
            "pollster",
            "date",
            "year",
            "region",
            *((logit(p) - logit(p, "prev_")).alias(f"swing_{p}") for p in LOGIT_PARTIES),
        )
        .join(weights, on=["pollster", "year", "region"])
        .group_by("year", "our_region")
        .agg(
            *(weighted_mean(f"swing_{p}", "weight") for p in LOGIT_PARTIES),
            pl.struct("pollster", "date").n_unique().alias("polls"),
        )
        .rename({"our_region": "region"})
    )

    # take out the province's swing, so each region is how it differs
    size = pl.col("total")
    return (
        prev.select("year", "region", "total")
        .join(swings, on=["year", "region"])
        .select(
            "year",
            "region",
            *(
                (
                    pl.col(f"swing_{p}")
                    - (size * pl.col(f"swing_{p}")).sum().over("year")
                    / size.filter(pl.col(f"swing_{p}").is_not_null()).sum().over("year")
                ).alias(f"region_poll_swing_{p}")
                for p in LOGIT_PARTIES
            ),
            "polls",
        )
        .sort("year", "region")
    )


def run(args: argparse.Namespace) -> int:
    missing = [
        p for p in (PROV_POLLS_CSV, REGIONAL_POLLS_CSV, REGION_WEIGHTS_CSV, REGIONS_CSV, MODEL_INPUT_PARQUET) if not p.exists()
    ]
    if missing:
        print(f"missing {', '.join(map(str, missing))} (run gen_dataset first?)", file=sys.stderr)
        return 1

    prov = read_csv(PROV_POLLS_CSV)
    last_dates = prov.group_by("year").agg(pl.col("date").max().alias("last_date"))

    province = (
        in_window(prov, last_dates)
        .group_by("year")
        .agg(*(pl.col(p).mean() for p in POLL_PARTIES), pl.len().alias("polls"))
        .sort("year")
    )

    # one row per pollster region and region of ours it overlaps
    weights = read_csv(REGION_WEIGHTS_CSV)
    regions = [c for c in weights.columns if c not in ("pollster", "year", "region")]
    weights = weights.unpivot(
        regions, index=["pollster", "year", "region"], variable_name="our_region", value_name="weight"
    ).filter(pl.col("weight") > 0)

    regional_polls = in_window(read_csv(REGIONAL_POLLS_CSV), last_dates)
    unweighted = regional_polls.join(weights, on=["pollster", "year", "region"], how="anti")
    if unweighted.height:
        print(
            "no region_weights for: "
            + ", ".join(f"{r['pollster']} {r['year']} {r['region']}" for r in unweighted.iter_rows(named=True)),
            file=sys.stderr,
        )
        return 1

    prev = previous_results(province["year"].to_list())
    swings = region_swings(regional_polls, weights, prev)

    # every region with a previous result should have a poll swing for every
    # party, or the model gets none for it
    uncovered = prev.select("year", "region").join(
        swings.filter(pl.all_horizontal(pl.col("^region_poll_swing_.*$").is_not_null())),
        on=["year", "region"],
        how="anti",
    )
    for row in uncovered.sort("year", "region").iter_rows(named=True):
        print(f"warning: no regional poll swing for {row['region']} in {row['year']}", file=sys.stderr)

    OUTPUT_PARQUET.parent.mkdir(exist_ok=True)
    province.write_parquet(OUTPUT_PARQUET)
    print(province)
    print(f"wrote {province.height} rows to {OUTPUT_PARQUET}")
    swings.write_parquet(REGION_SWINGS_PARQUET)
    print(f"wrote {swings.height} rows to {REGION_SWINGS_PARQUET}")

    return 0
