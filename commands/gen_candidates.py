"""Build who ran in each riding, and where incumbents ran, for every general election through 2026.

Takes no arguments. Run process_edata first. Past elections come from the
results; 2026 comes from raw_data/candidates_2026.csv, which has each party's
candidates per 2023 riding ("; " between several, blank for none) and the
sitting MLA, if they're running there, spelled as in the candidate columns.
The results start at 2013, so the MLAs elected in 2009 come from
raw_data/mlas_2009.csv, with each one's name as spelled in the 2013 results if
they ran again in the same riding (2009 and 2013 share a map). Writes two
files:

processed_data/candidates.parquet has one row per candidate on that
election's own ridings, with their party (grouped as in gen_dataset, plus
centrebc and onebc for 2026), whether they were elected (null for 2026), and
whether they were the incumbent. In 2017 through 2024 the incumbent is whoever
won that seat in the previous general election, matched by name wherever they
ran; in 2013 only in the same riding. In 2026 it's the sitting MLA, counting
floor crossings and by-elections. Names are "Last, First" before 2026, and
"First Last" as on Wikipedia in 2026.

processed_data/riding_candidates.parquet has one row per (election, 2023
riding). running_<party> says whether each model party (ndp, right, green,
other) ran: before 2024 the ridings were different, so it's whether
gen_dataset counts the party as contesting it, its candidates' old ridings
covering at least MIN_COVERAGE of the votes. incumbent_<party> is the share of
the riding's voters whose MLA from the previous election is on their ballot
again running for that party. Across a boundary change we only
know how each map's ridings split the 2023 riding, not how they overlap each
other, so we take an old riding and the new riding its MLA runs in to share
as much of the 2023 riding as they can, the smaller of their two shares.
"""

import argparse
import sys

import polars as pl

from commands.gen_dataset import PARTIES, tally_results
from commands.process_edata import MAP_VERSIONS
from commands.process_edata import OUTPUT_PARQUET as RIDING_RESULTS_PARQUET
from paths import PROCESSED_DATA_DIR, RAW_DATA_DIR

CANDIDATES_2026_CSV = RAW_DATA_DIR / "candidates_2026.csv"
MLAS_2009_CSV = RAW_DATA_DIR / "mlas_2009.csv"

CANDIDATES_PARQUET = PROCESSED_DATA_DIR / "candidates.parquet"
RIDING_CANDIDATES_PARQUET = PROCESSED_DATA_DIR / "riding_candidates.parquet"

# The model party each party group runs under. CentreBC and OneBC are new in
# 2026, with no past vote to swing from, so they go in with everyone else.
MODEL_PARTY: dict[str, str] = {
    "ndp": "ndp",
    "liberal": "right",
    "conservative": "right",
    "green": "green",
    "centrebc": "other",
    "onebc": "other",
    "other": "other",
}
RUNNING_PARTIES = ["green", "right", "other", "ndp"]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    pass


def read_2026() -> pl.DataFrame:
    """Read the 2026 file into one row per candidate."""
    wide = pl.read_csv(CANDIDATES_2026_CSV, infer_schema=False).fill_null("")
    parties = [c for c in wide.columns if c not in ("riding", "incumbent")]
    return (
        wide.unpivot(parties, index=["riding", "incumbent"], variable_name="party", value_name="candidate")
        .with_columns(pl.col("candidate").str.split("; "))
        .explode("candidate")
        .filter(pl.col("candidate") != "")
        .select(
            pl.lit(2026, dtype=pl.Int64).alias("year"),
            "riding",
            "party",
            "candidate",
            pl.lit(None, dtype=pl.Boolean).alias("elected"),
            (pl.col("candidate") == pl.col("incumbent")).alias("incumbent"),
        )
    )


def match_incumbents(candidates: pl.DataFrame, prev: int, year: int) -> pl.DataFrame:
    """Find where each winner of the prev election ran in year, as (prev_riding, riding, candidate) rows.

    Names don't always match exactly (middle names come and go, Greg is
    Gregory), so a winner matches a candidate with the same surname whose first
    name starts with theirs, or the other way around.
    """

    def keyed(frame: pl.DataFrame) -> pl.DataFrame:
        name = pl.col("candidate").str.split_exact(", ", 1)
        return frame.with_columns(
            name.struct.field("field_0").str.to_lowercase().alias("last"),
            name.struct.field("field_1").str.split(" ").list.first().str.to_lowercase().alias("first"),
        )

    winners = keyed(candidates.filter((pl.col("year") == prev) & pl.col("elected")))
    running = keyed(candidates.filter(pl.col("year") == year))
    return (
        winners.select(pl.col("riding").alias("prev_riding"), "last", "first")
        .join(running.select("riding", "candidate", "last", "first"), on="last", suffix="_now")
        .filter(pl.col("first").str.starts_with(pl.col("first_now")) | pl.col("first_now").str.starts_with(pl.col("first")))
        # a prefix can catch someone else too (Aman, Amandeep), so an exact
        # first name wins when there is one
        .with_columns(exact=pl.col("first") == pl.col("first_now"))
        .filter(pl.col("exact") | ~pl.col("exact").any().over("prev_riding"))
        .select("prev_riding", "riding", "candidate")
    )


def incumbency(results: pl.DataFrame, candidates: pl.DataFrame, matches: pl.DataFrame) -> pl.DataFrame:
    """Share of each 2023 riding's voters whose previous MLA is running again, per model party.

    matches has each past incumbent running again, as (year, prev_riding,
    riding, candidate).
    """
    # how each election's ridings split up each 2023 riding
    pieces = (
        results.group_by("year", "riding", "riding_2024")
        .agg(pl.col("votes").sum())
        .with_columns((pl.col("votes") / pl.col("votes").sum().over("year", "riding_2024")).alias("share"))
        .select("year", "riding", pl.col("riding_2024").alias("new_riding"), "share")
    )

    # the party each incumbent runs for now, which may not be the one they won for
    moved_all = matches.join(
        candidates.select("year", "riding", "candidate", pl.col("party").replace_strict(MODEL_PARTY)),
        on=["year", "riding", "candidate"],
    )

    years = sorted(results["year"].unique())
    shares = []
    for prev, year in zip(years, years[1:]):
        moved = moved_all.filter(pl.col("year") == year).drop("year")
        before = pieces.filter(pl.col("year") == prev).drop("year").rename({"riding": "prev_riding"})
        if MAP_VERSIONS[prev] == MAP_VERSIONS[year]:
            # same map: the MLA's old voters are only on their ballot if they ran in the same riding
            overlap = moved.filter(pl.col("riding") == pl.col("prev_riding")).join(before, on="prev_riding")
        else:
            after = pieces.filter(pl.col("year") == year).drop("year")
            overlap = (
                moved.join(before, on="prev_riding")
                .join(after, on=["riding", "new_riding"], suffix="_now")
                .with_columns(pl.min_horizontal("share", "share_now").alias("share"))
            )
        shares.append(overlap.select(pl.lit(year, dtype=pl.Int64).alias("year"), "new_riding", "party", "share"))

    # 2009 and 2013 share a map, but there are no 2009 results to split it up
    # with, so the first election's incumbents use its own
    shares.append(
        moved_all.filter(pl.col("year") == years[0])
        .join(pieces, on=["year", "riding"])
        .select("year", "new_riding", "party", "share")
    )

    # 2026 is on the 2024 map, and the file already says which sitting MLA runs where
    shares.append(
        candidates.filter((pl.col("year") == 2026) & pl.col("incumbent")).select(
            "year", pl.col("riding").alias("new_riding"), pl.col("party").replace_strict(MODEL_PARTY), share=pl.lit(1.0)
        )
    )

    return (
        pl.concat(shares)
        .group_by("year", pl.col("new_riding").alias("riding"))
        .agg((pl.col("share").filter(pl.col("party") == p).sum()).alias(f"incumbent_{p}") for p in RUNNING_PARTIES)
    )


def running(frame: pl.DataFrame) -> pl.DataFrame:
    """Widen (year, riding, party, ran) rows into a running_<party> per model party."""
    return (
        frame.with_columns(pl.col("party").replace_strict(MODEL_PARTY))
        .group_by("year", "riding")
        .agg((pl.col("ran") & (pl.col("party") == p)).any().alias(f"running_{p}") for p in RUNNING_PARTIES)
    )


def run(args: argparse.Namespace) -> int:
    missing = [p for p in (RIDING_RESULTS_PARQUET, CANDIDATES_2026_CSV, MLAS_2009_CSV) if not p.exists()]
    if missing:
        print(f"missing {', '.join(map(str, missing))} (run process_edata first?)", file=sys.stderr)
        return 1

    results = pl.read_parquet(RIDING_RESULTS_PARQUET).with_columns(
        pl.col("party").replace_strict(PARTIES, default="other")
    )
    candidates_2026 = read_2026()

    unknown = set(candidates_2026["party"]) - set(MODEL_PARTY)
    if unknown:
        print(f"unknown parties in {CANDIDATES_2026_CSV}: {', '.join(sorted(unknown))}", file=sys.stderr)
        return 1

    ridings = set(results.filter(pl.col("year") == results["year"].max())["riding_2024"])
    mismatched = ridings ^ set(candidates_2026["riding"])
    if mismatched:
        print(f"{CANDIDATES_2026_CSV} and the 2023 ridings differ: {', '.join(sorted(mismatched))}", file=sys.stderr)
        return 1

    named = pl.read_csv(CANDIDATES_2026_CSV, infer_schema=False).filter(pl.col("incumbent").fill_null("") != "")
    unmatched = named.join(candidates_2026.filter(pl.col("incumbent")), on="riding", how="anti")
    if unmatched.height:
        print(f"incumbents not among their riding's candidates: {', '.join(unmatched['riding'])}", file=sys.stderr)
        return 1

    # results are split into pieces of the old ridings inside each 2023 riding;
    # put each candidate back together on their own riding
    past = results.group_by("year", "riding", "party", "candidate").agg(pl.col("elected").any())

    years = sorted(past["year"].unique())
    mlas_2009 = pl.read_csv(MLAS_2009_CSV).filter(pl.col("candidate").is_not_null())
    matches = pl.concat(
        [
            mlas_2009.select(pl.lit(years[0], dtype=pl.Int64).alias("year"), pl.col("riding").alias("prev_riding"), "riding", "candidate"),
            *(
                match_incumbents(past, prev, year).with_columns(pl.lit(year, dtype=pl.Int64).alias("year"))
                for prev, year in zip(years, years[1:])
            ),
        ],
        how="diagonal",
    )
    unmatched = mlas_2009.join(past.filter(pl.col("year") == years[0]), on=["riding", "candidate"], how="anti")
    if unmatched.height:
        print(f"{MLAS_2009_CSV} names candidates not in the {years[0]} results:\n{unmatched}", file=sys.stderr)
        return 1
    doubled = matches.filter(pl.struct("year", "prev_riding").is_duplicated() | pl.struct("year", "riding", "candidate").is_duplicated())
    if doubled.height:
        print(f"ambiguous incumbent name matches:\n{doubled}", file=sys.stderr)
        return 1

    candidates = (
        pl.concat(
            [
                past.join(matches.select("year", "riding", "candidate", incumbent=pl.lit(True)), on=["year", "riding", "candidate"], how="left")
                .with_columns(pl.col("incumbent").fill_null(False)),
                candidates_2026,
            ]
        )
        .sort("year", "riding", "party", "candidate")
    )

    riding_candidates = (
        pl.concat(
            [
                running(tally_results(results).rename({"usable": "ran"})),
                running(candidates_2026.with_columns(ran=pl.lit(True))),
            ]
        )
        .join(incumbency(results, candidates, matches), on=["year", "riding"], how="left")
        .with_columns(pl.col(f"incumbent_{p}").fill_null(0.0) for p in RUNNING_PARTIES)
        .sort("year", "riding")
    )

    PROCESSED_DATA_DIR.mkdir(exist_ok=True)
    candidates.write_parquet(CANDIDATES_PARQUET)
    print(f"wrote {candidates.height} rows to {CANDIDATES_PARQUET}")
    riding_candidates.write_parquet(RIDING_CANDIDATES_PARQUET)
    print(f"wrote {riding_candidates.height} rows to {RIDING_CANDIDATES_PARQUET}")

    return 0
