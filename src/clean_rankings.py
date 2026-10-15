"""Build the weekly ranking table.

Official ATP ranking for 1973-1975, TML Rankings from 1976 (see section 8 of
notebooks/01_data_profiling.ipynb). Writes data/interim/rankings_clean.csv.
"""
import pandas as pd

from clean_data import INTERIM, LAST_SEASON, ROOT, read_csv_robust

RANKINGS_DIR = ROOT / "data" / "raw" / "rankings"

OFFICIAL_FOLDER = "Official ATP Rankings"
TML_FOLDER = "TML Rankings"
OFFICIAL_YEARS = range(1973, 1976)         # few snapshots, but the official No. 1
TML_YEARS = range(1976, LAST_SEASON + 1)   # weekly snapshots
MAX_RANK = 500   # the dashboard never needs the tail of the ranking
CLEAN_PLAYERS = INTERIM / "players_clean.csv"
# Spellings in the TML rankings that differ from the match files
RANKING_NAME_FIXES = {
    "Chris Lewis (NZL)": "Chris Lewis",
    "Teimuraz Kakoulia": "Teimuraz Kakulia",
}


def load_ranking_year(folder, year):
    """Read one season of one ranking source with explicit types."""
    path = next(RANKINGS_DIR.rglob(f"{folder}/{year}.csv"))
    df = read_csv_robust(path, dtype=str)                 # everything as text first
    df = df[df["date"] != "date"].copy()                  # repeated header row (TML 2025)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df["points"] = pd.to_numeric(df["points"], errors="coerce")
    df["source"] = folder
    if "name" not in df.columns:      # the official files may not have it
        df["name"] = pd.NA
    return df[["date", "id", "name", "rank", "points", "source"]]


def load_rankings():
    """Official source for the first years, TML for the rest."""
    frames = [load_ranking_year(OFFICIAL_FOLDER, y) for y in OFFICIAL_YEARS]
    frames += [load_ranking_year(TML_FOLDER, y) for y in TML_YEARS]
    rankings = pd.concat(frames, ignore_index=True)

    summary = rankings.groupby("source").agg(
        rows=("id", "size"),
        snapshots=("date", "nunique"),
        first=("date", "min"),
        last=("date", "max"),
    )
    print(summary.to_string())
    print(f"Total: {len(rankings):,} rows")
    return rankings

def keep_relevant_rows(rankings):
    """Keep the players of the cleaned table and the top MAX_RANK of each snapshot."""
    players = pd.read_csv(CLEAN_PLAYERS, usecols=["id"], dtype={"id": str})
    known = rankings["id"].isin(players["id"])
    in_top = rankings["rank"] <= MAX_RANK

    print(f"Rows below rank {MAX_RANK}: {(~in_top).sum():,}")
    print(f"Rows inside the top {MAX_RANK} of players missing from the cleaned players: "
          f"{(~known & in_top).sum():,}")
    kept = rankings[known & in_top].copy()
    print(f"Kept {len(kept):,} of {len(rankings):,} rows")
    return kept

def recover_missing_ids(rankings):
    """Fill the empty ids (TML files, 1976-1989) by matching the player name."""
    players = pd.read_csv(CLEAN_PLAYERS, usecols=["id", "name"], dtype=str)
    unambiguous = players[~players["name"].duplicated(keep=False)]   # shared names would be a guess
    lookup = dict(zip(unambiguous["name"], unambiguous["id"]))

    missing = rankings["id"].isna()
    names = rankings.loc[missing, "name"].replace(RANKING_NAME_FIXES)
    rankings.loc[missing, "id"] = names.map(lookup)

    still = rankings["id"].isna()
    print(f"Rows without id: {missing.sum():,} -> recovered {missing.sum() - still.sum():,}, "
          f"still without id: {still.sum():,}")
    print(f"  best rank among those still without id: {rankings.loc[still, 'rank'].min():.0f}")
    return rankings


def remove_duplicate_rows(rankings):
    """Exact copies are dropped; same player and week with different points keeps the higher."""
    n0 = len(rankings)
    rankings = rankings.drop_duplicates(["date", "id", "rank", "points"])
    n1 = len(rankings)
    rankings = (rankings.sort_values("points", ascending=False)
                        .drop_duplicates(["date", "id"])
                        .sort_values(["date", "rank"]))
    print(f"Exact duplicate rows removed: {n0 - n1:,}; "
          f"conflicting duplicates (kept higher points): {n1 - len(rankings):,}")
    return rankings


def run_checks(rankings):
    """Stop the script if the cleaned ranking breaks a rule we rely on."""
    assert rankings["rank"].notna().all(), "rows without a rank"
    assert (rankings["rank"] >= 1).all(), "rank below 1"
    assert not rankings.duplicated(["date", "id"]).any(), "a player twice in one snapshot"

    number_ones = rankings[rankings["rank"] == 1].groupby("date").size()
    snapshots = rankings["date"].nunique()
    print(f"Snapshots: {snapshots:,}; exactly one No. 1: {(number_ones == 1).sum():,}; "
          f"more than one: {(number_ones > 1).sum()}; none: {snapshots - len(number_ones)}")
    print("All checks passed")


def write_output(rankings):
    rankings["rank"] = rankings["rank"].astype(int)
    INTERIM.mkdir(parents=True, exist_ok=True)
    path = INTERIM / "rankings_clean.csv"
    rankings.to_csv(path, index=False)
    print(f"Wrote {len(rankings):,} rows x {rankings.shape[1]} columns -> {path.relative_to(ROOT)}")


def main():
    rankings = load_rankings()
    rankings = recover_missing_ids(rankings)
    rankings = keep_relevant_rows(rankings)
    rankings = remove_duplicate_rows(rankings)
    run_checks(rankings)
    write_output(rankings.drop(columns="name"))


if __name__ == "__main__":
    main()