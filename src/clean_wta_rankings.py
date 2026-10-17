"""Build the weekly WTA ranking table.

Reads the six decade files in data/raw/wta (1984 onwards) and writes
data/interim/wta_rankings_clean.csv.
"""
import pandas as pd

from clean_wta_data import INTERIM, LAST_SEASON, RAW_MATCHES, ROOT, read_csv_robust

# RAW_MATCHES is the data/raw/wta folder, which also holds the ranking files
DECADES = ["80s", "90s", "00s", "10s", "20s", "current"]
FIRST_RANKING_SEASON = 1984
MAX_RANK = 500   # the dashboard never needs the tail of the ranking
CLEAN_PLAYERS = INTERIM / "wta_players_clean.csv"


def load_rankings():
    """Stack the decade files with explicit types and keep the seasons we cover."""
    frames = [
        read_csv_robust(RAW_MATCHES / f"wta_rankings_{d}.csv",
                        dtype={"ranking_date": str, "player": str})
        for d in DECADES
    ]
    rankings = pd.concat(frames, ignore_index=True)
    rankings = rankings.rename(columns={"ranking_date": "date", "player": "id"})
    rankings["date"] = pd.to_datetime(rankings["date"], format="%Y%m%d")
    rankings["rank"] = pd.to_numeric(rankings["rank"], errors="coerce")
    rankings["points"] = pd.to_numeric(rankings["points"], errors="coerce")

    before = len(rankings)
    in_scope = rankings["date"].dt.year.between(FIRST_RANKING_SEASON, LAST_SEASON)
    rankings = rankings.loc[in_scope, ["date", "id", "rank", "points"]].copy()
    print(f"Loaded {before:,} rows; kept {len(rankings):,} "
          f"(seasons {FIRST_RANKING_SEASON}-{LAST_SEASON})")
    print(f"Snapshots: {rankings['date'].nunique():,}, "
          f"{rankings['date'].min().date()} to {rankings['date'].max().date()}")
    return rankings

def drop_incomplete_snapshots(rankings):
    """Drop the snapshots without a No. 1 (truncated lists in the source)."""
    best = rankings.groupby("date")["rank"].transform("min")
    incomplete = best > 1
    dates = sorted(rankings.loc[incomplete, "date"].dt.date.unique())
    print(f"Incomplete snapshots dropped: {len(dates)} {dates}")
    return rankings[~incomplete].copy()

def keep_relevant_rows(rankings):
    """Keep the players of the cleaned table and the top MAX_RANK of each snapshot."""
    players = pd.read_csv(CLEAN_PLAYERS, usecols=["id"], dtype={"id": str})
    known = rankings["id"].isin(players["id"])
    in_top = rankings["rank"] <= MAX_RANK

    print(f"Rows below rank {MAX_RANK}: {(~in_top).sum():,}")
    print(f"Rows inside the top {MAX_RANK} of players missing from the cleaned players: "
          f"{(~known & in_top).sum():,}")
    print(f"  best rank among them: {rankings.loc[~known & in_top, 'rank'].min():.0f}")
    kept = rankings[known & in_top].copy()
    print(f"Kept {len(kept):,} of {len(rankings):,} rows")
    return kept


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
    shared = number_ones[number_ones > 1]
    print(f"Snapshots: {snapshots:,}; exactly one No. 1: {(number_ones == 1).sum():,}; "
          f"more than one: {len(shared)}; none: {snapshots - len(number_ones)}")
    assert snapshots == len(number_ones), "a snapshot without a No. 1"
    # Known case: Seles shared the No. 1 with Graf after her return (Aug 1995 - 1996)
    assert set(shared.index.year) <= {1995, 1996}, "unexpected shared No. 1"
    print("All checks passed")


def write_output(rankings):
    rankings["rank"] = rankings["rank"].astype(int)
    INTERIM.mkdir(parents=True, exist_ok=True)
    path = INTERIM / "wta_rankings_clean.csv"
    rankings.to_csv(path, index=False)
    print(f"Wrote {len(rankings):,} rows x {rankings.shape[1]} columns -> {path.relative_to(ROOT)}")

def main():
    rankings = load_rankings()
    rankings = drop_incomplete_snapshots(rankings)
    rankings = keep_relevant_rows(rankings)
    rankings = remove_duplicate_rows(rankings)
    run_checks(rankings)
    write_output(rankings)

if __name__ == "__main__":
    main()