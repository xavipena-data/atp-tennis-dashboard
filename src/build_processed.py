"""Aggregate the cleaned ATP and WTA data into the tables the dashboard needs."""

from pathlib import Path

import pandas as pd

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"

# Interim file names per tour. ATP files have no prefix (historical), WTA files do.
FILES = {
    "atp": {"matches": "matches_clean.csv", "rankings": "rankings_clean.csv",
            "players": "players_clean.csv"},
    "wta": {"matches": "wta_matches_clean.csv", "rankings": "wta_rankings_clean.csv",
            "players": "wta_players_clean.csv"},
}

# Only the columns the Overview tab needs, read by name (never by position).
MATCH_COLUMNS = [
    "tourney_id", "tourney_name", "tourney_level", "level_group", "season",
    "round_clean", "outcome", "winner_id", "loser_id", "date",
]

FIRST_SEASON = 1968

LAST_SEASON = 2025

FREEZE_DAYS = 100  # a ranking that stays in force longer than this is the 2020 freeze

SLAMS = ["Australian Open", "Roland Garros", "Wimbledon", "US Open"]

# Names the source uses for the same major (1977 had two Australian Opens)
SLAM_ALIASES = {
    "Australian Open-1": "Australian Open",
    "Australian Open-2": "Australian Open",
    "Australian Open 1": "Australian Open",
    "Australian Open 2": "Australian Open",
    "Australian Championships": "Australian Open",
}

def load_matches(tour):
    """Load the cleaned matches of one tour with only the columns we use."""
    return pd.read_csv(
        INTERIM / FILES[tour]["matches"],
        usecols=MATCH_COLUMNS,
        dtype={"winner_id": str, "loser_id": str},
        low_memory=False,
    )

def build_seasons(matches):
    """One row per season: matches played and tournaments held.

    A match counts as played when it was completed or retired; walkovers are excluded.
    """
    played = matches[matches["outcome"].isin(["completed", "retired"])]
    seasons = (
        played.groupby("season")
        .agg(matches=("outcome", "size"), tournaments=("tourney_id", "nunique"))
        .reset_index()
    )
    return seasons

def build_titles(matches):
    """One row per player and season with at least one title.

    A title is a win in the final (round 'F') of any tour-level tournament.
    'slams' counts the titles that were Grand Slams.
    """
    finals = matches[matches["round_clean"] == "F"].copy()
    finals["slam"] = finals["level_group"] == "Grand Slam"
    titles = (
        finals.groupby(["winner_id", "season"])
        .agg(titles=("tourney_id", "size"), slams=("slam", "sum"))
        .reset_index()
        .rename(columns={"winner_id": "player_id"})
    )
    titles["slams"] = titles["slams"].astype(int)
    return titles


def check_titles(matches):
    """Print how many tournaments have a final, and whether any has more than one."""
    finals = matches[matches["round_clean"] == "F"]
    summary = pd.DataFrame({
        "tournaments": matches.groupby("level_group")["tourney_id"].nunique(),
        "with_final": finals.groupby("level_group")["tourney_id"].nunique(),
    }).fillna(0).astype(int)
    summary["without_final"] = summary["tournaments"] - summary["with_final"]
    print(summary.to_string())
    finals_per_tourney = finals.groupby("tourney_id").size()
    print("Tournaments with more than one final:", int((finals_per_tourney > 1).sum()))

def load_no1(tour):
    """Rows of the weekly rankings where the player is world No. 1."""
    rk = pd.read_csv(
        INTERIM / FILES[tour]["rankings"],
        usecols=["date", "id", "rank"],
        dtype={"id": str},
        parse_dates=["date"],
    )
    return rk[rk["rank"] == 1].sort_values(["date", "id"]).reset_index(drop=True)


def check_no1(no1):
    """Print the snapshot range, shared No. 1 weeks and the largest gaps between snapshots."""
    dates = pd.Series(no1["date"].unique())
    gaps = dates.diff().dt.days
    print("Snapshots with a No. 1:", len(dates), "| from", dates.min().date(), "to", dates.max().date())
    print("Snapshots with two No. 1:", int(no1.groupby("date").size().gt(1).sum()))
    print("Days between snapshots: median", int(gaps.median()), "| largest gaps:")
    big = gaps.nlargest(6)
    print(pd.DataFrame({
        "from": dates.shift(1)[big.index].dt.date,
        "to": dates[big.index].dt.date,
        "days": big.astype(int),
    }).to_string(index=False))

def days_by_season(no1, end):
    """Days each player was No. 1 in each calendar year, between 'date' and 'end'."""
    days = no1.assign(end=end)
    days["day"] = [list(pd.date_range(s, e - pd.Timedelta(days=1)))
                   for s, e in zip(days["date"], days["end"])]
    days = days.explode("day")
    days["season"] = pd.to_datetime(days["day"]).dt.year
    return days.groupby(["id", "season"]).size()


def build_weeks_no1(no1):
    """Weeks at No. 1 per player and season, in two versions.

    A ranking stays in force until the next one is published, so each snapshot holds
    from its own date up to the next snapshot (the last one for 7 days, never past the
    end of the last season). Two players sharing No. 1 each get the full time.
    - calendar_weeks: real elapsed time.
    - weeks: official count; a ranking in force for over FREEZE_DAYS (the 2020 freeze,
      when no rankings were published) counts as a single week.
    """
    dates = no1["date"].drop_duplicates().sort_values().reset_index(drop=True)
    last_end = min(dates.iloc[-1] + pd.Timedelta(days=7), pd.Timestamp(LAST_SEASON + 1, 1, 1))
    end_of = dict(zip(dates, dates.shift(-1).fillna(last_end)))

    end = no1["date"].map(end_of)
    frozen = (end - no1["date"]).dt.days > FREEZE_DAYS
    official_end = end.where(~frozen, no1["date"] + pd.Timedelta(days=7))

    weeks = pd.DataFrame({
        "calendar_weeks": days_by_season(no1, end),
        "weeks": days_by_season(no1, official_end),
    })
    return weeks.div(7).round(2).reset_index().rename(columns={"id": "player_id"})


def check_weeks(weeks, tour):
    """Print the top holders and the seasons whose calendar total is not about 52 weeks."""
    players = pd.read_csv(INTERIM / FILES[tour]["players"], usecols=["id", "name"], dtype={"id": str})
    top = (weeks.groupby("player_id")["weeks"].sum().sort_values(ascending=False)
           .head(8).reset_index().merge(players, left_on="player_id", right_on="id", how="left"))
    print(top[["name", "weeks"]].to_string(index=False))
    per_season = weeks.groupby("season")["calendar_weeks"].sum()
    odd = per_season[(per_season - 52.2).abs() > 0.4]
    print("Seasons whose calendar total is not about 52 weeks:")
    print(odd.round(2).to_string())  

def build_slam_champions(matches):
    """One row per Grand Slam final: who won each major in each season.

    'order' numbers the editions of the same major within a year (the Australian Open
    was held twice in 1977), so the dashboard can show both champions.
    """
    finals = matches[(matches["level_group"] == "Grand Slam") & (matches["round_clean"] == "F")].copy()
    finals["slam"] = finals["tourney_name"].replace(SLAM_ALIASES)
    unknown = set(finals["slam"]) - set(SLAMS)
    assert not unknown, f"Unexpected Grand Slam names: {unknown}"

    champs = (
        finals.rename(columns={"winner_id": "player_id"})
        [["season", "slam", "player_id", "date"]]
        .sort_values(["season", "slam", "date"])
    )
    champs["order"] = champs.groupby(["season", "slam"]).cumcount() + 1
    return champs.drop(columns="date")


def check_slams(champs):
    """Print the seasons that do not have exactly one champion per major."""
    per_season = champs.groupby("season").size()
    print("Grand Slam finals:", len(champs))
    print("Seasons without exactly 4 champions:")
    print(per_season[per_season != 4].to_string())    

def build_players_range(matches):
    """Distinct players with at least one match played, for every range of seasons.

    One row per (season_from, season_to) pair, so the dashboard can show the KPI for
    any period without having to add up players that appear in several seasons.
    """
    played = matches[matches["outcome"].isin(["completed", "retired"])]
    seen = pd.concat([
        played[["winner_id", "season"]].set_axis(["player_id", "season"], axis=1),
        played[["loser_id", "season"]].set_axis(["player_id", "season"], axis=1),
    ], ignore_index=True).drop_duplicates()

    seasons = range(FIRST_SEASON, LAST_SEASON + 1)
    present = (pd.crosstab(seen["player_id"], seen["season"])
               .reindex(columns=seasons, fill_value=0) > 0).to_numpy()

    rows = []
    for i, start in enumerate(seasons):
        # for a fixed start, a player counts from the first season they appear onwards
        counts = np.logical_or.accumulate(present[:, i:], axis=1).sum(axis=0)
        rows += [(start, start + j, int(c)) for j, c in enumerate(counts)]
    return pd.DataFrame(rows, columns=["season_from", "season_to", "players"])


def build_players(tour):
    """Name and country of every player in the cleaned players table."""
    players = pd.read_csv(INTERIM / FILES[tour]["players"],
                          usecols=["id", "name", "ioc"], dtype={"id": str})
    return players.rename(columns={"id": "player_id"})    

def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for tour in FILES:
        matches = load_matches(tour)
        print(tour.upper())

        # Matches played and tournaments held per season
        seasons = build_seasons(matches)
        seasons.to_csv(PROCESSED / f"{tour}_seasons.csv", index=False)

        # Distinct players for every range of seasons
        players_range = build_players_range(matches)
        players_range.to_csv(PROCESSED / f"{tour}_players_range.csv", index=False)
        full = players_range[(players_range["season_from"] == FIRST_SEASON) & (players_range["season_to"] == LAST_SEASON)]
        last = players_range[(players_range["season_from"] == LAST_SEASON) & (players_range["season_to"] == LAST_SEASON)]
        print("Players:", len(players_range), "ranges | all seasons:", int(full["players"].iloc[0]),
              "| 2025 only:", int(last["players"].iloc[0]))

        # Names and countries
        players = build_players(tour)
        players.to_csv(PROCESSED / f"{tour}_players.csv", index=False)

        # Titles per player and season
        check_titles(matches)
        titles = build_titles(matches)
        titles.to_csv(PROCESSED / f"{tour}_titles.csv", index=False)
        print("Title rows:", len(titles), "| total titles:", int(titles["titles"].sum()),
              "| Grand Slams:", int(titles["slams"].sum()))

        # Weeks at No. 1 per player and season
        no1 = load_no1(tour)
        check_no1(no1)
        weeks = build_weeks_no1(no1)
        weeks.to_csv(PROCESSED / f"{tour}_weeks_no1.csv", index=False)
        check_weeks(weeks, tour)

        # Grand Slam champions
        champs = build_slam_champions(matches)
        champs.to_csv(PROCESSED / f"{tour}_slam_champions.csv", index=False)
        check_slams(champs)
        print()


if __name__ == "__main__":
    main()