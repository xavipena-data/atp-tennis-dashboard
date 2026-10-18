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
    "surface", "minutes",
    "w_ace", "l_ace", "w_svpt", "l_svpt",
    "w_1stIn", "l_1stIn", "w_1stWon", "l_1stWon",
    "w_bpSaved", "l_bpSaved", "w_bpFaced", "l_bpFaced",
    "winner_ht", "loser_ht",
    "winner_rank", "loser_rank", "winner_age", "loser_age",
    "winner_hand", "loser_hand", "winner_name", "loser_name",
    "score", "winner_ioc", "loser_ioc",
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

SURFACES = ["Hard", "Clay", "Grass", "Carpet"]

MIN_CAREER_MATCHES = 50  # same rule as the Player profile tab

ROUND_CODES = {"R256": 0, "R128": 0, "R64": 1, "R32": 2, "R16": 3, "QF": 4, "SF": 5, "F": 6}  # 7 is the title

MAX_RANK = 500  # the ranking chart only shows the top 500

MIN_MEETINGS = 3  # a pair is kept in the head-to-head table from this many meetings

GAP_EDGES = [1, 6, 20, 50, 100]
UPSET_ROUNDS = {"R256": 0, "R128": 0, "R64": 1, "R32": 2, "R16": 3, "QF": 4, "SF": 5, "F": 6}

SERVE_SUMS = ["ace", "svpt", "1stIn", "1stWon", "bpSaved", "bpFaced"]

HEIGHT_EDGES = {"atp": [0, 180, 186, 191, 196, 300], "wta": [0, 165, 170, 175, 180, 300]}
AGE_EDGES = [0, 3, 6, 10, float("inf")]

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

def build_surface_matches(matches):
    """Matches played per season and surface; matches with unknown surface are left out."""
    played = matches[matches["outcome"].isin(["completed", "retired"])
                     & matches["surface"].isin(SURFACES)]
    out = played.groupby(["season", "surface"]).size().rename("matches").reset_index()
    out["surface"] = out["surface"].str.lower()
    return out


def build_surface_serve(matches):
    """Serve and match-length sums per season and surface, for completed matches.

    These are sums, not averages, so the dashboard can average any period exactly:
      aces per player per match = aces / (2 * matches)
      first-serve points won    = first_won / first_in
      match length              = minutes / minutes_matches
    """
    done = matches[(matches["outcome"] == "completed") & matches["surface"].isin(SURFACES)].copy()

    needed = ["w_svpt", "l_svpt", "w_ace", "l_ace", "w_1stIn", "l_1stIn", "w_1stWon", "l_1stWon"]
    serve = done[done[needed].notna().all(axis=1)].copy()
    serve["aces"] = serve["w_ace"] + serve["l_ace"]
    serve["first_in"] = serve["w_1stIn"] + serve["l_1stIn"]
    serve["first_won"] = serve["w_1stWon"] + serve["l_1stWon"]
    out = (serve.groupby(["season", "surface"])
           .agg(matches=("aces", "size"), aces=("aces", "sum"),
                first_in=("first_in", "sum"), first_won=("first_won", "sum"))
           .reset_index())

    # match length has its own coverage (WTA only since 2016), so it is counted separately
    length = (done[done["minutes"].notna()].groupby(["season", "surface"])
              .agg(minutes_matches=("minutes", "size"), minutes=("minutes", "sum"))
              .reset_index())

    out = out.merge(length, on=["season", "surface"], how="outer").fillna(0)
    out["surface"] = out["surface"].str.lower()
    numbers = ["matches", "aces", "first_in", "first_won", "minutes_matches", "minutes"]
    out[numbers] = out[numbers].astype(int)
    return out


def check_serve(serve):
    """Print the averages of the last ten seasons, to compare with what is known."""
    s = serve[serve["season"] >= 2016].groupby("surface")[
        ["matches", "aces", "first_in", "first_won", "minutes_matches", "minutes"]].sum()
    print(pd.DataFrame({
        "matches": s["matches"],
        "aces/player/match": (s["aces"] / (2 * s["matches"])).round(1),
        "1st serve won %": (s["first_won"] / s["first_in"] * 100).round(1),
        "minutes": (s["minutes"] / s["minutes_matches"]).round(0),
    }).to_string())

def build_player_surface(matches):
    """Wins and losses per player, season and surface, for players with enough matches.

    Walkovers and unknown surfaces are left out. A player is kept when they played at
    least MIN_CAREER_MATCHES matches in the tour (on any surface).
    """
    played = matches[matches["outcome"].isin(["completed", "retired"])]
    known = played[played["surface"].isin(SURFACES)]

    index = ["player_id", "season", "surface"]
    wins = known.groupby(["winner_id", "season", "surface"]).size().rename("w").rename_axis(index)
    losses = known.groupby(["loser_id", "season", "surface"]).size().rename("l").rename_axis(index)
    record = pd.concat([wins, losses], axis=1).fillna(0).astype(int)

    # one row per player and season, with a wins/losses pair per surface
    record = record.unstack("surface", fill_value=0)
    record.columns = [f"{kind}_{surface.lower()}" for kind, surface in record.columns]
    columns = [f"{kind}_{s.lower()}" for s in SURFACES for kind in ("w", "l")]
    record = record.reindex(columns=columns, fill_value=0).reset_index()

    career = pd.concat([played["winner_id"], played["loser_id"]]).value_counts()
    keep = career[career >= MIN_CAREER_MATCHES].index
    return record[record["player_id"].isin(keep)]


def check_player_surface(player_surface, tour):
    """Print the most active players with their win rate on each surface."""
    names = pd.read_csv(INTERIM / FILES[tour]["players"], usecols=["id", "name"],
                        dtype={"id": str}).set_index("id")["name"]
    total = player_surface.drop(columns="season").groupby("player_id").sum()
    total["matches"] = total.sum(axis=1)
    top = total.sort_values("matches", ascending=False).head(6)

    out = pd.DataFrame({"matches": top["matches"]})
    for s in SURFACES:
        s = s.lower()
        n = top[f"w_{s}"] + top[f"l_{s}"]
        out[s + " %"] = (top[f"w_{s}"] / n * 100).round(0)
    out.index = out.index.map(names)
    print(out.to_string())    

def listed_players(matches):
    """Ids of the players with at least MIN_CAREER_MATCHES played matches in the tour."""
    played = matches[matches["outcome"].isin(["completed", "retired"])]
    career = pd.concat([played["winner_id"], played["loser_id"]]).value_counts()
    return set(career[career >= MIN_CAREER_MATCHES].index)


def build_player_info(matches, tour, listed):
    """Name, hand, height, birth year and first/last season of each listed player."""
    players = pd.read_csv(INTERIM / FILES[tour]["players"], dtype=str).rename(columns={"id": "player_id"})
    played = matches[matches["outcome"].isin(["completed", "retired"])]

    # first and last season with a played match
    seen = pd.concat([played[["winner_id", "season"]].set_axis(["player_id", "season"], axis=1),
                      played[["loser_id", "season"]].set_axis(["player_id", "season"], axis=1)])
    career = seen.groupby("player_id")["season"].agg(first_season="min", last_season="max")

    # height: the most common value in the matches (a player can appear with two values)
    heights = pd.concat([played[["winner_id", "winner_ht"]].set_axis(["player_id", "height"], axis=1),
                         played[["loser_id", "loser_ht"]].set_axis(["player_id", "height"], axis=1)]).dropna()
    height = heights.groupby("player_id")["height"].agg(lambda s: s.mode().iat[0])

    # start from the listed ids so that nobody is lost if a player file lacks them
    info = pd.DataFrame({"player_id": sorted(listed)}).merge(players, on="player_id", how="left")
    info["birth_year"] = pd.to_datetime(info["birthdate"], errors="coerce").dt.year
    info["height_cm"] = info["player_id"].map(height)
    if "height" in info.columns:  # WTA also has the height in its player file
        info["height_cm"] = info["height_cm"].fillna(pd.to_numeric(info["height"], errors="coerce"))
    info["height_cm"] = info["height_cm"].round().astype("Int64")
    info["birth_year"] = info["birth_year"].astype("Int64")
    info = info.merge(career, left_on="player_id", right_index=True)
    return info[["player_id", "name", "hand", "height_cm", "birth_year", "first_season", "last_season"]]


def build_player_seasons(matches, listed):
    """Wins and losses per listed player and season, over all played matches."""
    played = matches[matches["outcome"].isin(["completed", "retired"])]
    index = ["player_id", "season"]
    wins = played.groupby(["winner_id", "season"]).size().rename("wins").rename_axis(index)
    losses = played.groupby(["loser_id", "season"]).size().rename("losses").rename_axis(index)
    table = pd.concat([wins, losses], axis=1).fillna(0).astype(int).reset_index()
    return table[table["player_id"].isin(listed)]


def check_player_pages(info, seasons, tour):
    """Career record of the most active players, and how many details are missing."""
    total = seasons.groupby("player_id")[["wins", "losses"]].sum()
    total["matches"] = total["wins"] + total["losses"]
    top = total.sort_values("matches", ascending=False).head(6)
    details = info.set_index("player_id")[["name", "hand", "height_cm", "birth_year", "first_season", "last_season"]]
    out = top.join(details).set_index("name")
    print(out.to_string())
    print("Listed:", len(info), "| without name:", int(info["name"].isna().sum()),
          "| hand:", int(info["hand"].isna().sum()), "| height:", int(info["height_cm"].isna().sum()),
          "| birth year:", int(info["birth_year"].isna().sum()), "(missing)")    

def build_player_slams(matches, listed):
    """Furthest round reached by each listed player at each Grand Slam, by season.

    Codes: 0 = first round ... 6 = lost the final, 7 = won the title (see ROUND_CODES).
    """
    # the third-place match is left out: both players already reached the semi-final
    played = matches[(matches["level_group"] == "Grand Slam")
                     & matches["outcome"].isin(["completed", "retired"])
                     & (matches["round_clean"] != "3rd/4th")].copy()
    played["slam"] = played["tourney_name"].replace(SLAM_ALIASES).map({name: i for i, name in enumerate(SLAMS)})
    played["code"] = played["round_clean"].map(ROUND_CODES)
    assert played["slam"].notna().all(), set(played.loc[played["slam"].isna(), "tourney_name"])
    unknown = set(played.loc[played["code"].isna(), "round_clean"])
    assert not unknown, f"Unmapped Grand Slam rounds: {unknown}"
    played["slam"] = played["slam"].astype(int)
    played["code"] = played["code"].astype(int)

    # a player reaches the round of every match they play; the champion goes one step further
    reached = pd.concat([
        played[["winner_id", "season", "slam", "code"]].rename(columns={"winner_id": "player_id"}),
        played[["loser_id", "season", "slam", "code"]].rename(columns={"loser_id": "player_id"}),
    ])
    finals = played[played["round_clean"] == "F"]
    champions = finals[["winner_id", "season", "slam"]].rename(columns={"winner_id": "player_id"})
    champions["code"] = 7
    reached = pd.concat([reached, champions])

    best = reached.groupby(["player_id", "season", "slam"])["code"].max().reset_index()
    table = best.pivot(index=["player_id", "season"], columns="slam", values="code")
    table = table.reindex(columns=range(len(SLAMS)))
    table.columns = [name.lower().replace(" ", "_") for name in SLAMS]
    table = table.astype("Int64").reset_index()
    return table[table["player_id"].isin(listed)]


def check_player_slams(slams, info):
    """Grand Slam titles of the most successful players, to compare with the records."""
    columns = [c for c in slams.columns if c not in ("player_id", "season")]
    titles = (slams[columns].fillna(-1) == 7).groupby(slams["player_id"]).sum().sum(axis=1)
    names = info.set_index("player_id")["name"]
    print(titles.sort_values(ascending=False).head(6).rename(index=names).to_string())
    print("Titles among listed players:", int(titles.sum()))

def build_player_rank(tour, listed):
    """Best ranking reached by each listed player in each month, top MAX_RANK only."""
    ranks = pd.read_csv(INTERIM / FILES[tour]["rankings"], usecols=["date", "id", "rank"],
                        dtype={"id": str}, parse_dates=["date"])
    ranks = ranks[ranks["id"].isin(listed) & (ranks["rank"] <= MAX_RANK)]
    ranks = ranks.assign(year=ranks["date"].dt.year, month=ranks["date"].dt.month)
    table = (ranks.groupby(["id", "year", "month"])["rank"].min().astype(int)
             .reset_index().rename(columns={"id": "player_id"}))
    return table


def check_player_rank(rank, info, weeks):
    """Best ranking of the players with most weeks at No. 1 (all should say 1)."""
    best = rank.sort_values(["rank", "year", "month"]).groupby("player_id").first()
    names = info.set_index("player_id")["name"]
    top = weeks.groupby("player_id")["weeks"].sum().sort_values(ascending=False).head(6).index
    print("Rows:", len(rank), "| players with a ranking:", rank["player_id"].nunique(), "of", len(info),
          "| years:", int(rank["year"].min()), "to", int(rank["year"].max()))
    print("Players whose best month was No. 1:", int((best["rank"] == 1).sum()))
    print(best.loc[top].rename(index=names).to_string())

def build_h2h(matches, listed):
    """Every played match between two listed players who met at least MIN_MEETINGS times.

    The two ids of a pair are always stored in the same order (player_a < player_b),
    so that A against B and B against A end up in the same group.
    """
    played = matches[matches["outcome"].isin(["completed", "retired"])]
    both = played[played["winner_id"].isin(listed) & played["loser_id"].isin(listed)].copy()

    a_won = both["winner_id"] < both["loser_id"]
    both["player_a"] = both["winner_id"].where(a_won, both["loser_id"])
    both["player_b"] = both["loser_id"].where(a_won, both["winner_id"])
    both["a_won"] = a_won.astype(int)

    meetings = both.groupby(["player_a", "player_b"])["season"].transform("size")
    both = both[meetings >= MIN_MEETINGS].copy()

    both["tourney"] = both["tourney_name"].replace(SLAM_ALIASES)
    both["slam"] = (both["level_group"] == "Grand Slam").astype(int)
    both["surface"] = both["surface"].where(both["surface"].isin(SURFACES), "").str.lower()
    both = both.sort_values(["player_a", "player_b", "date"])
    return both[["player_a", "player_b", "season", "tourney", "round_clean", "surface",
                 "a_won", "score", "slam"]]


def check_h2h(h2h, info):
    """The most played rivalries, to compare with the known records."""
    names = info.set_index("player_id")["name"]
    pairs = h2h.groupby(["player_a", "player_b"]).agg(meetings=("a_won", "size"), a_wins=("a_won", "sum"))
    top = pairs.sort_values("meetings", ascending=False).head(6).reset_index()
    top["rivalry"] = top["player_a"].map(names) + " vs " + top["player_b"].map(names)
    top["record"] = top["a_wins"].astype(str) + "-" + (top["meetings"] - top["a_wins"]).astype(str)
    print("Rows:", len(h2h), "| pairs:", len(pairs), "| average score length:", round(h2h["score"].str.len().mean(), 1))
    print(top[["rivalry", "meetings", "record"]].to_string(index=False))

def build_no1_stints(no1):
    """Periods in which each player held No. 1, with the official weeks of each period.

    A period is a run of rankings in force without a break. A ranking in force for over
    FREEZE_DAYS (the 2020 freeze) is a period of its own that counts as one week, so
    that the weeks add up to the official count of build_weeks_no1.
    """
    dates = no1["date"].drop_duplicates().sort_values().reset_index(drop=True)
    last_end = min(dates.iloc[-1] + pd.Timedelta(days=7), pd.Timestamp(LAST_SEASON + 1, 1, 1))
    end_of = dict(zip(dates, dates.shift(-1).fillna(last_end)))

    rows = no1.assign(end=no1["date"].map(end_of)).sort_values(["id", "date"]).reset_index(drop=True)
    length = (rows["end"] - rows["date"]).dt.days
    rows["frozen"] = length > FREEZE_DAYS
    rows["days"] = length.where(~rows["frozen"], 7)

    # a new period starts when the previous ranking of the player does not run straight
    # into this one, and around a freeze
    previous = rows.groupby("id")[["end", "frozen"]].shift()
    new = ((rows["date"] != previous["end"]) | rows["frozen"]
           | previous["frozen"].fillna(False).astype(bool))
    rows["period"] = new.cumsum()

    periods = rows.groupby("period").agg(player_id=("id", "first"), start=("date", "min"),
                                         end=("end", "max"), days=("days", "sum"))
    periods["weeks"] = (periods["days"] / 7).round(2)
    periods["start"] = periods["start"].dt.strftime("%Y-%m-%d")
    periods["end"] = periods["end"].dt.strftime("%Y-%m-%d")
    return periods[["player_id", "start", "end", "weeks"]].sort_values(["start", "player_id"]).reset_index(drop=True)


def build_year_end(no1, weeks):
    """The No. 1 at the last ranking of each year (if shared, the one with most weeks that year)."""
    last = no1.assign(season=no1["date"].dt.year).rename(columns={"id": "player_id"})
    last = last[last["date"] == last.groupby("season")["date"].transform("max")]
    last = last.merge(weeks[["player_id", "season", "calendar_weeks"]], on=["player_id", "season"], how="left")
    last = last.sort_values(["season", "calendar_weeks"], ascending=[True, False]).groupby("season").head(1)
    return last[["season", "player_id"]].sort_values("season").reset_index(drop=True)


def check_no1_tables(stints, year_end, weeks, info):
    """The longest periods, the match with weeks_no1 and the latest year-end No. 1."""
    names = info.set_index("player_id")["name"]
    top = stints.sort_values("weeks", ascending=False).head(5)
    print(top.assign(player=top["player_id"].map(names))[["player", "start", "end", "weeks"]].to_string(index=False))
    total = stints.groupby("player_id")["weeks"].sum()
    official = weeks.groupby("player_id")["weeks"].sum().reindex(total.index)
    print("Largest difference with weeks_no1:", round(float((total - official).abs().max()), 2))
    print(year_end.assign(player=year_end["player_id"].map(names)).tail(6)[["season", "player"]].to_string(index=False))

def upset_matches(matches):
    """Played matches where both players have a ranking and the two rankings differ."""
    played = matches[matches["outcome"].isin(["completed", "retired"])]
    ranked = played[played["winner_rank"].notna() & played["loser_rank"].notna()
                    & (played["winner_rank"] != played["loser_rank"])].copy()
    ranked["gap"] = (ranked["winner_rank"] - ranked["loser_rank"]).abs().astype(int)
    ranked["upset"] = ranked["winner_rank"] > ranked["loser_rank"]
    ranked["gap_bin"] = pd.cut(ranked["gap"], bins=GAP_EDGES + [float("inf")],
                               right=False, labels=False).astype(int)
    ranked["round_code"] = ranked["round_clean"].map(UPSET_ROUNDS)
    ranked["surf"] = ranked["surface"].map({s: i for i, s in enumerate(SURFACES)})
    return ranked


def by_scope(ranked):
    """Every match once as 'all' and Grand Slam matches again as 'gs'."""
    everything = ranked.assign(scope="all")
    slams = ranked[ranked["level_group"] == "Grand Slam"].assign(scope="gs")
    return pd.concat([everything, slams], ignore_index=True)

def build_upsets_gap(ranked):
    """Matches and upsets per season, scope and rank-gap group, with the summed gap of the upsets."""
    sc = by_scope(ranked)
    sc["upset_gap"] = sc["gap"].where(sc["upset"], 0)
    return (sc.groupby(["season", "scope", "gap_bin"])
              .agg(n=("upset", "size"), upsets=("upset", "sum"), gap_sum=("upset_gap", "sum"))
              .reset_index())


def build_upsets_round(ranked):
    """Matches and upsets per season, scope and round (0 = first round of a 128 draw ... 6 = final)."""
    sc = by_scope(ranked.dropna(subset=["round_code"]))
    sc["round_code"] = sc["round_code"].astype(int)
    return (sc.groupby(["season", "scope", "round_code"])
              .agg(n=("upset", "size"), upsets=("upset", "sum"))
              .reset_index())


def build_upsets_surface(ranked):
    """Matches and upsets per season, scope and surface (0 hard, 1 clay, 2 grass, 3 carpet)."""
    sc = by_scope(ranked.dropna(subset=["surf"]))
    sc["surf"] = sc["surf"].astype(int)
    return (sc.groupby(["season", "scope", "surf"])
              .agg(n=("upset", "size"), upsets=("upset", "sum"))
              .reset_index())


def check_upsets(ranked, gap, rnd, surf):
    """The three tables must agree with each other and with the matches they come from."""
    assert gap[gap["scope"] == "all"]["n"].sum() == len(ranked), "gap table does not add up"
    assert gap[gap["scope"] == "gs"]["n"].sum() == int((ranked["level_group"] == "Grand Slam").sum()), \
        "Grand Slam rows do not add up"
    for name, table in (("gap", gap), ("round", rnd), ("surface", surf)):
        assert (table["upsets"] <= table["n"]).all(), f"{name}: more upsets than matches"
    per = lambda t: t.groupby(["season", "scope"])["n"].sum()
    total = per(gap)
    for name, table in (("round", rnd), ("surface", surf)):
        assert (per(table) <= total.reindex(per(table).index)).all(), f"{name}: more matches than the gap table"
        print(f"{name} covers {per(table).sum() / total.sum() * 100:.1f}% of the matches")

def build_upsets_top(ranked, per_group=10, max_loser_rank=20):
    """The biggest upsets of every season: the beaten player was in the top 20 (max_loser_rank)."""
    shocks = ranked[ranked["upset"] & (ranked["loser_rank"] <= max_loser_rank)]
    sc = by_scope(shocks)
    sc = sc.sort_values(["season", "scope", "gap", "date"], ascending=[True, True, False, True])
    top = sc.groupby(["season", "scope"]).head(per_group).copy()
    top["surf"] = top["surf"].fillna(4).astype(int)
    top["winner_rank"] = top["winner_rank"].astype(int)
    top["loser_rank"] = top["loser_rank"].astype(int)
    top = top.rename(columns={"tourney_name": "tourney", "round_clean": "round",
                              "winner_name": "winner", "loser_name": "loser"})
    return top[["season", "scope", "tourney", "surf", "round", "winner", "winner_rank",
                "loser", "loser_rank", "gap", "score", "winner_ioc", "loser_ioc"]].reset_index(drop=True)


def check_upsets_top(top):
    """Complete names, at most ten per season and scope, and gaps that match the two rankings."""
    for column in ("tourney", "winner", "loser", "score"):
        assert top[column].notna().all(), f"upsets_top: missing {column}"
    assert top.groupby(["season", "scope"]).size().max() <= 10, "more than ten per group"
    assert (top["loser_rank"] + top["gap"] == top["winner_rank"]).all(), "gap does not match the rankings"
    biggest = top[top["scope"] == "all"].nlargest(5, "gap")
    print(biggest[["season", "tourney", "round", "winner", "winner_rank", "loser", "loser_rank"]].to_string(index=False))

def build_win_serve(matches):
    """Serve totals of the winners and of the losers per season and scope (all or Grand Slams)."""
    played = matches[matches["outcome"].isin(["completed", "retired"])]
    columns = [f"{side}_{stat}" for side in ("w", "l") for stat in SERVE_SUMS]
    known = played[columns].notna().all(axis=1) & (played["w_svpt"] > 0) & (played["l_svpt"] > 0)
    sc = by_scope(played[known])
    out = sc.groupby(["season", "scope"])[columns].sum().astype(int)
    out.insert(0, "n", sc.groupby(["season", "scope"]).size())
    return out.reset_index()


def check_win_serve(table):
    """Winners must serve better than losers; show the totals so they can be compared by eye."""
    everything = table[table["scope"] == "all"]
    n = int(everything["n"].sum())
    print("Serve matches:", n, "| from season", int(everything["season"].min()))
    measures = [("aces per match", "ace", None), ("first serves in %", "1stIn", "svpt"),
                ("first-serve points won %", "1stWon", "1stIn"), ("break points saved %", "bpSaved", "bpFaced")]
    for label, top, bottom in measures:
        value = {}
        for side in ("w", "l"):
            num = everything[f"{side}_{top}"].sum()
            value[side] = num / n if bottom is None else num / everything[f"{side}_{bottom}"].sum() * 100
        print(f"  {label}: winners {value['w']:.1f} | losers {value['l']:.1f}")
        if top in ("1stWon", "bpSaved"):
            assert value["w"] > value["l"], f"{label}: winners should be above losers"

def physical_flags(played, height_edges):
    """One row per match: appearances (n) and wins (w) of every height, age-gap and handedness group."""
    both_height = played["winner_ht"].notna() & played["loser_ht"].notna()
    winner_group = pd.cut(played["winner_ht"].where(both_height), bins=height_edges, right=False, labels=False)
    loser_group = pd.cut(played["loser_ht"].where(both_height), bins=height_edges, right=False, labels=False)
    flags = pd.DataFrame(index=played.index)
    for i in range(5):
        flags[f"h{i}_n"] = (winner_group == i).astype(int) + (loser_group == i).astype(int)
        flags[f"h{i}_w"] = (winner_group == i).astype(int)

    gap = (played["winner_age"] - played["loser_age"]).abs()
    age_group = pd.cut(gap.where(gap > 0), bins=AGE_EDGES, right=False, labels=False)
    younger_won = played["winner_age"] < played["loser_age"]
    for i in range(4):
        flags[f"a{i}_n"] = (age_group == i).astype(int)
        flags[f"a{i}_w"] = ((age_group == i) & younger_won).astype(int)

    lefty_won = (played["winner_hand"] == "L") & (played["loser_hand"] == "R")
    righty_won = (played["winner_hand"] == "R") & (played["loser_hand"] == "L")
    flags["hd_n"] = (lefty_won | righty_won).astype(int)
    flags["hd_w"] = lefty_won.astype(int)
    both_hands_known = played["winner_hand"].isin(["L", "R"]) & played["loser_hand"].isin(["L", "R"])
    flags["hk_n"] = both_hands_known.astype(int)
    return flags


def build_win_physical(matches, tour):
    """Height, age-gap and handedness groups per season, scope and surface (0 hard ... 3 carpet)."""
    played = matches[matches["outcome"].isin(["completed", "retired"]) & matches["surface"].isin(SURFACES)]
    flags = physical_flags(played, HEIGHT_EDGES[tour])
    data = pd.concat([played[["season", "level_group", "surface"]], flags], axis=1)
    data["surf"] = data["surface"].map({s: i for i, s in enumerate(SURFACES)})
    sc = by_scope(data)
    return sc.groupby(["season", "scope", "surf"])[list(flags.columns)].sum().reset_index()


def check_win_physical(table):
    """Wins never exceed appearances; show the win rate of every group over the whole history."""
    groups = [(f"h{i}", f"height {i}") for i in range(5)] + [(f"a{i}", f"age gap {i}") for i in range(4)] + [("hd", "left-handers")]
    everything = table[table["scope"] == "all"]
    for key, label in groups:
        assert (table[f"{key}_w"] <= table[f"{key}_n"]).all(), f"{label}: more wins than appearances"
        n = int(everything[f"{key}_n"].sum())
        print(f"  {label}: {n} appearances | win rate {everything[f'{key}_w'].sum() / n * 100:.1f}%")

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

        # Surfaces: matches per surface, serve and match length per surface
        surface_matches = build_surface_matches(matches)
        surface_matches.to_csv(PROCESSED / f"{tour}_surface_matches.csv", index=False)
        surface_serve = build_surface_serve(matches)
        surface_serve.to_csv(PROCESSED / f"{tour}_surface_serve.csv", index=False)
        print("Surface rows:", len(surface_matches), "| serve rows:", len(surface_serve))
        check_serve(surface_serve)

        # Surfaces: record of each player on each surface, by season
        player_surface = build_player_surface(matches)
        player_surface.to_csv(PROCESSED / f"{tour}_player_surface.csv", index=False)
        print("Player-surface rows:", len(player_surface),
              "| players:", player_surface["player_id"].nunique())
        check_player_surface(player_surface, tour)

        # Player pages: who is listed, their details and their record per season
        listed = listed_players(matches)
        player_info = build_player_info(matches, tour, listed)
        player_info.to_csv(PROCESSED / f"{tour}_player_info.csv", index=False)
        player_seasons = build_player_seasons(matches, listed)
        player_seasons.to_csv(PROCESSED / f"{tour}_player_seasons.csv", index=False)
        print("Player info:", len(player_info), "| season rows:", len(player_seasons))
        check_player_pages(player_info, player_seasons, tour)

        # Grand Slam record of each listed player
        player_slams = build_player_slams(matches, listed)
        player_slams.to_csv(PROCESSED / f"{tour}_player_slams.csv", index=False)
        print("Player Grand Slam rows:", len(player_slams))
        check_player_slams(player_slams, player_info)

        # Ranking over time of each listed player
        player_rank = build_player_rank(tour, listed)
        player_rank.to_csv(PROCESSED / f"{tour}_player_rank.csv", index=False)
        check_player_rank(player_rank, player_info, weeks)

        # Head-to-head: meetings between listed players who met MIN_MEETINGS times or more
        h2h = build_h2h(matches, listed)
        h2h.to_csv(PROCESSED / f"{tour}_h2h.csv", index=False)
        check_h2h(h2h, player_info)

        # Dynasties: periods at No. 1 and the year-end No. 1
        stints = build_no1_stints(no1)
        stints.to_csv(PROCESSED / f"{tour}_no1_stints.csv", index=False)
        year_end = build_year_end(no1, weeks)
        year_end.to_csv(PROCESSED / f"{tour}_no1_year_end.csv", index=False)
        print("No. 1 periods:", len(stints), "| year-end rows:", len(year_end))
        check_no1_tables(stints, year_end, weeks, player_info)

        # Upsets: rank gap, round and surface
        ranked = upset_matches(matches)
        upsets_gap = build_upsets_gap(ranked)
        upsets_gap.to_csv(PROCESSED / f"{tour}_upsets_gap.csv", index=False)
        upsets_round = build_upsets_round(ranked)
        upsets_round.to_csv(PROCESSED / f"{tour}_upsets_round.csv", index=False)
        upsets_surface = build_upsets_surface(ranked)
        upsets_surface.to_csv(PROCESSED / f"{tour}_upsets_surface.csv", index=False)
        print("Upset rows:", len(upsets_gap), "gap |", len(upsets_round), "round |", len(upsets_surface), "surface")
        check_upsets(ranked, upsets_gap, upsets_round, upsets_surface)

        upsets_top = build_upsets_top(ranked)
        upsets_top.to_csv(PROCESSED / f"{tour}_upsets_top.csv", index=False)
        print("Biggest upsets rows:", len(upsets_top))
        check_upsets_top(upsets_top)

        # Win factors: serve of the winners and the losers
        win_serve = build_win_serve(matches)
        win_serve.to_csv(PROCESSED / f"{tour}_win_serve.csv", index=False)
        print("Win-serve rows:", len(win_serve))
        check_win_serve(win_serve)

        win_physical = build_win_physical(matches, tour)
        win_physical.to_csv(PROCESSED / f"{tour}_win_physical.csv", index=False)
        print("Win-physical rows:", len(win_physical))
        check_win_physical(win_physical)

        print()

if __name__ == "__main__":
    main()