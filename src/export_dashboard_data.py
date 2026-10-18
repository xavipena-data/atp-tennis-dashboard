"""Export the processed tables as one compact JSON that the dashboard embeds."""

import json
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT = PROCESSED / "dashboard_data.json"

TOURS = ["atp", "wta"]
SLAMS = ["Australian Open", "Roland Garros", "Wimbledon", "US Open"]

SURFACES = ["hard", "clay", "grass", "carpet"]
PLAYER_SURFACE_COLUMNS = [f"{kind}_{s}" for s in SURFACES for kind in ("w", "l")]
COVERAGE = 0.5  # a statistic "is available" from the first season where half the matches have it

def read(tour, table):
    """Read one processed table; player ids are always text."""
    return pd.read_csv(PROCESSED / f"{tour}_{table}.csv",
                       dtype={"player_id": str, "player_a": str, "player_b": str})


def rows(df, columns):
    """A table as a list of plain Python lists, which is how it goes into the JSON."""
    return df[columns].astype(object).values.tolist()


def overview_data(tour):
    """Everything the Overview tab needs for one tour."""
    seasons = read(tour, "seasons")
    players_range = read(tour, "players_range")
    players = read(tour, "players")
    titles = read(tour, "titles")
    weeks = read(tour, "weeks_no1")
    champs = read(tour, "slam_champions")
    champs["slam"] = champs["slam"].map({name: i for i, name in enumerate(SLAMS)})

    # Keep only the players that appear in a table of this tab, to keep the file small
    used = (set(titles["player_id"]) | set(weeks["player_id"]) | set(champs["player_id"])
            | set(read(tour, "player_surface")["player_id"]))
    players = players[players["player_id"].isin(used)]

    return {
        "seasons": rows(seasons, ["season", "matches", "tournaments"]),
        "playersRange": rows(players_range, ["season_from", "season_to", "players"]),
        "players": {p.player_id: [p.name, p.ioc if pd.notna(p.ioc) else None]
                    for p in players.itertuples()},
        "titles": rows(titles, ["player_id", "season", "titles", "slams"]),
        "weeks": rows(weeks, ["player_id", "season", "weeks", "calendar_weeks"]),
        "slams": rows(champs, ["season", "slam", "player_id", "order"]),
    }

def first_covered(share):
    """First season in which at least COVERAGE of the played matches have the data."""
    return int(share[share >= COVERAGE].index.min())


def surfaces_data(tour):
    """Everything the Surfaces tab needs for one tour."""
    surface_matches = read(tour, "surface_matches")
    serve = read(tour, "surface_serve")
    player_surface = read(tour, "player_surface")

    # Matches per season: one row per season, one column per surface
    sm = (surface_matches.pivot(index="season", columns="surface", values="matches")
          .reindex(columns=SURFACES).fillna(0).astype(int).reset_index())

    # Serve and length: the surface goes in as its position in SURFACES (0 to 3)
    serve["surface"] = serve["surface"].map({s: i for i, s in enumerate(SURFACES)})

    # First season where at least half of the played matches carry serve / minutes data
    played = surface_matches.groupby("season")["matches"].sum()
    by_season = serve.groupby("season")[["matches", "minutes_matches"]].sum()
    meta = {"serveFrom": first_covered(by_season["matches"] / played),
            "minutesFrom": first_covered(by_season["minutes_matches"] / played)}

    # One entry per player: id and their seasons, each with wins/losses on the 4 surfaces
    py = [[pid, rows(group, ["season"] + PLAYER_SURFACE_COLUMNS)]
          for pid, group in player_surface.groupby("player_id")]

    return {
        "sm": rows(sm, ["season"] + SURFACES),
        "serve": rows(serve, ["season", "surface", "matches", "aces", "first_in",
                              "first_won", "minutes_matches", "minutes"]),
        "meta": meta,
        "py": py,
    }

def by_player(df, columns):
    """{player_id: [[...], [...]]}: the rows of each player, empty cells as null."""
    clean = df.astype(object).where(df.notna(), None)
    return {pid: rows(group, columns) for pid, group in clean.groupby("player_id")}


def rank_series(rank):
    """{player_id: [first year, first month (0 to 11), [rank of each month, 0 when not in the top 500]]}."""
    out = {}
    for pid, group in rank.groupby("player_id"):
        month = (group["year"] * 12 + group["month"] - 1).to_numpy()
        series = np.zeros(month.max() - month.min() + 1, dtype=int)
        series[month - month.min()] = group["rank"].to_numpy()
        year, first = divmod(int(month.min()), 12)
        out[pid] = [year, first, series.tolist()]
    return out


def h2h_data(h2h):
    """Meetings grouped by pair; tournament and round names go in lists and rows point to them."""
    tourneys = sorted(h2h["tourney"].unique())
    rounds = sorted(h2h["round_clean"].unique())
    surface_code = {s: i for i, s in enumerate(SURFACES)}  # 4 = surface not recorded
    h2h = h2h.assign(
        tourney=h2h["tourney"].map({name: i for i, name in enumerate(tourneys)}),
        round_clean=h2h["round_clean"].map({name: i for i, name in enumerate(rounds)}),
        surface=h2h["surface"].map(surface_code).fillna(4).astype(int),
        score=h2h["score"].fillna(""),
    )
    columns = ["season", "tourney", "round_clean", "surface", "a_won", "score"]
    pairs = [[a, b, rows(group, columns)] for (a, b), group in h2h.groupby(["player_a", "player_b"], sort=False)]
    return {"tourneys": tourneys, "rounds": rounds, "pairs": pairs}


def player_data(tour):
    """Everything the Player profile tab needs for one tour (surface records are in surf)."""
    info = read(tour, "player_info")
    details = {p.player_id: [p.hand if pd.notna(p.hand) else None,
                             int(p.height_cm) if pd.notna(p.height_cm) else None,
                             int(p.birth_year) if pd.notna(p.birth_year) else None,
                             int(p.first_season), int(p.last_season)]
               for p in info.itertuples()}
    slam_columns = [name.lower().replace(" ", "_") for name in SLAMS]
    return {
        "info": details,
        "seasons": by_player(read(tour, "player_seasons"), ["season", "wins", "losses"]),
        "slams": by_player(read(tour, "player_slams"), ["season"] + slam_columns),
        "rank": rank_series(read(tour, "player_rank")),
        "h2h": h2h_data(read(tour, "h2h")),
    }

def dynasties_data(tour):
    """The periods at No. 1 and the year-end No. 1 (weeks and birth years are already in the file)."""
    stints = read(tour, "no1_stints")
    year_end = read(tour, "no1_year_end")
    return {
        "stints": rows(stints, ["player_id", "start", "end", "weeks"]),
        "yearEnd": rows(year_end, ["season", "player_id"]),
    }

SCOPES = {"all": 0, "gs": 1}
HEIGHT_N = [f"h{i}_n" for i in range(5)]
AGE_N = [f"a{i}_n" for i in range(4)]
PHYSICAL_COLUMNS = (["season", "scope", "surf"]
                    + [f"h{i}_{k}" for i in range(5) for k in ("n", "w")]
                    + [f"a{i}_{k}" for i in range(4) for k in ("n", "w")]
                    + ["hd_n", "hd_w"])
SERVE_COLUMNS = ["season", "scope", "n"] + [f"{side}_{stat}" for stat in
                 ("ace", "svpt", "1stIn", "1stWon", "bpSaved", "bpFaced") for side in ("w", "l")]


def share(table, columns, played, divisor=1):
    """Share of each season's played matches that the table counts (scope 'all' rows only)."""
    everything = table[table["scope"] == "all"]
    return everything.groupby("season")[columns].sum().sum(axis=1) / divisor / played


def upsets_data(tour):
    """Everything the Upsets & win factors tab needs for one tour."""
    played = read(tour, "seasons").set_index("season")["matches"]
    gap = read(tour, "upsets_gap")
    serve = read(tour, "win_serve")
    physical = read(tour, "win_physical")
    top = read(tour, "upsets_top")
    top["score"] = top["score"].fillna("")
    top[["winner_ioc", "loser_ioc"]] = top[["winner_ioc", "loser_ioc"]].fillna("")

    # First season in which at least half of the played matches carry each statistic
        # First season of a stable run (6 seasons) in which at least half of the played matches carry each statistic
    meta = {"rankFrom": stable_from(share(gap, ["n"], played), played),
            "serveFrom": stable_from(share(serve, ["n"], played), played),
            "heightFrom": stable_from(share(physical, HEIGHT_N, played, divisor=2), played),
            "ageFrom": stable_from(share(physical, AGE_N, played), played),
            "handFrom": stable_from(share(physical, ["hk_n"], played), played)}

    code = lambda df: df.assign(scope=df["scope"].map(SCOPES))
    return {
        "meta": meta,
        "gap": rows(code(gap), ["season", "scope", "gap_bin", "n", "upsets", "gap_sum"]),
        "rd": rows(code(read(tour, "upsets_round")), ["season", "scope", "round_code", "n", "upsets"]),
        "sf": rows(code(read(tour, "upsets_surface")), ["season", "scope", "surf", "n", "upsets"]),
        "top": rows(code(top), ["season", "scope", "tourney", "surf", "round", "winner",
                                "winner_rank", "loser", "loser_rank", "gap", "score", "winner_ioc", "loser_ioc"]),
        "sv": rows(code(serve), SERVE_COLUMNS),
        "ph": rows(code(physical), PHYSICAL_COLUMNS),
    }

def stable_from(share, played, run=6):
    """First season from which this and the next run - 1 seasons all have at least COVERAGE of the data."""
    full = share.reindex(played.index).fillna(0) >= COVERAGE
    for season in full.index:
        if full.loc[season:season + run - 1].all():
            return int(season)
    raise ValueError("no stable run of seasons with enough data")

def main():
    data = {tour: {**overview_data(tour), "surf": surfaces_data(tour), "player": player_data(tour),
                   "dyn": dynasties_data(tour), "ups": upsets_data(tour)}
            for tour in TOURS}
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"),
                      default=lambda value: value.item())
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"Wrote {OUTPUT.name}: {len(text) / 1024:,.0f} KB")
    for tour in TOURS:
        print(tour.upper(), {key: len(value) for key, value in data[tour].items()})
        print("  surf:", {key: len(value) for key, value in data[tour]["surf"].items()})
        print("  meta:", data[tour]["surf"]["meta"])
        print("  player:", {key: len(value) for key, value in data[tour]["player"].items()})
        print("  h2h pairs:", len(data[tour]["player"]["h2h"]["pairs"]))
        print("  dyn:", {key: len(value) for key, value in data[tour]["dyn"].items()})
        print("  ups:", {key: len(value) for key, value in data[tour]["ups"].items()})
        print("  ups meta:", data[tour]["ups"]["meta"])

if __name__ == "__main__":
    main()