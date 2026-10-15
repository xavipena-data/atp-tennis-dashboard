"""Clean the raw players file and write data/interim/players_clean.csv.

Keeps only the players who appear in the cleaned matches. Name, nationality and
hand come from the match rows (better coverage); birth data comes from the
players file.
"""
import pandas as pd

from clean_data import INTERIM, ROOT

RAW_PLAYERS = ROOT / "data" / "raw" / "players" / "ATP_Database.csv"
CLEAN_MATCHES = INTERIM / "matches_clean.csv"


def load_players():
    """Read the players file (cp1252, ids as text so '000001' keeps its zeros)."""
    players = pd.read_csv(RAW_PLAYERS, encoding="cp1252", dtype={"id": str})
    print(f"Loaded {len(players):,} players")
    return players


def realign_shifted_birthdates(players):
    """Eight birthdates were written in `weight`: move them back to `birthdate`."""
    shifted = pd.to_numeric(players["weight"], errors="coerce") > 1000
    assert players.loc[shifted, "birthdate"].isna().all(), "would overwrite a birthdate"
    players.loc[shifted, "birthdate"] = players.loc[shifted, "weight"]
    players.loc[shifted, "weight"] = float("nan")
    print(f"Birthdates realigned from `weight`: {shifted.sum()}")
    return players


def parse_birthdate(players):
    """Turn 19930316.0 into the date 1993-03-16 (missing stays missing)."""
    raw = players["birthdate"]
    text = raw.dropna().astype(int).astype(str)
    parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce").reindex(players.index)
    invalid = parsed.isna() & raw.notna()
    print(f"Birthdates parsed: {parsed.notna().sum():,}; unparseable: {invalid.sum()}")
    players["birthdate"] = parsed
    return players

def players_from_matches():
    """One row per player seen in the cleaned matches: name, nationality, hand."""
    matches = pd.read_csv(CLEAN_MATCHES, low_memory=False,
                          dtype={"winner_id": str, "loser_id": str})
    sides = [
        matches[[f"{side}_id", f"{side}_name", f"{side}_ioc", f"{side}_hand"]]
        .set_axis(["id", "name", "ioc", "hand"], axis=1)
        for side in ("winner", "loser")
    ]
    both = pd.concat(sides, ignore_index=True)

    def most_common(values):
        values = values.dropna()
        return values.mode().iloc[0] if len(values) else pd.NA

    return both.groupby("id").agg(
        name=("name", most_common),
        ioc=("ioc", most_common),
        hand=("hand", most_common),
    ).reset_index()


def build_players_table(players):
    """Match-based columns + the birth data that only the players file has."""
    table = players_from_matches()
    unmatched = ~table["id"].isin(players["id"])
    print(f"Players in the matches that are missing from the players file: {unmatched.sum()}")

    extras = players[["id", "birthdate", "birthplace", "turnedpro", "backhand"]]
    table = table.merge(extras, on="id", how="left", validate="one_to_one")
    table["turnedpro"] = table["turnedpro"].astype("Int64")   # 2011.0 -> 2011

    print(f"Players table: {len(table):,} rows")
    print("\n% of players with the column filled:")
    print((table.notna().mean() * 100).round(1).to_string())
    print("\nhand:")
    print(table["hand"].value_counts(dropna=False).to_string())
    return table


def write_output(table):
    INTERIM.mkdir(parents=True, exist_ok=True)
    path = INTERIM / "players_clean.csv"
    table.to_csv(path, index=False)
    print(f"\nWrote {len(table):,} rows x {table.shape[1]} columns -> {path.relative_to(ROOT)}")


def main():
    players = load_players()
    players = realign_shifted_birthdates(players)
    players = parse_birthdate(players)
    table = build_players_table(players)
    write_output(table)


if __name__ == "__main__":
    main()

