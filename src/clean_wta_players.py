"""Clean the raw WTA players file and write data/interim/wta_players_clean.csv.

Keeps only the players who appear in the cleaned matches. Name, nationality,
hand and height come from the match rows (better coverage); the birth date comes
from the players file, which also fills the gaps in hand and height.
"""
import pandas as pd

from clean_wta_data import INTERIM, RAW_MATCHES, ROOT, read_csv_robust

# RAW_MATCHES is the data/raw/wta folder, which also holds the players file
RAW_PLAYERS = RAW_MATCHES / "wta_players.csv"
CLEAN_MATCHES = INTERIM / "wta_matches_clean.csv"


def load_players():
    """Read the players file (ids as text, like the ATP table)."""
    players = read_csv_robust(RAW_PLAYERS, dtype={"player_id": str}, low_memory=False)
    print(f"Loaded {len(players):,} players")
    return players


def parse_birthdate(players):
    """Turn 19800930.0 into the date 1980-09-30 (missing or impossible stays missing)."""
    raw = players["dob"]
    text = raw.dropna().astype(int).astype(str)
    parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce").reindex(players.index)
    invalid = parsed.isna() & raw.notna()
    print(f"Birthdates parsed: {parsed.notna().sum():,}; unparseable: {invalid.sum()}")
    players["birthdate"] = parsed
    return players

def players_from_matches():
    """One row per player seen in the cleaned matches: name, nationality, hand, height."""
    matches = pd.read_csv(CLEAN_MATCHES, low_memory=False,
                          dtype={"winner_id": str, "loser_id": str})
    sides = []
    for side in ("winner", "loser"):
        part = matches[[f"{side}_id", f"{side}_name", f"{side}_ioc",
                        f"{side}_hand", f"{side}_ht"]]
        sides.append(part.set_axis(["id", "name", "ioc", "hand", "height"], axis=1))
    both = pd.concat(sides, ignore_index=True)
    both["hand"] = both["hand"].mask(both["hand"] == "U")    # U means unknown

    def most_common(values):
        values = values.dropna()
        return values.mode().iloc[0] if len(values) else pd.NA

    return both.groupby("id").agg(
        name=("name", most_common),
        ioc=("ioc", most_common),
        hand=("hand", most_common),
        height=("height", most_common),
    ).reset_index()


def build_players_table(players):
    """Match-based columns + the birth date, and the file's hand/height as a fallback."""
    table = players_from_matches()
    unmatched = ~table["id"].isin(players["player_id"])
    print(f"Players in the matches that are missing from the players file: {unmatched.sum()}")

    extras = (players[["player_id", "birthdate", "hand", "height"]]
              .rename(columns={"player_id": "id", "hand": "hand_file", "height": "height_file"}))
    table = table.merge(extras, on="id", how="left", validate="one_to_one")

    hand_file = table.pop("hand_file").mask(lambda s: s == "U")
    table["hand"] = table["hand"].fillna(hand_file)
    table["height"] = table["height"].fillna(table.pop("height_file"))

    assert table["id"].is_unique
    print(f"Players table: {len(table):,} rows")
    print("\n% of players with the column filled:")
    print((table.notna().mean() * 100).round(1).to_string())
    print("\nhand:")
    print(table["hand"].value_counts(dropna=False).to_string())
    return table


def write_output(table):
    INTERIM.mkdir(parents=True, exist_ok=True)
    path = INTERIM / "wta_players_clean.csv"
    table.to_csv(path, index=False)
    print(f"\nWrote {len(table):,} rows x {table.shape[1]} columns -> {path.relative_to(ROOT)}")

def main():
    players = load_players()
    players = parse_birthdate(players)
    table = build_players_table(players)
    write_output(table)


if __name__ == "__main__":
    main()