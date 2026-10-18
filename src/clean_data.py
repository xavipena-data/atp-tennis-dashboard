"""Clean the raw TML match files and write one tidy table.

Reads data/raw/matches/<season>.csv, applies the rules documented in section 8
of notebooks/01_data_profiling.ipynb and writes data/interim/matches_clean.csv.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_MATCHES = ROOT / "data" / "raw" / "matches"
INTERIM = ROOT / "data" / "interim"

FIRST_SEASON = 1968
LAST_SEASON = 2025  # the 2026 file is stale (137 matches), so it is excluded

# Tournaments with a spelling that differs by more than casing
NAME_FIXES = {
    "NextGen Finals": "Next Gen Finals",
    "Next Gen ATP Finals": "Next Gen Finals",
}

# Scope (section 8): Davis Cup and Olympics by level, team events by name
EXCLUDED_LEVELS = {"D", "O"}
TEAM_COMPETITIONS = {
    "World Team Cup", "World Team Championship",
    "ATP Cup", "United Cup", "Laver Cup",  "Nations Cup",
    "WCT Aetna World Cup",
    "WCT World Cup",
    "Seoul Olympics",
}

# Coding glitches: (tournament, season, wrong level, right level)
LEVEL_FIXES = [
    ("Doha", 2024, "A", "250"),
    ("Los Cabos", 2024, "A", "250"),
]

# Individual events outside the normal tour ladder, whatever level they carry
SPECIAL_EVENTS = {"Grand Slam Cup", "Next Gen Finals"}

LEVEL_GROUPS = {
    "G": "Grand Slam",
    "F": "Tour Finals",
    "M": "Masters (ATP 1000)",
    "500": "ATP 500",
    "250": "ATP 250",
}

# Knockout rounds, earliest to latest. Anything else is not a bracket round.
BRACKET_ROUNDS = ["R256", "R128", "R64", "R32", "R16", "QF", "SF", "F"]
OTHER_ROUNDS = {"RR", "3rd/4th", "Fs", "Early round (unknown)"}

# Two rows with all of this equal are the same match entered twice
MATCH_KEY = ["tourney_id", "round", "winner_id", "loser_id", "score"]

def read_csv_robust(path, **kwargs):
    """Read a CSV as UTF-8; fall back to cp1252 if the file is not valid UTF-8."""
    try:
        return pd.read_csv(path, encoding="utf-8-sig", **kwargs)
    except UnicodeDecodeError:
        print(f"  (note) {Path(path).name} is not valid UTF-8 -> reading as cp1252")
        return pd.read_csv(path, encoding="cp1252", encoding_errors="replace", **kwargs)


def load_matches():
    """Read one file per season and stack them into a single table."""
    frames = []
    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        df = read_csv_robust(RAW_MATCHES / f"{season}.csv")
        df["season"] = season
        frames.append(df)
    matches = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(matches):,} rows x {matches.shape[1]} columns, "
          f"seasons {FIRST_SEASON}-{LAST_SEASON}")
    return matches

def normalise_tournament_names(matches):
    """Unify spellings of the same tournament (casing first, then explicit fixes)."""
    assert matches["tourney_name"].notna().all(), "unexpected missing tournament names"
    original = matches["tourney_name"]
    names = original.str.strip()

    # 1) Casing variants: for each lowercase key keep the most frequent spelling.
    canonical = {}
    for name in names.value_counts().index:      # most frequent first
        canonical.setdefault(name.lower(), name)
    names = names.str.lower().map(canonical)

    # 2) Spellings that differ by more than casing
    names = names.replace(NAME_FIXES)

    changes = (pd.DataFrame({"before": original, "after": names})
               .query("before != after").value_counts())
    print(f"Tournament names changed in {(original != names).sum():,} rows:")
    print(changes.to_string())
    matches["tourney_name"] = names
    return matches


def apply_scope(matches):
    """Drop Davis Cup, Olympics and team competitions."""
    before = len(matches)
    by_level = matches["tourney_level"].isin(EXCLUDED_LEVELS)
    by_name = matches["tourney_name"].isin(TEAM_COMPETITIONS)
    print(f"Scope: Davis Cup / Olympics {by_level.sum():,} rows; "
          f"team competitions {by_name.sum():,} rows "
          f"({(by_name & ~by_level).sum():,} not already dropped)")
    matches = matches[~(by_level | by_name)].copy()
    print(f"Scope: {before:,} -> {len(matches):,} rows")
    return matches

def fix_tourney_levels(matches):
    """Recode the tournaments whose level is wrong in a single season."""
    for name, season, wrong, right in LEVEL_FIXES:
        mask = ((matches["tourney_name"] == name)
                & (matches["season"] == season)
                & (matches["tourney_level"] == wrong))
        matches.loc[mask, "tourney_level"] = right
        print(f"Level fix: {name} {season} {wrong} -> {right} in {mask.sum()} rows")
    return matches


def add_level_group(matches):
    """Add `level_group`, a level that means the same thing in every era."""
    level = matches["tourney_level"]
    group = level.map(LEVEL_GROUPS)
    group = group.mask((level == "A") & (matches["season"] < 1990), "Tour event (pre-tiers)")
    group = group.mask(matches["tourney_name"].isin(SPECIAL_EVENTS), "Special event")

    unmapped = group.isna()
    assert not unmapped.any(), (
        "Levels without a group:\n"
        + matches.loc[unmapped, ["tourney_name", "tourney_level", "season"]]
        .drop_duplicates().to_string()
    )
    matches["level_group"] = group

    print("level_group:")
    print(group.value_counts().to_string())
    print("Special events by level:")
    print(matches[group == "Special event"]
          .groupby(["tourney_name", "tourney_level"]).size().to_string())
    return matches

def add_outcome(matches):
    """Add `outcome` (completed / retired / walkover) from the text of the score."""
    score = matches["score"].fillna("")
    has_digit = score.str.contains(r"\d")

    walkover = score.str.contains("W/O") | (score.str.contains("WEA") & ~has_digit)
    retired = (score.str.contains("RET|DEF") | (score.str.contains("WEA") & has_digit)) & ~walkover

    outcome = pd.Series("completed", index=matches.index)
    outcome = outcome.mask(retired, "retired").mask(walkover, "walkover")
    matches["outcome"] = outcome

    # A match that was never played cannot have a duration
    cleared = (walkover & matches["minutes"].notna()).sum()
    matches.loc[walkover, "minutes"] = float("nan")

    print("outcome:")
    print(outcome.value_counts().to_string())
    print(f"Minutes cleared on walkovers: {cleared:,}")
    return matches

def clean_rounds(matches):
    """Add `round_clean` and `round_order` (NaN when the round is not in the bracket)."""
    rnd = matches["round"]

    # In five 2007 tournaments "RR" replaces the rounds before the quarterfinals
    glitch = ((rnd == "RR")
              & (matches["level_group"] == "ATP 250")
              & (matches["season"] == 2007))
    matches["round_clean"] = rnd.mask(glitch, "Early round (unknown)")

    unexpected = set(matches["round_clean"].unique()) - set(BRACKET_ROUNDS) - OTHER_ROUNDS
    assert not unexpected, f"Unexpected round labels: {unexpected}"

    order = {name: i for i, name in enumerate(BRACKET_ROUNDS, start=1)}
    matches["round_order"] = matches["round_clean"].map(order)

    print("Rounds outside the knockout bracket:")
    outside = matches[matches["round_order"].isna()]
    print(outside.groupby(["round_clean", "level_group"]).size().to_string())
    return matches


def fill_surface(matches):
    """Label the missing surfaces as 'Unknown'."""
    missing = matches["surface"].isna()
    matches["surface"] = matches["surface"].fillna("Unknown")
    if missing.any():
        seasons = matches.loc[missing, "season"]
        print(f"Surface: {missing.sum():,} missing -> 'Unknown' "
              f"(seasons {seasons.min()}-{seasons.max()})")
    return matches

def fix_identical_player_ids(matches):
    """Fix rows where winner and loser share the same id (a data entry error)."""
    same = matches["winner_id"] == matches["loser_id"]
    if not same.any():
        return matches

    # Most frequent id of each name, using only the healthy rows
    ok = matches[~same]
    pairs = pd.concat([
        ok[["winner_name", "winner_id"]].set_axis(["name", "id"], axis=1),
        ok[["loser_name", "loser_id"]].set_axis(["name", "id"], axis=1),
    ])
    id_of = pairs.groupby("name")["id"].agg(lambda s: s.mode().iloc[0])

    for idx in matches.index[same]:
        for side in ("winner", "loser"):
            name = matches.at[idx, f"{side}_name"]
            correct = id_of.get(name)
            if correct is not None and correct != matches.at[idx, f"{side}_id"]:
                print(f"Id fix: {matches.at[idx, 'tourney_name']} {matches.at[idx, 'season']} "
                      f"{matches.at[idx, 'round']}, {side} {name}: "
                      f"{matches.at[idx, f'{side}_id']} -> {correct}")
                matches.at[idx, f"{side}_id"] = correct

    remaining = (matches["winner_id"] == matches["loser_id"]).sum()
    assert remaining == 0, f"{remaining} rows still have the same id for winner and loser"
    return matches

def drop_duplicate_matches(matches):
    """Drop matches entered twice (same tournament, round, players and score)."""
    dup = matches.duplicated(MATCH_KEY, keep="first")
    print(f"Duplicate matches dropped: {dup.sum():,}")
    return matches[~dup].copy()

def add_date(matches):
    """Turn tourney_date (e.g. 19680101) into a real date."""
    as_text = matches["tourney_date"].astype(int).astype(str)
    matches["date"] = pd.to_datetime(as_text, format="%Y%m%d")
    return matches


def run_checks(matches):
    """Stop the script if the cleaned table breaks a rule we rely on."""
    assert matches["season"].between(FIRST_SEASON, LAST_SEASON).all()
    assert matches["outcome"].isin(["completed", "retired", "walkover"]).all()
    assert matches["level_group"].notna().all()
    assert (matches["winner_id"] != matches["loser_id"]).all()
    assert not matches.duplicated(MATCH_KEY).any(), "duplicate matches left"


def write_output(matches):
    INTERIM.mkdir(parents=True, exist_ok=True)
    path = INTERIM / "matches_clean.csv"
    matches.to_csv(path, index=False)
    print(f"Wrote {len(matches):,} rows x {matches.shape[1]} columns -> {path.relative_to(ROOT)}")


def main():
    matches = load_matches()
    matches = normalise_tournament_names(matches)
    matches = apply_scope(matches)
    matches = fix_tourney_levels(matches)
    matches = add_level_group(matches)
    matches = add_outcome(matches)
    matches = clean_rounds(matches)
    matches = fill_surface(matches)
    matches = fix_identical_player_ids(matches)
    matches = drop_duplicate_matches(matches)
    matches = add_date(matches)
    run_checks(matches)
    write_output(matches)


if __name__ == "__main__":
    main()