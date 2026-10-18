"""Clean the raw WTA match files and write one tidy table.

Reads data/raw/wta/wta_matches_<season>.csv, applies the rules documented in
notebooks/02_wta_profiling.ipynb and writes data/interim/wta_matches_clean.csv.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_MATCHES = ROOT / "data" / "raw" / "wta"
INTERIM = ROOT / "data" / "interim"

FIRST_SEASON = 1968
LAST_SEASON = 2025  # the 2026 file is a partial season, so it is excluded

# Main-tour levels. G = Grand Slam, W = generic tour event of the pre-tier era
# (up to 1999), T1-T5 = tiers (up to 2008), PM / P / I = Premier Mandatory /
# Premier / International (2009 on), F = tour finals.
# Left out: D (Fed Cup), O (Olympics), CC (ITF circuits), E, 50+H, 35+H, J.
TOUR_LEVELS = {"G", "W", "I", "P", "PM", "T1", "T2", "T3", "T4", "T5", "F"}

# Team events that sit inside the main-tour levels. Ties between countries carry
# the pairing in the tournament name (e.g. "Athens Balkan Games BUL vs GRE");
# United Cup is a mixed team event.
TEAM_EVENT_PATTERN = r"\bvs\b|^United Cup$"

# Spellings the "most frequent wins" rule would get wrong (a tie, or the minority
# spelling is the correct one). Mapped after the casing step, so it works either way.
NAME_FIXES = {
    "Vina Del Mar": "Vina del Mar",
    "Cava Dei Tirreni": "Cava dei Tirreni",
    "Frinton-On-Sea": "Frinton-on-Sea",
    "Lee-On-Solent": "Lee-on-Solent",
    "S-Hertogenbosch": "s-Hertogenbosch",
}

# Scores that carry no information: unknown, an HTML leftover, or a score that a
# spreadsheet turned into a date (7-10 became "7-Oct", "Aug-00"...).
JUNK_SCORES = {"UNK", "&nbsp;"}
EXCEL_DATE_SCORE = r"\d{1,2}-[A-Za-z]{3}|[A-Za-z]{3}-\d{2}"

# Coding glitches: (tournament, season, wrong level, right level).
# The 2015 WTA Finals (8-player draw) is coded W instead of F.
LEVEL_FIXES = [
    ("WTA Finals", 2015, "W", "F"),
]

LEVEL_GROUPS = {"G": "Grand Slam", "F": "Tour Finals"}
OTHER_GROUP = "Tour event"

# Knockout rounds, earliest to latest. RR = round robin, BR = bronze / 3rd place.
BRACKET_ROUNDS = ["R128", "R64", "R32", "R16", "QF", "SF", "F"]
OTHER_ROUNDS = {"RR", "BR"}

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
        df = read_csv_robust(RAW_MATCHES / f"wta_matches_{season}.csv", low_memory=False)
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

    # 2) Spellings the first rule gets wrong
    names = names.replace(NAME_FIXES)

    changes = (pd.DataFrame({"before": original, "after": names})
               .query("before != after").value_counts())
    print(f"Tournament names changed in {(original != names).sum():,} rows:")
    print(changes.to_string())
    matches["tourney_name"] = names
    return matches


def apply_scope(matches):
    """Keep only the main-tour levels, and only individual (non-team) events."""
    before = len(matches)
    out = ~matches["tourney_level"].isin(TOUR_LEVELS)
    print("Scope: rows dropped by level:")
    print(matches.loc[out, "tourney_level"].value_counts().to_string())
    matches = matches[~out].copy()

    team = matches["tourney_name"].str.contains(
        TEAM_EVENT_PATTERN, case=False, regex=True, na=False
    )
    print(f"Scope: team-event rows dropped: {team.sum():,} "
          f"({matches.loc[team, 'tourney_name'].nunique()} event names)")
    matches = matches[~team].copy()

    print(f"Scope: {before:,} -> {len(matches):,} rows")
    return matches

def clean_scores(matches):
    """Set the unusable scores to missing (the match itself is kept)."""
    score = matches["score"].str.strip()
    junk = score.isin(JUNK_SCORES) | score.str.fullmatch(EXCEL_DATE_SCORE, na=False)
    print(f"Scores set to missing: {junk.sum():,} "
          f"({score[junk].nunique()} distinct values)")
    matches["score"] = score.mask(junk)
    return matches


def add_outcome(matches):
    """Add `outcome` (completed / retired / walkover) from the text of the score."""
    score = matches["score"].fillna("")

    walkover = score.str.contains("W/O", case=False)
    retired = score.str.contains(r"RET|DEF|ABD|\bAB\b", case=False) & ~walkover

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
    """Add `level_group`: Grand Slam, Tour Finals or any other tour event."""
    group = matches["tourney_level"].map(LEVEL_GROUPS).fillna(OTHER_GROUP)
    matches["level_group"] = group
    print("level_group:")
    print(group.value_counts().to_string())
    return matches


def clean_rounds(matches):
    """Add `round_clean` and `round_order` (NaN when the round is not in the bracket)."""
    matches["round_clean"] = matches["round"]

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

def drop_same_player_matches(matches):
    """Drop rows where winner and loser are the same player (data entry error)."""
    same = matches["winner_id"] == matches["loser_id"]
    for _, row in matches[same].iterrows():
        print(f"Same-player row dropped: {row['tourney_name']} {row['season']} "
              f"{row['round']}, {row['winner_name']}")
    return matches[~same].copy()


def drop_duplicate_matches(matches):
    """Drop matches entered twice (same tournament, round, players and score)."""
    dup = matches.duplicated(MATCH_KEY, keep="first")
    print(f"Duplicate matches dropped: {dup.sum():,}")
    return matches[~dup].copy()


def add_date(matches):
    """Turn tourney_date (e.g. 19750406) into a real date."""
    as_text = matches["tourney_date"].astype(int).astype(str)
    matches["date"] = pd.to_datetime(as_text, format="%Y%m%d")
    return matches


def run_checks(matches):
    """Stop the script if the cleaned table breaks a rule we rely on."""
    assert matches["season"].between(FIRST_SEASON, LAST_SEASON).all()
    assert matches["tourney_level"].isin(TOUR_LEVELS).all()
    assert matches["outcome"].isin(["completed", "retired", "walkover"]).all()
    assert matches["level_group"].notna().all()
    assert matches["surface"].notna().all()
    assert (matches["winner_id"] != matches["loser_id"]).all()
    assert not matches.duplicated(MATCH_KEY).any(), "duplicate matches left"


def write_output(matches):
    INTERIM.mkdir(parents=True, exist_ok=True)
    path = INTERIM / "wta_matches_clean.csv"
    matches.to_csv(path, index=False)
    print(f"Wrote {len(matches):,} rows x {matches.shape[1]} columns -> {path.relative_to(ROOT)}")

def main():
    matches = load_matches()
    matches = normalise_tournament_names(matches)
    matches = apply_scope(matches)
    matches = clean_scores(matches)
    matches = add_outcome(matches)
    matches = fix_tourney_levels(matches)
    matches = add_level_group(matches)
    matches = clean_rounds(matches)
    matches = fill_surface(matches)
    matches = drop_same_player_matches(matches)
    matches = drop_duplicate_matches(matches)
    matches = add_date(matches)
    run_checks(matches)
    write_output(matches)


if __name__ == "__main__":
    main()