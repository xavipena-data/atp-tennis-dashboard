"""Download raw ATP data from the TML databases.

Sources (both on GitHub, by Tennismylife):
  - TML-Database          -> one CSV per season (1968 onwards) and the players file
  - TML-Rankings-Database -> historical ATP rankings (downloaded as a ZIP)

TML-Database is originally inspired by Jeff Sackmann's tennis_atp project
(CC BY-NC-SA 4.0). Use is non-commercial. See README for attribution.

Output layout (inside data/raw/):
    matches/<year>.csv
    players/ATP_Database.csv
    rankings/<folder inside the rankings repo>/<file>.csv

Usage (from the repository root):
    python src/download_data.py                  # download missing files only
    python src/download_data.py --force          # overwrite existing files
    python src/download_data.py --no-rankings    # skip the rankings repo
"""

import argparse
import sys
import tempfile
import zipfile
from datetime import date
from pathlib import Path, PurePosixPath

import requests

MATCHES_BASE = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master"
PLAYERS_FILE = "ATP_Database.csv"
RANKINGS_ZIP = "https://github.com/Tennismylife/TML-Rankings-Database/archive/HEAD.zip"

FIRST_YEAR = 1968  # start of the Open Era
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
MATCHES_DIR = RAW_DIR / "matches"
PLAYERS_DIR = RAW_DIR / "players"
RANKINGS_DIR = RAW_DIR / "rankings"

MATCH_HEADER_START = "tourney_id"


def fetch_text_file(session: requests.Session, url: str, target: Path,
                    expected_start: str | None, force: bool) -> str:
    """Download one CSV. Returns 'ok', 'skipped', 'missing' or 'invalid'."""
    if target.exists() and not force:
        return "skipped"
    response = session.get(url, timeout=120)
    if response.status_code == 404:
        return "missing"
    response.raise_for_status()
    first_line = response.text.lstrip("\ufeff").split("\n", 1)[0]
    if expected_start and not first_line.startswith(expected_start):
        return "invalid"  # not the CSV we expect (e.g. an HTML error page)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(response.content)
    return "ok"


def download_matches(session: requests.Session, force: bool) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {"ok": [], "skipped": [], "missing": [], "invalid": []}
    for year in range(FIRST_YEAR, date.today().year + 1):
        name = f"{year}.csv"
        status = fetch_text_file(
            session, f"{MATCHES_BASE}/{name}", MATCHES_DIR / name, MATCH_HEADER_START, force
        )
        results[status].append(name)
        print(f"  [{status:>7}] matches/{name}")
    return results


def download_players(session: requests.Session, force: bool) -> str:
    status = fetch_text_file(
        session, f"{MATCHES_BASE}/{PLAYERS_FILE}", PLAYERS_DIR / PLAYERS_FILE, None, force
    )
    print(f"  [{status:>7}] players/{PLAYERS_FILE}")
    return status


def download_rankings(session: requests.Session, force: bool) -> list[str]:
    """Download the rankings repo ZIP and extract every CSV that sits in a folder."""
    print(f"  Downloading {RANKINGS_ZIP}")
    extracted: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "rankings.zip"
        with session.get(RANKINGS_ZIP, stream=True, timeout=180) as response:
            if response.status_code != 200:
                print(f"  [ failed] rankings ZIP: HTTP {response.status_code}")
                return extracted
            with open(zip_path, "wb") as fh:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        with zipfile.ZipFile(zip_path) as zf:
            for member in zf.namelist():
                parts = PurePosixPath(member).parts
                # parts[0] is the repo folder; keep CSVs inside sub-folders only
                # (root-level helper files such as url_rankings.csv are skipped).
                if member.endswith("/") or len(parts) < 3 or not parts[-1].lower().endswith(".csv"):
                    continue
                relative = Path(*parts[1:])
                target = RANKINGS_DIR / relative
                if target.exists() and not force:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member))
                extracted.append(str(relative))
    print(f"  Rankings files extracted: {len(extracted)}")
    return extracted


def report() -> None:
    years = sorted(int(p.stem) for p in MATCHES_DIR.glob("*.csv") if p.stem.isdigit())
    print("\n=== Summary ===")
    print(f"Target folder: {RAW_DIR}")
    if years:
        print(f"Match files: {len(years)} (years {years[0]}-{years[-1]})")
        gaps = sorted(set(range(years[0], years[-1] + 1)) - set(years))
        if gaps:
            print(f"  Missing years in range: {gaps}")
    else:
        print("Match files: none")
    print(f"Players file: {'yes' if (PLAYERS_DIR / PLAYERS_FILE).exists() else 'NO'}")
    ranking_files = sorted(p.relative_to(RANKINGS_DIR) for p in RANKINGS_DIR.rglob("*.csv"))
    print(f"Ranking files: {len(ranking_files)}")
    folders = sorted({str(p.parent) for p in ranking_files})
    for folder in folders:
        count = sum(1 for p in ranking_files if str(p.parent) == folder)
        sample = next(p.name for p in ranking_files if str(p.parent) == folder)
        print(f"  {folder}: {count} file(s), e.g. {sample}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    parser.add_argument("--no-rankings", action="store_true", help="skip the rankings repo")
    args = parser.parse_args()

    session = requests.Session()
    print("Matches (TML-Database):")
    match_results = download_matches(session, args.force)
    print("\nPlayers:")
    download_players(session, args.force)
    if not args.no_rankings:
        print("\nRankings (TML-Rankings-Database):")
        download_rankings(session, args.force)

    if match_results["invalid"]:
        print(f"\nWarning: unexpected content in {match_results['invalid']}")
    if not any(MATCHES_DIR.glob("*.csv")):
        sys.exit("No match files available. Check the connection or the source repository.")
    report()


if __name__ == "__main__":
    main()
