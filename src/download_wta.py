from pathlib import Path
import requests

BASE = "https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive/resolve/main/wta/"
DEST = Path("data/raw/wta")
DEST.mkdir(parents=True, exist_ok=True)

files = ["UPSTREAM_README.md", "wta_players.csv"]
files += [f"wta_matches_{year}.csv" for year in range(1968, 2026)]
files += [f"wta_rankings_{d}.csv" for d in ("80s", "90s", "00s", "10s", "20s", "current")]

for name in files:
    target = DEST / name
    if target.exists():
        print(f"ya existe  {name}")
        continue
    resp = requests.get(BASE + name, timeout=60)
    resp.raise_for_status()
    target.write_bytes(resp.content)
    print(f"descargado {name} ({len(resp.content):,} bytes)")