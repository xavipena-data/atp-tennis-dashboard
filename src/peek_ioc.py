from pathlib import Path
import pandas as pd

INTERIM = Path("data/interim")
FILES = {"atp": "matches_clean.csv", "wta": "wta_matches_clean.csv"}

counts = {}
for tour, name in FILES.items():
    m = pd.read_csv(INTERIM / name, usecols=["winner_ioc", "loser_ioc"])
    codes = pd.concat([m["winner_ioc"], m["loser_ioc"]])
    print(tour.upper(), "| distinct codes:", codes.nunique(), "| rows without a code:", int(codes.isna().sum()))
    counts[tour] = codes.value_counts()

table = pd.concat(counts, axis=1).fillna(0).astype(int)
table["total"] = table.sum(axis=1)
print(table.sort_values("total", ascending=False).to_string())