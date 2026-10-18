import pandas as pd
import numpy as np
from build_processed import INTERIM, FILES, load_matches

MIN_CAREER_MATCHES = 50

for tour in FILES:
    print(tour.upper())

    # columns available in each cleaned file
    for key in ("matches", "players", "rankings"):
        df = pd.read_csv(INTERIM / FILES[tour][key], nrows=2)
        print(" ", key, list(df.columns))

    # how big would the head-to-head table be?
    matches = load_matches(tour)
    played = matches[matches["outcome"].isin(["completed", "retired"])]
    career = pd.concat([played["winner_id"], played["loser_id"]]).value_counts()
    listed = set(career[career >= MIN_CAREER_MATCHES].index)
    both = played[played["winner_id"].isin(listed) & played["loser_id"].isin(listed)]
    first = np.where(both["winner_id"] < both["loser_id"], both["winner_id"], both["loser_id"])
    second = np.where(both["winner_id"] < both["loser_id"], both["loser_id"], both["winner_id"])
    meetings = both.groupby([first, second]).size()

    print("  listed players:", len(listed), "| matches between them:", len(both), "of", len(played))
    print("  pairs that met:", len(meetings))
    for k in (2, 3, 5):
        kept = meetings[meetings >= k]
        print(f"  pairs with {k}+ meetings: {len(kept)} | matches kept: {kept.sum()}")
    print()