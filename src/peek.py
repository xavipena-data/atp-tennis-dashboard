import pandas as pd
from build_processed import PROCESSED

for table in ["surface_matches", "surface_serve", "player_surface"]:
    df = pd.read_csv(PROCESSED / f"atp_{table}.csv")
    print(table, df.shape)
    print(df.head(4).to_string())
    print()