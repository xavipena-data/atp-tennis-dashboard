import pandas as pd
from build_processed import load_no1, build_no1_stints, INTERIM, FILES

no1 = load_no1("atp")
players = pd.read_csv(INTERIM / FILES["atp"]["players"], usecols=["id", "name"], dtype={"id": str})
names = players.set_index("id")["name"]
connors = players.loc[players["name"] == "Jimmy Connors", "id"].iloc[0]

# his periods as No. 1
stints = build_no1_stints(no1)
print(stints[stints["player_id"] == connors].to_string(index=False))

# who appears at No. 1 between mid 1974 and the end of 1977
window = no1[(no1["date"] >= "1974-07-01") & (no1["date"] <= "1977-12-31")]
print(window.assign(name=window["id"].map(names))
      .groupby("name").agg(first=("date", "min"), last=("date", "max"), snapshots=("date", "size")))