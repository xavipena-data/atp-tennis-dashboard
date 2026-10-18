import pandas as pd
from build_processed import load_matches, SLAM_ALIASES

for tour in ["atp", "wta"]:
    matches = load_matches(tour)
    slams = matches[matches["level_group"] == "Grand Slam"].copy()
    slams["slam"] = slams["tourney_name"].replace(SLAM_ALIASES)

    # most common surface of each Grand Slam in each season
    surface = (slams.dropna(subset=["surface"])
               .groupby(["slam", "season"])["surface"]
               .agg(lambda s: s.value_counts().index[0])
               .unstack("slam"))

    # seasons where the surface differs from the previous season
    previous = surface.shift()
    print(tour.upper())
    for slam in surface.columns:
        changed = surface[slam].ne(previous[slam]) & previous[slam].notna()
        for season in surface.index[changed]:
            print(f"  {slam} {season}: {previous.loc[season, slam]} -> {surface.loc[season, slam]}")
    print()

for tour in ["atp", "wta"]:
    matches = load_matches(tour)
    carpet = matches[matches["surface"] == "Carpet"]
    print(tour.upper(), "carpet:", carpet["season"].min(), "to", carpet["season"].max())
    print(carpet.groupby("season").size().tail(5).to_string())
    print()

wta = load_matches("wta")
late = wta[(wta["surface"] == "Carpet") & (wta["season"] >= 2009)]
print(late.groupby(["season", "tourney_name"]).size().to_string())