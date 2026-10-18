from build_processed import load_matches

for tour in ("atp", "wta"):
    m = load_matches(tour)
    m = m[m["outcome"].isin(["completed", "retired"])]
    print(f"\n{tour.upper()}")
    print("level_group:", m["level_group"].value_counts().to_dict())
    print("round_clean:", m["round_clean"].value_counts().to_dict())
    print("ranks:", m["winner_rank"].min(), "to", m["winner_rank"].max(),
          "| dtype:", m["winner_rank"].dtype)
    print("equal ranks:", int((m["winner_rank"] == m["loser_rank"]).sum()))