from build_processed import load_matches, upset_matches

for tour in ("atp", "wta"):
    matches = load_matches(tour)
    played = matches[matches["outcome"].isin(["completed", "retired"])]
    ranked = upset_matches(matches)
    print(f"\n{tour.upper()}: played {len(played)} | in the upsets table {len(ranked)}")
    print("upset rate:", round(ranked["upset"].mean() * 100, 1), "%")
    print(ranked.groupby("gap_bin")["upset"].agg(["size", "mean"]).round(3))
    print(ranked.groupby(ranked["season"] // 10 * 10)["upset"].mean().round(3))