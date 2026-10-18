from export_dashboard_data import read, share, first_covered, HEIGHT_N, AGE_N

for tour in ("atp", "wta"):
    played = read(tour, "seasons").set_index("season")["matches"]
    gap, serve, physical = read(tour, "upsets_gap"), read(tour, "win_serve"), read(tour, "win_physical")
    measures = {"rank": share(gap, ["n"], played), "serve": share(serve, ["n"], played),
                "height": share(physical, HEIGHT_N, played, divisor=2),
                "age": share(physical, AGE_N, played), "hand": share(physical, ["hk_n"], played)}
    print(f"\n{tour.upper()}")
    for name, s in measures.items():
        start = first_covered(s)
        later = s.reindex(played.index).fillna(0)
        later = later[later.index >= start]
        low = later[later < 0.5]
        print(f"  {name}: from {start} | later seasons below 50%:",
              {int(k): round(float(v) * 100) for k, v in low.items()} or "none")