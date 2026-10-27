import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from deviation_features import ROOT

OUT = ROOT / "outputs" / "v2"
V2 = ROOT / "data" / "synthetic_v2"
HOT_RATIO = 1.8
PERIODS = ["before 8", "morning peak 8-10:30", "midday 10:30-17", "evening peak 17-20"]
BLUE, ORANGE, RED, GREY = "#2E75B6", "#C55A11", "#C00000", "#7F7F7F"

t = pd.read_csv(OUT / "v2_trip_summary.csv")
secs = pd.read_csv(OUT / "v2_section_stats.csv", dtype={"shape_id": str})
zones = pd.read_csv(V2 / "planted_slow_zones.csv", dtype={"shape_id": str})

p = (t[t["period"].isin(PERIODS)].groupby("period")
       .agg(trips=("trip_id", "size"), delay=("delay_pct", "mean")).reindex(PERIODS))

fig, ax = plt.subplots(2, 2, figsize=(14, 9))

a = ax[0, 0]
a.bar(range(len(p)), p["delay"], color=[BLUE, ORANGE, BLUE, ORANGE])
a.set_xticks(range(len(p)))
a.set_xticklabels([x.replace(" ", "\n", 1) for x in p.index], fontsize=9)
for i, (n, d) in enumerate(zip(p["trips"], p["delay"])):
    a.text(i, d + 1, f"{d:.0f}%\n({n} trips)", ha="center", fontsize=9)
a.set_ylim(0, p["delay"].max() * 1.3)
a.set_ylabel("Delay (% over scheduled duration)")
a.set_title("Average delay vs timetable, by time of day (orange = planted peak)")

b = ax[0, 1]
b.axvspan(8, 10.5, color="#F4B183", alpha=0.35, label="planted peak windows")
b.axvspan(17, 20, color="#F4B183", alpha=0.35)
b.scatter(t["mid_hour"], t["delay_pct"], s=18, color=BLUE)
b.set_xlabel("Hour of day (middle of the journey)")
b.set_ylabel("Delay (%)")
b.set_title("Delay of each of the 120 trips")
b.legend(loc="upper left")


def profile(axis, route):
    z = zones[zones["route_short_name"] == route].iloc[0]
    s = secs[secs["shape_id"] == z["shape_id"]].sort_values("start_m")
    s = s[(s["end_m"] >= z["start_m"] - 1500) & (s["start_m"] <= z["end_m"] + 1500)].copy()
    s["mid_km"] = (s["start_m"] + s["end_m"]) / 2000
    overlap = (s["start_m"] < z["end_m"]) & (s["end_m"] > z["start_m"])
    found = bool((s["hotspot"] & overlap).any())

    axis.axvspan(z["start_m"] / 1000, z["end_m"] / 1000, color="#F4B183", alpha=0.5,
                 label="planted slow zone")
    axis.axhline(HOT_RATIO, ls="--", color=GREY, label=f"hotspot level ({HOT_RATIO}x)")
    axis.plot(s["mid_km"], s["mean_ratio"], color=BLUE, lw=1, zorder=1)
    normal = s[~s["near_stop"] & ~s["hotspot"]]
    stop = s[s["near_stop"]]
    hot = s[s["hotspot"]]
    axis.scatter(normal["mid_km"], normal["mean_ratio"], color=BLUE, s=30, zorder=2, label="normal section")
    axis.scatter(stop["mid_km"], stop["mean_ratio"], facecolors="none", edgecolors=GREY, s=40,
                 zorder=2, label="near a bus stop (excluded)")
    axis.scatter(hot["mid_km"], hot["mean_ratio"], color=RED, s=50, zorder=3, label="hotspot")
    axis.set_xlabel("Distance along route (km)")
    axis.set_ylabel("Section time vs the trip's typical section")
    axis.set_title(f"{route}: planted zone {'FOUND' if found else 'MISSED'}")
    axis.legend(fontsize=8, loc="upper left")

    print("=" * 70)
    print(f"{route}: planted zone {z['start_m']:.0f}-{z['end_m']:.0f} m -> {'FOUND' if found else 'MISSED'}")
    print(s[["start_m", "end_m", "trips", "mean_ratio", "slow_share", "near_stop", "hotspot"]]
          .round(2).to_string(index=False))
    print()


profile(ax[1, 0], "285-B")
profile(ax[1, 1], "KMT-NYD")

fig.suptitle("Phase 3 on synthetic v2 data (planted patterns, so results can be checked)")
fig.tight_layout()
fig.savefig(OUT / "v2_phase3_overview.png", dpi=130)

print("DELAY BY TIME OF DAY")
print(p.round(1).to_string())
print()
print("Saved:", OUT / "v2_phase3_overview.png")