import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from deviation_features import ROOT, haversine_m

OUT = ROOT / "outputs"
V2 = OUT / "v2"
SYN2 = ROOT / "data" / "synthetic_v2"
DASH = ROOT / "dashboard"
DASH.mkdir(exist_ok=True)


def records(df):
    return df.astype(object).where(df.notna(), None).to_dict("records")


def reliability_block(d):
    y = d["eval_is_outlier"].astype(bool)
    flag = d["reliability_flag"].astype(bool)
    return dict(
        pings=int(len(d)), outliers=int(y.sum()),
        tn=int(((~y) & (~flag)).sum()), fp=int(((~y) & flag).sum()),
        fn=int((y & (~flag)).sum()), tp=int((y & flag).sum()),
        precision=float(precision_score(y, flag)), recall=float(recall_score(y, flag)),
        f1=float(f1_score(y, flag)), auc=float(roc_auc_score(y, d["reliability_outlier_prob"])),
    )


# ---------- GPS reliability and deviation detection ----------
v2 = pd.read_csv(V2 / "v2_deviation_candidates.csv")
v1 = pd.read_csv(OUT / "deviation_candidates_exact.csv")
v1h = v1[v1["split"] == "held_out"]
reliability = {"v1_heldout": reliability_block(v1h), "v2_all": reliability_block(v2)}
detector = {
    "v1_heldout": {k: int(v) for k, v in v1h["candidate_label"].value_counts().items()},
    "v2_all": {k: int(v) for k, v in v2["candidate_label"].value_counts().items()},
}

# ---------- Route matching: nearest-only vs sequence-aware ----------
new = pd.read_csv(V2 / "v2_map_matched_exact.csv", dtype={"shape_id": str},
                  usecols=["trip_id", "ping_seq", "shape_id", "matched_along_m", "true_dist_m",
                           "matched_lat", "matched_lon", "true_lat", "true_lon",
                           "is_outlier", "match_distance_m"])
old = pd.read_csv(V2 / "v2_map_matched_nearest_only.csv", dtype={"shape_id": str},
                  usecols=["shape_id", "matched_segment_index", "matched_lat", "matched_lon"])
sh = pd.read_csv(ROOT / "data" / "gtfs" / "shapes.txt",
                 usecols=["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"],
                 dtype={"shape_id": str})
sh = sh[sh["shape_id"].isin(old["shape_id"].unique())]
old["along_m"] = np.nan
for sid, s in sh.groupby("shape_id"):
    s = s.sort_values("shape_pt_sequence")
    lat, lon = s["shape_pt_lat"].to_numpy(), s["shape_pt_lon"].to_numpy()
    cum = np.concatenate([[0.0], np.cumsum(haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:]))])
    mk = old["shape_id"] == sid
    seg = old.loc[mk, "matched_segment_index"].to_numpy()
    old.loc[mk, "along_m"] = cum[seg] + haversine_m(
        lat[seg], lon[seg], old.loc[mk, "matched_lat"].to_numpy(), old.loc[mk, "matched_lon"].to_numpy())


def err_block(e):
    return dict(median_m=float(e.median()), p95_m=float(e.quantile(0.95)),
                max_m=float(e.max()), over_100m=int((e > 100).sum()))


pos_err = haversine_m(new["true_lat"], new["true_lon"], new["matched_lat"], new["matched_lon"])
ok = ~new["is_outlier"]
matching = {
    "pings": int(len(new)),
    "nearest_only": err_block((old["along_m"] - new["true_dist_m"]).abs()),
    "sequence_aware": err_block((new["matched_along_m"] - new["true_dist_m"]).abs()),
    "ping_to_route_mean_m": float(new["match_distance_m"].mean()),
    "ping_to_route_median_m": float(new["match_distance_m"].median()),
    "over_30m": int((new["match_distance_m"] > 30).sum()),
    "true_to_matched_mean_m": float(pos_err[ok].mean()),
    "true_to_matched_p95_m": float(pos_err[ok].quantile(0.95)),
}

# ---------- Mobility ----------
trips = pd.read_csv(V2 / "v2_trip_summary.csv")
tcols = ["trip_id", "route_short_name", "period", "mid_hour", "delay_pct", "duration_min",
         "scheduled_duration_min", "observed_speed_kmph", "timetable_speed_kmph",
         "observed_distance_m", "planned_distance_m", "has_slow_zone"]
period_order = ["before 8", "morning peak 8-10:30", "midday 10:30-17", "evening peak 17-20"]
per = (trips[trips["period"].isin(period_order)].groupby("period")
       .agg(trips=("trip_id", "size"), mean_delay_pct=("delay_pct", "mean"),
            observed_speed=("observed_speed_kmph", "mean"),
            timetable_speed=("timetable_speed_kmph", "mean"))
       .reindex(period_order).reset_index())
mobility = {
    "trips": records(trips[tcols].round(2)),
    "periods": records(per.round(2)),
    "routes": records(pd.read_csv(V2 / "v2_route_summary.csv").round(2)),
    "distance_err_pct_mean": float((100 * (trips["observed_distance_m"] / trips["planned_distance_m"] - 1)).mean()),
}

# ---------- Hotspots ----------
hot = pd.read_csv(V2 / "v2_hotspot_table.csv")
zones = pd.read_csv(SYN2 / "planted_slow_zones.csv")
zones["found"] = zones["zone_id"].isin(set(hot["planted_zone"].dropna()))
secs = pd.read_csv(V2 / "v2_section_stats.csv")[
    ["route_short_name", "start_m", "end_m", "trips", "mean_ratio", "slow_share", "near_stop", "hotspot"]]
hotspots = {
    "hotspots": records(hot.round(4)),
    "zones": records(zones[["zone_id", "route_short_name", "start_m", "end_m",
                            "centre_lat", "centre_lon", "found"]].round(5)),
    "sections": records(secs.round(3)),
}
network = new[["matched_lat", "matched_lon"]].iloc[::8].round(5).values.tolist()

peaks = pd.read_csv(SYN2 / "planted_peak_windows.csv")
meta = {
    "routes": int(trips["route_short_name"].nunique()), "trips": int(len(trips)),
    "pings": int(len(new)), "reliability_threshold": 0.62,
    "peaks": records(peaks),
}

data = dict(meta=meta, reliability=reliability, detector=detector, matching=matching,
            mobility=mobility, hotspots=hotspots, network=network)
path = DASH / "data.json"
with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, default=lambda o: o.item() if hasattr(o, "item") else str(o), allow_nan=False)

print("Saved:", path, f"({path.stat().st_size / 1024:.0f} KB)")
print()
print("META:", {k: v for k, v in meta.items() if k != "peaks"})
print()
print("ROUTE MATCHING, error along the route vs truth")
for k in ("nearest_only", "sequence_aware"):
    print(" ", k, {a: round(b, 1) for a, b in matching[k].items()})
print("  ping-to-route mean/median (m):", round(matching["ping_to_route_mean_m"], 2),
      round(matching["ping_to_route_median_m"], 2), "| over 30 m:", matching["over_30m"])
print()
for name in ("v1_heldout", "v2_all"):
    r = reliability[name]
    print(f"RELIABILITY {name}: precision {r['precision']:.3f} recall {r['recall']:.3f} "
          f"F1 {r['f1']:.3f} AUC {r['auc']:.3f} | tn/fp/fn/tp {r['tn']}/{r['fp']}/{r['fn']}/{r['tp']}")
print()
print("DETECTOR LABELS:", detector)
print()
print("DELAY BY PERIOD")
print(per.round(1).to_string(index=False))
print("Distance error vs route length (mean %):", round(mobility["distance_err_pct_mean"], 2))
print()
print(f"HOTSPOTS: {len(hot)} found | planted zones found {int(zones['found'].sum())} of {len(zones)} "
      f"| network points: {len(network)}")