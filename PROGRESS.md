\# Smart Route Classification for Public Transport and Mobility Analysis

Final-year CS AI/ML project. Windows, project root `C:\\smart route classification`, virtual env `.venv`.

All data is \*\*synthetic\*\* (generated from real BMTC GTFS route shapes). Nothing is validated on real buses.



\## Phase status

| Phase | Scope | Status |

|---|---|---|

| 1 | Data prep and map matching | Done. Matcher upgraded twice (see below) |

| 2 | GPS reliability model and false-deviation detection | Done |

| 3 | Mobility analysis (distance, duration, speed, delay, patterns) | In progress. Dataset v2 built; trip-level and hotspot results obtained |

| 4 | Recurring hotspots and integrated dashboard | Not started (hotspot search prototype exists) |

| 5 | UI/UX preview of live tracking with demo data | Not started. Live feed stays future scope |



\## Key results so far

\*\*Reliability model\*\* (Random Forest, tuned threshold 0.62, features: route\_distance\_m, time\_gap\_s, step\_distance\_m, implied\_speed\_kmh, speed\_change\_kmh, heading\_change\_deg)

\- Original 10 held-out trips: precision 0.726, recall 0.411, F1 0.525, ROC-AUC 0.797

\- 120 unseen v2 trips: precision 0.747, recall 0.433, F1 0.549, ROC-AUC 0.800

\- Limitation: most injected noise stays near the route, so recall is about 43 percent



\*\*False-deviation detector\*\* (rules: >30 m off route, run of 3+ pings, at most half flagged unreliable; labels: likely\_gps\_noise, potential\_route\_deviation, uncertain, normal)

\- Synthetic data has no real detours, so the correct answer is 0 deviations. Final result on 120 v2 trips: 0 deviations.

\- First run on v1 gave 5 false deviations. Cause: map matcher missed an 830 m segment. Fixed by exact search.

\- `is\_outlier`, `true\_lat`, `true\_lon` are used only for offline evaluation, never in detector logic.



\*\*Map matching\*\* (three versions)

1\. Midpoint search (8 nearest segment midpoints): fails on very long segments.

2\. Exact search over all segments (`src/map\_matching\_exact.py`): fixed long-segment errors. v1 data uses this.

3\. Sequence-aware Viterbi matcher (`src/v2\_map\_matching\_sequence.py`): fixes wrong-pass matches on routes that revisit the same road. Pings wrong by more than 100 m: 198 -> 18; largest error 3073 m -> 170 m.



\*\*Synthetic data findings\*\*

\- v1 generator (`data/synthetic/synthetic\_gps\_generator.py`) stops randomly (not at bus stops), so speed is about 0.5 of planned by design. No hotspots or time-of-day patterns exist in v1. Valid v1 results: distance within about 1 percent of planned.

\- v2 generator (`data/synthetic/synthetic\_gps\_generator\_v2.py`, output in `data/synthetic\_v2/`): 120 trips, 20 routes x 6 trips, stops at real GTFS stops, planted rush-hour slowdowns (8:00-10:30 and 17:00-20:00), 8 planted 250 m slow zones, timetable durations for delay measurement. BEMLA-12 has no GTFS intermediate stops, so nearby stops (within 30 m) are used (`stop\_source` column).



\*\*v2 mobility results\*\*

\- Rush-hour delay recovered: morning peak 57.6 percent, evening peak 67.8 percent, midday 20.8 percent (against timetable).

\- Duration measured within 0.4 percent of truth on average.

\- Hotspot search (200 m sections, ratio >= 1.8, >= 60 percent of trips slow, sections within 30 m of a stop excluded): found 6 of 8 planted zones, 0 false alarms (after the sequence-aware matcher). Z02 and Z06 missed because a bus stop sits inside the zone and its section is excluded. Thresholds were not tuned to rescue them.



\## Run order (v2 pipeline)

1\. `python data\\synthetic\\synthetic\_gps\_generator\_v2.py`

2\. `python src\\v2\_map\_matching.py` then `python src\\v2\_map\_matching\_sequence.py` (final matcher; overwrites `outputs\\v2\\v2\_map\_matched\_exact.csv`, backup kept as `v2\_map\_matched\_nearest\_only.csv`)

3\. `python src\\v2\_reliability\_detector.py`

4\. `python src\\v2\_mobility\_trips.py` ( it was last run before the sequence-aware matcher)

5\. `python src\\v2\_hotspots.py`

Shared helpers: `src\\deviation\_features.py` (features, model loading), `src\\detect\_deviations.py` (detector rules).

v1 pipeline: `map\_matching.py`, `gps\_reliability\_model.py`, `tune\_gps\_threshold.py`, `map\_matching\_exact.py`, `detect\_deviations\_exact.py`, `mobility\_\*.py`.



\## Known limitations (state these in the report)

\- All data synthetic; `is\_outlier` is injected noise, not real route deviation.

\- Model probabilities are not calibrated reliability scores.

\- v1 and v2 share the same noise settings, so generalisation to other noise types is untested.

\- Standing detection (under 3 km/h) is a rough estimate; pings are 10-30 s apart.

\- Original generator has hard-coded paths from the author's earlier environment.

- v2 observed distance is 0.6 percent below route length on average (worst trip 8.8 percent on a short route). Cause confirmed: the stretch before the first ping and after the last ping is not covered. Distance is measured along the route (matched_along_m); straight lines between pings cut corners and added a further 1.1 percent shortfall.



\## Next steps


1\. Phase 3 charts on v2 (delay by period, speed by route, section speed profiles).

2\. Phase 4: hotspot map, recurring problems, dashboard combining route classification, reliability, deviation detection and mobility.

3\. Phase 5: clearly labelled demo UI for live tracking (no real-time or ETA claims).

