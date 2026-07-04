#!/usr/bin/env python3
"""Measure the AOD550 calibration baseline for the excess convention.

The engine's aerosol input expresses EXCESS over the air the khayt edge
factor was calibrated in (the desert campaign cluster; see
validation/RESULTS_CRITERION_SITES.md section 9.7 and
crates/twilight-weather/src/aod.rs AOD_BASELINE_550). The baseline must
be in the SAME retrieval scale as the live values (CAMS via Open-Meteo)
so the product's absolute bias cancels in the difference - an OPAC type
constant is not in that scale.

This script measures the CAMS fajr-hour AOD550 distribution at the four
desert calibration sites over recent winter mornings (the campaigns
selected CLEAR mornings, so the calibrated air sits near the clean
decile). Run of 2026-07-03, n = 1080 mornings 2023-2025:

    mecca          p10=0.130 p25=0.180 median=0.240
    hail           p10=0.070 p25=0.110 median=0.170
    riyadh_desert  p10=0.120 p25=0.180 median=0.340
    aswan          p10=0.090 p25=0.120 median=0.170
    CLUSTER        p10=0.090 p25=0.140 median=0.210

Baseline chosen: 0.10 (the cluster clean decile). Independent
cross-check: section 9.6 measured the Assiut campaign's implied excess
at ~0.08; CAMS reads ~0.15-0.20 at the Assiut fajr hour in the campaign
season, implying a baseline of 0.07-0.12.

Usage: python3 tools/aod_baseline_survey.py
(network; ~24 small requests to the free Open-Meteo air-quality API)
"""

import json
import urllib.request

# Desert calibration cluster sites and their approximate fajr hour (UTC).
SITES = [
    ("mecca", 21.4225, 39.8262, 2),
    ("hail", 27.517, 41.70, 2),
    ("riyadh_desert", 25.763, 46.5, 2),
    ("aswan", 24.088, 32.90, 3),
]

# Winter-morning windows (clear-season, matching the campaigns' bias
# toward winter dates) across the CAMS archive years.
WINDOWS = [
    ("2023-01-01", "2023-02-28"),
    ("2023-12-01", "2023-12-31"),
    ("2024-01-01", "2024-02-28"),
    ("2024-12-01", "2024-12-31"),
    ("2025-01-01", "2025-02-28"),
    ("2025-12-01", "2025-12-31"),
]

API = "https://air-quality-api.open-meteo.com/v1/air-quality"


def series(lat, lon, start, end):
    url = (f"{API}?latitude={lat}&longitude={lon}"
           f"&hourly=aerosol_optical_depth&start_date={start}&end_date={end}"
           f"&timezone=UTC")
    d = json.load(urllib.request.urlopen(url, timeout=30))
    t = d["hourly"]["time"]
    v = d["hourly"]["aerosol_optical_depth"]
    return {tt: vv for tt, vv in zip(t, v) if vv is not None}


def main():
    all_vals = []
    for name, lat, lon, hour in SITES:
        s = {}
        for a, b in WINDOWS:
            s.update(series(lat, lon, a, b))
        morning = sorted(v for t, v in s.items()
                         if t.split("T")[1] == f"{hour:02d}:00")
        n = len(morning)
        if n == 0:
            print(f"{name:14s} NO DATA")
            continue
        q = lambda p: morning[min(int(p * n), n - 1)]
        all_vals.extend(morning)
        print(f"{name:14s} n={n} min={morning[0]:.3f} p10={q(0.10):.3f} "
              f"p25={q(0.25):.3f} median={q(0.5):.3f} p75={q(0.75):.3f} "
              f"max={morning[-1]:.3f}")
    all_vals.sort()
    n = len(all_vals)
    q = lambda p: all_vals[min(int(p * n), n - 1)]
    print(f"{'CLUSTER':14s} n={n} min={all_vals[0]:.3f} p10={q(0.10):.3f} "
          f"p25={q(0.25):.3f} median={q(0.5):.3f}")
    print("\nbaseline recommendation: cluster p10 ~ 0.10 "
          "(clear-morning-selected campaigns sit near the clean decile)")


if __name__ == "__main__":
    main()
