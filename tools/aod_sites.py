#!/usr/bin/env python3
"""Measured-AOD validation of the section-9 sites through the live AOD path.

For each site the engine runs on the CAMPAIGN date (so the comparators
are the section-9 pristine/bracket rows at identical geometry and moon),
while the measured AOD comes from a REPRESENTATIVE RECENT date in the
same season: the Open-Meteo/CAMS air-quality archive begins 2022-07-29
(measured; see crates/twilight-weather/src/aod.rs AOD_ARCHIVE_START),
so the 2010-2015 campaign mornings themselves have no archived AOD.
This substitution is the documented caveat of every row, not a hidden
assumption.

The measured value enters exactly as the production path will feed it:
excess over the calibration baseline (aod.rs AOD_BASELINE_550 = 0.10,
provenance in tools/aod_baseline_survey.py), through the pipeline's
interim env switches TWILIGHT_AOD_EXCESS_550 / TWILIGHT_AOD_SIGMA_550
(the one-line CLI flag wiring is documented in
twilight_weather::AtmosphericParams::aod_sigma_550). TWILIGHT_KHAYT_DEBUG
is set so the per-run sigma-term split lands in the raw output.

Usage:
    python3 tools/aod_sites.py [--out DIR] [--jobs N] [--only SITE]

Cache: raw stdout+stderr per run beside the TSV (rerun-safe); AOD API
bodies cached under data/weather (same file naming as the Rust fetcher).
TWILIGHT_CLI pins the binary against parallel rebuilds.
"""

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import re
import subprocess
import sys
import urllib.request

CLI = pathlib.Path(os.environ.get(
    "TWILIGHT_CLI",
    pathlib.Path(__file__).resolve().parent.parent / "target/release/twilight-cli"))

REPO = pathlib.Path(__file__).resolve().parent.parent
AOD_CACHE = REPO / "data/weather"
AOD_BASELINE_550 = 0.10  # keep in sync with twilight-weather aod.rs

# name -> (lat, lon, elev_m, [(engine_date, aod_date, fajr_hour_utc)], extra_args)
# extra_args carry the optical CARRIER type (the excess rides its
# optics) exactly as section 9 chose it per site background.
SITES = {
    # Assiut agricultural Nile valley (campaign 2012-2014, observed
    # fajr 13.665; section 9.6). Carrier: continental-average optics.
    "assiut": (27.167, 31.167, 70, [
        ("2013-01-15", "2026-01-15", 3),
        ("2013-04-15", "2026-04-15", 2),
        ("2013-10-15", "2025-10-15", 3),
    ], ["--aerosol", "continental-average"]),
    # Tubruq sea background (campaign 2007-2008, observed 13.43-13.48;
    # section 9.5). Carrier: maritime-clean optics + sea albedo, as in
    # the section 9.5 marine row. 2026-04-15 is a measured dust event
    # (AOD 0.38) and is included deliberately as the per-date scatter.
    "tubruq": (32.078, 23.983, 25, [
        ("2010-01-15", "2026-01-15", 4),
        ("2010-04-15", "2026-04-15", 4),
        ("2010-07-15", "2025-07-15", 4),
    ], ["--albedo", "0.06", "--aerosol", "maritime-clean"]),
    # Birmingham OpenFajr winter row (panel 12.9; veiled engine 12.85,
    # section 9.3). The veil is the winter mechanism; the measured
    # December AOD sits below the baseline, so the honest expectation
    # is excess 0 (no double-correction on top of the veil, 9.4).
    "birmingham": (52.44, -1.95, 150, [
        ("2015-12-10", "2025-12-10", 6),
    ], ["--skyglow"]),
    # Mecca control: the calibration cluster itself. A clean-season
    # measured AOD must leave it essentially unmoved (pristine 14.96).
    "mecca": (21.4225, 39.8262, 300, [
        ("2015-12-21", "2025-12-21", 2),
    ], []),
}

_TIME_LINE = (r":\s+(\S+?)\s+(?:±([^\s(]+)\s+)?(?:\(\+1d\)\s+)?"
              r"\(SZA ([\d.]+).?, depression ([\d.]+)")
PATTERNS = {
    "khayt_fajr": re.compile(r"Fajr \(khayt al-abyad\)" + _TIME_LINE),
    "legacy_fajr": re.compile(r"Fajr \(true dawn\)" + _TIME_LINE),
}
SIGMA_TERMS = re.compile(
    r"khayt sigma terms fajr_sadiq: crossing ([\d.]+), skyglow_cal ([\d.]+), "
    r"skyglow_duty ([\d.]+), aerosol ([\d.]+) -> total ([\d.]+) deg")


def fetch_aod(lat, lon, date, hour):
    """CAMS AOD550 at the fajr hour; body cached like the Rust fetcher."""
    AOD_CACHE.mkdir(parents=True, exist_ok=True)
    cache = AOD_CACHE / f"aod_{lat:.3f}_{lon:.3f}_{date}.json"
    if cache.exists():
        body = cache.read_text()
    else:
        url = (f"https://air-quality-api.open-meteo.com/v1/air-quality"
               f"?latitude={lat}&longitude={lon}&hourly=aerosol_optical_depth"
               f"&start_date={date}&end_date={date}&timezone=UTC")
        body = urllib.request.urlopen(url, timeout=30).read().decode()
        cache.write_text(body)
    d = json.loads(body)
    target = f"{date}T{hour:02d}:00"
    for t, v in zip(d["hourly"]["time"], d["hourly"]["aerosol_optical_depth"]):
        if t == target:
            if v is None:
                raise RuntimeError(f"AOD null at {target} (out of archive?)")
            return v
    raise RuntimeError(f"hour {target} missing from AOD series")


def run_one(name, lat, lon, elev, engine_date, aod_date, hour, extra, outdir):
    aod = fetch_aod(lat, lon, aod_date, hour)
    excess = max(0.0, aod - AOD_BASELINE_550)
    sigma = 0.03 + 0.20 * aod  # aod.rs aod_sigma_envelope
    raw = outdir / f"{name}_{engine_date}_aod{aod_date}.txt"
    if raw.exists() and "depression" in raw.read_text():
        out = raw.read_text()
    else:
        cmd = [str(CLI), "pray", f"--lat={lat}", f"--lon={lon}",
               f"--elevation={elev}", f"--date={engine_date}"] + extra
        env = dict(os.environ,
                   TWILIGHT_AOD_EXCESS_550=f"{excess:.4f}",
                   TWILIGHT_AOD_SIGMA_550=f"{sigma:.4f}",
                   TWILIGHT_KHAYT_DEBUG="1")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=3600, env=env)
        except subprocess.TimeoutExpired:
            return {"site": name, "engine_date": engine_date, "error": "timeout"}
        out = proc.stdout + proc.stderr
        raw.write_text(
            f"# cmd: {' '.join(cmd)}\n# TWILIGHT_AOD_EXCESS_550={excess:.4f} "
            f"TWILIGHT_AOD_SIGMA_550={sigma:.4f} (AOD {aod:.2f} @ {aod_date} "
            f"{hour:02d}UTC, baseline {AOD_BASELINE_550})\n" + out)
        if proc.returncode != 0:
            return {"site": name, "engine_date": engine_date,
                    "error": f"exit {proc.returncode}"}
    row = {"site": name, "engine_date": engine_date, "aod_date": aod_date,
           "aod_550": round(aod, 3), "excess_550": round(excess, 3),
           "aod_sigma": round(sigma, 3), "extra": " ".join(extra)}
    for key, pat in PATTERNS.items():
        m = pat.search(out)
        if m:
            row[key + "_time"] = m.group(1)
            row[key + "_pm_min"] = m.group(2) or ""
            row[key + "_dep"] = float(m.group(4))
    m = SIGMA_TERMS.search(out)
    if m:
        row.update(sigma_crossing=float(m.group(1)),
                   sigma_skyglow_cal=float(m.group(2)),
                   sigma_skyglow_duty=float(m.group(3)),
                   sigma_aerosol=float(m.group(4)),
                   sigma_total_deg=float(m.group(5)))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append")
    ap.add_argument("--out", default=str(REPO / "validation/aod_runs"))
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, (lat, lon, elev, dates, extra) in SITES.items():
        if args.only and name not in args.only:
            continue
        for engine_date, aod_date, hour in dates:
            jobs.append((name, lat, lon, elev, engine_date, aod_date, hour, extra))

    rows = []
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_one, *j, outdir): j for j in jobs}
        for fut in cf.as_completed(futs):
            row = fut.result()
            rows.append(row)
            print(json.dumps(row), flush=True)

    rows.sort(key=lambda r: (r["site"], r["engine_date"]))
    tsv = outdir / "aod_runs.tsv"
    cols = ["site", "engine_date", "aod_date", "aod_550", "excess_550",
            "aod_sigma", "khayt_fajr_dep", "khayt_fajr_time",
            "khayt_fajr_pm_min", "legacy_fajr_dep", "sigma_crossing",
            "sigma_skyglow_cal", "sigma_skyglow_duty", "sigma_aerosol",
            "sigma_total_deg", "extra"]
    with tsv.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"\nwrote {tsv} ({len(rows)} rows)", file=sys.stderr)


if __name__ == "__main__":
    main()
