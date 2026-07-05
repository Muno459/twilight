#!/usr/bin/env python3
"""Recalibrate the khayt appearance edge factor on the FINAL engine.

Protocol (the constant's definition, applied on the frozen current
binary): run the Mecca-region calibration cluster (Riyadh KACST, Hail,
Aswan) across a factor ladder via TWILIGHT_KHAYT_EDGE_APPEARANCE, and
pick the factor minimizing the cluster's modality-matched residual
(naked-eye targets for Riyadh/Hail, camera for Aswan against the
legacy/instrument output). Nothing outside the cluster is consulted:
every other campaign remains a genuine test at the chosen factor.

Outputs one line per (site, date, factor) to the cache dir (fresh
directory: the old edge_factor cache belongs to the b776776 engine)
and a RECAL summary at the end.

Usage: python3 tools/recalibrate_edge_factor.py [--jobs 2]
       [--factors 40,45,50,55,60,65]
"""

import argparse
import concurrent.futures as cf
import json
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLI = ROOT / "target/release/twilight-cli"
CACHE = ROOT / "validation/criterion_runs/edge_factor_v2"

# (site, lat, lon, elev, dates, eye_target_dep, instrument_target_dep)
CLUSTER = [
    ("riyadh_desert", 25.763, 46.5, 540,
     ["2004-01-15", "2004-04-15", "2004-07-15", "2004-10-15"], 14.6, 14.5),
    ("hail", 27.517, 41.70, 1000,
     ["2014-10-15", "2015-01-15", "2015-04-15"], 14.01, None),
    ("aswan", 24.088, 32.90, 100,
     ["2016-01-12", "2016-01-14", "2016-01-16"], None, 14.90),
]

KHAYT_RE = re.compile(r"Fajr \(khayt al-abyad\).*?depression ([\d.]+)")
LEGACY_RE = re.compile(r"Fajr \(threshold\).*depression ([\d.]+)|legacy.*?([\d.]+)")


def run_one(site, lat, lon, elev, date, factor):
    out = CACHE / f"{site}_{date}_f{factor:g}.txt"
    if out.exists():
        txt = out.read_text()
    else:
        import os
        env = dict(os.environ)
        env["TWILIGHT_KHAYT_EDGE_APPEARANCE"] = str(factor)
        r = subprocess.run(
            [str(CLI), "pray", "--lat", str(lat), "--lon", str(lon),
             "--elevation", str(elev), "--date", date, "--scattering", "hybrid"],
            capture_output=True, text=True, env=env, timeout=7200,
        )
        txt = r.stdout
        out.write_text(txt)
    m = KHAYT_RE.search(txt)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--factors", default="40,45,50,55,60,65")
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    factors = [float(f) for f in a.factors.split(",")]

    jobs = []
    for site, lat, lon, elev, dates, eye, cam in CLUSTER:
        for d in dates:
            for f in factors:
                jobs.append((site, lat, lon, elev, d, f))
    results = {}
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(run_one, *j): j for j in jobs}
        for fut in cf.as_completed(futs):
            site, _, _, _, d, f = futs[fut]
            try:
                dep = fut.result()
            except Exception as e:
                dep = None
                print(f"  ERROR {site} {d} f{f}: {e}", flush=True)
            results[(site, d, f)] = dep
            print(f"  {site} {d} f{f:g}: {dep}", flush=True)

    # Cluster residual per factor: mean over sites of (site-mean khayt
    # minus the site's naked-eye target); Aswan uses the camera target
    # against the khayt output as an upper anchor with half weight.
    print("\nRECAL,factor,rms_residual,detail")
    best = None
    for f in factors:
        terms = []
        for site, _, _, _, dates, eye, cam in CLUSTER:
            deps = [results.get((site, d, f)) for d in dates]
            deps = [x for x in deps if x is not None]
            if not deps:
                continue
            mean = sum(deps) / len(deps)
            target = eye if eye is not None else cam
            w = 1.0 if eye is not None else 0.5
            terms.append((w, mean - target, site, mean))
        if not terms:
            continue
        rms = (sum(w * r * r for w, r, _, _ in terms) / sum(w for w, _, _, _ in terms)) ** 0.5
        detail = " ".join(f"{s}:{m:.2f}({r:+.2f})" for _, r, s, m in terms)
        print(f"RECAL,{f:g},{rms:.3f},{detail}")
        if best is None or rms < best[1]:
            best = (f, rms)
    print(f"\nBEST factor on final engine: {best[0]:g} (weighted RMS {best[1]:.3f} deg)")
    (CACHE / "RECAL_SUMMARY.json").write_text(json.dumps(
        {"best_factor": best[0], "rms": best[1],
         "ladder": {str(f): {f"{s}_{d}": results.get((s, d, f))
                             for s, _, _, _, ds, _, _ in CLUSTER for d in ds}
                    for f in factors}}, indent=1))


if __name__ == "__main__":
    main()
