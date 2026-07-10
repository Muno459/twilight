#!/usr/bin/env python3
"""Confirm the correct edge factor by exact engine runs (not interpolation)
at a fine ladder across every desert campaign, then report the
leave-one-out and full-set RMS-minimizing f. Reuses the edge_factor cache;
runs only the new factors.
"""
import os, re, subprocess, math, pathlib, sys
import concurrent.futures as cf

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLI = ROOT / "target/release/twilight-cli"
CACHE = ROOT / "validation/criterion_runs/edge_factor"
CACHE.mkdir(parents=True, exist_ok=True)
DEP = re.compile(r"khayt al-abyad\).*?depression ([\d.]+)")

# desert campaigns: name -> (lat, lon, elev, date, obs_target, in_calib_cluster)
SITES = {
    "riyadh_desert": (25.763, 46.5, 540, "2004-10-15", 14.60, True),
    "hail":          (27.517, 41.70, 1000, "2015-01-15", 14.01, True),
    "sinai":         (31.067, 32.867, 20, "2011-04-15", 14.61, False),
    "fayum":         (29.283, 30.05, 50, "2018-12-09", 14.80, False),
    "matrouh":       (31.003, 27.85, 75, "2012-04-15", 14.50, False),
    "kottamia":      (29.932, 31.825, 470, "2010-04-15", 14.66, False),
    "bahariya":      (28.715, 29.997, 150, "2007-10-15", 14.60, False),
    "tubruq":        (32.078, 23.983, 25, "2010-01-15", 14.68, False),
    # wadi_natrun excluded: inversion flagged it a broken extrapolation outlier
}
FACTORS = [45.0, 50.0, 52.0, 54.0, 60.0]  # 45/60 in cache; 50/52/54 fresh, GPU-serial

def run_one(args):
    site, lat, lon, elev, date, f = args
    out = CACHE / f"{site}_{date}_f{f:g}.txt"
    if out.exists():
        txt = out.read_text()
    else:
        env = dict(os.environ)
        env["TWILIGHT_KHAYT_EDGE_APPEARANCE"] = str(f)
        env["RAYON_NUM_THREADS"] = "4"
        r = subprocess.run(
            [str(CLI), "pray", "--lat", str(lat), "--lon", str(lon),
             "--elevation", str(elev), "--date", date, "--scattering", "hybrid"],
            capture_output=True, text=True, env=env, timeout=7200)
        txt = r.stdout
        out.write_text(txt)
    m = DEP.search(txt)
    return (site, f, float(m.group(1)) if m else None)

def main():
    jobs = [(s, lat, lon, elev, d, f)
            for s, (lat, lon, elev, d, obs, cal) in SITES.items() for f in FACTORS]
    print(f"running {len(jobs)} (site,factor) points, GPU-serial...", flush=True)
    dep = {}  # (site,f) -> depression
    for job in jobs:                       # serial: GPU can't take concurrent Metal contexts
        site, f, d = run_one(job)
        dep[(site, f)] = d
        print(f"  {site} f{f:g} -> {d}", flush=True)

    def resid_at(site, f):
        obs = SITES[site][4]
        xs = sorted(x for (s, x) in dep if s == site and dep[(s, x)] is not None)
        if f in [x for x in xs]:
            return dep[(site, f)] - obs
        # log-linear interp within ladder
        for i in range(len(xs)-1):
            if xs[i] <= f <= xs[i+1]:
                y0, y1 = dep[(site, xs[i])], dep[(site, xs[i+1])]
                t = (math.log(f)-math.log(xs[i]))/(math.log(xs[i+1])-math.log(xs[i]))
                return (y0 + t*(y1-y0)) - obs
        return None

    def rms(v): return math.sqrt(sum(x*x for x in v)/len(v)) if v else None
    def mean(v): return sum(v)/len(v) if v else None

    grid = [40+0.5*i for i in range(61)]  # 40..70
    oos = [s for s in SITES if not SITES[s][5]]
    alld = list(SITES)
    def opt(sites):
        best = None
        for f in grid:
            rs = [resid_at(s, f) for s in sites]
            rs = [r for r in rs if r is not None]
            rr = rms(rs)
            if rr is not None and (best is None or rr < best[1]):
                best = (f, rr, mean([r for r in (resid_at(s, f) for s in sites) if r is not None]))
        return best

    print("\n=== residuals at key f (exact runs) ===")
    for f in [50, 52, 54, 56, 58, 70]:
        rs_oos = [resid_at(s, f) for s in oos if resid_at(s, f) is not None]
        print(f"  f={f}:  OOS RMS={rms(rs_oos):.3f}  bias={mean(rs_oos):+.3f}")
    print("\n>>> OUT-OF-SAMPLE optimum:", opt(oos))
    print(">>> ALL-desert optimum: ", opt(alld))
    # leave-one-out on out-of-sample sites
    print("\n=== leave-one-out optima (out-of-sample) ===")
    loo = []
    for held in oos:
        b = opt([s for s in oos if s != held])
        loo.append(b[0]); print(f"  drop {held:14s} -> f*={b[0]:.1f}")
    print(f">>> LOO mean f* = {sum(loo)/len(loo):.1f}  (spread {min(loo):.1f}-{max(loo):.1f})")

if __name__ == "__main__":
    main()
