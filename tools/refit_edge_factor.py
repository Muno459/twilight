#!/usr/bin/env python3
"""Re-fit the khayt appearance edge factor f from the CACHED ladder runs,
for an arbitrary vector of observed targets. No new GPU runs: parse
dep(f) from every cached edge_factor_v1 run, fit a robust log-linear
curve dep = a + b*log10(f) per site, invert to the implied f per site,
and RMS-minimize the out-of-sample f over the six test-desert sites.

Purpose: quantify how far the OOS optimum moves if the Kottamia-cluster
targets are mean+2SD upper bounds (as shipped) vs central means. The
whole 70->56 recalibration hinges on those six targets.

Usage:
  python3 tools/refit_edge_factor.py                 # shipped targets + downshift sweep
  python3 tools/refit_edge_factor.py --targets kottamia=14.2,tubruq=14.1,...
"""
import re, math, pathlib, sys, argparse

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / "validation/criterion_runs/edge_factor"
DEP = re.compile(r"khayt al-abyad\).*?depression ([\d.]+)", re.S)

# OOS test-desert sites (NOT in the Mecca calibration cluster) with the
# SHIPPED observed targets (the ones under mean-vs-2SD suspicion).
OOS = {
    "sinai":    14.61,
    "fayum":    14.80,
    "matrouh":  14.50,
    "kottamia": 14.66,
    "bahariya": 14.60,
    "tubruq":   14.68,
}
# Calibration-cluster desert anchors (shown for reference; Hail's 14.01
# is a confirmed central mean, Riyadh 14.60 a mean).
CAL = {"riyadh_desert": 14.60, "hail": 14.01}


def ladder(site):
    """Return sorted [(f, dep)] parsed from every cached run for the site."""
    pts = []
    for p in sorted(CACHE.glob(f"{site}_*_f*.txt")):
        m = re.search(rf"{re.escape(site)}_.*_f([0-9.]+)\.txt$", p.name)
        if not m:
            continue
        f = float(m.group(1))
        dm = DEP.search(p.read_text())
        if dm:
            pts.append((f, float(dm.group(1))))
    return sorted(pts)


def loglinfit(pts):
    """OLS fit dep = a + b*log10(f); returns (a, b, resid_rms, n)."""
    xs = [math.log10(f) for f, _ in pts]
    ys = [d for _, d in pts]
    n = len(xs)
    mx = sum(xs) / n; my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    rr = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return a, b, rr, n


FITS = {}
for s in list(OOS) + list(CAL):
    pts = ladder(s)
    FITS[s] = (loglinfit(pts), pts)


def dep_at(site, f):
    (a, b, _, _), _ = FITS[site]
    return a + b * math.log10(f)


def implied_f(site, target):
    (a, b, _, _), _ = FITS[site]
    return 10 ** ((target - a) / b)


def oos_opt(targets, sites):
    """RMS-minimizing f over the given sites for the given target dict."""
    best = None
    f = 40.0
    while f <= 90.0001:
        rs = [dep_at(s, f) - targets[s] for s in sites]
        rms = math.sqrt(sum(r * r for r in rs) / len(rs))
        bias = sum(rs) / len(rs)
        if best is None or rms < best[1]:
            best = (f, rms, bias)
        f += 0.25
    return best


def report(targets, label):
    sites = list(OOS)
    b = oos_opt(targets, sites)
    print(f"\n=== {label} ===")
    print(f"{'site':14s} {'target':>7s} {'implied_f':>10s}  fit dep=a+b*log10f (b, residRMS, n)")
    impls = []
    for s in sites:
        (a, bb, rr, n), _ = FITS[s]
        imf = implied_f(s, targets[s])
        impls.append(imf)
        print(f"{s:14s} {targets[s]:7.2f} {imf:10.1f}  b={bb:+.3f} rr={rr:.3f} n={n}")
    print(f"  implied-f mean {sum(impls)/len(impls):.1f}  spread {min(impls):.1f}-{max(impls):.1f}")
    print(f"  >>> OOS RMS-optimal f = {b[0]:.1f}  (RMS {b[1]:.3f}, bias {b[2]:+.3f})")
    return b[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="")
    a = ap.parse_args()

    print("Cal-cluster anchors (reference): Hail 14.01 (mean), Riyadh 14.60 (mean)")
    for s in CAL:
        print(f"  {s:14s} implied_f @ {CAL[s]:.2f} = {implied_f(s, CAL[s]):.1f}")

    report(dict(OOS), "SHIPPED targets (possible upper bounds) -> current basis")

    if a.targets:
        t = dict(OOS)
        for kv in a.targets.split(","):
            k, v = kv.split("=")
            t[k.strip()] = float(v)
        report(t, "USER-SUPPLIED targets (central means)")

    # Sensitivity: uniform downshift of all six OOS targets (upper-bound
    # -> mean) by delta.
    print("\n=== sensitivity: uniform downshift delta (upper-bound -> mean) ===")
    print(f"{'delta':>6s} {'OOS f*':>8s}")
    d = 0.0
    while d <= 0.61:
        t = {s: OOS[s] - d for s in OOS}
        print(f"{d:6.2f} {oos_opt(t, list(OOS))[0]:8.1f}")
        d += 0.1
