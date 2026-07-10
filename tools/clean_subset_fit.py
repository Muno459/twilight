#!/usr/bin/env python3
"""Lock the edge factor on a CONSISTENT central-mean, clear-sky basis.

Uses the seasonal Mecca-cluster ladder (RECAL_SUMMARY.json, Riyadh/Hail/
Aswan across 3-4 dates each) plus Kottamia's single-date ladder
(edge_factor v1), inverting each site's confirmed CENTRAL MEAN to its
implied f, and reporting the RMS-optimal f over defensible subsets.

Confirmed central means (naked-eye true dawn / camera):
  Riyadh   14.58 (n=13, mean)                 -- pristine
  Aswan    14.90 (camera, n=5, mean)          -- pristine
  Kottamia 14.665 (n=4, mean; NOT an upper bound)  -- pristine, OOS
  Hail     14.01 (n=32 SELECTED good-vis, mean)    -- pristine, deep (1000 m)
Contaminated/heterogeneous (NOT clear-sky comparable, shown for context):
  Tubruq 13.14 (n=623 all-nights), Assiut 11.25 (light-polluted),
  Matrouh 13.41 (n=4), Bahariya 13.81, Sinai ~13.66 (median+sigma basis)
"""
import json, math, re, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
RECAL = json.loads((ROOT / "validation/criterion_runs/edge_factor_v2/RECAL_SUMMARY.json").read_text())
V1 = ROOT / "validation/criterion_runs/edge_factor"
DEP = re.compile(r"khayt al-abyad\).*?depression ([\d.]+)", re.S)

# --- seasonal-mean dep(f) per Mecca-cluster site from RECAL ladder ---
def seasonal_ladder(prefix):
    out = {}
    for f, sites in RECAL["ladder"].items():
        vals = [v for k, v in sites.items() if k.startswith(prefix) and v is not None]
        if vals:
            out[float(f)] = sum(vals) / len(vals)
    return dict(sorted(out.items()))

def v1_ladder(site):
    pts = {}
    for p in sorted(V1.glob(f"{site}_*_f*.txt")):
        m = re.search(rf"_f([0-9.]+)\.txt$", p.name)
        dm = DEP.search(p.read_text())
        if m and dm:
            pts[float(m.group(1))] = float(dm.group(1))
    return dict(sorted(pts.items()))

def loglinfit(ladder):
    xs = [math.log10(f) for f in ladder]; ys = list(ladder.values())
    n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
    sxx = sum((x-mx)**2 for x in xs); sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    b = sxy/sxx; a = my - b*mx
    return a, b

LAD = {
    "riyadh":   seasonal_ladder("riyadh_desert"),
    "hail":     seasonal_ladder("hail"),
    "aswan":    seasonal_ladder("aswan"),
    "kottamia": v1_ladder("kottamia"),
}
FIT = {s: loglinfit(l) for s, l in LAD.items()}
MEAN = {"riyadh": 14.58, "aswan": 14.90, "kottamia": 14.665, "hail": 14.01}

def dep_at(s, f):
    a, b = FIT[s]; return a + b*math.log10(f)
def implied_f(s):
    a, b = FIT[s]; return 10 ** ((MEAN[s]-a)/b)
def opt(sites):
    best = None; f = 40.0
    while f <= 90.0001:
        rs = [dep_at(s, f)-MEAN[s] for s in sites]
        rms = math.sqrt(sum(r*r for r in rs)/len(rs))
        if best is None or rms < best[1]: best = (f, rms)
        f += 0.25
    return best

print("Per-site implied f at confirmed CENTRAL MEAN (clear-sky basis):")
for s in ["riyadh", "aswan", "kottamia", "hail"]:
    print(f"  {s:9s} mean {MEAN[s]:.3f}  ->  implied f = {implied_f(s):5.1f}   "
          f"(ladder f-range {min(LAD[s]):.0f}-{max(LAD[s]):.0f}, {len(LAD[s])} pts)")
print()
for label, sites in [
    ("pristine sea-level trio {riyadh,aswan,kottamia}", ["riyadh", "aswan", "kottamia"]),
    ("pristine + Hail (adds 1000 m altitude site)",     ["riyadh", "aswan", "kottamia", "hail"]),
    ("Kottamia alone (independent OOS central mean)",    ["kottamia"]),
]:
    f, rms = opt(sites)
    print(f"  OOS RMS-optimal f = {f:5.1f}  (RMS {rms:.3f})   <- {label}")
