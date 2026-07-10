#!/usr/bin/env python3
"""Write validation/deep_regime_results.csv in the exact compare_deep()
format from the cached 128-seed twilight caches + cached MYSTIC refs,
covering every cell whose twilight cache exists (1d tau*=1,3 all wl;
field tau*=1 wl550; field tau*=3 if/when its cache lands). This gives the
correct 128-seed 1d rows now, without blocking on the slow, non-critical
field-tau3 cell. The running orchestrator will later overwrite with the
identical 1d rows plus the field-tau3 pair.
"""
import json, math, pathlib, csv, sys

DEEP = pathlib.Path(__file__).resolve().parent.parent / "validation/deep"
OUT = pathlib.Path(__file__).resolve().parent.parent / "validation/deep_regime_results.csv"
SZAS = [101.0, 103.0]; TAUS = [1.0, 3.0]; WLS = [450.0, 550.0, 650.0]; FLOOR = 0.05

def mean_se(vals):
    n = len(vals); m = sum(vals)/n
    if n < 2: return m, 0.0
    sd = math.sqrt(sum((x-m)**2 for x in vals)/(n-1))
    return m, sd/math.sqrt(n)

def tw_cache(path, tau):
    p = DEEP / f"twilight_{path}_tau{tau:g}.json"
    if not p.exists(): return None
    d = json.loads(p.read_text())
    return {tuple(float(x) for x in k.split(":")): mean_se(v) for k, v in d["rows"].items()}

def read_spc(tag, name):
    p = DEEP / tag / name
    if not p.exists(): return None
    parts = p.read_text().split()
    return float(parts[4]) * 1e-3 if len(parts) >= 5 else None

def mystic(tau, sza, wl):
    tag = f"deep_mystic_tau{tau:g}_sza{sza:g}_wl{wl:g}"
    rad = read_spc(tag, "mc.rad.spc"); se = read_spc(tag, "mc.rad.std.spc")
    nph = 300000000
    done = DEEP / tag / "case.done"
    if done.exists():
        for l in done.read_text().splitlines():
            if l.startswith("mc_photons"): nph = int(l.split()[1])
    return rad, se, nph

tw = {(t, p): tw_cache(p, t) for t in TAUS for p in ("1d", "field")}
n_pass = n_fail = n_low = 0
rows = []
for ts in TAUS:
    for sza in SZAS:
        for wl in WLS:
            my, myse, nph = mystic(ts, sza, wl)
            for path in ("1d", "field"):
                tbl = tw[(ts, path)]
                if tbl is None or (sza, wl) not in tbl:
                    continue
                m, se = tbl[(sza, wl)]
                if my is None or my <= 0:
                    continue
                myse = myse or 0.0
                band = 3.0*math.sqrt(se*se + myse*myse) + FLOOR*my
                if band > 0.5*my: v = "LOW-POWER"; n_low += 1
                elif abs(m-my) <= band: v = "PASS"; n_pass += 1
                else: v = "FAIL"; n_fail += 1
                rows.append([ts, sza, wl, path, f"{m:.4e}", f"{se:.2e}",
                             f"{my:.4e}", f"{myse:.2e}", f"{nph:.0e}",
                             f"{m/my:.4f}", f"{band/my:.4f}", v])

if "--write" in sys.argv:
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau_star","sza","wl","path","tw","tw_se","mystic",
                    "mystic_se","mystic_photons","tw_over_mystic","band_over_mystic","verdict"])
        w.writerows(rows)
    print(f"WROTE {OUT} ({len(rows)} rows)")
print(f"gate: {n_pass} PASS / {n_fail} FAIL / {n_low} LOW-POWER  ({len(rows)} cells)")
for r in rows:
    print(f"  tau*={r[0]:g} sza{r[1]:g} {r[2]:g} {r[3]:5s}: {r[9]}x band {r[10]} [{r[11]}]")
