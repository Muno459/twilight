#!/usr/bin/env python3
"""Reproduce the deep-referee verdict for the 1d cells from the 128-seed
twilight caches + cached MYSTIC refs, WITHOUT waiting for the (uncached,
slow) field-tau3 cell. Mirrors compare_deep()'s exact gate logic:
  band = 3*sqrt(se_tw^2 + se_my^2) + 0.05*my
  LOW-POWER if band > 0.5*my; else PASS if |tw-my|<=band; else FAIL.
This drives paper1 tab:variance (tau*=3 zenith=1d) + the tau*=1 summary.
"""
import json, math, pathlib, statistics

DEEP = pathlib.Path(__file__).resolve().parent.parent / "validation/deep"

def mean_se(vals):
    n = len(vals); m = sum(vals)/n
    if n < 2: return m, 0.0
    sd = math.sqrt(sum((x-m)**2 for x in vals)/(n-1))
    return m, sd/math.sqrt(n)

def tw_cache(path, tau):
    d = json.loads((DEEP / f"twilight_{path}_tau{tau:g}.json").read_text())
    return {tuple(float(x) for x in k.split(":")): mean_se(v) for k, v in d["rows"].items()}, d["seeds"]

def read_spc(tag, name):
    p = DEEP / tag / name
    if not p.exists(): return None
    parts = p.read_text().split()
    return float(parts[4]) * 1e-3 if len(parts) >= 5 else None

def mystic(tau, sza, wl):
    tag = f"deep_mystic_tau{tau:g}_sza{sza:g}_wl{wl:g}"
    return read_spc(tag, "mc.rad.spc"), read_spc(tag, "mc.rad.std.spc")

for tau in (1.0, 3.0):
    tw, seeds = tw_cache("1d", tau)
    print(f"\n=== tau*={tau:g}  1d (zenith)  [{seeds} seeds] ===")
    print(f"{'SZA/nm':<10}{'tw/my':>9}{'band/my':>9}{'verdict':>12}   my")
    for sza in (101.0, 103.0):
        for wl in (450.0, 550.0, 650.0):
            m, se = tw[(sza, wl)]
            my, myse = mystic(tau, sza, wl)
            myse = myse or 0.0
            band = 3.0*math.sqrt(se*se + myse*myse) + 0.05*my
            if band > 0.5*my: v = "LOW-POWER"
            elif abs(m-my) <= band: v = "PASS"
            else: v = "FAIL"
            print(f"{sza:.0f}/{wl:.0f}".ljust(10) +
                  f"{m/my:>9.2f}{band/my:>9.2f}{v:>12}   {my:.3e}+-{myse:.1e}")
