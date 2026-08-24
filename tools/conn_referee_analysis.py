#!/usr/bin/env python3
"""Gate the connection-estimator field re-referee against MYSTIC.

The published deep-tier field rows are the PRE-connection analog
estimator at 1024 (tau*=1) and 512 (tau*=3) seeds. The voxel-field
connection estimator (FIELD_CONNECTIONS_PLAN.md, merged 4ccd737) gated
the same cells at 128 seeds on the original verification box (G-FC-3).
This script gates a fresh 128-seed connection-arm campaign, run on this
machine, against both the cached MYSTIC references and (where replicas
exist) the inverse-variance pooled references, and prints the analog
rows alongside for the estimator-to-estimator comparison.

Input: a DEEPCSV dump from
    DEEP_PATH=field DEEP_SEEDS=128 DEEP_PHOTONS=16000 \
    DEEP_SZAS=101,103 DEEP_WLS=550 DEEP_TAU_STAR={1,3} \
    cargo test -p twilight-cpu --release deep_referee_runner -- --ignored --nocapture

Usage: python tools/conn_referee_analysis.py <deepcsv-file>
"""
import csv
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TIER = ROOT / "artifacts/2026-07_deep_closure/deep_tier/deep_regime_results.csv"
REPLICAS = ROOT / "artifacts/2026-07_deep_closure/mystic_replicas"
CACHED = ROOT / "validation/deep"


def read_spc(path):
    t = path.read_text().strip().splitlines()
    return float(t[0].split()[-1]) if t else None


def pooled_ref(tau, sza, wl):
    """Cached + replicas by inverse variance, in the tier's W scale."""
    ests = []
    c = CACHED / f"deep_mystic_tau{tau:g}_sza{sza:g}_wl{wl:g}"
    v, s = read_spc(c / "mc.rad.spc"), read_spc(c / "mc.rad.std.spc")
    if v is not None:
        ests.append((v, s))
    for d in sorted(REPLICAS.glob(f"tau{tau:g}_sza{sza:g}_wl{wl:g}_r*")):
        v, s = read_spc(d / "mc.rad.spc"), read_spc(d / "mc.rad.std.spc")
        if v is not None:
            ests.append((v, s))
    if len(ests) < 2:
        return None, None
    w = [1.0 / (s * s) for _, s in ests]
    m = sum(wi * v for wi, (v, _) in zip(w, ests)) / sum(w)
    return m * 1e-3, math.sqrt(1.0 / sum(w)) * 1e-3


def main():
    src = pathlib.Path(sys.argv[1])
    per = {}
    for line in src.read_text().splitlines():
        if not line.startswith("DEEPCSV,"):
            continue
        _, path, tau, seed, sza, wl, rad = line.split(",")
        per.setdefault((float(tau), float(sza), float(wl)), []).append(float(rad))

    tier = {}
    for r in csv.DictReader(open(TIER)):
        k = (float(r["tau_star"]), float(r["sza"]), float(r["wl"]), r["path"])
        tier[k] = r

    print(f"{'cell':<16}{'n':>5}{'conn mean':>12}{'se':>10}{'cv':>7}"
          f"{'vs cached':>11}{'band':>7}{'verdict':>9}"
          f"{'vs pooled':>11}{'analog(row)':>13}{'GFC3':>7}")
    print("-" * 118)
    gfc3 = {(1.0, 103.0): 1.185, (3.0, 103.0): 1.229}
    for (tau, sza, wl), vals in sorted(per.items()):
        n = len(vals)
        m = sum(vals) / n
        var = sum((x - m) ** 2 for x in vals) / n
        se = math.sqrt(var / n)
        cv = math.sqrt(var) / m if m > 0 else float("nan")
        row = tier.get((tau, sza, wl, "field"))
        my = float(row["mystic"])
        my_se = float(row["mystic_se"])
        ratio = m / my
        band = (3 * math.sqrt(se**2 + my_se**2) + 0.05 * my) / my
        constraining = band < 0.5
        ok = abs(ratio - 1) <= band
        verdict = ("PASS" if ok else "FAIL") if constraining else "LOW-POWER"
        pm, ps = pooled_ref(tau, sza, wl)
        vs_pooled = ""
        if pm:
            pband = (3 * math.sqrt(se**2 + ps**2) + 0.05 * pm) / pm
            pratio = m / pm
            pok = abs(pratio - 1) <= pband
            vs_pooled = f"{pratio:.3f}{'P' if pok else 'F'}"
        analog = float(row["tw_over_mystic"])
        g = gfc3.get((tau, sza), "")
        print(f"tau{tau:g} {sza:g} {wl:g} f {n:>5}{m:>12.4e}{se:>10.2e}"
              f"{cv:>7.2f}{ratio:>11.3f}{band:>7.3f}{verdict:>9}"
              f"{vs_pooled:>11}{analog:>13.3f}{g!s:>7}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
