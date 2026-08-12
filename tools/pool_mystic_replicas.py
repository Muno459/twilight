#!/usr/bin/env python3
"""Pool the fresh-seed MYSTIC replicas into the deep-tier reference.

The deep-regime gate is

    |m - r| <= 3*sqrt(se_m^2 + se_r^2) + 0.05*r

and in 15 of the 16 cells the REFEREE standard error se_r, not the
model's se_m, is the dominant term (measured: se_r/r 5.9-10.7 percent
against se_m/m 2.5-5.9 percent). Model seeds therefore buy almost
nothing; the reference is the limiting instrument.

For the four tau*=3 cells that carry two independent fresh-seed 1e9
replicas each (artifacts/2026-07_deep_closure/mystic_replicas/), the
cached run and its replicas are three independent estimates of the same
quantity. This script

  1. tests whether their scatter is consistent with their reported
     sigmas (a chi-square-per-degree-of-freedom check: MC standard
     errors in a heavy-tailed regime are exactly where reported sigmas
     go optimistic, and the paper's 3-sigma bands assume they do not);
  2. pools them by inverse variance into a tighter reference; and
  3. recomputes the gate against the pooled reference.

Native units are libRadtran mW/(m^2 sr nm); the deep tier CSV carries
the same numbers scaled by 1e-3. Ratios are unit-free, so pooling is
done in native units and compared against the CSV in its own scale.

Usage:  python tools/pool_mystic_replicas.py
"""
import csv
import math
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHED = ROOT / "validation/deep"
REPLICAS = ROOT / "artifacts/2026-07_deep_closure/mystic_replicas"
TIER = ROOT / "artifacts/2026-07_deep_closure/deep_tier/deep_regime_results.csv"

# libRadtran mW -> the deep tier CSV's W scale.
UNIT = 1e-3


def read_spc(path):
    """Last whitespace field of a one-line libRadtran .spc file."""
    txt = path.read_text().strip().splitlines()
    if not txt:
        return None
    return float(txt[0].split()[-1])


def cell_estimates(tau, sza, wl):
    """Cached reference plus every fresh-seed replica, as (value, se)."""
    out = []
    cached = CACHED / f"deep_mystic_tau{tau}_sza{sza}_wl{wl}"
    v = read_spc(cached / "mc.rad.spc")
    s = read_spc(cached / "mc.rad.std.spc")
    if v is not None:
        out.append(("cached", v, s))
    for d in sorted(REPLICAS.glob(f"tau{tau}_sza{sza}_wl{wl}_r*")):
        v = read_spc(d / "mc.rad.spc")
        s = read_spc(d / "mc.rad.std.spc")
        if v is not None:
            out.append((d.name.rsplit("_", 1)[-1], v, s))
    return out


def pool(estimates):
    """Inverse-variance pooled mean, its se, and the consistency chi2/dof."""
    w = [1.0 / (s * s) for _, _, s in estimates]
    mean = sum(wi * v for wi, (_, v, _) in zip(w, estimates)) / sum(w)
    se = math.sqrt(1.0 / sum(w))
    dof = len(estimates) - 1
    chi2 = sum(wi * (v - mean) ** 2 for wi, (_, v, _) in zip(w, estimates))
    return mean, se, chi2, dof


def main():
    rows = list(csv.DictReader(open(TIER)))
    # (tau, sza, wl) that actually carry replicas
    have = set()
    for d in REPLICAS.glob("tau*_sza*_wl*_r*"):
        m = re.match(r"tau(\d+)_sza(\d+)_wl(\d+)_r\d+", d.name)
        if m:
            have.add(m.groups())

    if not have:
        print("no replicas found", file=sys.stderr)
        return 1

    print("Replica consistency and inverse-variance pooling")
    print("=" * 78)
    print(f"{'cell':<22}{'n':>2}{'pooled':>12}{'se%':>7}{'was se%':>9}"
          f"{'chi2/dof':>10}{'max dev':>9}")
    print("-" * 78)

    pooled_ref = {}
    for tau, sza, wl in sorted(have):
        est = cell_estimates(tau, sza, wl)
        if len(est) < 2:
            continue
        mean, se, chi2, dof = pool(est)
        # largest deviation of any single estimate from the pooled mean,
        # in units of that estimate's own reported sigma
        maxdev = max(abs(v - mean) / s for _, v, s in est)
        cached_se_pct = 100.0 * est[0][2] / est[0][1]
        print(f"tau{tau} sza{sza} wl{wl}{'':<6}{len(est):>2}{mean*UNIT:>12.4e}"
              f"{100*se/mean:>7.1f}{cached_se_pct:>9.1f}"
              f"{chi2/dof:>10.2f}{maxdev:>9.1f}")
        pooled_ref[(tau, sza, wl)] = (mean * UNIT, se * UNIT)

    print()
    print("Gate against the pooled reference (1d and field paths)")
    print("=" * 78)
    print(f"{'cell':<26}{'ratio was':>10}{'ratio now':>10}"
          f"{'band was':>10}{'band now':>10}  verdict")
    print("-" * 78)

    improved = []
    for r in rows:
        key = (str(int(float(r["tau_star"]))), str(int(float(r["sza"]))),
               str(int(float(r["wl"]))))
        if key not in pooled_ref:
            continue
        ref, ref_se = pooled_ref[key]
        m, m_se = float(r["tw"]), float(r["tw_se"])
        band = (3.0 * math.sqrt(m_se ** 2 + ref_se ** 2) + 0.05 * ref) / ref
        ratio = m / ref
        ok = abs(ratio - 1.0) <= band
        constraining = band < 0.5
        verdict = ("PASS" if ok else "FAIL") if constraining else "LOW-POWER"
        cell = f"tau{key[0]} {key[1]} {key[2]} {r['path']}"
        print(f"{cell:<26}{float(r['tw_over_mystic']):>10.3f}{ratio:>10.3f}"
              f"{float(r['band_over_mystic']):>10.3f}{band:>10.3f}  {verdict}")
        improved.append((float(r["band_over_mystic"]), band))

    if improved:
        was = sum(a for a, _ in improved) / len(improved)
        now = sum(b for _, b in improved) / len(improved)
        print("-" * 78)
        print(f"mean band {was:.3f} -> {now:.3f} "
              f"({100*(1-now/was):.0f} percent tighter) over {len(improved)} cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
