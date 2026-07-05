#!/usr/bin/env python3
"""Analysis for RESULTS_CRITERION_SITES.md section 9 (background modeling).

Reads the raw run outputs cached in this directory (pristine sweep +
--background sweep from tools/criterion_sites.py) and prints:
  1. the Birmingham skyglow veil response curve,
  2. the Birmingham before/after delta table + winter/full RMS,
  3. Tubruq sea/desert albedo table,
  4. Assiut aerosol table.
Gitignored artifact; regenerate tables with `python3 analyze_background.py`.
"""

import math
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
KHAYT = re.compile(r"Fajr \(khayt al-abyad\):\s+\S+\s+\(SZA [\d.]+.?, depression ([\d.]+)")
LEGACY = re.compile(r"Fajr \(true dawn\):\s+\S+\s+\(SZA [\d.]+.?, depression ([\d.]+)")
GLOW = re.compile(r"Skyglow:\s+(.*)")


def khayt(name):
    """khayt fajr depression from a cached raw output, or None."""
    p = HERE / name
    if not p.exists():
        return None
    m = KHAYT.search(p.read_text())
    return float(m.group(1)) if m else None


def veil(radiance_nw):
    """Additive skyglow veil at the khayt band (3 deg elev) [cd/m^2]."""
    lift = 1.0 + 8.0 * (1.0 - math.cos(math.radians(87.0))) ** 2.2
    return 0.092e-3 * radiance_nw ** 0.72 * lift


# OpenFajr panel values (RESULTS section 3).
PANEL = {
    "2015-01-11": 13.0, "2015-01-24": 12.9, "2015-02-22": 13.7,
    "2015-02-27": 13.0, "2015-04-20": 15.0, "2015-04-27": 13.7,
    "2015-05-13": 13.0, "2015-05-27": 13.0, "2015-06-07": 12.6,
    "2015-06-22": 12.5, "2015-06-30": 12.3, "2015-07-06": 13.0,
    "2015-07-18": 13.8, "2015-08-16": 14.3, "2015-09-06": 14.9,
    "2015-09-23": 14.6, "2015-11-13": 13.5, "2015-12-10": 12.9,
    "2015-12-25": 12.6,
}
WINTER = ["2015-01-11", "2015-01-24", "2015-02-22", "2015-02-27",
          "2015-11-13", "2015-12-10", "2015-12-25"]

TUBRUQ_DATES = ["2010-01-15", "2010-04-15", "2010-07-15"]
ASSIUT_DATES = ["2013-01-15", "2013-04-15", "2013-10-15"]


def rms(deltas):
    ds = [d for d in deltas if d is not None]
    return math.sqrt(sum(d * d for d in ds) / len(ds)) if ds else float("nan")


def birmingham_curve():
    print("== 1. Birmingham veil response curve (khayt fajr depression) ==")
    print("config           veil_cd_m2   Jan24   Jun22   Dec10")
    rows = [
        ("pristine", 0.0, "birmingham_{d}.txt"),
        # Corrected-units emulation (R_emul = R_true/14677): the veil
        # column is the PHYSICAL ring veil these runs apply.
        ("fixB4 (R=2)", veil(2), "birmingham_b4_fix_{d}_skyglow_radiance_1.363e-4.txt"),
        ("fixB5 (R=6)", veil(6), "birmingham_b5_fix_{d}_skyglow_radiance_4.088e-4.txt"),
        ("fixB6 (R=15)", veil(15), "birmingham_b6_fix_{d}_skyglow_radiance_1.0221e-3.txt"),
        ("fixAtlas 3.6mcd", 3.595e-3 * 8.108, "birmingham_atlas_fix_{d}_skyglow_radiance_0.011086.txt"),
        # 1000x-bugged veils (kept as extreme-veil response points):
        ("bug R=0.2", veil(0.2) * 1000, "birmingham_veil_r02_{d}_skyglow_radiance_0.2.txt"),
        ("bug R=0.4", veil(0.4) * 1000, "birmingham_veil_r04_{d}_skyglow_radiance_0.4.txt"),
        ("bug R=0.8", veil(0.8) * 1000, "birmingham_veil_r08_{d}_skyglow_radiance_0.8.txt"),
        ("bug R=2", veil(2) * 1000, "birmingham_veil_r2_{d}_skyglow_radiance_2.txt"),
        ("bug R=6", veil(6) * 1000, "birmingham_veil_r6_{d}_skyglow_radiance_6.txt"),
        ("bug B6 R=15", veil(15) * 1000, "birmingham_glow_{d}_skyglow_bortle_6.txt"),
        ("bug atlas", 3.595 * 8.108, "birmingham_atlas_{d}_skyglow.txt"),
    ]
    for label, v, pat in rows:
        vals = [khayt(pat.format(d=d))
                for d in ["2015-01-24", "2015-06-22", "2015-12-10"]]
        print(f"{label:16s} {v:10.2e}  " +
              "  ".join(f"{x:5.2f}" if x else "  -  " for x in vals))
    print()


def birmingham_table(glow_pat, label, dates=None):
    dates = dates or list(PANEL)
    print(f"== Birmingham {label}: per-date deltas ==")
    print("date        panel  pristine  bg-model  old_delta  new_delta")
    old_d, new_d, old_w, new_w = [], [], [], []
    for d in dates:
        p = PANEL[d]
        pri = khayt(f"birmingham_{d}.txt")
        new = khayt(glow_pat.format(d=d))
        od = (pri - p) if pri is not None else None
        nd = (new - p) if new is not None else None
        old_d.append(od)
        new_d.append(nd)
        if d in WINTER:
            old_w.append(od)
            new_w.append(nd)
        print(f"{d}  {p:5.1f}  {pri if pri else float('nan'):7.2f}  "
              f"{new if new else float('nan'):7.2f}  "
              f"{od if od is not None else float('nan'):+8.2f}  "
              f"{nd if nd is not None else float('nan'):+8.2f}")
    print(f"winter RMS: pristine {rms(old_w):.2f} -> {label} {rms(new_w):.2f}")
    print(f"all-row RMS ({len([d for d in new_d if d is not None])} dates): "
          f"pristine {rms(old_d):.2f} -> {label} {rms(new_d):.2f}")
    print(f"mean delta: pristine {sum(d for d in old_d if d is not None)/len(old_d):+.2f} -> "
          f"{label} {sum(d for d in new_d if d is not None)/max(1,len([d for d in new_d if d is not None])):+.2f}")
    print()


def tubruq():
    print("== 3. Tubruq: sea vs desert background (khayt fajr) ==")
    print("config             Jan15   Apr15   Jul15   mean")
    configs = [
        ("pristine a=0.15", "tubruq_{d}.txt"),
        ("sea a=0.06", "tubruq_sea_{d}_albedo_0.06.txt"),
        ("sea+marine aer", "tubruq_sea_marine_{d}_albedo_0.06_aerosol_maritime-clean.txt"),
        ("desert a=0.30", "tubruq_desert_alb_{d}_albedo_0.30.txt"),
    ]
    for label, pat in configs:
        vals = [khayt(pat.format(d=d)) for d in TUBRUQ_DATES]
        got = [v for v in vals if v is not None]
        mean = sum(got) / len(got) if got else float("nan")
        print(f"{label:18s} " +
              "  ".join(f"{v:5.2f}" if v else "  -  " for v in vals) +
              f"  {mean:5.2f}")
    print("observed: sea bg 13.43-13.48, desert bg 14.66-14.70")
    print()


def assiut():
    print("== 4. Assiut agricultural (khayt fajr) ==")
    print("config             Jan15   Apr15   Oct15   mean")
    configs = [
        ("pristine", "assiut_{d}.txt"),
        ("cont-clean .05", "assiut_agri_clean_{d}_aerosol_continental-clean.txt"),
        ("cont-avg .12", "assiut_aer_seasons_{d}_aerosol_continental-average.txt"),
    ]
    for label, pat in configs:
        vals = [khayt(pat.format(d=d)) for d in ASSIUT_DATES]
        got = [v for v in vals if v is not None]
        mean = sum(got) / len(got) if got else float("nan")
        print(f"{label:18s} " +
              "  ".join(f"{v:5.2f}" if v else "  -  " for v in vals) +
              f"  {mean:5.2f}")
    print("observed: 13.665 (2012-2014 campaign mean)")
    g = HERE / "assiut_glow_2013-01-15_skyglow.txt"
    if g.exists():
        for m in GLOW.finditer(g.read_text()):
            print("assiut skyglow probe:", m.group(1))
        v = khayt("assiut_glow_2013-01-15_skyglow.txt")
        print(f"assiut atlas-skyglow khayt (city-pixel veil, Jan): {v}")
    print()


DBG = re.compile(
    r"khayt sza\s+([\d.]+): spread\s+([\d.e+-]+) \(\+-([\d.e+-]+)\) "
    r"central\s+([\d.e+-]+) ahmar\s+([\d.e+-]+)")


def margin_curves():
    """Aligned pristine-vs-veiled spread margin curves (TWILIGHT_KHAYT_DEBUG)."""
    def load(name):
        p = HERE / name
        if not p.exists() or not p.stat().st_size:
            return {}
        return {float(m.group(1)): (float(m.group(2)), float(m.group(4)))
                for m in DBG.finditer(p.read_text())}
    a = load("dbg_pristine_dec10.err")
    b = load("dbg_r08_dec10.err")
    if not a:
        return
    print("== Dec 10 spread-margin curves (dep = SZA-90) ==")
    print("sza     dep   spread_pristine  spread_r08   ratio   central_pristine")
    for sza in sorted(set(a) | set(b)):
        pa = a.get(sza)
        pb = b.get(sza)
        ratio = (pa[0] / pb[0]) if pa and pb and pb[0] > 0 else float("nan")
        print(f"{sza:6.2f} {sza-90:5.2f}  {pa[0] if pa else float('nan'):12.4g}"
              f"  {pb[0] if pb else float('nan'):12.4g}  {ratio:6.2f}"
              f"  {pa[1] if pa else float('nan'):12.4g}")
    print()


if __name__ == "__main__":
    margin_curves()
    birmingham_curve()
    dates10 = ["2015-01-11", "2015-01-24", "2015-02-22", "2015-02-27",
               "2015-04-20", "2015-06-22", "2015-09-23", "2015-11-13",
               "2015-12-10", "2015-12-25"]
    birmingham_table("birmingham_atlas_{d}_skyglow.txt",
                     "measured atlas, BUGGED 1000x veil", dates10)
    birmingham_table(
        "birmingham_atlas_fix_{d}_skyglow_radiance_0.011086.txt",
        "measured atlas, corrected units")
    birmingham_table(
        "birmingham_aer_cavg_{d}_aerosol_continental-average.txt",
        "continental-average aerosol")
    tubruq()
    assiut()
