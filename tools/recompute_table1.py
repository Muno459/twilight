#!/usr/bin/env python3
"""Recompute paper Table 1 (tab:campaigns) + its medians from the
current-engine f=56 runs. One uniform rule, one engine build.

Engine value per row = seasonal MEAN of the modality-matched depression:
  eye  -> khayt al-abyad ; inst -> legacy true-dawn.
Paired rows (Tubruq sea, Aswan 1909) reuse the same clear-sky run as
their sibling and are scored against a different observed value, exactly
as the shipped table does (both Tubruq rows shared one Engine number).
Residual = Engine - observed_central.

Medians follow the paper's definitions: 14 scored rows (Wadi Al-Natrun
'inside' and Sana'a 'bracket' excluded); independent = scored minus the
three dagger calibration rows; per-modality eye/instrument.
"""
import pathlib, re, statistics, sys

RUNS = pathlib.Path(__file__).resolve().parent.parent / "validation/criterion_runs"

def parse(site, date):
    f = RUNS / f"{site}_{date}.txt"
    if not f.exists(): return None
    t = f.read_text()
    kh = re.search(r"khayt al-abyad\).*?depression ([\d.]+)", t)
    lg = re.search(r"true dawn\).*?depression ([\d.]+)", t)
    return (float(kh.group(1)) if kh else None,
            float(lg.group(1)) if lg else None)

def mean_mod(runs, mod):
    """runs: list of (site,date). Return seasonal mean of khayt (eye) or legacy (inst)."""
    vals = []
    for s, d in runs:
        p = parse(s, d)
        if p is None: print(f"  MISSING {s}_{d}", file=sys.stderr); continue
        v = p[0] if mod == "eye" else p[1]
        if v is not None: vals.append(v)
    return round(statistics.mean(vals), 2) if vals else None

R = lambda site, dates: [(site, d) for d in dates]

# (display, runs, modality, observed_central, dagger?, scored?)
ROWS = [
    ("Riyadh desert (KACST)", R("riyadh_desert", ["2004-01-15","2004-04-15","2004-07-15","2004-10-15"]), "eye", 14.6,  True,  True),
    ("Hail",                  R("hail", ["2014-10-15","2015-01-15","2015-04-15"]),                        "eye", 14.01, True,  True),
    ("Aswan 2016, camera",    R("aswan", ["2016-01-12","2016-01-14","2016-01-16"]),                       "inst",14.90, True,  True),
    ("North Sinai",           R("sinai", ["2011-01-15","2011-04-15","2011-10-15"]),                       "eye", 14.61, False, True),
    ("Fayum",                 R("fayum", ["2018-12-09","2018-12-10","2019-12-19"]),                       "eye", 14.8,  False, True),
    ("Matrouh",               R("matrouh", ["2012-01-15","2012-04-15","2012-10-15"]),                     "eye", 14.5,  False, True),
    ("Kottamia",              R("kottamia", ["2010-01-15","2010-04-15","2010-10-15"]),                    "eye", 14.66, False, True),
    ("Bahariya",              R("bahariya", ["2007-01-15","2007-04-15","2007-10-15"]),                    "eye", 14.6,  False, True),
    ("Wadi Al-Natrun",        R("wadi_natrun", ["2017-01-15","2017-04-15","2017-10-15"]),                 "eye", None,  False, False),
    ("Tubruq (desert)",       R("tubruq", ["2010-01-15","2010-04-15","2010-07-15"]),                      "eye", 14.68, False, True),
    ("Tubruq (sea)",          R("tubruq", ["2010-01-15","2010-04-15","2010-07-15"]),                      "eye", 13.455,False, True),
    ("Assiut (agricultural)", R("assiut", ["2013-01-15","2013-04-15","2013-10-15"]),                      "eye", 13.665,False, True),
    ("Sana'a, 2200 m",        R("sanaa", ["2003-11-23","2003-11-24","2003-11-28"]),                       "eye", None,  False, False),
    ("Depok, SQM knee",       R("depok", ["2015-06-15","2015-07-01","2015-07-15"]),                       "inst",14.0,  False, True),
    ("Malaysia SQM",          R("kuala_lipis",["2007-11-10","2007-12-29","2008-02-09"])+R("merang",["2007-05-08"])+R("port_klang",["2008-04-07"]), "inst", 14.19, False, True),
    ("Aswan 1909, camera",    R("aswan", ["2016-01-12","2016-01-14","2016-01-16"]),                       "inst",14.25, False, True),
]

print(f"{'Site':<24}{'Engine':>8}{'Obs':>9}{'Resid':>8}  mod  {'scored':>6}")
print("-"*62)
scored, indep, eye, inst = [], [], [], []
for disp, runs, mod, obs, dagger, is_scored in ROWS:
    eng = mean_mod(runs, mod)
    resid = round(eng - obs, 2) if (eng is not None and obs is not None) else None
    tag = "dagger" if dagger else ("scored" if is_scored else "unscored")
    print(f"{disp:<24}{str(eng):>8}{str(obs):>9}{str(resid):>8}  {mod:<4} {tag:>8}")
    if is_scored and resid is not None:
        scored.append(resid)
        if not dagger:
            indep.append(resid)
            # paper's eye/instrument medians are over the INDEPENDENT set
            (eye if mod == "eye" else inst).append(resid)

med = lambda x: round(statistics.median([abs(v) for v in x]), 3) if x else None
print("-"*62)
print(f"eye ({len(eye)}):         median |resid| = {med(eye)}")
print(f"instrument ({len(inst)}):  median |resid| = {med(inst)}")
print(f"pooled ({len(scored)}):     median |resid| = {med(scored)}")
print(f"independent ({len(indep)}): median |resid| = {med(indep)}")
