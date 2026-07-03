#!/usr/bin/env python3
"""Multi-site validation sweep of the fajr/isha criterion.

Runs `twilight-cli pray` (release build) for each published observation
campaign site/date, parses the khayt and legacy depressions from stdout,
and writes one TSV row per run. Raw stdout is kept next to the TSV for
audit. Sites and dates mirror the campaigns documented in
validation/RESULTS_CRITERION_SITES.md.

Usage:
    python3 tools/criterion_sites.py [--only SITE] [--out DIR] [--jobs N]
    python3 tools/criterion_sites.py --background   # non-desert background runs

The default set (SITES) is the PRISTINE sweep documented in sections 1-8
of RESULTS_CRITERION_SITES.md; --background selects BACKGROUND_SITES,
the reference-sky background-modeling runs of section 9 (skyglow veil
curve, measured-atlas Birmingham, Tubruq sea/desert albedo, Assiut
agricultural aerosol). Cache is per raw-output file, so both modes share
one --out directory without collision.
"""

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import re
import subprocess
import sys

# TWILIGHT_CLI overrides the binary path: pin a snapshot binary so a
# parallel rebuild of target/release cannot change engines mid-sweep.
CLI = pathlib.Path(os.environ.get(
    "TWILIGHT_CLI",
    pathlib.Path(__file__).resolve().parent.parent / "target/release/twilight-cli"))

# name -> (lat, lon, elevation_m, [dates], extra_args)
SITES = {
    # OpenFajr Birmingham UK: camera at 52.44 N, -1.95 E (paper footnote 11),
    # ~150 m; panel year Dec 2014 - Dec 2015. Dates below are panel dates.
    "birmingham": (52.44, -1.95, 150, [
        "2015-01-11", "2015-01-24", "2015-02-22", "2015-02-27",
        "2015-04-20", "2015-04-27", "2015-05-13", "2015-05-27",
        "2015-06-07", "2015-06-22", "2015-06-30", "2015-07-06",
        "2015-07-18", "2015-08-16", "2015-09-06", "2015-09-23",
        "2015-11-13", "2015-12-10", "2015-12-25",
    ], []),
    # Mecca: engine calibration site, run for seasonal spread context.
    "mecca": (21.4225, 39.8262, 300, [
        "2015-03-20", "2015-06-21", "2015-12-21",
    ], []),
    # Hail, Saudi Arabia (Khalifa/Hassan/Taha 2018): 27 31 N, 41 42 E,
    # desert background, campaign 2014-2015 (~80 mornings, 32 selected).
    "hail": (27.517, 41.70, 1000, [
        "2014-10-15", "2015-01-15", "2015-04-15",
    ], []),
    # Aswan, Egypt calibrated camera (Adv. Space Res. 2024): Jan 12-16 2016.
    "aswan": (24.088, 32.90, 100, [
        "2016-01-12", "2016-01-14", "2016-01-16",
    ], []),
    # North Sinai, Egypt (Hassan/Issa/Mousa/Abdel-Hadi 2016): 31 04 N,
    # 32 52 E, desert background, campaign 2010-2012.
    "sinai": (31.067, 32.867, 20, [
        "2011-01-15", "2011-04-15", "2011-10-15",
    ], []),
    # Assiut, Egypt (same paper): 27 10 N, 31 10 E, agricultural background,
    # campaign 2012-2014.
    "assiut": (27.167, 31.167, 70, [
        "2013-01-15", "2013-04-15", "2013-10-15",
    ], []),
    # Bani-Hoshiesh, 30 km east of Sana'a, Yemen (Sultan 2004): 2200 m,
    # naked-eye team, Nov 23-28 2003 after heavy rain (very clear).
    "sanaa": (15.50, 44.40, 2200, [
        "2003-11-23", "2003-11-24", "2003-11-28",
    ], []),
    # Deep desert 170 km from Riyadh (Al Mostafa et al. 2005, KACST):
    # 25 45 41 N, 540 m (longitude misprinted in the source; deep desert
    # near Riyadh). One year, twice monthly, naked eye + Nikon D70.
    "riyadh_desert": (25.763, 46.5, 540, [
        "2004-01-15", "2004-04-15", "2004-07-15", "2004-10-15",
    ], []),
    # Wadi al Hitan, Fayum, Egypt (Rashed et al. 2022): 29 17 N, 30 03 E,
    # 50 m; group trips Dec 9-11 2018 and Dec 19 2019; SQM + naked eye.
    "fayum": (29.283, 30.05, 50, [
        "2018-12-09", "2018-12-10", "2019-12-19",
    ], []),
    # Matrouh, Egypt (Hassan et al. 2013/2014): 31 0.2 N, 27 51 E, 75 m,
    # sea-desert background, multi-year photoelectric + naked eye.
    "matrouh": (31.003, 27.85, 75, [
        "2012-01-15", "2012-04-15", "2012-10-15",
    ], []),
    # Kottamia, Egypt (Issa & Hassan 2011; Hassan et al. 2014): 29 55.9 N,
    # 31 49.5 E, 470 m, desert.
    "kottamia": (29.932, 31.825, 470, [
        "2010-01-15", "2010-04-15", "2010-10-15",
    ], []),
    # Bahariya oasis, Egypt (Issa & Hassan 2008 II/III; Hassan 2014):
    # 28 42.9 N, 29 59.82 E, 150 m, desert.
    "bahariya": (28.715, 29.997, 150, [
        "2007-01-15", "2007-04-15", "2007-10-15",
    ], []),
    # Wadi Al-Natrun, Egypt (Semeida & Hassan 2018): 30 30 N, 30 09 E,
    # 30 m, desert.
    "wadi_natrun": (30.50, 30.15, 30, [
        "2017-01-15", "2017-04-15", "2017-10-15",
    ], []),
    # Tubruq, Libya (Hassan et al. 2009 sea bg 2007-2008; Hassan &
    # Abdel-Hadi 2015 desert bg 2009-2013): 32 05 N, 23 59 E, 10-40 m.
    "tubruq": (32.078, 23.983, 25, [
        "2010-01-15", "2010-04-15", "2010-07-15",
    ], []),
    # Depok, Indonesia (Saksono & Fulazzaky 2020): 6 26 54 S, 106 48 08 E,
    # 50-140 m; SQM, 26 days June-July 2015.
    "depok": (-6.448, 106.802, 100, [
        "2015-06-15", "2015-07-01", "2015-07-15",
    ], []),
    # Malaysia SQM campaign (Abdel-Hadi & Hassan 2022, IJAA 12:7-29):
    # May 2007 - April 2008, SQM-LE. Dawn dates at their actual sites.
    "kuala_lipis": (4.183, 102.05, 75, [
        "2007-11-10", "2007-12-29", "2008-02-09",
    ], []),
    "merang": (5.517, 102.95, 42, ["2007-05-08"], []),
    "port_klang": (3.0, 101.40, 46, ["2008-04-07"], []),
    # Lembang/Bosscha, Indonesia (Herdiwijaya 2020, JPCS 1523 012007):
    # SQM first-departure criterion class, 2011-2018 moonless nights.
    "lembang": (-6.824, 107.617, 1310, [
        "2015-03-15", "2015-06-15", "2015-09-15",
    ], []),
    # Sensitivity checks.
    "hail_aer_desert": (27.517, 41.70, 1000, ["2015-01-15"],
                        ["--aerosol", "desert"]),
    # The OpenFajr camera site is inside a Bortle ~6 city; the khayt
    # reference sky there has an artificial floor that the pristine
    # clear-sky run lacks. Run every panel date with the skyglow model.
    "birmingham_glow": (52.44, -1.95, 150, [
        "2015-01-11", "2015-01-24", "2015-02-22", "2015-02-27",
        "2015-04-20", "2015-04-27", "2015-05-13", "2015-05-27",
        "2015-06-07", "2015-06-22", "2015-06-30", "2015-07-06",
        "2015-07-18", "2015-08-16", "2015-09-06", "2015-09-23",
        "2015-11-13", "2015-12-10", "2015-12-25",
    ], ["--skyglow", "--bortle", "6"]),
    # Assiut sits in the agricultural, populated Nile valley; the naked
    # eye result there (13.665) is ~1 deg shallower than the desert
    # cluster, which the campaign attributes to the background.
    "assiut_aer": (27.167, 31.167, 70, ["2013-01-15"],
                   ["--aerosol", "continental-average"]),
    # (extended over the full campaign season in BACKGROUND_SITES below)
    # Fine scan for the decisive OpenFajr dates: near the June turnaround
    # the khayt crossing is cliff-shaped and quantizes to the 0.5 deg
    # grid; 0.25 deg halves that.
    "birmingham_fine": (52.44, -1.95, 150, [
        "2015-04-20", "2015-06-22", "2015-06-30", "2015-09-06",
        "2015-12-25",
    ], ["--sza-step", "0.25"]),
}

# ── Reference-sky background modeling (RESULTS section 9) ──────────
#
# The three misses of the pristine sweep (Assiut agricultural, Tubruq
# sea background, Birmingham urban winter) share one variable: the
# campaign's reference ("black thread") sky is not the pristine desert
# the criterion was calibrated against. These runs model each site's
# actual background with the engine's existing hooks (--skyglow /
# --radiance, --albedo, --aerosol). Nothing here retunes the criterion.
#
# The skyglow veil at the khayt 3-deg band elevation is
#   veil [cd/m^2] = 0.092e-3 * R^0.72 * 8.11   (R in nW/cm^2/sr)
# (bortle.rs radiance->zenith fit x angular.rs Duriscoe lift A=8, B=2.2).
BIRMINGHAM_VEIL_DATES = ["2015-01-24", "2015-06-22", "2015-12-10"]
BACKGROUND_SITES = {
    # Step 1: the veil response curve. Three radiance levels between
    # pristine and the cached Bortle-6 bracket (15 nW), on the worst
    # winter miss (Jan 24), a dark winter row (Dec 10) and the June
    # persistent-twilight row that must not degrade (Jun 22).
    # NOTE: this r08-r6 ladder predates the unit-bug discovery below;
    # its applied ring veils are 0.64-2.7 cd/m^2 (1000x the intended
    # values). Kept as the bug's dose-response evidence; the
    # corrected-units ladder is the *_fix set at the bottom.
    "birmingham_veil_r08": (52.44, -1.95, 150, BIRMINGHAM_VEIL_DATES,
                            ["--skyglow", "--radiance", "0.8"]),
    "birmingham_veil_r2": (52.44, -1.95, 150, BIRMINGHAM_VEIL_DATES,
                           ["--skyglow", "--radiance", "2"]),
    "birmingham_veil_r6": (52.44, -1.95, 150, BIRMINGHAM_VEIL_DATES,
                           ["--skyglow", "--radiance", "6"]),
    # Step 2: the MEASURED skyglow (Lorenz 2024 atlas at the camera
    # pixel: 3.595 mcd/m^2 artificial zenith, Bortle 7 equivalent;
    # tile cached under data/skyglow). Winter + shoulder + June.
    "birmingham_atlas": (52.44, -1.95, 150, [
        "2015-01-11", "2015-01-24", "2015-02-22", "2015-02-27",
        "2015-04-20", "2015-06-22", "2015-09-23", "2015-11-13",
        "2015-12-10", "2015-12-25",
    ], ["--skyglow"]),
    # Spectrum sensitivity: 2015 Birmingham was still sodium-dominated
    # (LED PFI conversion mid-rollout); all-HPS spectrum, one date.
    "birmingham_atlas_led0": (52.44, -1.95, 150, ["2015-12-10"],
                              ["--skyglow", "--led-fraction", "0"]),
    # Step 3: the OTHER honest non-desert input for a UK winter - the
    # site's real aerosol. The desert calibration cluster is clean dry
    # air; UK boundary-layer aerosol (AERONET UK climatology AOD550
    # ~0.08-0.15) is bracketed by continental-clean (0.05) and
    # continental-average (0.12). At the khayt band's 3-deg elevation
    # the slant airmass ~15 turns AOD 0.12 into ~e^-1.8 dimming of the
    # dawn band, a first-order effect no veil can mimic. Winter + June
    # (the miss rows and the must-not-degrade rows); the corrected
    # skyglow (*_fix below) carries the full-year claim.
    "birmingham_aer_cavg": (52.44, -1.95, 150, [
        "2015-01-11", "2015-01-24", "2015-02-22", "2015-02-27",
        "2015-06-07", "2015-06-22", "2015-06-30",
        "2015-11-13", "2015-12-10", "2015-12-25",
    ], ["--aerosol", "continental-average"]),
    "birmingham_aer_cclean": (52.44, -1.95, 150, [
        "2015-01-11", "2015-01-24", "2015-02-22", "2015-02-27",
        "2015-06-22", "2015-11-13", "2015-12-10", "2015-12-25",
    ], ["--aerosol", "continental-clean"]),
    # Tubruq: same site, two campaign backgrounds. Sea-facing dawn
    # (2007-2008 campaign): Mediterranean albedo ~0.06, optional marine
    # boundary-layer aerosol along the dawn path. Desert-facing
    # (2009-2013): dry sand albedo ~0.30 (pristine run used 0.15).
    "tubruq_sea": (32.078, 23.983, 25,
                   ["2010-01-15", "2010-04-15", "2010-07-15"],
                   ["--albedo", "0.06"]),
    "tubruq_sea_marine": (32.078, 23.983, 25,
                          ["2010-01-15", "2010-04-15", "2010-07-15"],
                          ["--albedo", "0.06", "--aerosol", "maritime-clean"]),
    "tubruq_desert_alb": (32.078, 23.983, 25,
                          ["2010-01-15", "2010-04-15", "2010-07-15"],
                          ["--albedo", "0.30"]),
    # Assiut agricultural Nile valley: continental aerosol brackets
    # (clean AOD 0.05, average AOD 0.12) over the campaign season, plus
    # one measured-atlas skyglow probe (the coordinate is Assiut city).
    "assiut_aer_seasons": (27.167, 31.167, 70,
                           ["2013-01-15", "2013-04-15", "2013-10-15"],
                           ["--aerosol", "continental-average"]),
    "assiut_agri_clean": (27.167, 31.167, 70,
                          ["2013-01-15", "2013-04-15", "2013-10-15"],
                          ["--aerosol", "continental-clean"]),
    "assiut_glow": (27.167, 31.167, 70, ["2013-01-15"], ["--skyglow"]),
    # ── Corrected-units skyglow emulation ────────────────────────────
    # The khayt skyglow veil carries a 1000x unit bug: bortle.rs
    # radiance_to_zenith_luminance returns mcd/m^2 (its doc says so),
    # quick_estimate stores that number into SkyglowResult::
    # zenith_luminance whose doc says cd/m^2, and pipeline.rs adds
    # zenith_luminance * angular lift onto cd/m^2 patch luminances.
    # Verified empirically: TWILIGHT_KHAYT_DEBUG margin ratios at deep
    # night for --radiance 0.8 give a ~110x target inflation where
    # correct units predict 1.9x (validation/criterion_runs/dbg_*.err).
    # veil scales as R^0.72, so R_emul = R_true / 1000^(1/0.72)
    # = R_true / 14677 makes the buggy path compute the PHYSICALLY
    # CORRECT khayt veil (legacy spectral injection then ~vanishes;
    # khayt fajr vs the naked-eye/panel output is what these compare).
    # Atlas Birmingham 3.595 mcd artificial zenith (= R_true 162.7 nW)
    # -> R_emul 0.011086. Full panel year:
    "birmingham_atlas_fix": (52.44, -1.95, 150, [
        "2015-01-11", "2015-01-24", "2015-02-22", "2015-02-27",
        "2015-04-20", "2015-04-27", "2015-05-13", "2015-05-27",
        "2015-06-07", "2015-06-22", "2015-06-30", "2015-07-06",
        "2015-07-18", "2015-08-16", "2015-09-06", "2015-09-23",
        "2015-11-13", "2015-12-10", "2015-12-25",
    ], ["--skyglow", "--radiance", "0.011086"]),
    # Corrected-units Bortle brackets (true R = 2, 6, 15 nW):
    "birmingham_b4_fix": (52.44, -1.95, 150, BIRMINGHAM_VEIL_DATES,
                          ["--skyglow", "--radiance", "1.363e-4"]),
    "birmingham_b5_fix": (52.44, -1.95, 150, BIRMINGHAM_VEIL_DATES,
                          ["--skyglow", "--radiance", "4.088e-4"]),
    "birmingham_b6_fix": (52.44, -1.95, 150, BIRMINGHAM_VEIL_DATES,
                          ["--skyglow", "--radiance", "1.0221e-3"]),
}

# Every time line may carry an optional uncertainty token (e.g.
# "+-0.5min" or the +-<0.1min floor) and an optional "(+1d)" marker
# between the time and "(SZA ...". [^(\s]* tolerates the former; the
# (?:\(\+1d\)\s+)? group the latter (pre-existing lines used \S* which
# silently failed on the uncertainty suffix: the legacy_fajr TSV gap).
_TIME_LINE = r":\s+(\S+?)\s+(?:[^(\s]+\s+)?(?:\(\+1d\)\s+)?\(SZA ([\d.]+).?, depression ([\d.]+)"
PATTERNS = {
    "khayt_fajr": re.compile(r"Fajr \(khayt al-abyad\)" + _TIME_LINE),
    "false_dawn_from": re.compile(r"false dawn \(al-fajr al-kadhib\) visible from (\S+)"),
    "khayt_isha_ahmar": re.compile(r"Isha \(shafaq ahmar\)" + _TIME_LINE),
    "khayt_isha_abyad": re.compile(r"Isha \(shafaq abyad\)" + _TIME_LINE),
    "legacy_fajr": re.compile(r"Fajr \(true dawn\)" + _TIME_LINE),
    "legacy_isha_abyad": re.compile(r"Isha \(al-abyad\)" + _TIME_LINE),
    "legacy_isha_ahmar": re.compile(r"Isha \(al-ahmar\)" + _TIME_LINE),
}


def run_one(name, lat, lon, elev, date, extra, outdir):
    raw = outdir / f"{name}_{date}{'_' + '_'.join(a.strip('-') for a in extra) if extra else ''}.txt"
    if raw.exists() and "depression" in raw.read_text():
        out = raw.read_text()
    else:
        # Use --opt=value forms: clap rejects negative values passed as
        # separate tokens ("--lon -1.95" parses -1.95 as a flag).
        cmd = [str(CLI), "pray", f"--lat={lat}", f"--lon={lon}",
               f"--elevation={elev}", f"--date={date}"] + extra
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=2400)
        except subprocess.TimeoutExpired:
            return {"site": name, "date": date, "error": "timeout"}
        out = proc.stdout + proc.stderr
        raw.write_text(out)
        if proc.returncode != 0:
            return {"site": name, "date": date, "error": f"exit {proc.returncode}"}
    row = {"site": name, "date": date, "extra": " ".join(extra)}
    for key, pat in PATTERNS.items():
        m = pat.search(out)
        if m:
            if key == "false_dawn_from":
                row[key] = m.group(1)
            else:
                row[key + "_time"] = m.group(1)
                row[key + "_dep"] = float(m.group(3))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", help="restrict to site name(s)")
    ap.add_argument("--out", default=str(
        pathlib.Path(__file__).resolve().parent.parent
        / "validation/criterion_runs"))
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--background", action="store_true",
                    help="run the reference-sky background-modeling set "
                         "(RESULTS section 9) instead of the pristine sweep")
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    sites = BACKGROUND_SITES if args.background else SITES
    jobs = []
    for name, (lat, lon, elev, dates, extra) in sites.items():
        if args.only and name not in args.only:
            continue
        for date in dates:
            jobs.append((name, lat, lon, elev, date, extra))

    rows = []
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(run_one, *j, outdir): j for j in jobs}
        for fut in cf.as_completed(futs):
            row = fut.result()
            rows.append(row)
            print(json.dumps(row), flush=True)

    rows.sort(key=lambda r: (r["site"], r["date"]))
    tsv = outdir / ("criterion_runs_background.tsv" if args.background
                    else "criterion_runs.tsv")
    cols = ["site", "date", "khayt_fajr_dep", "legacy_fajr_dep",
            "khayt_isha_ahmar_dep", "khayt_isha_abyad_dep",
            "legacy_isha_ahmar_dep", "legacy_isha_abyad_dep",
            "khayt_fajr_time", "false_dawn_from", "extra"]
    with tsv.open("w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"\nwrote {tsv} ({len(rows)} rows)", file=sys.stderr)


if __name__ == "__main__":
    main()
