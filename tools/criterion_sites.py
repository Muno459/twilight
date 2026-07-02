#!/usr/bin/env python3
"""Multi-site validation sweep of the fajr/isha criterion.

Runs `twilight-cli pray` (release build) for each published observation
campaign site/date, parses the khayt and legacy depressions from stdout,
and writes one TSV row per run. Raw stdout is kept next to the TSV for
audit. Sites and dates mirror the campaigns documented in
validation/RESULTS_CRITERION_SITES.md.

Usage:
    python3 tools/criterion_sites.py [--only SITE] [--out DIR] [--jobs N]
"""

import argparse
import concurrent.futures as cf
import json
import pathlib
import re
import subprocess
import sys

CLI = pathlib.Path(__file__).resolve().parent.parent / "target/release/twilight-cli"

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
    # Fine scan for the decisive OpenFajr dates: near the June turnaround
    # the khayt crossing is cliff-shaped and quantizes to the 0.5 deg
    # grid; 0.25 deg halves that.
    "birmingham_fine": (52.44, -1.95, 150, [
        "2015-04-20", "2015-06-22", "2015-06-30", "2015-09-06",
        "2015-12-25",
    ], ["--sza-step", "0.25"]),
}

PATTERNS = {
    "khayt_fajr": re.compile(
        r"Fajr \(khayt al-abyad\):\s+(\S+)\s+\(SZA ([\d.]+).?, depression ([\d.]+)"),
    "false_dawn_from": re.compile(r"false dawn \(al-fajr al-kadhib\) visible from (\S+)"),
    "khayt_isha_ahmar": re.compile(
        r"Isha \(shafaq ahmar\):\s+(\S+)\s+\(SZA ([\d.]+).?, depression ([\d.]+)"),
    "khayt_isha_abyad": re.compile(
        r"Isha \(shafaq abyad\):\s+(\S+)\s+\(SZA ([\d.]+).?, depression ([\d.]+)"),
    "legacy_fajr": re.compile(
        r"Fajr \(true dawn\):\s+(\S+)\s+\(SZA ([\d.]+).?, depression ([\d.]+)"),
    "legacy_isha_abyad": re.compile(
        r"Isha \(al-abyad\):\s+(\S+)\s+\S*\s*\(SZA ([\d.]+).?, depression ([\d.]+)"),
    "legacy_isha_ahmar": re.compile(
        r"Isha \(al-ahmar\):\s+(\S+)\s+\S*\s*\(SZA ([\d.]+).?, depression ([\d.]+)"),
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
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
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
    ap.add_argument("--out", default="/private/tmp/claude-501/-Users-mostafamahdi/"
                    "512438d9-285c-48d2-93ab-b35b52762c14/scratchpad/criterion_runs")
    ap.add_argument("--jobs", type=int, default=3)
    args = ap.parse_args()

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, (lat, lon, elev, dates, extra) in SITES.items():
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
    tsv = outdir / "criterion_runs.tsv"
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
