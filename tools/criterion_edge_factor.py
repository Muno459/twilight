#!/usr/bin/env python3
"""Edge-factor constraint sweep: implied-factor inversion + zodiacal
cross-prediction for the khayt appearance edge factor.

The khayt criterion's appearance edge factor (KhaytParams::
edge_factor_appearance = 45.0, effective 45 x k_contrast 0.4 = 18x over
the Blackwell TVI disc threshold) was calibrated ONCE against the Mecca
desert cluster. This tool attacks that single tuned number two ways:

ATTACK 1 (implied-factor inversion): for each khayt-matched naked-eye
campaign of validation/RESULTS_CRITERION_SITES.md, run the engine at
several edge factors bracketing 45 (via the TWILIGHT_KHAYT_EDGE_
APPEARANCE env knob added in khayt.rs for exactly this analysis), fit
the site's factor-to-depression response, and INVERT: the factor that
would exactly reproduce the site's observation. A tight cluster of
implied factors across sites/latitudes/seasons pins the constant; a
wide scatter or latitude trend falsifies it.

ATTACK 3 (zodiacal cross-prediction): the same psychophysics drives the
false-dawn (kadhib) detection on the zodiacal wedge, which was never
tuned. Run a dark desert site at increasing artificial skyglow
(--skyglow --radiance at the Bortle 4/5/6 radiance equivalents used in
RESULTS_CRITERION_SITES.md section 9.3) and record where the engine
stops reporting kadhib; compare against the published Bortle limit of
naked-eye zodiacal-light visibility (clearly evident at Bortle 4, hints
at Bortle 5, gone at Bortle 6; Bortle 2001).

Usage:
    python3 tools/criterion_edge_factor.py            # attack 1 runs + analysis
    python3 tools/criterion_edge_factor.py --attack3  # zodiacal ladder
    python3 tools/criterion_edge_factor.py --analyze  # re-parse cache only
    python3 tools/criterion_edge_factor.py --only hail --jobs 2

Engine binary: $TWILIGHT_CLI (pin a snapshot; a parallel rebuild of
target/release must not change engines mid-sweep). Always a release
build. Raw stdout+stderr cached per run under
validation/criterion_runs/edge_factor/; delete a file to force a rerun.
All factors (45 included) run fresh on the pinned binary: a measured
engine drift (hail f45: 14.65 HEAD vs 14.46 pristine cache) rules out
factor-45 cache reuse; the old cache only supplies the differential
seasonal (date-to-mean) adjustment, which is drift-insensitive.
"""

import argparse
import concurrent.futures as cf
import json
import math
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLI = pathlib.Path(os.environ.get("TWILIGHT_CLI",
                                  ROOT / "target/release/twilight-cli"))
RUNDIR = ROOT / "validation/criterion_runs/edge_factor"
PRISTINE = ROOT / "validation/criterion_runs"

# Appended to every engine invocation (set by --cpu); the CPU path is the
# published-number reference, the GPU path a faster cross-check.
EXTRA_ARGS = []

# The calibrated default under attack. MUST match the engine's shipped
# KhaytParams default (khayt.rs pins it by test): run_one only sets the
# override env var when factor != CAL_FACTOR, so a stale value here makes
# every "CAL_FACTOR" run silently execute at whatever the binary ships.
# 45 was the pre-hyperaccuracy value; the 2026-07-07 refit shipped 56.
CAL_FACTOR = 56.0
LADDER = [25.0, 60.0, 80.0]  # new-run factors; CAL_FACTOR from cache/verify

# ── Attack 1 sites ───────────────────────────────────────────────────
# One date per campaign, chosen as the pristine-sweep date whose
# factor-45 khayt lies closest to the site's factor-45 seasonal mean
# (minimizes the seasonal adjustment below). obs = the campaign's
# naked-eye khayt-matched angle from RESULTS_CRITERION_SITES.md sec. 1.
#
# role:
#  desert  - naked-eye desert campaign; invert with seasonal adjustment
#  panel   - Birmingham OpenFajr panel date; obs is date-specific
#  bracket - Sana'a two-event bracket; report factor BOUNDS only
#  cliff   - compressed-twilight row; document factor (in)sensitivity
# name: (lat, lon, elev, date, extra_args, obs, obs_err, role, factors)
SITES = {
    "riyadh_desert": (25.763, 46.5, 540, "2004-10-15", [], 14.60, 0.30,
                      "desert", LADDER),
    "hail": (27.517, 41.70, 1000, "2015-01-15", [], 14.01, 0.32,
             "desert", LADDER + [35.0]),  # +35: log-linearity probe
    "sinai": (31.067, 32.867, 20, "2011-04-15", [], 14.61, None,
              "desert", LADDER),
    "fayum": (29.283, 30.05, 50, "2018-12-09", [], 14.80, None,
              "desert", LADDER),
    "matrouh": (31.003, 27.85, 75, "2012-04-15", [], 14.50, None,
                "desert", LADDER),
    "kottamia": (29.932, 31.825, 470, "2010-04-15", [], 14.66, 0.20,
                 "desert", LADDER),
    "bahariya": (28.715, 29.997, 150, "2007-10-15", [], 14.60, None,
                 "desert", LADDER),
    "wadi_natrun": (30.50, 30.15, 30, "2017-01-15", [], 14.57, None,
                    "desert", LADDER),
    "tubruq": (32.078, 23.983, 25, "2010-01-15", [], 14.68, None,
               "desert", LADDER),
    # Sana'a: Sultan's two events bracket tabayyun (18.8 first glow
    # merged with zodiacal light > tabayyun > 13.2 colors divergence).
    "sanaa": (15.50, 44.40, 2200, "2003-11-24", [], (13.2, 18.8), None,
              "bracket", [25.0, 80.0]),
    # Birmingham OpenFajr panel dates spanning the year. Backgrounds per
    # RESULTS sec. 9: spring/autumn peak mornings match the PRISTINE sky
    # (street lights dimmed at those hours); the winter morning needs the
    # measured atlas veil (post-unit-fix engine: plain --skyglow).
    "birmingham_apr": (52.44, -1.95, 150, "2015-04-20", [], 15.0, None,
                       "panel", LADDER),
    "birmingham_sep": (52.44, -1.95, 150, "2015-09-23", [], 14.6, None,
                       "panel", LADDER),
    "birmingham_dec_glow": (52.44, -1.95, 150, "2015-12-10",
                            ["--skyglow"], 12.9, None,
                            "panel", [25.0, 45.0, 80.0]),
    # June compressed-twilight row: khayt is cliff-shaped there; expect
    # factor INSENSITIVITY (documented, not inverted).
    "birmingham_jun": (52.44, -1.95, 150, "2015-06-30", [], 12.3, None,
                       "cliff", [25.0, 80.0]),
}

# Map to the pristine-sweep site names (for factor-45 cache reuse and
# the seasonal mean).
PRISTINE_NAME = {
    "birmingham_apr": "birmingham", "birmingham_sep": "birmingham",
    "birmingham_jun": "birmingham",
}

# NOTE: an initial verification pass (hail 2015-01-15 factor 45, fresh
# HEAD binary vs the pristine-sweep cache) measured 14.65 vs 14.46: the
# engines DIVERGED between the criterion-sites sweep and HEAD, so every
# ladder point runs fresh and the old cache contributes only the
# differential seasonal adjustment. The drift per site is printed by
# the analysis for the record.

# ── Attack 3: zodiacal / kadhib skyglow ladder ──────────────────────
# Dark tropical desert (Mecca coordinates, the calibration site, where
# the pristine run detects kadhib). Radiance in nW/cm^2/sr; Bortle
# equivalents follow RESULTS_CRITERION_SITES.md sec. 9.3 (R=2 -> B4,
# R=6 -> B5, R=15 -> B6). R=0.5 is a rural B3 probe.
A3_SITE = ("mecca_zodiacal", 21.4225, 39.8262, 300, "2015-12-21")
A3_RADIANCES = [None, 0.5, 2.0, 6.0, 15.0]

_TIME = r":\s+(\S+?)\s+(?:[^(\s]+\s+)?(?:\(\+1d\)\s+)?\(SZA ([\d.]+).?, depression ([\d.]+)"
RE_KHAYT = re.compile(r"Fajr \(khayt al-abyad\)" + _TIME)
RE_LEGACY = re.compile(r"Fajr \(true dawn\)" + _TIME)
RE_KADHIB = re.compile(r"false dawn \(al-fajr al-kadhib\) visible from (\S+)")
RE_TZ = re.compile(r"Timezone:.*UTC([+-]\d\d):(\d\d)")
RE_SUNRISE = re.compile(r"Sunrise:\s+(\S+)")
RE_BORTLE = re.compile(r"[Bb]ortle[^\d]*(\d)")


def run_one(name, lat, lon, elev, date, factor, extra):
    """One engine run at one edge factor; returns raw output (cached).

    NO reuse of the pristine-sweep factor-45 cache: the verification
    rows measured a real engine drift (hail 2015-01-15: fresh 14.65 vs
    cached 14.46) between the criterion-sites engine and HEAD, so every
    ladder point (45 included) runs fresh on the pinned binary. The
    old-engine cache is still used, but only for the DIFFERENTIAL
    seasonal adjustment (date-to-mean offset), which is drift-
    insensitive to first order.
    """
    tag = "" if not extra else "_" + "_".join(a.strip("-") for a in extra)
    raw = RUNDIR / f"{name}_{date}_f{factor:g}{tag}.txt"
    # Engine output and cache files are UTF-8. Decoding them with the
    # Windows locale codepage (the text=True default there) turns the
    # degree sign into two characters and silently breaks RE_KHAYT, so
    # every depression parses as None on Windows.
    if raw.exists() and "depression" in raw.read_text(encoding="utf-8"):
        return raw.read_text(encoding="utf-8")
    env = dict(os.environ)
    if factor != CAL_FACTOR:
        env["TWILIGHT_KHAYT_EDGE_APPEARANCE"] = str(factor)
    env["TWILIGHT_KHAYT_DEBUG"] = "1"  # margin curves kept for audit
    cmd = [str(CLI), "pray", f"--lat={lat}", f"--lon={lon}",
           f"--elevation={elev}", f"--date={date}"] + extra + EXTRA_ARGS
    proc = subprocess.run(cmd, capture_output=True,
                          timeout=2400, env=env)
    out = (proc.stdout.decode("utf-8", "replace")
           + proc.stderr.decode("utf-8", "replace"))
    raw.write_text(out, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"{name} {date} f{factor}: exit {proc.returncode}")
    return out


def khayt_dep(out):
    m = RE_KHAYT.search(out)
    return float(m.group(3)) if m else None


# ── NOAA solar position (Meeus low-accuracy, <0.01 deg) ─────────────
def solar_depression_utc(lat, lon, jd_utc):
    d = jd_utc - 2451545.0
    g = math.radians((357.529 + 0.98560028 * d) % 360)
    q = (280.459 + 0.98564736 * d) % 360
    lam = math.radians(q + 1.915 * math.sin(g) + 0.020 * math.sin(2 * g))
    e = math.radians(23.439 - 0.00000036 * d)
    ra = math.degrees(math.atan2(math.cos(e) * math.sin(lam), math.cos(lam)))
    dec = math.asin(math.sin(e) * math.sin(lam))
    gmst = (280.46061837 + 360.98564736629 * d) % 360
    ha = math.radians((gmst + lon - ra) % 360)
    latr = math.radians(lat)
    elev = math.asin(math.sin(latr) * math.sin(dec)
                     + math.cos(latr) * math.cos(dec) * math.cos(ha))
    return -math.degrees(elev)


def to_jd(date, hms, utc_off_h):
    y, mo, dy = (int(x) for x in date.split("-"))
    h, mi, s = (int(x) for x in hms.split(":"))
    frac = (h + mi / 60 + s / 3600 - utc_off_h) / 24
    a = (14 - mo) // 12
    yy = y + 4800 - a
    mm = mo + 12 * a - 3
    jdn = dy + (153 * mm + 2) // 5 + 365 * yy + yy // 4 - yy // 100 + yy // 400 - 32045
    return jdn - 0.5 + frac


RE_DBG = re.compile(
    r"khayt sza\s+([\d.]+): spread\s+([\d.e+-]+)\s+\(\+-[\d.e+-]+\)\s+central\s+([\d.e+-]+)")


def margin_crossings(out, fajr_sza):
    """Central and spread margin=1 crossings [deg SZA] from the
    TWILIGHT_KHAYT_DEBUG coarse dump. The dump has one block per side
    of the night; the MORNING block is the one whose spread crossing
    lies nearest the printed khayt fajr SZA. Log-linear interpolation
    between bracketing points (the solver's smooth path applied to the
    coarse curves). Returns (spread_sza, central_sza) or (None, None).

    This reads the DETECTION MACHINERY directly (the margin curves ARE
    the criterion); the stdout kadhib line additionally applies a
    verdict-classification rule (central deeper than spread by > 0.2
    deg) whose reporting in-flight HEAD work may gate differently.
    """
    lines = RE_DBG.findall(out)
    if not lines:
        return None, None
    blocks, cur, prev = [], [], -1.0
    for sza, sp, ce in lines:
        sza = float(sza)
        if sza <= prev and cur:
            blocks.append(cur)
            cur = []
        cur.append((sza, float(sp), float(ce)))
        prev = sza
    blocks.append(cur)

    def crossing(pts, col):
        best = None
        for p0, p1 in zip(pts, pts[1:]):
            a, b = p0[1 + col], p1[1 + col]
            if a >= 1.0 > b:
                if b > 0.0:
                    la, lb = math.log(a), math.log(b)
                    best = p0[0] + (p1[0] - p0[0]) * la / (la - lb)
                else:
                    best = 0.5 * (p0[0] + p1[0])
        return best

    scored = []
    for blk in blocks:
        s = crossing(blk, 0)
        c = crossing(blk, 1)
        if s is not None:
            scored.append((abs(s - fajr_sza), s, c))
    if not scored:
        # No spread crossing at all (fully veiled): still report the
        # central crossing if any block has one.
        cs = [crossing(blk, 1) for blk in blocks]
        cs = [c for c in cs if c is not None]
        return None, max(cs) if cs else None
    _, s, c = min(scored)
    return s, c


def kadhib_depression(out, lat, lon, date):
    """Kadhib onset depression from its printed clock time via NOAA
    solar position, self-validated against the printed khayt pair."""
    mk = RE_KADHIB.search(out)
    mf = RE_KHAYT.search(out)
    mtz = RE_TZ.search(out)
    if not (mk and mf and mtz):
        return None, None
    off = int(mtz.group(1)) + int(mtz.group(2)) / 60 * (1 if int(mtz.group(1)) >= 0 else -1)
    # Validation: printed khayt (time, depression) pair.
    dep_f_engine = float(mf.group(3))
    dep_f_noaa = solar_depression_utc(lat, lon, to_jd(date, mf.group(1), off))
    resid = dep_f_noaa - dep_f_engine
    dep_k = solar_depression_utc(lat, lon, to_jd(date, mk.group(1), off)) - resid
    return dep_k, abs(resid)


# ── Attack 1 analysis ────────────────────────────────────────────────
def fit_loglinear(pairs):
    """dep = a + b*log10(f); returns (a, b, rms residual)."""
    xs = [math.log10(f) for f, _ in pairs]
    ys = [d for _, d in pairs]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    rms = math.sqrt(sum((a + b * x - y) ** 2 for x, y in zip(xs, ys)) / n)
    return a, b, rms


def invert(pairs, a, b, target):
    """Factor reproducing `target` depression: monotone piecewise-linear
    interpolation in log10(factor) between the bracketing ladder points;
    global log-linear fit only for extrapolation beyond the ladder."""
    pts = sorted(((math.log10(f), d) for f, d in pairs))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        lo, hi = min(y0, y1), max(y0, y1)
        if lo <= target <= hi and abs(y1 - y0) > 1e-9:
            return 10 ** (x0 + (target - y0) / (y1 - y0) * (x1 - x0)), False
    return 10 ** ((target - a) / b), True


def analyze(rows, pristine_tsv):
    """rows: {site: {factor: dep}}. Returns per-site inversion records.

    Two implied factors per site:
      implied_head - invert the (seasonally adjusted) observation on the
        FRESH HEAD ladder. Reflects HEAD's behaviour, including any
        radiance drift since the calibration engine.
      implied_cal  - the same inversion in the CALIBRATION frame:
        the fresh curve is shifted back by the per-site f45 drift
        (dep45_fresh - old_f45_at_date) so the inversion is effectively
        against the engine the 45 was tuned on. This is the clean
        cross-site universality metric (a global drift cancels).
    """
    # Old-engine factor-45 values from the pristine sweep TSV: the
    # differential seasonal offset AND the per-site drift reference.
    old45, means = {}, {}
    for line in pristine_tsv.read_text().splitlines()[1:]:
        f = line.split("\t")
        if len(f) > 10 and f[2] and not f[10]:
            means.setdefault(f[0], []).append(float(f[2]))
            old45[(f[0], f[1])] = float(f[2])
    recs = []
    for site, (lat, lon, elev, date, extra, obs, err, role, _) in SITES.items():
        deps = rows.get(site, {})
        pairs = sorted((f, d) for f, d in deps.items() if d is not None)
        if len(pairs) < 2:
            continue
        a, b, rms = fit_loglinear(pairs)
        dep45 = deps.get(CAL_FACTOR)
        pn = PRISTINE_NAME.get(site, site)
        seas = 0.0
        if role == "desert" and pn in means and (pn, date) in old45:
            seas = old45[(pn, date)] - sum(means[pn]) / len(means[pn])
        # Per-site HEAD-vs-calibration f45 drift (deg), when a clean
        # (no-skyglow) old f45 exists at this exact date.
        drift = None
        if not extra and dep45 is not None and (pn, date) in old45:
            drift = dep45 - old45[(pn, date)]
        rec = dict(site=site, lat=lat, date=date, role=role, obs=obs,
                   err=err, dep45=dep45, seas=seas, drift=drift,
                   a=a, b=b, rms=rms, pairs=pairs)
        if role in ("desert", "panel") and b < -1e-6:
            target = obs + seas
            rec["implied_head"], rec["extrap"] = invert(pairs, a, b, target)
            if drift is not None:
                # Calibration frame: solve g_fresh(f) = target + drift.
                rec["implied_cal"], _ = invert(pairs, a, b, target + drift)
                if err:
                    rec["cal_lo"], _ = invert(pairs, a, b, target + drift + err)
                    rec["cal_hi"], _ = invert(pairs, a, b, target + drift - err)
        elif role == "bracket" and b < -1e-6:
            lo_dep, hi_dep = obs
            rec["factor_hi"], _ = invert(pairs, a, b, lo_dep)  # shallow edge
            rec["factor_lo"], _ = invert(pairs, a, b, hi_dep)  # deep edge
        recs.append(rec)
    return recs


def curve_at(rec, factor):
    """Site's khayt depression at `factor`: piecewise-linear in log10 f
    on the measured ladder, global fit for extrapolation."""
    pts = sorted((math.log10(f), d) for f, d in rec["pairs"])
    x = math.log10(factor)
    if x <= pts[0][0] or x >= pts[-1][0]:
        return rec["a"] + rec["b"] * x
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 + (x - x0) / (x1 - x0) * (y1 - y0)
    return rec["a"] + rec["b"] * x


def leave_one_out(recs):
    """Recalibrate the edge factor on each single desert site (its
    calibration-frame implied factor), then predict every OTHER desert
    site with that factor and report how far the prediction moves from
    the 45 baseline (deg) and its residual against the seasonally
    adjusted, drift-referenced observation. All in the calibration
    frame so the inversion is single-engine-consistent."""
    desert = [r for r in recs if r["role"] == "desert" and "implied_cal" in r]
    out = []
    for cal in desert:
        fi = cal["implied_cal"]
        shifts, deltas = [], []
        for r in desert:
            if r is cal:
                continue
            pred = curve_at(r, fi) - r["drift"]        # back to cal frame
            base = curve_at(r, CAL_FACTOR) - r["drift"]
            shifts.append(abs(pred - base))
            deltas.append(pred - (r["obs"] + r["seas"]))
        out.append(dict(cal=cal["site"], factor=fi,
                        max_shift=max(shifts),
                        mad=sum(abs(d) for d in deltas) / len(deltas),
                        max_abs_delta=max(abs(d) for d in deltas)))
    return out


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx * sy == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append")
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--attack3", action="store_true")
    ap.add_argument("--cpu", action="store_true",
                    help="run the engine on the CPU reference path")
    ap.add_argument("--analyze", action="store_true",
                    help="parse cache only; no new runs")
    args = ap.parse_args()
    RUNDIR.mkdir(parents=True, exist_ok=True)
    if args.cpu:
        EXTRA_ARGS.append("--cpu")

    if args.attack3:
        name, lat, lon, elev, date = A3_SITE
        print(f"# Attack 3: kadhib vs skyglow at {name} {date}")
        for r in A3_RADIANCES:
            extra = [] if r is None else ["--skyglow", f"--radiance={r}"]
            tag = "pristine" if r is None else f"R{r:g}"
            out = run_one(name, lat, lon, elev, date, CAL_FACTOR, extra)
            dep_k, resid = kadhib_depression(out, lat, lon, date)
            dep_f = khayt_dep(out)
            # Machinery-level crossings straight from the margin curves.
            s_sza, c_sza = margin_crossings(out, 90.0 + dep_f if dep_f else 105.0)
            mb = RE_BORTLE.search(out)
            print(json.dumps(dict(
                radiance=r, tag=tag, khayt_fajr=dep_f,
                stdout_kadhib_dep=None if dep_k is None else round(dep_k, 2),
                noaa_resid=None if resid is None else round(resid, 3),
                margin_spread_dep=None if s_sza is None else round(s_sza - 90, 2),
                margin_central_dep=None if c_sza is None else round(c_sza - 90, 2),
                central_minus_spread=None if None in (s_sza, c_sza)
                else round(c_sza - s_sza, 2),
                bortle=mb.group(1) if mb else None,
                stdout_kadhib_line=dep_k is not None)))
        return

    # ── Attack 1 ──
    jobs = []
    for site, (lat, lon, elev, date, extra, obs, err, role, factors) in SITES.items():
        if args.only and site not in args.only:
            continue
        fs = list(factors) + [CAL_FACTOR]
        for f in fs:
            jobs.append((site, lat, lon, elev, date, f, extra))
    rows = {}
    if not args.analyze:
        with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(run_one, *j): j for j in jobs}
            for fut in cf.as_completed(futs):
                site, _, _, _, date, f, extra = futs[fut]
                dep = khayt_dep(fut.result())
                rows.setdefault(site, {})[f] = dep
                print(json.dumps({"site": site, "date": date,
                                  "factor": f, "khayt": dep}), flush=True)
    else:
        for site, lat, lon, elev, date, f, extra in jobs:
            tag = "" if not extra else "_" + "_".join(a.strip("-") for a in extra)
            raw = RUNDIR / f"{site}_{date}_f{f:g}{tag}.txt"
            if raw.exists():
                rows.setdefault(site, {})[f] = khayt_dep(raw.read_text())

    recs = analyze(rows, PRISTINE / "criterion_runs.tsv")

    # ── Engine-drift report: fresh HEAD f45 vs the pristine cache ──
    drifts = [r["drift"] for r in recs if r.get("drift") is not None]
    print("\n# HEAD-vs-calibration f45 drift (deg; + = HEAD deeper/earlier):")
    for r in recs:
        if r.get("drift") is not None:
            print(f"#   {r['site']:20s} fresh {r['dep45']:.2f} "
                  f"cache {r['dep45'] - r['drift']:.2f}  {r['drift']:+.2f}")
    if drifts:
        print(f"#   mean drift {sum(drifts)/len(drifts):+.2f} deg "
              f"[{min(drifts):+.2f}..{max(drifts):+.2f}], n={len(drifts)}")

    # ── Per-site implied factors ──
    print("\n# site | role | lat | obs | b(dep/dex) | rms | "
          "implied_head | implied_cal [lo..hi]")
    for r in recs:
        base = (f"{r['site']:20s} {r['role']:7s} {r['lat']:6.2f} "
                f"{str(r['obs']):13s} b={r['b']:6.2f} rms={r['rms']:.3f}")
        if "implied_head" in r:
            cal = (f" cal={r['implied_cal']:.1f}"
                   + (f" [{r['cal_lo']:.0f}..{r['cal_hi']:.0f}]"
                      if "cal_lo" in r else "")
                   if "implied_cal" in r else " cal=n/a")
            xt = " (extrap)" if r.get("extrap") else ""
            print(f"{base} head={r['implied_head']:.1f}{cal}{xt}")
        elif "factor_lo" in r:
            print(f"{base} bracket bounds [{r['factor_lo']:.1f}..{r['factor_hi']:.0f}]")
        else:
            print(f"{base} (no inversion: role={r['role']})")

    # ── Desert distribution (the universality test) ──
    for frame, key in (("HEAD", "implied_head"), ("CALIBRATION", "implied_cal")):
        fs = [r[key] for r in recs
              if r["role"] == "desert" and key in r and not r.get("extrap")]
        lats = [r["lat"] for r in recs
                if r["role"] == "desert" and key in r and not r.get("extrap")]
        if len(fs) < 3:
            continue
        lfs = [math.log10(f) for f in fs]
        n = len(fs)
        gm = 10 ** (sum(lfs) / n)
        sd = math.sqrt(sum((x - sum(lfs) / n) ** 2 for x in lfs) / (n - 1))
        print(f"\n# {frame} frame desert implied factors: n={n} "
              f"geo-mean={gm:.1f} spread={sd:.3f} dex (x/{10**sd:.2f}) "
              f"min={min(fs):.1f} max={max(fs):.1f}")
        print(f"#   latitude trend Pearson r(log10 f, lat) = "
              f"{pearson(lfs, lats):.2f} over {min(lats):.1f}..{max(lats):.1f} N")

    print("\n# leave-one-out (recalibrate on each desert site, calibration frame):")
    for l in leave_one_out(recs):
        print(f"#   cal={l['cal']:15s} f={l['factor']:5.1f} "
              f"max pred shift={l['max_shift']:.2f} deg "
              f"mean|delta|={l['mad']:.2f} max|delta|={l['max_abs_delta']:.2f}")


if __name__ == "__main__":
    main()
