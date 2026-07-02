#!/usr/bin/env python3
"""Measured-sky referee for the deep-twilight tail (SZA > 102 deg).

Compares the engine's zenith twilight brightness decay against PUBLISHED,
MEASURED twilight skies, in the depression range (5-20 deg) where no public
Monte Carlo RT referee converges. Companion to tools/validate_libradtran.py,
which stops where MYSTIC stops converging (~SZA 100-104).

All referee data is embedded below as literals; the script fetches NOTHING
at runtime. Full extraction provenance is in the comments next to each
literal. Referees:

  R1  Patat, Ugolnikov & Postylyakov 2006, "UBVRI twilight sky brightness
      at ESO-Paranal", A&A 455, 385-393. Calibrated zenith V and B surface
      brightness vs sun zenith distance zeta, quadratic fits over
      95 <= zeta <= 105 deg from 1083 FORS1/VLT twilight flats (2005) plus
      3388 long exposures (1999-2005) reaching zeta = 112 deg.
  R2  Koomen, Lock, Packer, Scolnik, Tousey & Hulburt 1952, "Measurements
      of the brightness of the twilight sky", JOSA 42, 353-356. Photopic
      zenith luminance vs solar altitude, tabulated 0 to -15 deg at
      Sacramento Peak NM (2800 m) and rural Maryland (30 m).
  R3  Night floors: Grauer & Grauer 2021 (Sci Rep 11, 23893; CCIDSS SQM
      floor at deep solar minimum) and Patat 2003a (A&A 400, 1183; Paranal
      V floor at solar maximum). NOTE: the Grauer papers contain NO
      twilight-vs-depression data (they discard everything brighter than
      18 deg depression); they referee the night FLOOR only.
  R4  hnsky.org single-night SQM twilight fit (tertiary, weak provenance,
      flagged as such below).

Engine side: `twilight-cli compare` (spectral radiance, 380-780 nm at
10 nm) run multi-seed in hybrid mode (exact single scattering + MC orders
2+), plus `twilight-cli sqm predict` for the SQM-band rails, which reuses
the engine's own luminance -> mag/arcsec^2 convention end to end.

Zero point: the engine's documented SQM rail (crates/twilight-skyglow/src/
bortle.rs, luminance_to_sqm) is

    mag/arcsec^2 = 12.58 - 2.5 * log10(L [cd/m^2])

the standard V-band surface-brightness <-> luminance relation
(0 mag/arcsec^2 == 1.08e5 cd/m^2, Garstang convention), exact for a
solar-type spectrum. The same zero point serves the V comparison here, so
the V referee and the engine's SQM campaign share ONE rail; this script's
Python re-implementation of that rail is cross-checked against `sqm
predict` output at runtime (agreement < 0.001 mag, see check_zero_point).

Usage (from the repo root; requires target/release/twilight-cli, so run
`cargo build --release` first):

    python3 tools/validate_measured_twilight.py
    python3 tools/validate_measured_twilight.py --photons 2000 --seeds 8

Engine CSVs are cached in --workdir keyed by the exact CLI configuration;
delete files there to force re-simulation (a full cold run at the default
photons/seeds is roughly 1.5 h on 8 performance cores). Output: a markdown
report on stdout (the basis of validation/RESULTS_MEASURED_SKY.md).
"""

import argparse
import math
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Engine rails (transcribed from the Rust sources; keep in sync)
# ---------------------------------------------------------------------------

# Wavelength grid of `compare` output: crates/twilight-data/src/builder.rs,
# DEFAULT_WAVELENGTHS_NM = 380..780 nm at 10 nm (41 samples).
ENGINE_WL_NM = [380.0 + 10.0 * i for i in range(41)]

# CIE 1924 photopic V(lambda), 10 nm subsample of the engine's 5 nm table
# (crates/twilight-threshold/src/vision.rs, PHOTOPIC_V, indices 0,2,...,80).
# The engine integrates radiance * V(lambda) trapezoidally with K_m =
# 683.002 lm/W (crates/twilight-threshold/src/luminance.rs); on the 10 nm
# engine grid the interpolated table values are exactly these.
CIE_V_10NM = [
    0.0000390, 0.0001200, 0.0003960, 0.0012100, 0.0040000, 0.0116000,
    0.0230000, 0.0380000, 0.0600000, 0.0910000, 0.1390000, 0.2080000,
    0.3230000, 0.5030000, 0.7100000, 0.8620000, 0.9540000, 0.9950000,
    0.9950000, 0.9520000, 0.8700000, 0.7570000, 0.6310000, 0.5030000,
    0.3810000, 0.2650000, 0.1750000, 0.1070000, 0.0610000, 0.0320000,
    0.0170000, 0.0082100, 0.0041000, 0.0020900, 0.0010500, 0.0005200,
    0.0002490, 0.0001200, 0.0000600, 0.0000300, 0.0000150,
]
KM_PHOTOPIC = 683.002  # lm/W, luminance.rs

# Engine luminance -> mag/arcsec^2 zero point (bortle.rs, luminance_to_sqm).
SQM_ZP = 12.58

# Engine airglow rail (crates/twilight-threshold/src/night_sky.rs):
# zenith airglow = (90 + 0.43 * F10.7) S10, S10_TO_CD = 0.69e-6 cd/m^2 per
# S10, no extinction at zenith (airmass factor is (X-1) = 0). Used here
# ONLY to restate the engine's night floor (computed at the F10.7 = 130
# default of `sqm predict`) at a referee's solar-activity epoch. This is a
# transcription of the engine's own parameterization, not a new model.
S10_TO_CD = 0.69e-6


def engine_zenith_airglow_cd(f107):
    return (90.0 + 0.43 * f107) * S10_TO_CD


def refloor(l_floor_cd, f107_from, f107_to):
    """Engine night-floor luminance moved from one F10.7 epoch to another
    by swapping only the engine's own airglow term."""
    return l_floor_cd - engine_zenith_airglow_cd(f107_from) \
        + engine_zenith_airglow_cd(f107_to)


def photopic_luminance(radiance_w_m2_sr_nm):
    """cd/m^2 from spectral radiance on the engine grid; mirrors
    twilight_threshold::luminance::photopic_luminance (trapezoid, K_m)."""
    r = radiance_w_m2_sr_nm
    assert len(r) == len(ENGINE_WL_NM)
    integral = 0.0
    for i in range(len(r) - 1):
        dw = ENGINE_WL_NM[i + 1] - ENGINE_WL_NM[i]
        integral += 0.5 * (r[i] * CIE_V_10NM[i] + r[i + 1] * CIE_V_10NM[i + 1]) * dw
    return KM_PHOTOPIC * integral


def lum_to_mag(l_cd):
    return SQM_ZP - 2.5 * math.log10(l_cd) if l_cd > 0 else float("inf")


# ---------------------------------------------------------------------------
# B-band synthetic photometry (for the Patat B referee)
# ---------------------------------------------------------------------------

# Bessell 1990 (PASP 102, 1181, Table 2) B passband response at 10 nm,
# verified digit-for-digit 2026-07-02 against the SVO Filter Profile
# Service ascii table (Generic/Bessell.B). The 360/370 nm points (0.000,
# 0.030) fall below the engine's 380 nm grid edge; with the 0.134 at 380
# nm the truncated wing carries ~1-2% of the response integral for a blue
# twilight spectrum, i.e. < 0.02 mag, noted in the report.
BESSELL_B_NM = [380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480,
                490, 500, 510, 520, 530, 540, 550, 560]
BESSELL_B_RESP = [0.134, 0.567, 0.920, 0.978, 1.000, 0.978, 0.935, 0.853,
                  0.740, 0.640, 0.536, 0.424, 0.325, 0.235, 0.150, 0.095,
                  0.043, 0.009, 0.000]

# Zero point: band-mean flux density of a B = 0 source, f_lambda = 6.32e-9
# erg/cm^2/s/A (Bessell, Castelli & Plez 1998, A&A 333, 231, Table A2;
# standard value, recalled). Cross-check (fetched 2026-07-02): the SVO
# Filter Profile Service quotes a Vega-spectrum-computed ZP of 6.13268e-9
# for the same response, 0.033 mag from the BCP98 value; that difference
# plus ~0.05-0.1 mag from band-mean f_lambda on a steep twilight spectrum
# form the stated B zero-point systematic (~0.1 mag), small against the
# ~1.2 mag/deg decay under test.
B_ZP_FLAM = 6.32e-9  # erg cm^-2 s^-1 A^-1 for B = 0

SR_PER_ARCSEC2 = (math.pi / 180.0 / 3600.0) ** 2
# W/m^2/sr/nm -> erg/cm^2/s/A/arcsec^2 (1 W/m^2/nm = 100 erg/cm^2/s/A).
RAD_TO_FLAM_ARCSEC2 = 100.0 * SR_PER_ARCSEC2


B_RESP_INTEGRAL = sum(
    0.5 * (BESSELL_B_RESP[i] + BESSELL_B_RESP[i + 1])
    * (BESSELL_B_NM[i + 1] - BESSELL_B_NM[i])
    for i in range(len(BESSELL_B_NM) - 1)
)


def b_band_flux(radiance_w_m2_sr_nm):
    """Response-weighted band integral of radiance [W/m^2/sr]."""
    grid = dict(zip(ENGINE_WL_NM, radiance_w_m2_sr_nm))
    num = 0.0
    for i in range(len(BESSELL_B_NM) - 1):
        w0, w1 = BESSELL_B_NM[i], BESSELL_B_NM[i + 1]
        s0, s1 = BESSELL_B_RESP[i], BESSELL_B_RESP[i + 1]
        num += 0.5 * (grid[float(w0)] * s0 + grid[float(w1)] * s1) * (w1 - w0)
    return num


def b_flux_to_mag(band_flux):
    if band_flux <= 0.0:
        return float("inf")
    mean_flam = (band_flux / B_RESP_INTEGRAL) * RAD_TO_FLAM_ARCSEC2
    return -2.5 * math.log10(mean_flam / B_ZP_FLAM)


# ---------------------------------------------------------------------------
# Referee R1: Patat, Ugolnikov & Postylyakov 2006 (ESO-Paranal)
# ---------------------------------------------------------------------------
# Extraction provenance: transcribed 2026-07-02 from arXiv astro-ph/0604128
# and cross-checked against the published PDF mirrored at
# https://www.eso.org/~fpatat/science/skybright/twilight.pdf (identical).
# Their Sec. 4 / Table 1: zenith surface brightness fitted over
# 95 <= zeta <= 105 deg (zeta = sun zenith distance = 90 + depression) as
#   b(zeta) = a0 + a1*(zeta - 95) + a2*(zeta - 95)^2   [mag/arcsec^2]
# sigma = RMS of the data about the fit; gamma (+- error) = slope of a
# separate LINEAR fit over 95 <= zeta <= 100 [mag/deg]. Data: FORS1/VLT,
# zenith selection |alpha| <= 40 deg, no color/airmass correction,
# internal photometric error < 1%.
PATAT_FIT = {
    #        a0     a1      a2    sigma  gamma gamma_err
    "U": (11.78, 1.376, -0.039, 0.24, 1.23, 0.01),
    "B": (11.84, 1.411, -0.041, 0.12, 1.24, 0.01),
    "V": (11.84, 1.518, -0.057, 0.18, 1.14, 0.02),
    "R": (11.40, 1.567, -0.064, 0.29, 1.09, 0.03),
    "I": (10.93, 1.470, -0.062, 0.40, 0.94, 0.03),
}
PATAT_FIT_RANGE = (95.0, 105.0)
# Sec. 4: "In all passbands, the night sky brightness level is reached at
# around zeta = 105-106 deg."
PATAT_MERGE_ZETA = (105.0, 106.0)

# Dark night-sky zenith brightness the twilight curves merge into, adopted
# by Patat 2006 from Patat 2003a (A&A 400, 1183, Table 4: "zenith corrected
# average sky brightness during dark time", Apr 2000 - Sep 2001 = cycle 23
# SOLAR MAXIMUM): (mean, rms, min, max) in mag/arcsec^2. Patat 2003a quotes
# the literature full-solar-cycle spread as ~0.5 mag in B and V (Walker
# 1988); his own F10.7 regression over the baseline is consistent with zero
# (B: 0.08+-0.13, V: 0.07+-0.11 mag).
PATAT_NIGHT_SKY = {
    "U": (22.28, 0.22, 21.89, 22.61),
    "B": (22.64, 0.18, 22.19, 23.02),
    "V": (21.61, 0.20, 20.99, 22.10),
    "R": (20.87, 0.19, 20.38, 21.45),
    "I": (19.71, 0.25, 19.08, 20.53),
}
PATAT_FLOOR_F107 = 180.0  # approx. cycle-23-max mean F10.7 for 2000-2001

# Paranal site for the engine runs. Patat's mean site pressure is 743 hPa
# (2635 m). Paranal aerosols are pristine desert-coast (AOD550 typically
# 0.02-0.05): bracketed by the engine's continental-clean (AOD550 = 0.05,
# primary) and none (AOD 0, sensitivity). O3 column set to 260 DU (typical
# subtropical value; engine default is 345 DU). Albedo engine default 0.15.
PARANAL = {"lat": -24.63, "lon": -70.40, "elev": 2635.0}


def patat_quadratic(band, zeta):
    a0, a1, a2 = PATAT_FIT[band][:3]
    x = zeta - 95.0
    return a0 + a1 * x + a2 * x * x


# ---------------------------------------------------------------------------
# Referee R2: Koomen et al. 1952 (JOSA 42, 353)
# ---------------------------------------------------------------------------
# Extraction provenance: transcribed 2026-07-02 from a 400-dpi render of the
# original scanned paper (Squarespace/talmudology mirror of JOSA 42, 353).
# Zenith (P = 90 deg) columns of Tables I and II; identical in all six
# azimuth blocks of each table (six-fold internal confirmation). Unit:
# candles per square foot; instrument: 1P22 photomultiplier + green filter
# matched to the LIGHT-ADAPTED (photopic) eye, 1.5 deg FOV, calibrated
# against a Macbeth illuminometer. Their p. 353 caveat: below ~0.003 c/ft^2
# (H below about -7.5 deg) strict photopic photometry no longer describes
# the eye, but the METER stayed photopic; values are as printed.
# Sacramento Peak NM (2800 m): seven clear moonless evenings, May-June
# 1951. Maryland (30 m, rural): Jan-Mar 1951. Vertical photopic sun
# transmission: 85-90% Sacramento Peak, 75-85% Maryland (their Sec. p.354);
# engine aerosol choices below reproduce those windows.
CFT2_TO_CD_M2 = 10.7639  # 1 candle/ft^2 = 10.7639 cd/m^2 (unit identity)

KOOMEN_H_DEG = [0.0, -3.0, -6.0, -9.0, -12.0, -15.0]  # solar altitude
KOOMEN_SACPEAK_CFT2 = [8.0, 1.0, 0.022, 0.00075, 0.000076, 0.000020]
KOOMEN_MARYLAND_CFT2 = [15.0, 2.0, 0.06, 0.0015, 0.00012, 0.00004]
# Abstract + p. 355: "For H from about -3 to -11 deg the entire sky changed
# in brightness at about the same rate of a factor of 10 for each 2 deg
# change in H" = 1.25 mag/deg.
KOOMEN_SLOPE_MAG_PER_DEG = 1.25
KOOMEN_SLOPE_RANGE = (-11.0, -3.0)
# Fig. 1 night asymptote at H = -18..-20 deg: ~1.3-1.5e-5 c/ft^2. FLAGGED:
# read off the log-axis figure (digitization), not a printed number.
KOOMEN_NIGHT_ASYMPTOTE_CFT2 = (1.3e-5, 1.5e-5)

SACPEAK = {"lat": 32.787, "lon": -105.820, "elev": 2800.0}
MARYLAND = {"lat": 39.0, "lon": -76.8, "elev": 30.0}

# ---------------------------------------------------------------------------
# Referee R3: night floors
# ---------------------------------------------------------------------------
# Grauer & Grauer 2021, Sci Rep 11, 23893 (arXiv 2112.01664; verified
# against the PMC full text). Cosmic Campground IDSS, New Mexico (Table 1:
# 33.4793 N, 108.9226 W, 1634 m, SQM-LU-DL, artificial glow 0.632 ucd/m^2 =
# negligible). Campaign 2018-09-04 to 2020-04-30, deep cycle 24/25 solar
# minimum, campaign-mean F10.7 = 69.77 +- 2.45 sfu (their Sec. 4).
#   - darkest night 2019-02-07/08 (JD 2458522), LST 11-14 h:
#     22.128 mpsas (stdev 0.016)
#   - ten darkest nights, average minimum: 22.07 mpsas (stdev 0.03)
#   - nightly airglow span within solar minimum alone: > 0.5 mpsas;
#     2019 PASP companion (131, 114508): 2018 full range 21.99-21.13.
# SQM absolute calibration: +-10% (~0.1 mag); differential < 0.03 mag.
GRAUER_CCIDSS = {"lat": 33.4793, "lon": -108.9226, "elev": 1634.0,
                 "date": "2019-02-07",
                 "darkest_mpsas": 22.128, "darkest_sd": 0.016,
                 "ten_night_mpsas": 22.07, "ten_night_sd": 0.03,
                 "f107": 69.77}
SQM_PREDICT_F107 = 130.0  # hard-coded default in cmd `sqm predict` (main.rs)

# ---------------------------------------------------------------------------
# Referee R4 (tertiary): hnsky.org SQM twilight fit
# ---------------------------------------------------------------------------
# https://www.hnsky.org/sqm_twilight.htm (Han Kleijn; fetched 2026-07-02).
# Single night 2017-06-26, Unihedron SQM-L, exact site undisclosed
# (Netherlands presumed; engine run uses 52.0 N 5.0 E, 0 m), site best-ever
# SQM 20.7 mpsas. Fit for sun elevation x in [-12, 0]:
#   SQM = -1.057 * x + 6.7489   (slope 1.057 mag per degree of depression)
# The page's second branch (-12..-18) cannot come from that midsummer night
# at Dutch latitude (sun depression never exceeds ~14 deg) and is ignored.
# NOT peer reviewed; amateur single-night data; shape check only.
HNSKY_SLOPE = 1.057
HNSKY_INTERCEPT = 6.7489
HNSKY_SITE = {"lat": 52.0, "lon": 5.0, "elev": 0.0, "date": "2017-06-26"}
# Skyglow input derived from the page's own numbers: total floor 20.7 mpsas
# = 0.00057 cd/m^2 minus its quoted natural sky 0.00022 cd/m^2 leaves
# ~0.35 mcd/m^2 artificial; inverting the engine's Falchi rail
# (0.092 * R^0.72 mcd, bortle.rs) gives R ~ 6.4 nW/cm^2/sr.
HNSKY_RADIANCE = 6.4

# ---------------------------------------------------------------------------
# Engine runners (subprocess wrappers around the release CLI, with caching)
# ---------------------------------------------------------------------------

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(REPO, "target", "release", "twilight-cli")


def run_to_cache(cache_path, cmd, out_flag=False):
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        with open(cache_path) as f:
            return f.read()
    print(f"  running: {' '.join(cmd)}", file=sys.stderr)
    if out_flag:
        subprocess.run(cmd + ["--out", cache_path], capture_output=True,
                       text=True, check=True)
        with open(cache_path) as f:
            return f.read()
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    with open(cache_path, "w") as f:
        f.write(out)
    return out


def compare_run(workdir, tag, site, szas, aerosol, o3_du, scattering,
                photons, seed):
    """One `compare` pass; returns {sza: [radiance at the 41 wavelengths]}."""
    cmd = [
        CLI, "compare",
        f"--lat={site['lat']}", f"--lon={site['lon']}",
        "-e", str(site["elev"]),
        "--aerosol", aerosol,
        "--scattering", scattering,
        "--sza", ",".join(f"{z:g}" for z in szas),
    ]
    if scattering != "single":
        cmd += ["--photons", str(photons), "--seed-salt", str(seed)]
    if o3_du is not None:
        cmd += ["--o3-du", str(o3_du)]
    suffix = "single" if scattering == "single" \
        else f"{scattering}_p{photons}_s{seed}"
    text = run_to_cache(os.path.join(workdir, f"{tag}_{suffix}.csv"), cmd)
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or line.startswith("sza_deg") or not line.strip():
            continue
        sza, _vz, _ra, wl, rad = line.split(",")
        out.setdefault(float(sza), {})[float(wl)] = float(rad)
    return {z: [d[w] for w in ENGINE_WL_NM] for z, d in out.items()}


def sqm_predict_run(workdir, tag, site, date, scattering="single",
                    photons=100, step_min=5, extra=()):
    """One `sqm predict` night; returns [(sza, total_cd, mag)]. NOTE: the
    sqm subcommand has no --seed-salt; hybrid runs are seed 0. MC scatter
    for those is quoted from the matched multi-seed `compare` runs."""
    cmd = [
        CLI, "sqm", "predict",
        f"--lat={site['lat']}", f"--lon={site['lon']}",
        "-e", str(site["elev"]), "--date", date,
        "--scattering", scattering, "--photons", str(photons),
        "--step-min", str(step_min),
    ] + list(extra)
    cache = os.path.join(workdir, f"sqm_{tag}_{scattering}_p{photons}.csv")
    text = run_to_cache(cache, cmd, out_flag=True)
    pts = []
    for line in text.splitlines():
        parts = line.strip().split(",")
        if len(parts) == 5 and not line.startswith(("#", "time_local")):
            pts.append((float(parts[2]), float(parts[3]), float(parts[4])))
    return pts


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def mean_se(values):
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, math.sqrt(var / n)


def linfit_slope(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx


# ---------------------------------------------------------------------------
# Cross-check: the Python photopic rail against the engine's sqm rail
# ---------------------------------------------------------------------------

def check_zero_point(workdir):
    """`sqm predict` converts via mesopic luminance, which equals photopic
    above 5 cd/m^2. Compare the Python photopic mag from `compare` spectra
    against the sqm mag at matching bright-twilight SZAs of the same
    atmosphere (US Standard, no aerosol, default O3). Must agree < 0.01."""
    pts = sqm_predict_run(workdir, "paranal_floor_2005-12-01", PARANAL,
                          "2005-12-01")
    bright = [(z, cd, m) for z, cd, m in pts[:4] if cd > 5.0]
    szas = [round(z, 3) for z, _, _ in bright]
    grid = compare_run(workdir, "xcheck_paranal_none", PARANAL, szas,
                       "none", None, "single", 1, 0)
    worst = 0.0
    for (z, _cd, m_engine) in bright:
        m_py = lum_to_mag(photopic_luminance(grid[round(z, 3)]))
        worst = max(worst, abs(m_py - m_engine))
    return worst, pts


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def band_mags(grids, band, floor_cd=0.0):
    """Per-SZA (mean mag, SE mag, n) across seeds. Averaging happens in
    luminance/flux space (MC noise is heavy-tailed in mag space); SE is
    propagated to magnitudes as 1.0857 * SE_L / L."""
    out = {}
    for z in sorted(grids[0].keys()):
        if band == "V":
            vals = [photopic_luminance(g[z]) + floor_cd for g in grids]
            m, se = mean_se(vals)
            mag = lum_to_mag(m)
        else:
            vals = [b_band_flux(g[z]) for g in grids]
            m, se = mean_se(vals)
            mag = b_flux_to_mag(m)
        out[z] = (mag, 1.0857 * se / m if m > 0 else 0.0, len(vals))
    return out


def sec_patat(workdir, grids_cc, grids_none, grid_single, floor_cd):
    print("\n## R1: Patat et al. 2006, Paranal V and B\n")
    print(f"Engine floor added to V: {floor_cd:.3e} cd/m^2 "
          f"({lum_to_mag(floor_cd):.2f} mpsas, sqm predict Paranal "
          f"2005-12-01 darkest, F10.7=130)\n")

    v_tot = band_mags(grids_cc, "V", floor_cd)
    v_none = band_mags(grids_none, "V", floor_cd)
    b_twi = band_mags(grids_cc, "B", 0.0)

    def patat_b_twilight(z):
        """Patat's B fit with his own B night floor (22.64, solar max)
        subtracted in flux space: the twilight-only measured estimate.
        Only meaningful where the fit is above the floor."""
        tot = 10 ** (-0.4 * patat_quadratic("B", z))
        fl = 10 ** (-0.4 * PATAT_NIGHT_SKY["B"][0])
        return -2.5 * math.log10(tot - fl) if tot > fl else None

    print("| zeta | depr | engine V (cc+floor) | SE | engine V (none+floor) | Patat V | dV | engine B (twilight) | Patat B fit | Patat B twi-only | dB(twi) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for z in sorted(v_tot):
        in_fit = 95 <= z <= 105
        pv = patat_quadratic("V", z) if in_fit else None
        pb = patat_quadratic("B", z) if in_fit else None
        pbt = patat_b_twilight(z) if in_fit else None
        ev, sev, _ = v_tot[z]
        en, _, _ = v_none[z]
        eb, seb, _ = b_twi[z]
        pvs = f"{pv:.2f}" if pv else "-"
        pbs = f"{pb:.2f}" if pb else "-"
        pbts = f"{pbt:.2f}" if pbt else "-"
        dvs = f"{ev - pv:+.2f}" if pv else "-"
        dbs = f"{eb - pbt:+.2f}" if pbt else "-"
        print(f"| {z:.0f} | {z-90:.0f} | {ev:.2f} | {sev:.2f} | {en:.2f} | "
              f"{pvs} | {dvs} | {eb:.2f} | {pbs} | {pbts} | {dbs} |")

    # Shape metrics
    for label, lo, hi in [("95-100 (Patat gamma range)", 95, 100),
                          ("96-106 (deep-tail shape)", 96, 106)]:
        zs = [z for z in sorted(v_tot) if lo <= z <= hi]
        s_v = linfit_slope(zs, [v_tot[z][0] for z in zs])
        zs_p = [z for z in zs if z <= 105]
        s_pv = linfit_slope(zs_p, [patat_quadratic("V", z) for z in zs_p])
        # B is twilight-only on the engine side: cap at 103, past which
        # Patat's fit is >15% night floor in B.
        zs_b = [z for z in zs if z <= 103]
        s_b = linfit_slope(zs_b, [b_twi[z][0] for z in zs_b])
        s_pb = linfit_slope(zs_b, [patat_quadratic("B", z) for z in zs_b])
        print(f"\nSlope {label}: engine V {s_v:.3f} vs Patat V {s_pv:.3f} "
              f"(gamma_V = 1.14 +- 0.02 over 95-100); "
              f"engine B {s_b:.3f} vs Patat B {s_pb:.3f} "
              f"(gamma_B = 1.24 +- 0.01), both mag/deg")

    # Floor + merge point
    fl_v = PATAT_NIGHT_SKY["V"]
    eng_floor_at_max = lum_to_mag(refloor(floor_cd, SQM_PREDICT_F107,
                                          PATAT_FLOOR_F107))
    print(f"\nFloor: engine {lum_to_mag(floor_cd):.2f} (F10.7=130); restated "
          f"at Patat's solar-max epoch (F10.7~{PATAT_FLOOR_F107:.0f}) via the "
          f"engine airglow rail: {eng_floor_at_max:.2f}; Patat 2003a V = "
          f"{fl_v[0]} +- {fl_v[1]} (range {fl_v[2]}-{fl_v[3]})")
    merged = [z for z in sorted(v_tot)
              if v_tot[z][0] > lum_to_mag(floor_cd) - 0.1]
    if merged:
        print(f"Merge: engine V within 0.1 mag of floor from zeta = "
              f"{merged[0]:.0f}; Patat: 105-106 (all bands)")
    # single-scattering collapse note
    v_s = {z: lum_to_mag(photopic_luminance(grid_single[z]))
           for z in sorted(grid_single)}
    z_dead = next((z for z in sorted(v_s) if v_s[z] > 23.0), None)
    print(f"Single scattering alone falls below the night floor at zeta ~ "
          f"{z_dead:.0f}: the 96-106 range is decided by MC multiple "
          f"scattering (Patat's own finding: single-scatter drops below the "
          f"night sky between 99 and 100 deg).")


def sec_koomen(workdir, grids_sp, grids_md, sp_floor_cd, md_floor_cd):
    print("\n## R2: Koomen et al. 1952, photopic zenith luminance\n")
    print("Engine natural night floor added at both sites (sqm predict "
          f"darkest, F10.7=130): Sacramento Peak {sp_floor_cd:.2e} cd/m^2, "
          f"Maryland {md_floor_cd:.2e} cd/m^2. No artificial-skyglow term "
          "is added for 1951 Maryland (no measured value exists); the "
          "deepest Maryland rows inherit that unknown.\n")
    for name, grids, table, site_floor in [
            ("Sacramento Peak (2800 m, May-Jun 1951, continental-clean)",
             grids_sp, KOOMEN_SACPEAK_CFT2, sp_floor_cd),
            ("Maryland (30 m, Jan-Mar 1951, continental-average)",
             grids_md, KOOMEN_MARYLAND_CFT2, md_floor_cd)]:
        print(f"\n### {name}\n")
        print("| H (deg) | engine L (cd/m^2) | SE | measured L (cd/m^2) | delta (mag) |")
        print("|---|---|---|---|---|")
        engine_pts = []
        for h, cft2 in zip(KOOMEN_H_DEG, table):
            z = 90.0 - h
            vals = [photopic_luminance(g[z]) for g in grids]
            if site_floor is not None:
                vals = [v + site_floor for v in vals]
            m, se = mean_se(vals)
            meas = cft2 * CFT2_TO_CD_M2
            dmag = -2.5 * math.log10(m / meas)
            engine_pts.append((h, m))
            print(f"| {h:.0f} | {m:.3e} | {se:.1e} | {meas:.3e} | {dmag:+.2f} |")
        # decay slope over their -3..-11 window (engine points -3,-6,-9)
        pts = [(h, m) for h, m in engine_pts if -11 <= h <= -3]
        slope = linfit_slope([h for h, _ in pts],
                             [-2.5 * math.log10(m) for _, m in pts])
        print(f"\nDecay rate H in [-9, -3]: engine {slope:.2f} mag/deg vs "
              f"Koomen 'factor 10 per 2 deg' = 1.25 mag/deg over [-11, -3]")
    lo, hi = KOOMEN_NIGHT_ASYMPTOTE_CFT2
    l18 = mean_se([photopic_luminance(g[108.0]) for g in grids_sp])[0] \
        + sp_floor_cd
    print(f"\nNight asymptote (Fig. 1, DIGITIZED): "
          f"{lo * CFT2_TO_CD_M2:.2e}-{hi * CFT2_TO_CD_M2:.2e} cd/m^2 "
          f"({lum_to_mag(hi * CFT2_TO_CD_M2):.2f}-"
          f"{lum_to_mag(lo * CFT2_TO_CD_M2):.2f} mpsas); engine Sacramento "
          f"Peak twilight+floor at H = -18: {l18:.2e} cd/m^2 "
          f"({lum_to_mag(l18):.2f} mpsas; floor {lum_to_mag(sp_floor_cd):.2f} "
          f"at F10.7=130; 1951 was the cycle-18 declining phase).")


def sec_grauer(workdir):
    print("\n## R3: night floors (Grauer CCIDSS solar-min; Patat solar-max)\n")
    g = GRAUER_CCIDSS
    pts = sqm_predict_run(workdir, "ccidss_2019-02-07",
                          {k: g[k] for k in ("lat", "lon", "elev")},
                          g["date"])
    darkest_cd, darkest_mag = min(((cd, m) for _z, cd, m in pts))
    adj_cd = refloor(darkest_cd, SQM_PREDICT_F107, g["f107"])
    print(f"Engine CCIDSS {g['date']} darkest: {darkest_mag:.2f} mpsas at "
          f"F10.7=130 (sqm predict default); restated at the campaign's "
          f"measured F10.7 = {g['f107']} via the engine airglow rail: "
          f"{lum_to_mag(adj_cd):.2f} mpsas.")
    print(f"Measured: {g['darkest_mpsas']} +- {g['darkest_sd']} (that exact "
          f"night); {g['ten_night_mpsas']} +- {g['ten_night_sd']} (ten-night "
          f"average of minima). SQM absolute calibration +-0.1 mag.")
    print("Solar-cycle spread: Grauer 2019/2021 measure >0.5 mpsas nightly "
          "airglow span WITHIN solar minimum (2018 range 21.99-21.13); "
          "literature full-cycle spread ~0.4-0.65 mag (Benn & Ellison, "
          "Krisciunas), Patat 2003a: ~0.5 mag (Walker 1988). Past depression "
          "~14-16 deg the measured sky is airglow-dominated and any single-"
          "epoch comparison carries that spread.")


def sec_hnsky(workdir):
    print("\n## R4 (tertiary): hnsky.org SQM twilight slope\n")
    pts = sqm_predict_run(workdir, "hnsky_2017-06-26",
                          {k: HNSKY_SITE[k] for k in ("lat", "lon", "elev")},
                          HNSKY_SITE["date"], scattering="hybrid",
                          photons=1000,
                          extra=["--radiance", str(HNSKY_RADIANCE)])
    sel = [(z - 90.0, m) for z, _cd, m in pts if 90.0 <= z <= 102.0]
    slope = linfit_slope([d for d, _ in sel], [m for _, m in sel])
    d12 = [m for d, m in sel if 11.0 <= d <= 12.5]
    print(f"Engine (sqm predict hybrid, seed 0, radiance {HNSKY_RADIANCE} "
          f"nW/cm^2/sr from the site's own floor): slope over depression "
          f"0-12 = {slope:.3f} mag/deg; hnsky fit slope {HNSKY_SLOPE} "
          f"mag/deg (intercept {HNSKY_INTERCEPT}).")
    if d12:
        hn = HNSKY_SLOPE * 12.0 + HNSKY_INTERCEPT
        print(f"At depression 12: engine {sum(d12)/len(d12):.2f} vs hnsky "
              f"fit {hn:.2f} mpsas.")
    print("FLAGS: single amateur night, site coordinates assumed (52N 5E), "
          "SQM-L (not LU), fit includes the site's light-pollution floor.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Measured-sky referee for the deep-twilight tail")
    ap.add_argument("--workdir",
                    default=os.path.join(REPO, "validation", "measured_sky_runs"))
    ap.add_argument("--photons", type=int, default=2000,
                    help="MC secondary rays per LOS step (hybrid)")
    ap.add_argument("--seeds", type=int, default=8,
                    help="seed-salt passes for the Paranal grid")
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)

    print("# Measured-sky referee: engine vs published twilight data")
    print(f"\nconfig: photons={args.photons} seeds={args.seeds} "
          f"workdir={args.workdir}")

    worst, paranal_night = check_zero_point(args.workdir)
    print(f"\nZero-point cross-check (Python photopic rail vs engine sqm "
          f"rail, bright twilight): max |dmag| = {worst:.4f} "
          f"({'OK' if worst < 0.01 else 'FAIL'})")
    assert worst < 0.01, "zero-point rails diverged; do not trust the tables"
    paranal_floor_cd = min(cd for _z, cd, _m in paranal_night)

    szas_fine = [90.0 + i for i in range(21)]
    szas_koomen = [90.0, 93.0, 96.0, 99.0, 102.0, 105.0, 108.0]
    seeds = list(range(1, args.seeds + 1))

    grids_cc = [compare_run(args.workdir, "paranal_cc", PARANAL, szas_fine,
                            "continental-clean", 260, "hybrid",
                            args.photons, s) for s in seeds]
    grids_none = [compare_run(args.workdir, "paranal_none", PARANAL,
                              szas_fine, "none", 260, "hybrid",
                              args.photons, s) for s in seeds[:2]]
    grid_single = compare_run(args.workdir, "paranal_cc", PARANAL, szas_fine,
                              "continental-clean", 260, "single", 0, 0)
    grids_sp = [compare_run(args.workdir, "sacpeak_cc", SACPEAK, szas_koomen,
                            "continental-clean", None, "hybrid",
                            args.photons, s) for s in seeds[:4]]
    grids_md = [compare_run(args.workdir, "maryland_ca", MARYLAND,
                            szas_koomen, "continental-average", None,
                            "hybrid", args.photons, s) for s in seeds[:4]]
    sp_night = sqm_predict_run(args.workdir, "sacpeak_floor_1951-06-01",
                               SACPEAK, "1951-06-01")
    sp_floor_cd = min(cd for _z, cd, _m in sp_night)
    md_night = sqm_predict_run(args.workdir, "maryland_floor_1951-02-01",
                               MARYLAND, "1951-02-01")
    md_floor_cd = min(cd for _z, cd, _m in md_night)

    sec_patat(args.workdir, grids_cc, grids_none, grid_single,
              paranal_floor_cd)
    sec_koomen(args.workdir, grids_sp, grids_md, sp_floor_cd, md_floor_cd)
    sec_grauer(args.workdir)
    sec_hnsky(args.workdir)


if __name__ == "__main__":
    main()
