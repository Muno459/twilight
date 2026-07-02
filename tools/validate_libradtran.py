#!/usr/bin/env python3
"""Validate twilight's radiative transfer against libRadtran (uvspec).

This is the project's primary external truth anchor. It generates matched
uvspec input decks, runs both codes over the same (SZA, view) grid, and
reports per-wavelength relative errors with pass/fail tolerances.

Tiers
-----
1a. Rayleigh-only sky radiance vs DISORT (pseudospherical), SZA 60-95.
    Isolates geometry + Rayleigh phase + optics. Target |rel err| < 5%.
1b. Rayleigh-only sky radiance vs MYSTIC (spherical 1D MC), SZA 90-108.
    The twilight regime proper - DISORT plane-parallel/PS is not valid much
    past SZA ~95; only a spherical solver is. Target |rel err| < 10%
    (MYSTIC MC noise + twilight sensitivity). THIS TIER ANSWERS THE
    100-km-CEILING QUESTION: if twilight's radiance collapses at SZA >= 104
    while MYSTIC's does not, the ceiling (or missing multiple scattering)
    is quantified directly.
2.  Column optics: direct + diffuse irradiance at SZA 60-85 sweeping
    O3 in {220, 347, 546} DU and aerosol on/off. Tests gas absorption and
    aerosol extinction. Target < 5% (clear), < 10% (aerosol).
3.  Full field with aerosol vs MYSTIC, then V(lambda) luminance vs SZA and
    the implied threshold-crossing SZAs - the photometric bottom line.

Install libRadtran (not packaged for brew; build from source):
    curl -LO http://www.libradtran.org/download/libRadtran-2.0.6.tar.gz
    tar xzf libRadtran-2.0.6.tar.gz && cd libRadtran-2.0.6
    ./configure && make            # gfortran + flex/bison required
    export LIBRADTRAN_DIR=$PWD     # this script reads it

Usage:
    python3 tools/validate_libradtran.py --tier 1a          # run + compare
    python3 tools/validate_libradtran.py --tier 1a --decks-only   # just emit decks
    python3 tools/validate_libradtran.py --all

Conventions (verified against the libRadtran 2.x manual):
  - uvspec `umu` is the cosine of the polar angle of the RADIANCE direction:
    umu < 0 is downwelling (ground observer looking up). Looking at zenith
    => umu = -1.0. twilight's view_zenith t (deg from straight up) maps to
    umu = -cos(t * pi/180).
  - uvspec `phi` / `phi0`: phi - phi0 = 0 looks toward the sun (principal
    plane), 180 anti-solar. twilight's rel_azimuth maps to phi = phi0 +
    rel_azimuth.
  - uvspec radiance output `uu` is in mW/m^2/nm/sr when `source solar` is in
    mW/m^2/nm (the default for most bundled solar files) - the script
    normalizes both codes to W/m^2/sr/nm and ALSO offers --shape-only to
    compare shapes when the solar files differ.
  - Solar spectrum: twilight uses TSIS-1 HSRS. Pass --solar-file to point
    uvspec at a TSIS file for absolute comparison; otherwise the script
    falls back to atlas_plus_modtran and warns that absolute differences
    of a few % are expected (use --shape-only).
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TWILIGHT_CLI = REPO / "target" / "release" / "twilight-cli"
OUT_DIR = REPO / "validation"

# Matched to twilight's grid: 380-780 nm. uvspec wavelength range.
# Env-overridable for the overnight 1e8 campaign.
MC_BACKWARD_PHOTONS = int(os.environ.get("MC_BACKWARD_PHOTONS", "10000000"))
TW_DEEP_PHOTONS = int(os.environ.get("TW_DEEP_PHOTONS", "4000"))
WL_MIN, WL_MAX = 380.0, 780.0

# twilight default surface albedo (crates/twilight-cli pray/compare default)
ALBEDO = 0.15

# Default O3 column: twilight's *actual* standard-profile column. The
# docstring value "~347 DU" in twilight-weather is known to disagree with
# the integral of the embedded profile (~546 DU) - pass the value you want
# tested; the tier-2 sweep covers both.
O3_DU_DEFAULT = None  # None = leave both codes on their native defaults


def libradtran_dir() -> Path | None:
    env = os.environ.get("LIBRADTRAN_DIR")
    if env and (Path(env) / "bin" / "uvspec").exists():
        return Path(env)
    uvspec = shutil.which("uvspec")
    if uvspec:
        return Path(uvspec).resolve().parent.parent
    return None


def deck_common(lrt: Path | None, solar_file: str | None, o3_du: float | None,
                rayleigh_only: bool) -> str:
    """Common deck header shared by all tiers."""
    data = (lrt / "data") if lrt else Path("<LIBRADTRAN_DIR>/data")
    lines = [
        f"data_files_path {data}/",
        f"atmosphere_file {data}/atmmod/afglus.dat   # US Standard 1976",
        f"albedo {ALBEDO}",
        f"wavelength {WL_MIN} {WL_MAX}",
        "mol_abs_param crs                  # cross-section absorption (UV/vis)",
    ]
    if solar_file:
        lines.insert(2, f"source solar {solar_file}")
    else:
        lines.insert(2, f"source solar {data}/solar_flux/apm_1nm   # atlas_plus_modtran 1nm")
    if rayleigh_only:
        lines.append("no_absorption mol                  # pure Rayleigh tier")
    elif o3_du is not None:
        lines.append(f"mol_modify O3 {o3_du:.1f} DU")
    return "\n".join(lines)


def deck_radiance(common: str, solver: str, sza: float, umus: list[float],
                  phis: list[float], wl: tuple[float, float] | None = None) -> str:
    """Radiance deck: ground observer looking up."""
    umu_s = " ".join(f"{u:.6f}" for u in umus)
    phi_s = " ".join(f"{p:.1f}" for p in phis)
    solver_lines = {
        "disort": "rte_solver disort\nnumber_of_streams 16\npseudospherical",
        "mystic": ("rte_solver montecarlo\nmc_spherical 1D\n"
                   "mc_photons 2000000\nmc_vroom on"),
        # Backward mode: traces from the zenith sensor toward the sun,
        # which is the only tractable geometry at SZA >= 98 (forward MC
        # goes photon-starved: the sky there is 1e-7..1e-9 of TOA).
        # Verified against forward MYSTIC at SZA 95 before use.
        "mystic-backward": ("rte_solver montecarlo\nmc_spherical 1D\n"
                            f"mc_photons {MC_BACKWARD_PHOTONS}\n"
                            "mc_backward\nmc_vroom on"),
    }[solver]
    body = common
    if wl is not None:
        # Replace the broadband wavelength line for single-wavelength MC runs.
        body = "\n".join(
            f"wavelength {wl[0]:.1f} {wl[1]:.1f}" if l.startswith("wavelength ") else l
            for l in common.splitlines()
        )
    return f"""{body}
{solver_lines}
sza {sza:.2f}
phi0 0.0
umu {umu_s}
phi {phi_s}
zout 0.0
output_user lambda uu
quiet
"""


def deck_irradiance(common: str, sza: float) -> str:
    return f"""{common}
rte_solver disort
number_of_streams 16
pseudospherical
sza {sza:.2f}
zout 0.0
output_user lambda edir edn
quiet
"""


def run_uvspec(lrt: Path, deck: str, tag: str) -> str:
    OUT_DIR.mkdir(exist_ok=True)
    inp = OUT_DIR / f"{tag}.inp"
    out = OUT_DIR / f"{tag}.out"
    inp.write_text(deck)
    with open(inp) as fi, open(out, "w") as fo:
        r = subprocess.run([str(lrt / "bin" / "uvspec")], stdin=fi, stdout=fo,
                           stderr=subprocess.PIPE, text=True, cwd=OUT_DIR)
    if r.returncode != 0:
        sys.exit(f"uvspec failed for {tag}:\n{r.stderr[:2000]}")
    return out.read_text()


def run_twilight_compare(szas, view_zeniths, rel_azimuths, rayleigh_only,
                         o3_du, scattering="single", photons=10000,
                         no_refraction=False) -> dict:
    """Returns {(sza, vz, ra, wl): radiance_W_m2_sr_nm}."""
    if not TWILIGHT_CLI.exists():
        sys.exit("Build first: cargo build --release -p twilight-cli")
    cmd = [
        str(TWILIGHT_CLI), "compare",
        "--sza", ",".join(f"{s:g}" for s in szas),
        "--view-zenith", ",".join(f"{v:g}" for v in view_zeniths),
        "--rel-azimuth", ",".join(f"{r:g}" for r in rel_azimuths),
        "--scattering", scattering,
        "--photons", str(photons),
    ]
    if rayleigh_only:
        cmd.append("--rayleigh-only")
    cmd.append("--fast")  # scalar radiance (I); polarization correction ~0.5-2%
    if no_refraction:
        cmd.append("--no-refraction")
    if o3_du is not None:
        cmd += ["--o3-du", f"{o3_du:g}"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"twilight-cli compare failed:\n{r.stderr[:2000]}")
    table = {}
    reader = csv.reader(io.StringIO(r.stdout))
    for row in reader:
        if not row or row[0].startswith("#") or row[0] == "sza_deg":
            continue
        sza, vz, ra, wl, rad = (float(x) for x in row)
        table[(sza, vz, ra, wl)] = rad
    return table


def parse_uvspec_radiance(text: str, umus: list[float], phis: list[float]) -> dict:
    """Parse `output_user lambda uu` blocks.

    For multiple umu/phi, uvspec prints, per wavelength, lines of uu values.
    Format (radiance block): lambda, then for each umu a line:
       umu  u0u  uu(phi1) uu(phi2) ...
    We parse defensively and return {(wl, umu, phi): uu}.
    Units: mW/m^2/nm/sr with the default solar files -> convert to W later.
    """
    out = {}
    lines = [l for l in text.splitlines() if l.strip()]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        # wavelength header line has 1 + len(umus)*0 columns when output_user
        # is lambda uu: actually uvspec emits "lambda  uu(umu1,phi1) ..." in
        # a flat row when both lists are short. Handle the flat-row case:
        if len(parts) == 1 + len(umus) * len(phis):
            wl = float(parts[0])
            k = 1
            for u in umus:
                for p in phis:
                    out[(wl, u, p)] = float(parts[k]) * 1e-3  # mW -> W
                    k += 1
            i += 1
            continue
        i += 1
    if not out:
        sys.exit("Could not parse uvspec radiance output - inspect validation/*.out "
                 "and adapt parse_uvspec_radiance() to your libRadtran version.")
    return out


def compare_tier1(lrt, solver, szas, tol, shape_only, solar_file):
    """Tier 1a/1b: Rayleigh-only radiance."""
    view_zeniths = [0.0, 75.0]
    rel_azimuths = [0.0, 90.0, 180.0]
    umus = [-math.cos(math.radians(v)) for v in view_zeniths]
    common = deck_common(lrt, solar_file, None, rayleigh_only=True)

    print(f"\n=== Tier 1 ({solver}, Rayleigh-only, SZA {szas[0]}-{szas[-1]}) ===")
    # twilight hybrid = exact order-1 + MC orders 2+ (all orders), matching
    # DISORT's full field. Single-scatter-only vs full DISORT differs by the
    # multiple-scattering fraction (30-60% for Rayleigh) and is NOT a fair test.
    tw = run_twilight_compare(szas, view_zeniths, rel_azimuths, True, None,
                              scattering="hybrid", photons=300)

    n_pass = n_fail = 0
    rows = []
    for sza in szas:
        deck = deck_radiance(common, solver, sza, umus, rel_azimuths)
        tag = f"tier1_{solver}_sza{sza:g}"
        if lrt is None:
            (OUT_DIR / f"{tag}.inp").parent.mkdir(exist_ok=True)
            (OUT_DIR / f"{tag}.inp").write_text(deck)
            continue
        lr = parse_uvspec_radiance(run_uvspec(lrt, deck, tag), umus, rel_azimuths)
        # normalize per-(vz,ra): optionally shape-only (scale by 550nm ratio)
        for vz, umu in zip(view_zeniths, umus):
            for ra in rel_azimuths:
                pairs = []
                for (wl_l, u_l, p_l), v_l in lr.items():
                    if abs(u_l - umu) < 1e-6 and abs(p_l - ra) < 1e-6:
                        # nearest twilight wavelength (10nm grid)
                        wl_t = round(wl_l / 10) * 10
                        v_t = tw.get((sza, vz, ra, float(wl_t)))
                        if v_t is not None and v_l > 0:
                            pairs.append((wl_l, v_t, v_l))
                if not pairs:
                    continue
                scale = 1.0
                if shape_only:
                    mid = min(pairs, key=lambda x: abs(x[0] - 550))
                    if mid[1] > 0:
                        scale = mid[2] / mid[1]
                for wl, v_t, v_l in pairs:
                    rel = abs(v_t * scale - v_l) / v_l
                    ok = rel <= tol
                    n_pass += ok
                    n_fail += not ok
                    rows.append((sza, vz, ra, wl, v_t * scale, v_l, rel, ok))

    if lrt is None:
        print(f"  uvspec not found - decks written to {OUT_DIR}/ (run with "
              f"LIBRADTRAN_DIR set to compare)")
        return True
    _report(rows, n_pass, n_fail, tol)
    return n_fail == 0



def compare_tier1b_mystic(lrt, szas, tol, shape_only, solar_file,
                          solver="mystic", tw_photons=500):
    """Tier 1b: deep-twilight ZENITH radiance vs MYSTIC spherical MC.

    One single-direction (zenith) MYSTIC run per (SZA, wavelength): MYSTIC
    reports radiance in mc.rad.spc (stdout uu columns are zero), one row
    per wavelength, and multi-direction sampling is awkward (mc_vroom
    conflicts; only the first direction lands in the .spc). Zenith is also
    the canonical observable for the deep-twilight/100-km-ceiling check.
    """
    wls = [450.0, 550.0, 650.0]
    print(f"\n=== Tier 1b (MYSTIC spherical, zenith, SZA {szas[0]}-{szas[-1]}) ===")
    if lrt is None:
        print("  decks-only mode: install libRadtran and set LIBRADTRAN_DIR")
        return True

    common = deck_common(lrt, solar_file, None, rayleigh_only=True)
    # Straight shadow rays: MYSTIC mc_spherical does not refract - remove
    # twilight's refraction for the apples-to-apples run.
    tw = run_twilight_compare(szas, [0.0], [0.0], True, None,
                              scattering="hybrid", photons=tw_photons,
                              no_refraction=True)

    n_pass = n_fail = 0
    rows = []
    for sza in szas:
        for w in wls:
            deck = deck_radiance(common, solver, sza, [-1.0], [0.0], wl=(w, w))
            tag = f"tier1b_sza{sza:g}_wl{w:g}"
            run_uvspec(lrt, deck, tag)
            spc = OUT_DIR / "mc.rad.spc"
            if not spc.exists():
                sys.exit(f"MYSTIC did not produce {spc}")
            # rows: wl ix iy iz radiance  (single direction)
            val = None
            for line in spc.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 5 and abs(float(parts[0]) - w) < 0.51:
                    val = float(parts[4]) * 1e-3  # mW -> W
            if val is None:
                continue
            v_t = tw.get((sza, 0.0, 0.0, w))
            if v_t is None:
                continue
            rows.append([sza, 0.0, 0.0, w, v_t, val])

    # shape normalization across each SZA using 550nm
    out_rows = []
    for sza in szas:
        sub = [r for r in rows if r[0] == sza and r[5] > 0]
        if not sub:
            print(f"  SZA {sza:5.1f}: MYSTIC returned zero radiance (deeper than its reach?)")
            continue
        scale = 1.0
        if shape_only:
            mid = min(sub, key=lambda r: abs(r[3] - 550))
            if mid[4] > 0:
                scale = mid[5] / mid[4]
        for r in sub:
            rel = abs(r[4] * scale - r[5]) / r[5]
            ok = rel <= tol
            n_pass += ok
            n_fail += not ok
            out_rows.append((r[0], r[1], r[2], r[3], r[4] * scale, r[5], rel, ok))
            print(f"  SZA {r[0]:5.1f} wl {r[3]:5.0f}: twilight={r[4]*scale:.4e} mystic={r[5]:.4e} rel={rel:6.1%}{'' if ok else '  <-- FAIL'}")

    _report(out_rows, n_pass, n_fail, tol)
    return n_fail == 0

# ---------------------------------------------------------------------------
# G2: explicit-cloud slab referee (docs/3D_TRANSPORT_PLAN.md gate G2)
# ---------------------------------------------------------------------------
# twilight's cloud channel is DELTA-EDDINGTON SCALED at build time
# (twilight-data builder::add_cloud_layer): for unscaled (tau, ssa, g) the
# medium actually carries
#   tau_ext* = tau * (1 - ssa*g^2)          extinction OD of the deck
#   ssa*     = (1-g^2)*ssa / (1 - ssa*g^2)  single-scattering albedo
#   g*       = g / (1+g)                    HG asymmetry
# with the scattering part in a gray per-shell cloud channel and the
# absorption part folded into the shell optics. The referee therefore
# solves the SAME transport problem by configuring the libRadtran water
# cloud with tau_ext*, ssa*, g* and a Henyey-Greenstein phase function
# (wc_properties hu is HG by construction; wc_modify tau/ssa/gg set).
# This gates the CHAIN MACHINERY, not the delta-scaling approximation.
G2_CLOUD_SSA = 0.999   # twilight water-cloud preset constants
G2_CLOUD_G = 0.85
G2_CLOUD_BASE_KM = 1.0  # afglus grid levels -> exact layer match
G2_CLOUD_TOP_KM = 2.0
G2_F = G2_CLOUD_G * G2_CLOUD_G
G2_DE_SCALE = 1.0 - G2_CLOUD_SSA * G2_F
G2_SSA_STAR = (1.0 - G2_F) * G2_CLOUD_SSA / G2_DE_SCALE
G2_G_STAR = G2_CLOUD_G / (1.0 + G2_CLOUD_G)
G2_TAU_STARS = [1.0, 3.0, 10.0]     # scaled extinction ODs under test
G2_SZAS = [30.0, 60.0, 85.0]        # 85 = stretch (pseudospherical disort)
G2_WLS = [450.0, 550.0, 650.0]
G2_VZ = [0.0, 60.0]                 # zenith + one off-zenith
G2_RA = [0.0, 180.0]
G2_SEEDS = list(range(1, 7))
G2_PHOTONS = {"hybrid": 2000, "multiple": 8000}
G2_MYSTIC_PHOTONS = int(os.environ.get("G2_MYSTIC_PHOTONS", "4000000"))


def g2_solar_file() -> Path:
    """uvspec solar file = twilight's exact TSIS-1 10nm table (mW/m^2/nm).

    Duplicated from crates/twilight-data/src/solar_spectrum.rs so the two
    codes share the identical solar scale (absolute comparison, no
    shape-only normalization needed).
    """
    irr = [1.119, 1.068, 1.527, 1.714, 1.744, 1.638, 1.810, 2.087, 2.024,
           1.948, 2.005, 1.946, 1.940, 1.889, 1.863, 1.843, 1.824, 1.848,
           1.833, 1.803, 1.780, 1.694, 1.704, 1.693, 1.639, 1.636, 1.594,
           1.580, 1.544, 1.515, 1.486, 1.438, 1.413, 1.389, 1.360, 1.323,
           1.296, 1.265, 1.194, 1.244, 1.216]
    p = OUT_DIR / "g2_solar_tsis_tw.dat"
    lines = ["# twilight TSIS-1 HSRS 10nm table (solar_spectrum.rs), mW/m^2/nm"]
    for i, e in enumerate(irr):
        lines.append(f"{380 + 10 * i:.1f} {e * 1000:.3f}")
    p.write_text("\n".join(lines) + "\n")
    return p


def g2_wc_file() -> Path:
    """1D wc file: one layer between base and top (layer interpretation).

    LWC/reff are placeholders; wc_modify tau/ssa/gg override everything.
    """
    p = OUT_DIR / "g2_wc_slab.dat"
    p.write_text(f"# z(km) LWC(g/m^3) reff(um)\n"
                 f"{G2_CLOUD_TOP_KM:.1f} 0.0 0.0\n"
                 f"{G2_CLOUD_BASE_KM:.1f} 0.1 10.0\n")
    return p


def g2_deck(lrt: Path, solar: Path, wc: Path, tau_star: float | None,
            solver: str, sza: float, wl: float, umus: list[float],
            phis: list[float]) -> str:
    data = lrt / "data"
    cloud = ""
    if tau_star is not None:
        cloud = (f"wc_file 1D {wc}\n"
                 f"wc_properties hu\n"
                 f"wc_modify tau set {tau_star:.6f}\n"
                 f"wc_modify ssa set {G2_SSA_STAR:.6f}\n"
                 f"wc_modify gg set {G2_G_STAR:.6f}\n")
    solver_lines = {
        "disort": "rte_solver disort\nnumber_of_streams 32\npseudospherical",
        "mystic": f"rte_solver montecarlo\nmc_photons {G2_MYSTIC_PHOTONS}\nmc_std",
    }[solver]
    umu_s = " ".join(f"{u:.6f}" for u in umus)
    phi_s = " ".join(f"{p:.1f}" for p in phis)
    return f"""data_files_path {data}/
atmosphere_file {data}/atmmod/afglus.dat
source solar {solar} per_nm
albedo {ALBEDO}
wavelength {wl:.1f} {wl:.1f}
mol_abs_param crs
no_absorption mol
{cloud}{solver_lines}
sza {sza:.2f}
phi0 0.0
umu {umu_s}
phi {phi_s}
zout 0.0
output_user lambda uu
quiet
"""


def g2_run_twilight(tau_star: float | None, mode: str, seed: int) -> dict:
    """One twilight compare run over the full G2 grid. Returns
    {(sza, vz, ra, wl): W/m^2/sr/nm}."""
    cmd = [
        str(TWILIGHT_CLI), "compare",
        "--sza", ",".join(f"{s:g}" for s in G2_SZAS),
        "--view-zenith", ",".join(f"{v:g}" for v in G2_VZ),
        "--rel-azimuth", ",".join(f"{r:g}" for r in G2_RA),
        "--rayleigh-only", "--fast", "--no-refraction",
        "--scattering", mode,
        "--photons", str(G2_PHOTONS[mode]),
        "--seed-salt", str(seed),
    ]
    if tau_star is not None:
        tau_unscaled = tau_star / G2_DE_SCALE
        cmd += ["--cloud-tau", f"{tau_unscaled:.6f}",
                "--cloud-base-km", f"{G2_CLOUD_BASE_KM:g}",
                "--cloud-top-km", f"{G2_CLOUD_TOP_KM:g}",
                "--cloud-ssa", f"{G2_CLOUD_SSA:g}",
                "--cloud-g", f"{G2_CLOUD_G:g}"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"twilight-cli compare failed ({mode}, tau*={tau_star}):\n"
                 f"{r.stderr[:2000]}")
    table = {}
    for row in csv.reader(io.StringIO(r.stdout)):
        if not row or row[0].startswith("#") or row[0] == "sza_deg":
            continue
        sza, vz, ra, wl, rad = (float(x) for x in row)
        if wl in G2_WLS:
            table[(sza, vz, ra, wl)] = rad
    return table


def g2_mean_se(vals: list[float]) -> tuple[float, float]:
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, math.sqrt(var / n)


def compare_g2(lrt):
    """G2 referee campaign: twilight chain estimators vs disort/MYSTIC on
    the delta-Eddington-scaled homogeneous cloud slab."""
    if lrt is None:
        sys.exit("G2 needs libRadtran (set LIBRADTRAN_DIR)")
    if not TWILIGHT_CLI.exists():
        sys.exit("Build first: cargo build --release -p twilight-cli")

    solar = g2_solar_file()
    wc = g2_wc_file()
    umus = [-math.cos(math.radians(v)) for v in G2_VZ]

    print("=== G2: explicit-cloud slab referee ===")
    print(f"  scaled deck constants: de_scale={G2_DE_SCALE:.7f} "
          f"ssa*={G2_SSA_STAR:.7f} g*={G2_G_STAR:.7f}")
    for ts in G2_TAU_STARS:
        print(f"  tau*={ts:g}: twilight --cloud-tau {ts / G2_DE_SCALE:.6f} "
              f"(unscaled); uvspec wc_modify tau set {ts:.6f}")

    cases: list[float | None] = [None] + G2_TAU_STARS  # None = clear anchor

    # ---- twilight side: hybrid + multiple, multi-seed --------------------
    tw: dict = {}   # (tau_star, mode) -> {(sza,vz,ra,wl): (mean, se)}
    from concurrent.futures import ThreadPoolExecutor
    for tau_star in cases:
        for mode in ("hybrid", "multiple"):
            # hybrid compare runs are single-threaded; parallelize seeds.
            workers = 6 if mode == "hybrid" else 2
            with ThreadPoolExecutor(max_workers=workers) as ex:
                tables = list(ex.map(
                    lambda s: g2_run_twilight(tau_star, mode, s), G2_SEEDS))
            merged = {}
            for key in tables[0]:
                merged[key] = g2_mean_se([t[key] for t in tables])
            tw[(tau_star, mode)] = merged
            label = "clear" if tau_star is None else f"tau*={tau_star:g}"
            print(f"  twilight {mode:8s} {label}: {len(G2_SEEDS)} seeds done")

    # ---- disort side -----------------------------------------------------
    dis: dict = {}  # (tau_star, sza, vz, ra, wl) -> W/m^2/sr/nm
    for tau_star in cases:
        for sza in G2_SZAS:
            for wl in G2_WLS:
                tag = (f"g2_disort_"
                       f"{'clear' if tau_star is None else f'tau{tau_star:g}'}"
                       f"_sza{sza:g}_wl{wl:g}")
                deck = g2_deck(lrt, solar, wc, tau_star, "disort", sza, wl,
                               umus, G2_RA)
                out = run_uvspec(lrt, deck, tag)
                lr = parse_uvspec_radiance(out, umus, G2_RA)
                for vz, umu in zip(G2_VZ, umus):
                    for ra in G2_RA:
                        for (wl_l, u_l, p_l), v in lr.items():
                            if abs(u_l - umu) < 1e-6 and abs(p_l - ra) < 1e-6:
                                dis[(tau_star, sza, vz, ra, wl)] = v
    print("  disort referee done")

    # ---- MYSTIC MC cross-check (one direction per run) -------------------
    mystic: dict = {}  # (tau_star, sza, vz, ra, wl) -> (rad, std)
    mystic_cases = [(3.0, 60.0, 0.0, 0.0, 550.0),
                    (3.0, 30.0, 0.0, 0.0, 550.0),
                    (10.0, 60.0, 0.0, 0.0, 550.0),
                    (10.0, 30.0, 0.0, 0.0, 550.0),
                    (10.0, 60.0, 60.0, 0.0, 550.0),
                    (1.0, 60.0, 0.0, 0.0, 550.0)]
    for (ts, sza, vz, ra, wl) in mystic_cases:
        tag = f"g2_mystic_tau{ts:g}_sza{sza:g}_vz{vz:g}_ra{ra:g}_wl{wl:g}"
        umu = -math.cos(math.radians(vz))
        deck = g2_deck(lrt, solar, wc, ts, "mystic", sza, wl, [umu], [ra])
        run_uvspec(lrt, deck, tag)
        rad = std = None
        spc = OUT_DIR / "mc.rad.spc"
        stdf = OUT_DIR / "mc.rad.std.spc"
        if spc.exists():
            parts = spc.read_text().split()
            if len(parts) >= 5:
                rad = float(parts[4]) * 1e-3
        if stdf.exists():
            parts = stdf.read_text().split()
            if len(parts) >= 5:
                std = float(parts[4]) * 1e-3
        if rad is not None:
            mystic[(ts, sza, vz, ra, wl)] = (rad, std or 0.0)
            print(f"  MYSTIC tau*={ts:g} sza={sza:g} vz={vz:g}: "
                  f"{rad:.4e} +- {std or 0:.1e}")

    # ---- table ------------------------------------------------------------
    out_csv = OUT_DIR / "g2_results.csv"
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["tau_star", "sza", "vz", "ra", "wl",
                    "tw_hybrid", "tw_hybrid_se", "tw_multiple",
                    "tw_multiple_se", "disort", "mystic", "mystic_std",
                    "hyb_over_disort", "mul_over_disort"])
        for tau_star in cases:
            ts_key = 0.0 if tau_star is None else tau_star
            for sza in G2_SZAS:
                for vz in G2_VZ:
                    for ra in G2_RA:
                        if vz == 0.0 and ra != 0.0:
                            continue  # zenith is azimuth-degenerate
                        for wl in G2_WLS:
                            h, hse = tw[(tau_star, "hybrid")][(sza, vz, ra, wl)]
                            m, mse = tw[(tau_star, "multiple")][(sza, vz, ra, wl)]
                            d = dis[(tau_star, sza, vz, ra, wl)]
                            my = mystic.get((ts_key, sza, vz, ra, wl))
                            w.writerow([ts_key, sza, vz, ra, wl,
                                        f"{h:.6e}", f"{hse:.2e}",
                                        f"{m:.6e}", f"{mse:.2e}",
                                        f"{d:.6e}",
                                        f"{my[0]:.6e}" if my else "",
                                        f"{my[1]:.2e}" if my else "",
                                        f"{h / d:.4f}" if d > 0 else "",
                                        f"{m / d:.4f}" if d > 0 else ""])
    print(f"  full table: {out_csv}")

    # console summary at 550
    print("\n  550 nm summary (ratio twilight/disort, mean of seeds):")
    print("  tau*   sza  vz  ra   hybrid/disort  multiple/disort")
    for tau_star in cases:
        ts_key = 0.0 if tau_star is None else tau_star
        for sza in G2_SZAS:
            for vz, ra in [(0.0, 0.0), (60.0, 0.0), (60.0, 180.0)]:
                h, hse = tw[(tau_star, "hybrid")][(sza, vz, ra, 550.0)]
                m, mse = tw[(tau_star, "multiple")][(sza, vz, ra, 550.0)]
                d = dis[(tau_star, sza, vz, ra, 550.0)]
                print(f"  {ts_key:5g} {sza:5g} {vz:3g} {ra:4g}   "
                      f"{h / d:6.3f}+-{hse / d:.3f}   {m / d:6.3f}+-{mse / d:.3f}")
    return True


def _report(rows, n_pass, n_fail, tol):
    worst = sorted(rows, key=lambda r: -r[6])[:10]
    print(f"  {n_pass} pass / {n_fail} fail (tol {tol:.0%})")
    if worst:
        print("  worst points: sza  vz   ra    wl     twilight      libradtran   rel")
        for sza, vz, ra, wl, vt, vl, rel, ok in worst:
            flag = "" if ok else "  <-- FAIL"
            print(f"   {sza:5.1f} {vz:4.0f} {ra:5.0f} {wl:6.1f}  {vt:.4e}  {vl:.4e}  {rel:6.1%}{flag}")
    out = OUT_DIR / "report.csv"
    with open(out, "a") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)
    print(f"  full rows appended to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=["1a", "1b", "1b-deep", "2", "3", "g2"],
                    default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--decks-only", action="store_true",
                    help="emit uvspec decks without running (no libRadtran needed)")
    ap.add_argument("--shape-only", action="store_true",
                    help="normalize at 550nm to compare spectral shape only "
                         "(use when solar files differ)")
    ap.add_argument("--solar-file", default=None,
                    help="path to a TSIS-1 solar file for uvspec (absolute compare)")
    args = ap.parse_args()

    lrt = None if args.decks_only else libradtran_dir()
    if lrt is None and not args.decks_only:
        print("NOTE: uvspec not found (set LIBRADTRAN_DIR). Emitting decks only.\n"
              "Install: see module docstring.", file=sys.stderr)

    OUT_DIR.mkdir(exist_ok=True)
    ok = True
    tiers = ["1a", "1b"] if args.all else ([args.tier] if args.tier else ["1a"])
    for tier in tiers:
        if tier == "1a":
            ok &= compare_tier1(lrt, "disort", [60, 70, 80, 85, 90, 95],
                                tol=0.05, shape_only=args.shape_only,
                                solar_file=args.solar_file)
        elif tier == "1b":
            ok &= compare_tier1b_mystic(lrt, [95, 98, 100, 102, 104, 106],
                                        tol=0.10, shape_only=args.shape_only,
                                        solar_file=args.solar_file)
        elif tier == "1b-deep":
            # The deep campaign: MYSTIC backward from the zenith sensor
            # (forward is photon-starved past 98) at 1e7 photons, with a
            # heavier twilight side. SZA 95 is repeated as the anchor
            # where forward MYSTIC already passed.
            ok &= compare_tier1b_mystic(lrt, [95, 96, 98, 100, 102, 104, 106],
                                        tol=0.15, shape_only=args.shape_only,
                                        solar_file=args.solar_file,
                                        solver="mystic-backward",
                                        tw_photons=TW_DEEP_PHOTONS)
        elif tier == "g2":
            ok &= compare_g2(lrt)
        elif tier in ("2", "3"):
            print(f"Tier {tier}: deck templates not yet automated - "
                  "see module docstring for the design.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
