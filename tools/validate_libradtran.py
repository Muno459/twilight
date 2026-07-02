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


# ---------------------------------------------------------------------------
# G3-CLOUD-TWILIGHT: the uniform 1-2 km deck at TWILIGHT geometry vs
# spherical BACKWARD MYSTIC (external referee for the "clouds at twilight
# geometry" confidence row). Same delta-scaled same-problem construction
# as G2 (wc_properties hu + wc_modify tau/ssa/gg set == twilight's
# internally delta-Eddington-scaled gray cloud channel), but the referee
# is MYSTIC with mc_spherical 1D + mc_backward + mc_vroom + mc_std, the
# only public referee that is valid past SZA ~95. Zenith view, absolute
# comparison on the shared TSIS-1 10 nm solar table, no refraction on
# either side, twilight's 150 km atmosphere vs afglus's 120 km top (the
# residual geometry difference is < the stat bands through SZA 101).
#
# Expected per docs/RESULTS.md: the hybrid estimator is one-sided LOW
# under the deck at SZA >= 97 (analog-under-cloud starvation, 0.37-0.45x
# class, converges from below). That is REPORTED as the documented
# limitation, not gated. Multiple (independent analog estimator) is
# gated everywhere its own variance allows.
# ---------------------------------------------------------------------------
G3_SZAS = [95.0, 97.0, 99.0, 101.0]
G3_TAU_BOTH = [1.0, 3.0]            # both estimators, all G3_SZAS
G3_TAU_MULT_ONLY = [10.0]           # multiple-only (hybrid limitation documented)
G3_TAU10_SZAS = [95.0, 97.0]        # tau*=10 runtime cap: keep 95-97 coverage
G3_WLS = [450.0, 550.0, 650.0]
G3_SEEDS = list(range(1, 7))
G3_HYB_PHOTONS = int(os.environ.get("G3_HYB_PHOTONS", "16000"))
# multiple: per-SZA-group photon tiers (deep SZA needs far more analog rays)
G3_MUL_PHOTONS = {95.0: 500_000, 97.0: 500_000,
                  99.0: 2_000_000, 101.0: 2_000_000}
# MYSTIC backward: 1e7 where it resolves in seconds-minutes, 1e8 deep
G3_MYSTIC_PHOTONS = {95.0: 10_000_000, 97.0: 10_000_000,
                     99.0: 100_000_000, 101.0: 100_000_000}
G3_MYSTIC_WORKERS = int(os.environ.get("G3_MYSTIC_WORKERS", "5"))
G3_SYS_FLOOR = 0.05  # systematic band floor: solar-table interpolation,
                     # afglus 120 km top vs twilight 150 km, scalar (--fast)
G3_SMOKE = os.environ.get("G3_SMOKE") == "1"
if G3_SMOKE:
    G3_SZAS = [95.0, 97.0]
    G3_TAU_BOTH = [1.0]
    G3_TAU_MULT_ONLY = []
    G3_WLS = [550.0]
    G3_SEEDS = [1, 2]
    G3_HYB_PHOTONS = 1000
    G3_MUL_PHOTONS = {95.0: 50_000, 97.0: 50_000}
    G3_MYSTIC_PHOTONS = {95.0: 1_000_000, 97.0: 1_000_000}


def g3_mystic_deck(lrt: Path, solar: Path, wc: Path, tau_star: float,
                   sza: float, wl: float, photons: int) -> str:
    data = lrt / "data"
    return f"""data_files_path {data}/
atmosphere_file {data}/atmmod/afglus.dat
source solar {solar} per_nm
albedo {ALBEDO}
wavelength {wl:.1f} {wl:.1f}
mol_abs_param crs
no_absorption mol
wc_file 1D {wc}
wc_properties hu
wc_modify tau set {tau_star:.6f}
wc_modify ssa set {G2_SSA_STAR:.6f}
wc_modify gg set {G2_G_STAR:.6f}
rte_solver montecarlo
mc_spherical 1D
mc_photons {photons}
mc_backward
mc_vroom on
mc_std
sza {sza:.2f}
phi0 0.0
umu -1.000000
phi 0.0
zout 0.0
output_user lambda uu
quiet
"""


def g3_run_mystic_case(lrt: Path, solar: Path, wc: Path, tau_star: float,
                       sza: float, wl: float):
    """One spherical-backward MYSTIC run in its own work dir (uvspec drops
    mc.rad.spc/randomseed in cwd; per-case dirs make the pool safe).
    Returns (rad_W, se_W, photons)."""
    photons = G3_MYSTIC_PHOTONS[sza]
    tag = f"g3_mystic_tau{tau_star:g}_sza{sza:g}_wl{wl:g}"
    workdir = OUT_DIR / "g3" / tag
    workdir.mkdir(parents=True, exist_ok=True)
    deck = g3_mystic_deck(lrt, solar, wc, tau_star, sza, wl, photons)
    inp = workdir / "case.inp"
    done = workdir / "case.done"  # written only after a completed run:
    # cache validity is tied to completion, not to deck presence (a killed
    # run leaves case.inp but no marker).
    cached = (done.exists() and done.read_text() == deck
              and (workdir / "mc.rad.spc").exists())
    if not cached:
        done.unlink(missing_ok=True)
        inp.write_text(deck)
        with open(inp) as fi, open(workdir / "case.out", "w") as fo:
            r = subprocess.run([str(lrt / "bin" / "uvspec")], stdin=fi,
                               stdout=fo, stderr=subprocess.PIPE, text=True,
                               cwd=workdir)
        if r.returncode != 0:
            # Degrade gracefully (NO-REF row) so a killed/failed deep run
            # cannot take the whole campaign down with it.
            print(f"  MYSTIC {tag}: FAILED/KILLED (rc={r.returncode}) - "
                  f"dropped\n{r.stderr[:500]}", flush=True)
            return None, None, photons
        done.write_text(deck)

    def read_spc(name):
        p = workdir / name
        if not p.exists():
            return None
        parts = p.read_text().split()
        return float(parts[4]) * 1e-3 if len(parts) >= 5 else None  # mW -> W

    rad = read_spc("mc.rad.spc")
    se = read_spc("mc.rad.std.spc")
    print(f"  MYSTIC {tag}: rad={rad if rad is not None else float('nan'):.4e} "
          f"se={se if se is not None else float('nan'):.1e} "
          f"({photons:.0e} photons)", flush=True)
    return rad, se, photons


def g3_run_twilight(tau_star: float, mode: str, seed: int,
                    szas: list[float], photons: int) -> dict:
    """One twilight compare run, zenith only. {(sza, wl): W/m^2/sr/nm}."""
    cmd = [
        str(TWILIGHT_CLI), "compare",
        "--sza", ",".join(f"{s:g}" for s in szas),
        "--view-zenith", "0", "--rel-azimuth", "0",
        "--rayleigh-only", "--fast", "--no-refraction",
        "--scattering", mode,
        "--photons", str(photons),
        "--seed-salt", str(seed),
        "--cloud-tau", f"{tau_star / G2_DE_SCALE:.6f}",
        "--cloud-base-km", f"{G2_CLOUD_BASE_KM:g}",
        "--cloud-top-km", f"{G2_CLOUD_TOP_KM:g}",
        "--cloud-ssa", f"{G2_CLOUD_SSA:g}",
        "--cloud-g", f"{G2_CLOUD_G:g}",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"twilight-cli compare failed ({mode}, tau*={tau_star}, "
                 f"seed {seed}):\n{r.stderr[:2000]}")
    table = {}
    for row in csv.reader(io.StringIO(r.stdout)):
        if not row or row[0].startswith("#") or row[0] == "sza_deg":
            continue
        sza, vz, ra, wl, rad = (float(x) for x in row)
        if wl in G3_WLS:
            table[(sza, wl)] = rad
    return table


def compare_g3_cloud_twilight(lrt):
    """G3 campaign: twilight chain estimators vs spherical backward MYSTIC
    on the delta-scaled uniform deck at SZA 95-101, zenith view."""
    if lrt is None:
        sys.exit("g3-cloud-twilight needs libRadtran (set LIBRADTRAN_DIR)")
    if not TWILIGHT_CLI.exists():
        sys.exit("Build first: cargo build --release -p twilight-cli")
    from concurrent.futures import ThreadPoolExecutor

    solar = g2_solar_file()
    wc = g2_wc_file()
    mult_taus = G3_TAU_BOTH + G3_TAU_MULT_ONLY

    def szas_for(tau_star):
        return G3_TAU10_SZAS if tau_star in G3_TAU_MULT_ONLY else G3_SZAS

    print("=== G3: clouds at twilight geometry (spherical backward MYSTIC) ===")
    print(f"  scaled constants: de_scale={G2_DE_SCALE:.7f} "
          f"ssa*={G2_SSA_STAR:.7f} g*={G2_G_STAR:.7f}")
    for ts in mult_taus:
        print(f"  tau*={ts:g}: twilight --cloud-tau {ts / G2_DE_SCALE:.6f}; "
              f"uvspec wc_modify tau set {ts:.6f}; SZAs {szas_for(ts)}")

    # ---- MYSTIC referee pool (started first; runs while twilight runs) ----
    mystic_cases = [(ts, sza, wl) for ts in mult_taus
                    for sza in szas_for(ts) for wl in G3_WLS]
    mystic_pool = ThreadPoolExecutor(max_workers=G3_MYSTIC_WORKERS)
    mystic_futs = {c: mystic_pool.submit(g3_run_mystic_case, lrt, solar, wc, *c)
                   for c in mystic_cases}

    # ---- twilight hybrid: one grid run per seed, seeds in parallel --------
    tw: dict = {}  # (tau, mode) -> {(sza, wl): (mean, se)}
    for ts in G3_TAU_BOTH:
        with ThreadPoolExecutor(max_workers=6) as ex:
            tables = list(ex.map(
                lambda s: g3_run_twilight(ts, "hybrid", s, G3_SZAS,
                                          G3_HYB_PHOTONS), G3_SEEDS))
        tw[(ts, "hybrid")] = {k: g2_mean_se([t[k] for t in tables])
                              for k in tables[0]}
        print(f"  twilight hybrid   tau*={ts:g}: {len(G3_SEEDS)} seeds x "
              f"{G3_HYB_PHOTONS} photons done", flush=True)

    # ---- twilight multiple: photon tiers by SZA group ---------------------
    for ts in mult_taus:
        merged: dict = {}
        groups: dict[int, list[float]] = {}
        for sza in szas_for(ts):
            groups.setdefault(G3_MUL_PHOTONS[sza], []).append(sza)
        for photons, szas in sorted(groups.items()):
            with ThreadPoolExecutor(max_workers=3) as ex:
                tables = list(ex.map(
                    lambda s: g3_run_twilight(ts, "multiple", s, szas,
                                              photons), G3_SEEDS))
            for k in tables[0]:
                merged[k] = g2_mean_se([t[k] for t in tables])
        tw[(ts, "multiple")] = merged
        print(f"  twilight multiple tau*={ts:g}: {len(G3_SEEDS)} seeds done",
              flush=True)

    # ---- collect MYSTIC ----------------------------------------------------
    mystic = {c: f.result() for c, f in mystic_futs.items()}
    mystic_pool.shutdown()

    # ---- gate + table -------------------------------------------------------
    # Band per point: 3 x combined SE + a flat systematic floor. Verdicts:
    #   multiple: PASS/FAIL, except LOW-POWER when the band exceeds half the
    #             referee value (the comparison has no statistical power).
    #   hybrid:   gated at SZA 95 only; SZA >= 97 under the deck is the
    #             documented one-sided-low limitation -> KNOWN-LIM (reported,
    #             not counted). tau* 10 not run for hybrid at all.
    out_csv = OUT_DIR / "g3_cloud_twilight_results.csv"
    n_pass = n_fail = n_lowpow = n_knownlim = 0
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["tau_star", "sza", "wl",
                    "tw_hybrid", "tw_hybrid_se", "tw_multiple",
                    "tw_multiple_se", "mystic", "mystic_se",
                    "mystic_photons", "hyb_over_mystic", "mul_over_mystic",
                    "band_mul", "verdict_hybrid", "verdict_multiple"])
        for ts in mult_taus:
            for sza in szas_for(ts):
                for wl in G3_WLS:
                    my, myse, nph = mystic[(ts, sza, wl)]
                    hyb = tw.get((ts, "hybrid"), {}).get((sza, wl))
                    mul = tw[(ts, "multiple")][(sza, wl)]
                    if my is None or my <= 0:
                        w.writerow([ts, sza, wl,
                                    f"{hyb[0]:.4e}" if hyb else "",
                                    f"{hyb[1]:.2e}" if hyb else "",
                                    f"{mul[0]:.4e}", f"{mul[1]:.2e}",
                                    "0", "", f"{nph:.0e}", "", "", "",
                                    "NO-REF", "NO-REF"])
                        continue
                    myse = myse or 0.0

                    def verdict(mean, se, gated, known_lim):
                        band = 3.0 * math.sqrt(se * se + myse * myse) \
                            + G3_SYS_FLOOR * my
                        if known_lim:
                            return "KNOWN-LIM", band
                        if band > 0.5 * my:
                            return "LOW-POWER", band
                        if not gated:
                            return "INFO", band
                        return ("PASS" if abs(mean - my) <= band else "FAIL",
                                band)

                    v_h = band_h = None
                    if hyb is not None:
                        v_h, band_h = verdict(hyb[0], hyb[1], gated=(sza < 97),
                                              known_lim=(sza >= 97))
                        if v_h == "KNOWN-LIM":
                            n_knownlim += 1
                        elif v_h == "PASS":
                            n_pass += 1
                        elif v_h == "FAIL":
                            n_fail += 1
                        elif v_h == "LOW-POWER":
                            n_lowpow += 1
                    v_m, band_m = verdict(mul[0], mul[1], gated=True,
                                          known_lim=False)
                    if v_m == "PASS":
                        n_pass += 1
                    elif v_m == "FAIL":
                        n_fail += 1
                    else:
                        n_lowpow += 1
                    w.writerow([ts, sza, wl,
                                f"{hyb[0]:.4e}" if hyb else "",
                                f"{hyb[1]:.2e}" if hyb else "",
                                f"{mul[0]:.4e}", f"{mul[1]:.2e}",
                                f"{my:.4e}", f"{myse:.2e}", f"{nph:.0e}",
                                f"{hyb[0] / my:.4f}" if hyb else "",
                                f"{mul[0] / my:.4f}",
                                f"{band_m / my:.4f}",
                                v_h or "", v_m])
                    hyb_s = (f"hyb {hyb[0] / my:5.2f}x [{v_h}]"
                             if hyb else "hyb   -  ")
                    print(f"  tau*={ts:4g} SZA {sza:5.1f} wl {wl:3.0f}: "
                          f"{hyb_s}  mul {mul[0] / my:5.2f}x [{v_m}]  "
                          f"my={my:.3e}+-{myse:.1e}", flush=True)

    print(f"\n  gate: {n_pass} pass / {n_fail} fail / {n_lowpow} low-power "
          f"/ {n_knownlim} known-lim (hybrid SZA>=97, documented "
          f"analog-under-cloud starvation)")
    print(f"  full table: {out_csv}")
    return n_fail == 0


# ---------------------------------------------------------------------------
# G3-CUBE (transport plan G2b): synthetic 3D cube at daytime SZA vs MYSTIC 3D.
#
# STATUS: decks-prepared. The campaign cannot execute against the public
# libRadtran and the current twilight CLI, for two independent reasons that
# this tier documents and reproduces:
#
#   1. Referee gap: the public libRadtran 2.0.6 ships MYSTIC with
#      HAVE_MYSTIC3D compiled out. A 3D wc_file parses and loads, but the
#      first photon touching a 3D layer aborts with "Error! you are not
#      allowed to use mystic 3D!" (libsrc_c/mystic.c, travel_tau guard),
#      and `mc_sample_grid` is not even tokenized in src/uvspec_lex.l.
#      Full 3D MYSTIC is distributed by LMU on collaboration terms only.
#   2. twilight CLI gap: `compare` (the radiance surface all referee tiers
#      drive) does not accept --cloud-field; only `pray` does, and pray
#      emits prayer times, not radiances. External refereeing of the voxel
#      transport path needs compare --cloud-field (or an equivalent
#      radiance-grid surface).
#
# What IS delivered, ready to run the moment a full-MYSTIC build (or SHDOM/
# I3RC) plus the compare surface exist:
#   - a 16x16x1 km^3-cell domain (plane-parallel periodic in MYSTIC; the
#     4x4 km cloudy block is 6 km from its periodic images, so daytime
#     shadows at SZA <= 60 cannot wrap) with per-cell EXPLICIT delta-scaled
#     gray optics (wc3D flag 1: ext g ssa - the same-problem construction,
#     no microphysics parameterization in the loop),
#   - the byte-identical twilight Cloud3DField sidecar for the same cube
#     (via tools/cloud3d_common.write_field), using the ice preset of
#     crates/twilight-data cloud_field_builder.rs (ssa 0.97, g 0.77,
#     r_eff 30 um, rho_ice 0.917e6 g/m^3) inverted so the field carries
#     scaled extinction tau*/km identical to the referee cells,
#   - decks for SZA {30, 60} x view {zenith, 60 deg slant} x pixel
#     {cloud-center, clear} x 550 nm, backward MYSTIC.
# ---------------------------------------------------------------------------
G3C_SSA_ICE = 0.97          # crates/twilight-data cloud_field_builder.rs
G3C_G_ICE = 0.77
G3C_RHO_ICE = 0.917e6       # g/m^3
G3C_R_EFF = 30e-6           # m
G3C_F = G3C_G_ICE * G3C_G_ICE
G3C_DE = 1.0 - G3C_SSA_ICE * G3C_F
G3C_SSA_STAR = (1.0 - G3C_F) * G3C_SSA_ICE / G3C_DE
G3C_G_STAR = G3C_G_ICE / (1.0 + G3C_G_ICE)
G3C_TAU_STAR = 3.0          # scaled extinction OD across the 1 km cube depth
G3C_N = 16                  # 16 x 16 cells, 1 km each
G3C_BLOCK = (7, 10)         # 1-based inclusive ix/iy range of the cloudy block
G3C_BASE_KM, G3C_TOP_KM = 1.0, 2.0
G3C_SZAS = [30.0, 60.0]
G3C_VIEWS = [(0.0, "vz0"), (60.0, "vz60")]
G3C_PIXELS = [((G3C_N // 2, G3C_N // 2), "cloud"), ((2, 2), "clear")]
G3C_WL = 550.0
G3C_PHOTONS = 4_000_000


def g3cube_emit(lrt):
    """Emit the ready-to-run G2b cube artifacts and demonstrate the blockers."""
    out = OUT_DIR / "g3cube"
    out.mkdir(parents=True, exist_ok=True)
    solar = g2_solar_file()

    # --- referee cube: per-cell explicit scaled optics (flag 1) -----------
    lo, hi = G3C_BLOCK
    lines = [f"{G3C_N} {G3C_N} 1 1",
             f"1.0 1.0 {G3C_BASE_KM:g} {G3C_TOP_KM:g}"]
    for iy in range(lo, hi + 1):
        for ix in range(lo, hi + 1):
            lines.append(f"{ix} {iy} 1 {G3C_TAU_STAR:.6f} "
                         f"{G3C_G_STAR:.7f} {G3C_SSA_STAR:.7f}")
    wc3d = out / "wc3d_cube.dat"
    wc3d.write_text("\n".join(lines) + "\n")

    data = (lrt / "data") if lrt else Path("<LIBRADTRAN_DIR>/data")
    n_decks = 0
    for sza in G3C_SZAS:
        for vz, vtag in G3C_VIEWS:
            for (px, py), ptag in G3C_PIXELS:
                deck = f"""# G2b cube referee deck - REQUIRES full MYSTIC (HAVE_MYSTIC3D):
# the public libRadtran 2.0.6 rejects mc_sample_grid at parse time and
# aborts 3D transport with "you are not allowed to use mystic 3D!".
data_files_path {data}/
atmosphere_file {data}/atmmod/afglus.dat
source solar {solar} per_nm
albedo {ALBEDO}
wavelength {G3C_WL:.1f} {G3C_WL:.1f}
mol_abs_param crs
no_absorption mol
wc_file 3D {wc3d}
rte_solver montecarlo
mc_photons {G3C_PHOTONS}
mc_backward {px} {py} {px} {py}
mc_sample_grid {G3C_N} {G3C_N} 1 1
mc_vroom on
mc_std
sza {sza:.2f}
phi0 0.0
umu {-math.cos(math.radians(vz)):.6f}
phi 0.0
zout 0.0
output_user lambda uu
quiet
"""
                (out / f"g3cube_sza{sza:g}_{vtag}_{ptag}.inp").write_text(deck)
                n_decks += 1

    # --- twilight sidecar: the SAME cube as a Cloud3DField ----------------
    # field_from_iwc_grid: beta = 3 IWC / (2 rho r_eff); the gray channel
    # carries beta * de_scale (* ssa* split). Invert for IWC such that the
    # cell's scaled extinction equals G3C_TAU_STAR per km.
    beta_per_iwc_km = 3.0 / (2.0 * G3C_RHO_ICE * G3C_R_EFF) * 1e3  # km^-1 per g/m^3
    iwc_cloud = G3C_TAU_STAR / G3C_DE / beta_per_iwc_km            # g/m^3
    sys.path.insert(0, str(REPO / "tools"))
    import numpy as np
    from cloud3d_common import write_field
    nz_src, top_m = 4, 4000.0   # slab boundaries 4/3/2/1/0 km: index 2 = 1-2 km
    iwc = np.zeros((nz_src, G3C_N, G3C_N), dtype=np.float32)
    iwc[2, lo - 1:hi, lo - 1:hi] = iwc_cloud   # y-symmetric block: N-S flip safe
    heights = np.linspace(top_m, 0.0, nz_src)  # only [0] (top_m) enters the header
    dlat = 1.0 / 111.32
    lat0 = -(G3C_N / 2) * dlat                 # south edge; domain centered on (0,0)
    write_field(out / "cube_field.bin", iwc, heights, lat0, lat0, dlat, dlat,
                "2026-07-02T12:00:00Z", "g3-cube synthetic (validate_libradtran)")

    print("=== G3-CUBE (G2b): synthetic 3D cube - decks prepared ===")
    print(f"  cube: {hi - lo + 1}x{hi - lo + 1} km block, {G3C_BASE_KM:g}-"
          f"{G3C_TOP_KM:g} km, per-cell ext*={G3C_TAU_STAR:g}/km "
          f"g*={G3C_G_STAR:.7f} ssa*={G3C_SSA_STAR:.7f} "
          f"(ice preset de_scale={G3C_DE:.6f})")
    print(f"  sidecar IWC for the same scaled optics: {iwc_cloud:.6f} g/m^3")
    print(f"  {n_decks} decks + {wc3d.name} + cube_field.bin(.json) in {out}/")

    # --- demonstrate blocker 1 empirically ---------------------------------
    # Two distinct public-build failures, both probed:
    #   (a) the full 16x16 deck: mc_sample_grid is rejected at parse time,
    #       and without it the 3D backward setup crashes (SIGBUS) before
    #       transport - the 3D sampling machinery is simply not there;
    #   (b) a minimal 2x2 single-pixel deck survives setup and hits the
    #       explicit guard "you are not allowed to use mystic 3D!"
    #       (HAVE_MYSTIC3D compiled out).
    if lrt is not None:
        wc_min = out / "_probe_wc3d_2x2.dat"
        wc_min.write_text("2 2 1 1\n1.0 1.0 1.0 2.0\n"
                          f"1 1 1 {G3C_TAU_STAR:.6f} "
                          f"{G3C_G_STAR:.7f} {G3C_SSA_STAR:.7f}\n")
        probe = out / "_probe_minimal3d.inp"
        base = (out / "g3cube_sza30_vz0_cloud.inp").read_text().splitlines()
        keep = []
        for l in base:
            if l.startswith("mc_sample_grid"):
                continue
            if l.startswith("wc_file 3D"):
                l = f"wc_file 3D {wc_min}"
            if l.startswith("mc_backward "):
                l = "mc_backward 1 1 1 1"
            if l.startswith("mc_photons"):
                l = "mc_photons 1000"
            keep.append(l)
        probe.write_text("\n".join(keep) + "\n")
        with open(probe) as fi:
            r = subprocess.run([str(lrt / "bin" / "uvspec")], stdin=fi,
                               capture_output=True, text=True, cwd=out)
        blocked = "not allowed to use mystic 3D" in (r.stderr or "")
        print(f"  public-referee probe (minimal 3D deck): rc={r.returncode} "
              f"{'-> confirmed: HAVE_MYSTIC3D compiled out' if blocked else '(unexpected: inspect stderr)'}")
        if not blocked:
            print(r.stderr[:800])
    print("  twilight CLI gap: `compare` has no --cloud-field (only `pray`); "
          "voxel-path radiances are not externally refereeable until that "
          "surface exists.")
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
    ap.add_argument("--tier", choices=["1a", "1b", "1b-deep", "2", "3", "g2",
                                       "g3-cloud-twilight", "g3-cube"],
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
        elif tier == "g3-cloud-twilight":
            ok &= compare_g3_cloud_twilight(lrt)
        elif tier == "g3-cube":
            ok &= g3cube_emit(lrt)
        elif tier in ("2", "3"):
            print(f"Tier {tier}: deck templates not yet automated - "
                  "see module docstring for the design.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
