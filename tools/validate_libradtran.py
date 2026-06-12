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
    ap.add_argument("--tier", choices=["1a", "1b", "1b-deep", "2", "3"], default=None)
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
        elif tier in ("2", "3"):
            print(f"Tier {tier}: deck templates not yet automated - "
                  "see module docstring for the design.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
