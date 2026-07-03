#!/usr/bin/env python3
"""g3-cube SHDOM referee: transport plan gate G2b, clouds in TRUE 3D
against a public external 3D solver.

The blocked referee (full MYSTIC 3D, collocation-only) is replaced by
Frank Evans' SHDOM (Spherical Harmonic Discrete Ordinate Method), a
public, deterministic, fully 3D RTE solver. Same-problem construction:
the EXACT g3-cube medium the twilight field path transports (per-cell
delta-scaled gray optics on the byte-matched Cloud3DField sidecar) is
written into an SHDOM tabulated-phase-function property file over the
identical Rayleigh background, and downwelling ground radiances are
compared pixel-for-pixel at daytime SZA.

SHDOM PROVENANCE AND BUILD (documented per campaign policy)
  source   https://nit.coloradolinux.com/~evans/shdom/shdom.tar.gz
  sha256   5843979eee701b654a57f940acf3b5d7a0a93ec4300120e8bf39c46c0af4ca76
  version  polarized-SHDOM beta distribution, Updatelist head 13MAY15
           (gzip timestamp 2015-05-14; shdomsub1.f dated 2015-05-14)
  build    gfortran 15.2 (Homebrew), no MPI, no netcdf:
             make shdom propgen make_mie_table plotscattab \\
               FFLAGS="-O3 -fallow-argument-mismatch -std=legacy -w"
           (put.c helper needs: cc -std=c89 -Wno-implicit-int -o put put.c)
  smoke    bundled les2y21 case via run_mono_les (PATH must include the
           build dir): converges in 24 iterations, 0.81 s CPU, radiance
           field positive and structured; accepted as build verification.

SAME-PROBLEM CONSTRUCTION (all constants replicated from the Rust
source, files cited)
  - Cloud voxels: the twilight field path carries the DELTA-SCALED CLOUD
    SCATTERING extinction only (cloud_field_builder.rs field_from_iwc_grid,
    cloud_field.rs header): sigma_s* = beta * de * ssa* with
    beta = 3 IWC / (2 rho_ice r_eff), IWC = 0.129493 g/m^3 (f32 as
    stored), rho_ice 0.917e6 g/m^3, r_eff 30 um, ice preset ssa 0.97
    g 0.77 => de-scale 0.424887, ssa* 0.9293930, g* = 0.77/1.77
    = 0.4350282. Cloud ABSORPTION IS DROPPED by the field path (pipeline
    build_atmosphere comment, plan open decision 2). The PRIMARY SHDOM
    referee medium is therefore the medium twilight actually transports:
    ext = sigma_s* (= 2.78818/km), ssa_cloud = 1, HG g*. A SECONDARY
    SHDOM variant with ext* = beta*de = 3.000/km, ssa* = 0.9293930
    externally quantifies the dropped-absorption approximation.
  - Phase function: HG expands exactly in Legendre, chi_l = (2l+1) g^l
    (SHDOM convention chi_0 = 1 implied, chi_1 = 3g). NLEG = 30 terms:
    truncation |chi_30| < 1e-7 at g* = 0.435. Rayleigh phase is
    chi_2 = 0.5 exactly (pure (1+cos^2), matching twilight's scalar
    phase; the King factor enters the cross-section only, as in
    spectrum.rs). Mixed cells combine the two by scattering fraction.
  - Rayleigh background: twilight's exact staircase (builder.rs
    build_clear_sky): 55 shells on DEFAULT_ALTITUDES_KM, per-shell
    beta = sigma_Bodhaine(wl) * n(midpoint), n log-linear on the USSA-76
    table (atmosphere_profiles.rs), sigma per spectrum.rs (Peck-Reeder
    n-1, Bates King factor, N_s = 2.546899e19). SHDOM z-grid samples the
    staircase; column-tau agreement is printed by --stage shdom.
  - Solar scaling: SHDOM SOLARFLUX is flux on a HORIZONTAL surface
    (shdom.txt), so SOLARFLUX = F_TSIS(wl) * cos(SZA) with F_TSIS from
    solar_spectrum.rs (1.848 at 550 nm, 2.087 at 450 nm). SHDOM
    radiances are then in W/m^2/sr/nm, directly comparable to
    `twilight-cli compare` output.
  - Surface: Lambertian albedo 0.15 both codes. Scalar transport both
    codes (twilight --fast, SHDOM NSTOKES=1); refraction disabled on the
    twilight side (--no-refraction); no gas absorption, no aerosol
    (--rayleigh-only; SHDOM medium is Rayleigh + cloud only).

GEOMETRY (axes mapped once, stated here)
  SHDOM +X = east (+lon), +Y = north (+lat). twilight solar azimuth 270
  (sun in the west) => beam travels east => SOLARAZ = 0. Views are
  toward the sun (rel_azimuth 0), so received radiance propagates
  east+down: SHDOM output angles (mu, phi) = (-1, 0) zenith and
  (-0.5, 0) for the 60-degree slant. SHDOM is plane-parallel with
  periodic horizontal BCs; twilight is spherical with a finite field
  footprint. The clear-sky anchor (both codes, same configuration, no
  cloud) measures this geometry+config systematic; cloud gates are
  normalized by it and it is added to the band.

FOOTPRINT NOTE (why the campaign field is 64x64 km, not the original
16x16 sidecar): outside its footprint the Cloud3DField answers with the
horizontal-MEAN column (cloud_field.rs background_column). For the
16x16 cube that fallback is a tau ~0.17 haze everywhere beyond 16 km,
which SHDOM does not have; on a 64x64 km field the fallback drops to
tau 0.012 and every sun ray that matters at SZA <= 60 stays inside the
footprint. The 4x4x1 km block itself is unchanged. SHDOM's cube domain
is 32x32 km, so its periodic images sit >= 28 km from the block; the
residual image effects are <1 percent and folded into the stated floor.
The deck sidecar is 16x16 uniform (fallback = same deck = infinite
deck, matching SHDOM's periodic uniform deck exactly).

BLOCK EDGES IN SHDOM: property values live on grid points with
(tri)linear interpolation between them, so a sharp cube edge cannot be
represented exactly. Boundary grid points carry HALF the block value
(and quarter/eighth on edges/corners), which conserves the integral of
extinction through every face exactly; the edge is smeared over one
grid spacing (0.5 km horizontal, 0.04 km vertical). The block-edge
pixel quantifies what remains of this representation difference.

USAGE (stages cache; rerun any stage safely)
  python3 tools/validate_shdom.py --stage fields    # write sidecars
  python3 tools/validate_shdom.py --stage shdom     # property files + SHDOM runs
  python3 tools/validate_shdom.py --stage twilight  # compare runs (cached CSV)
  python3 tools/validate_shdom.py --stage report    # tables + verdict
  python3 tools/validate_shdom.py                   # all of the above
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cloud3d_common import write_field  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
G3 = REPO / "validation" / "g3cube"
SHDOM_DIR = G3 / "shdom"
TWI_DIR = G3 / "twi"
TWILIGHT_BIN = Path(os.environ.get(
    "TWILIGHT_BIN", REPO / "target" / "release" / "twilight-cli"))
SHDOM_BIN_DEFAULT = Path.home() / "tools-build" / "shdom" / "shdom"

# ── Constants replicated from the Rust source ────────────────────────
# crates/twilight-data/src/atmosphere_profiles.rs
ALTITUDE_GRID_KM = np.array([
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
    24.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0,
    75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 110.0, 120.0, 130.0, 140.0,
    150.0,
])
US_STD_NUMBER_DENSITY = np.array([
    2.547e19, 2.311e19, 2.093e19, 1.891e19, 1.703e19, 1.532e19, 1.373e19,
    1.227e19, 1.093e19, 9.711e18, 8.598e18, 7.585e18, 6.486e18, 5.543e18,
    4.738e18, 4.049e18, 3.462e18, 2.960e18, 2.529e18, 2.162e18, 1.849e18,
    1.573e18, 1.341e18, 1.143e18, 9.759e17, 8.334e17, 3.828e17, 1.757e17,
    8.283e16, 4.084e16, 2.135e16, 1.181e16, 6.439e15, 3.393e15, 1.722e15,
    8.300e14, 3.838e14, 1.714e14, 7.116e13, 2.920e13, 1.189e13, 2.144e12,
    5.107e11, 1.930e11, 9.322e10, 5.186e10,
])
US_STD_TEMPERATURE_K = np.array([
    288.15, 281.65, 275.15, 268.65, 262.15, 255.65, 249.15, 242.65,
    236.15, 229.65, 223.15, 216.65, 216.65, 216.65, 216.65, 216.65,
    216.65, 216.65, 216.65, 216.65, 216.65, 217.65, 218.65, 219.65,
    220.65, 221.65, 226.65, 237.05, 251.05, 264.15, 270.65, 260.65,
    247.05, 233.05, 219.15, 208.40, 198.64, 188.89, 186.87, 188.42,
    195.08, 240.00, 360.00, 469.27, 559.63, 634.39,
])
# crates/twilight-data/src/builder.rs (shell boundaries; 55 shells)
DEFAULT_ALTITUDES_KM = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0,
    9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0,
    20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0,
    42.0, 44.0, 46.0, 48.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0,
    85.0, 90.0, 95.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0,
])
# crates/twilight-data/src/solar_spectrum.rs (TSIS-1 HSRS)
F_TSIS = {550.0: 1.848, 450.0: 2.087}

# Cloud microphysics chain (cloud_field_builder.rs field_from_iwc_grid)
IWC_F32 = float(np.float32(0.129493))  # g/m^3 as stored in the sidecar
RHO_ICE = 0.917e6
R_EFF = 30e-6
SSA_ICE = 0.97
G_ICE = 0.77
F_PEAK = G_ICE * G_ICE
DE_SCALE = 1.0 - SSA_ICE * F_PEAK              # 0.424887
SSA_STAR = (1.0 - F_PEAK) * SSA_ICE / DE_SCALE  # 0.9293930
G_STAR = G_ICE / (1.0 + G_ICE)                  # 0.4350282
BETA_UNSCALED = 3.0 * IWC_F32 / (2.0 * RHO_ICE * R_EFF)   # 1/m
EXT_STAR_KM = BETA_UNSCALED * DE_SCALE * 1000.0           # ~3.000 /km
# What the field actually carries (f32-rounded like the builder stores it):
SIGMA_SCAT_KM = float(np.float32(BETA_UNSCALED * DE_SCALE * SSA_STAR)) * 1000.0

ALBEDO = 0.15
NLEG = 30
DLAT = 1.0 / 111.32  # deg per km, FieldGeometry convention (equator)

# Case matrix
SZAS = [30.0, 60.0]
VZS = [0.0, 60.0]
WAVELENGTHS = [550.0, 450.0]
# twilight observers: (name, lat_offset_km, lon_offset_km, elevation_m).
# Edge/gap sit at 1 m elevation: at elevation 0 the ECEF radius of a
# (lat, lon, 0) observer can round one ulp BELOW EARTH_RADIUS_M at some
# coordinates, and the multiple estimator then kills every photon
# (radiance exactly 0; hybrid is unaffected; (0, 0) is exact and safe).
# Engine bug reported upstream; 1 m is ~1e-4 of the Rayleigh scale
# height, far below every band in this campaign.
OBSERVERS = [("center", 0.0, 0.0, 0.0), ("edge", 0.0, -1.5, 1.0),
             ("gap", -6.5, -6.5, 1.0)]
# SHDOM cube pixel for each observer (32 km domain, block center at 16,16)
SHDOM_PIXEL = {"center": (16.0, 16.0), "edge": (14.5, 16.0), "gap": (9.5, 9.5)}

CUBE_FIELD = G3 / "cube_field_64.bin"
DECK_FIELD = G3 / "deck_field_16.bin"


# ── Rayleigh replication (spectrum.rs) ───────────────────────────────
def rayleigh_cross_section_cm2(wl_nm: float) -> float:
    lam_um = wl_nm / 1000.0
    lam_cm = wl_nm * 1e-7
    s2 = 1.0 / (lam_um * lam_um)
    n_minus_1 = (5791817.0 / (238.0185 - s2) + 167909.0 / (57.362 - s2)) * 1e-8
    n_s = 2.546899e19
    f_king = 1.0480 + 0.00013 * (550.0 - wl_nm) / 150.0
    n2m1 = 2.0 * n_minus_1 + n_minus_1 * n_minus_1
    ll = n2m1 / (n2m1 + 3.0)
    return 24.0 * math.pi**3 / (n_s * n_s * lam_cm**4) * ll * ll * f_king


def number_density_at(alt_km: float) -> float:
    """molecules/cm^3, log-linear like atmosphere_profiles.rs."""
    logn = np.log(US_STD_NUMBER_DENSITY)
    if alt_km <= ALTITUDE_GRID_KM[0]:
        return float(US_STD_NUMBER_DENSITY[0])
    if alt_km >= ALTITUDE_GRID_KM[-1]:
        return float(US_STD_NUMBER_DENSITY[-1])
    return float(np.exp(np.interp(alt_km, ALTITUDE_GRID_KM, logn)))


def temperature_at(alt_km: float) -> float:
    return float(np.interp(alt_km, ALTITUDE_GRID_KM, US_STD_TEMPERATURE_K))


def shell_beta_km(wl_nm: float) -> np.ndarray:
    """Per-shell Rayleigh extinction [1/km]: twilight's exact staircase."""
    sig_m2 = rayleigh_cross_section_cm2(wl_nm) * 1e-4
    mids = 0.5 * (DEFAULT_ALTITUDES_KM[:-1] + DEFAULT_ALTITUDES_KM[1:])
    return np.array(
        [sig_m2 * number_density_at(m) * 1e6 * 1000.0 for m in mids]
    )


def staircase_beta_km(z_km: float, betas: np.ndarray) -> float:
    """Rayleigh extinction of the twilight shell containing z."""
    s = int(np.searchsorted(DEFAULT_ALTITUDES_KM, z_km, side="right")) - 1
    s = max(0, min(s, len(betas) - 1))
    return float(betas[s])


# ── SHDOM z grid ─────────────────────────────────────────────────────
def shdom_z_grid() -> np.ndarray:
    below = [0.0, 0.25, 0.5, 0.75, 0.96]
    cloud = [1.0, 1.04, 1.25, 1.5, 1.75, 1.96, 2.0]
    above = [2.04, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75]
    above += [5.5 + i for i in range(15)]        # 5.5 .. 19.5
    above += [21.0 + 2 * i for i in range(15)]   # 21 .. 49
    above += [52.5, 57.5, 62.5, 67.5, 72.5, 77.5, 80.0]
    return np.array(below + cloud + above)


def cloud_weight(v: float, lo: float, hi: float) -> float:
    """Half-value boundary sampling of the block indicator: 1 strictly
    inside (lo, hi), 0.5 exactly on the boundary, else 0. Conserves the
    integral of the trilinearly interpolated field exactly."""
    if lo < v < hi:
        return 1.0
    if v == lo or v == hi:
        return 0.5
    return 0.0


# ── Sidecar fields (twilight side) ───────────────────────────────────
def write_sidecars() -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    heights = np.array([4000.0, 3000.0, 2000.0, 1000.0])  # top-down tops

    # 64x64 km isolated cube: block cells 30..33 (offsets -2..2 km).
    iwc = np.zeros((4, 64, 64), dtype=np.float32)
    iwc[2, 30:34, 30:34] = IWC_F32  # level 2 = 1..2 km (top-down order)
    write_field(str(CUBE_FIELD), iwc, heights, -32 * DLAT, -32 * DLAT,
                DLAT, DLAT, now, "g3-cube 64km synthetic (validate_shdom)")

    # 16x16 km uniform deck (background column = deck => infinite deck).
    iwc = np.zeros((4, 16, 16), dtype=np.float32)
    iwc[2, :, :] = IWC_F32
    write_field(str(DECK_FIELD), iwc, heights, -8 * DLAT, -8 * DLAT,
                DLAT, DLAT, now, "g3-deck 16km synthetic (validate_shdom)")
    print(f"wrote {CUBE_FIELD.name} (64x64, block cells 30..33) and "
          f"{DECK_FIELD.name} (16x16 uniform); IWC {IWC_F32:.8f} g/m^3")


# ── SHDOM property files ─────────────────────────────────────────────
def hg_legendre(g: float, nleg: int) -> np.ndarray:
    return np.array([(2 * l + 1) * g**l for l in range(1, nleg + 1)])


def mixed_phase(w_hg: float) -> np.ndarray:
    chi = w_hg * hg_legendre(G_STAR, NLEG)
    chi[1] += (1.0 - w_hg) * 0.5  # Rayleigh chi_2
    return chi


def write_property_file(path: Path, medium: str, wl_nm: float,
                        absorbing: bool) -> None:
    """medium: cube | deck | clear. absorbing=False is the PRIMARY
    referee (the medium twilight carries: pure scattering sigma_s*);
    absorbing=True carries ext*, ssa* (dropped-absorption variant)."""
    betas = shell_beta_km(wl_nm)
    zgrid = shdom_z_grid()
    nz = len(zgrid)
    if medium == "cube":
        nx = ny = 64
        dx = 0.5
        xs = np.arange(nx) * dx
    else:
        nx = ny = 8
        dx = 2.0
        xs = np.arange(nx) * dx

    ext_c_km = EXT_STAR_KM if absorbing else SIGMA_SCAT_KM
    ssa_c = SSA_STAR if absorbing else 1.0

    # Phase function table: index 1 = Rayleigh; mixed functions keyed by
    # the HG scattering fraction (rounded to avoid float-key dust).
    phase_index = {}
    phase_rows = ["2 0.0 0.5"]  # Rayleigh: chi1=0, chi2=0.5

    def phase_for(w_hg: float) -> int:
        key = round(w_hg, 9)
        if key == 0.0:
            return 1
        if key not in phase_index:
            chi = mixed_phase(w_hg)
            phase_rows.append(
                str(NLEG) + " " + " ".join(f"{c:.7e}" for c in chi))
            phase_index[key] = len(phase_rows)
        return phase_index[key]

    wz = np.array([cloud_weight(z, 1.0, 2.0) for z in zgrid])
    if medium == "cube":
        # block at [14,18] km in both axes
        wx = np.array([cloud_weight(x, 14.0, 18.0) for x in xs])
    elif medium == "deck":
        wx = np.ones(nx)
    else:
        wx = np.zeros(nx)

    lines = []
    for iz, z in enumerate(zgrid):
        bray = staircase_beta_km(z, betas)
        temp = temperature_at(z)
        for iy in range(ny):
            for ix in range(nx):
                w = wx[ix] * wx[iy] * wz[iz] if medium == "cube" else wz[iz] * wx[ix]
                ec = ext_c_km * w
                ext = bray + ec
                if ec > 0.0:
                    scat_c = ec * ssa_c
                    ssa = (bray + scat_c) / ext
                    ip = phase_for(scat_c / (bray + scat_c))
                else:
                    ssa = 1.0
                    ip = 1
                lines.append(
                    f"{ix+1} {iy+1} {iz+1} {temp:.2f} {ext:.6e} "
                    f"{ssa:.7f} {ip}")

    with open(path, "w") as f:
        f.write("T\n")
        f.write(f"{nx} {ny} {nz}\n")
        f.write(f"{dx} {dx} " + " ".join(f"{z:.4f}" for z in zgrid) + "\n")
        f.write(f"{len(phase_rows)}\n")
        f.write("\n".join(phase_rows) + "\n")
        f.write("\n".join(lines) + "\n")

    # Column-tau bookkeeping for the report.
    twi_tau = float(np.sum(betas * np.diff(DEFAULT_ALTITUDES_KM)))
    shdom_tau = float(np.trapezoid(
        [staircase_beta_km(z, betas) for z in zgrid], zgrid))
    print(f"  {path.name}: nz={nz} phase_fns={len(phase_rows)} "
          f"rayleigh column tau twilight={twi_tau:.6f} "
          f"shdom-grid={shdom_tau:.6f} "
          f"(delta {100*(shdom_tau/twi_tau-1):+.3f}%)")


def run_shdom(shdom_bin: Path, name: str, prp: Path, sza: float,
              wl_nm: float, force: bool) -> Path:
    out = SHDOM_DIR / f"{name}.rad"
    if out.exists() and not force:
        return out
    with open(prp) as f:
        f.readline()
        nx, ny, nz = f.readline().split()
    mu0 = math.cos(math.radians(sza))
    flux_h = F_TSIS[wl_nm] * mu0  # SOLARFLUX is on a horizontal surface
    dx = 0.5 if "cube" in name else 2.0
    lines = [
        name[:60], str(prp), "NONE", "NONE", "NONE", "NONE",
        "1",                      # NSTOKES
        f"{nx} {ny} {nz}",
        "16 32",                  # NMU NPHI
        "0",                      # BCFLAG periodic
        "0",                      # IPFLAG 3D
        "T",                      # DELTAM
        "P",                      # GRIDTYPE from property file
        "S",                      # solar source
        f"{flux_h:.6f} {-mu0:.8f} 0.0",   # SOLARFLUX SOLARMU SOLARAZ
        "0.0",                    # SKYRAD
        f"{ALBEDO}",              # GNDALBEDO
        f"{wl_nm/1000.0:.4f}",    # WAVELEN um
        "0.020 0.002",            # SPLITACC SHACC
        "T 1.0E-5 200",           # ACCEL SOLACC MAXITER
        "1", "R",
        f"0.0 {dx} {dx} 0.0 0.0 2  -1.0 0.0  -0.5 0.0",
        str(out),
        "NONE",                   # netcdf output
        "12000",                  # MAX_TOTAL_MB
        "2.4",                    # ADAPT_GRID_FACTOR
        "0.8",                    # NUM_SH_TERM_FACTOR
        "1.5",                    # CELL_TO_POINT_RATIO
    ]
    log = SHDOM_DIR / f"{name}.log"
    with open(log, "w") as lf:
        r = subprocess.run([str(shdom_bin)], input="\n".join(lines) + "\n",
                           text=True, stdout=lf, stderr=subprocess.STDOUT,
                           cwd=SHDOM_DIR)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"SHDOM run {name} failed; see {log}")
    return out


def parse_shdom_radiances(path: Path) -> dict:
    """{(mu, phi): {(x, y): radiance}}"""
    res = {}
    cur = None
    for line in path.read_text().splitlines():
        if line.startswith("!"):
            m = re.search(r"^!\s+(-?[\d.]+)\s+(-?[\d.]+)\s+<-\s+\(mu,phi\)",
                          line)
            if m:
                cur = (float(m.group(1)), float(m.group(2)))
                res[cur] = {}
            continue
        if cur is None:
            continue
        parts = line.split()
        if len(parts) >= 3:
            x, y, rad = float(parts[0]), float(parts[1]), float(parts[2])
            res[cur][(round(x, 3), round(y, 3))] = rad
    return res


def shdom_value(rads: dict, vz: float, x: float, y: float) -> float:
    mu = -1.0 if vz == 0.0 else -0.5
    for (m, p), grid in rads.items():
        if abs(m - mu) < 1e-4:
            key = (round(x, 3), round(y, 3))
            if key in grid:
                return grid[key]
            raise KeyError(f"pixel {key} not in SHDOM output")
    raise KeyError(f"angle mu={mu} not in SHDOM output")


def stage_shdom(shdom_bin: Path, force: bool) -> None:
    SHDOM_DIR.mkdir(parents=True, exist_ok=True)
    for wl in WAVELENGTHS:
        w = int(wl)
        for medium, absorbing in [("cube", False), ("cube", True),
                                  ("deck", False), ("deck", True),
                                  ("clear", False)]:
            tag = f"{medium}{'_abs' if absorbing else ''}_{w}"
            prp = SHDOM_DIR / f"{tag}.prp"
            if not prp.exists() or force:
                write_property_file(prp, medium, wl, absorbing)
            for sza in SZAS:
                name = f"{tag}_sza{int(sza)}"
                print(f"  shdom {name} ...", flush=True)
                run_shdom(shdom_bin, name, prp, sza, wl, force)
    print("SHDOM stage complete (cached in validation/g3cube/shdom/)")


# ── twilight runs ────────────────────────────────────────────────────
def twilight_case_name(medium: str, obs: str, est: str, seed: int) -> str:
    return f"{medium}_{obs}_{est}_s{seed}"


def run_twilight_case(medium: str, obs_name: str, lat: float, lon: float,
                      elev_m: float, est: str, seed: int, photons: dict,
                      force: bool) -> Path:
    TWI_DIR.mkdir(parents=True, exist_ok=True)
    out = TWI_DIR / (twilight_case_name(medium, obs_name, est, seed) + ".csv")
    if out.exists() and out.stat().st_size > 0 and not force:
        return out
    cmd = [
        str(TWILIGHT_BIN), "compare",
        # equals-form: clap rejects space-separated negative values
        # ("--lon -0.013..." parses "-0..." as an unknown flag)
        f"--lat={lat:.10f}", f"--lon={lon:.10f}",
        "--elevation", str(elev_m),
        "--sza", ",".join(str(s) for s in SZAS),
        "--view-zenith", ",".join(str(v) for v in VZS),
        "--rel-azimuth", "0", "--solar-azimuth", "270",
        "--albedo", str(ALBEDO), "--rayleigh-only",
        "--scattering", est, "--photons", str(photons[est]),
        "--fast", "--no-refraction", "--seed-salt", str(seed),
    ]
    if medium == "cube":
        cmd += ["--cloud-field", str(CUBE_FIELD)]
    elif medium == "deck":
        cmd += ["--cloud-field", str(DECK_FIELD)]
    env = dict(os.environ)
    # The multiple estimator fans out over the 41 wavelengths with
    # rayon; cap per-process threads so concurrent runs do not thrash.
    env.setdefault("RAYON_NUM_THREADS", "4")
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO,
                       env=env)
    if r.returncode != 0:
        raise RuntimeError(f"twilight {out.stem} failed:\n{r.stderr[-2000:]}")
    out.write_text(r.stdout)
    return out


def twilight_matrix() -> list:
    cases = []
    for medium in ["cube", "deck", "clear"]:
        obs_list = (OBSERVERS if medium == "cube"
                    else [("center", 0.0, 0.0, 0.0)])
        for oname, dlat_km, dlon_km, elev_m in obs_list:
            for est in ["hybrid", "multiple"]:
                cases.append((medium, oname, dlat_km * DLAT, dlon_km * DLAT,
                              elev_m, est))
    return cases


def stage_twilight(photons: dict, seeds: int, jobs: int, force: bool) -> None:
    if not TWILIGHT_BIN.exists():
        raise RuntimeError(f"missing {TWILIGHT_BIN}; cargo build --release")
    tasks = []
    for medium, oname, lat, lon, elev_m, est in twilight_matrix():
        for seed in range(seeds):
            tasks.append((medium, oname, lat, lon, elev_m, est, seed))
    print(f"twilight stage: {len(tasks)} runs "
          f"(photons {photons}, {jobs} concurrent)")
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {
            ex.submit(run_twilight_case, m, o, la, lo, el, e, s, photons,
                      force): (m, o, e, s)
            for (m, o, la, lo, el, e, s) in tasks
        }
        for f, key in futs.items():
            f.result()
            print(f"  done {key}", flush=True)


def read_twilight_csv(path: Path) -> dict:
    """{(sza, vz, wl): radiance}"""
    res = {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or line.startswith("sza_deg") or not line:
            continue
        sza, vz, _ra, wl, rad = line.split(",")
        if float(wl) in WAVELENGTHS:
            res[(float(sza), float(vz), float(wl))] = float(rad)
    return res


# ── report ───────────────────────────────────────────────────────────
def collect() -> tuple:
    twi = {}  # (medium, obs, est, sza, vz, wl) -> (mean, se, n)
    for medium, oname, _la, _lo, _el, est in twilight_matrix():
        seeds = sorted(TWI_DIR.glob(
            twilight_case_name(medium, oname, est, 0)[:-1] + "*.csv"))
        per_seed = [read_twilight_csv(p) for p in seeds]
        if not per_seed:
            continue
        for key in per_seed[0]:
            vals = np.array([d[key] for d in per_seed if key in d])
            twi[(medium, oname, est) + key] = (
                float(vals.mean()),
                float(vals.std(ddof=1) / math.sqrt(len(vals)))
                if len(vals) > 1 else 0.0,
                len(vals),
            )
    sh = {}  # (medium_tag, sza, vz, wl, obs) -> radiance
    for wl in WAVELENGTHS:
        w = int(wl)
        for tag in [f"cube_{w}", f"cube_abs_{w}", f"deck_{w}",
                    f"deck_abs_{w}", f"clear_{w}"]:
            for sza in SZAS:
                p = SHDOM_DIR / f"{tag}_sza{int(sza)}.rad"
                if not p.exists():
                    continue
                rads = parse_shdom_radiances(p)
                for vz in VZS:
                    if tag.startswith("cube"):
                        for oname, (x, y) in SHDOM_PIXEL.items():
                            sh[(tag, sza, vz, wl, oname)] = shdom_value(
                                rads, vz, x, y)
                    else:
                        sh[(tag, sza, vz, wl, "center")] = shdom_value(
                            rads, vz, 0.0, 0.0)
    return twi, sh


def stage_report() -> None:
    twi, sh = collect()
    print(f"\ncube optics: sigma_s*={SIGMA_SCAT_KM:.6f}/km (primary, "
          f"pure scattering), ext*={EXT_STAR_KM:.6f}/km ssa*={SSA_STAR:.7f} "
          f"(absorbing variant), g*={G_STAR:.7f}\n")

    for wl in WAVELENGTHS:
        w = int(wl)
        print(f"## {w} nm")
        # Clear-sky anchor
        print("\nClear-sky anchor (twilight/SHDOM, no cloud; the "
              "geometry+config systematic):")
        print("| est | sza | vz | twilight [W/m2/sr/nm] | SHDOM | ratio |")
        print("|---|---|---|---|---|---|")
        anchor = {}
        for est in ["hybrid", "multiple"]:
            for sza in SZAS:
                for vz in VZS:
                    t = twi.get(("clear", "center", est, sza, vz, wl))
                    s = sh.get((f"clear_{w}", sza, vz, wl, "center"))
                    if not t or not s:
                        continue
                    a = t[0] / s
                    anchor[(est, sza, vz)] = a
                    print(f"| {est} | {int(sza)} | {int(vz)} | "
                          f"{t[0]:.5e} +- {t[1]:.1e} | {s:.5e} | {a:.4f} |")

        # Absolute + normalized per-case table
        print("\nCloud cases (ratio = twilight/SHDOM primary; norm = "
              "ratio/anchor; band = 3 SE + |anchor-1| + 5%):")
        print("| medium | pixel | est | sza | vz | twilight | SHDOM | "
              "SHDOM_abs | ratio | norm | band | gate |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for medium in ["cube", "deck"]:
            obs_list = ([o[0] for o in OBSERVERS] if medium == "cube"
                        else ["center"])
            for oname in obs_list:
                for est in ["hybrid", "multiple"]:
                    for sza in SZAS:
                        for vz in VZS:
                            t = twi.get((medium, oname, est, sza, vz, wl))
                            s = sh.get((f"{medium}_{w}", sza, vz, wl, oname))
                            sa = sh.get(
                                (f"{medium}_abs_{w}", sza, vz, wl, oname))
                            a = anchor.get((est, sza, vz))
                            if not t or not s or a is None:
                                continue
                            ratio = t[0] / s
                            norm = ratio / a
                            se_rel = t[1] / t[0] if t[0] else 0.0
                            band = 3 * se_rel + abs(a - 1.0) + 0.05
                            gate = "PASS" if abs(norm - 1.0) <= band else "FAIL"
                            print(f"| {medium} | {oname} | {est} | {int(sza)}"
                                  f" | {int(vz)} | {t[0]:.4e}+-{t[1]:.1e} | "
                                  f"{s:.4e} | "
                                  f"{sa:.4e} | {ratio:.3f} | {norm:.3f} | "
                                  f"{band:.3f} | {gate} |"
                                  if sa is not None else
                                  f"| {medium} | {oname} | {est} | {int(sza)}"
                                  f" | {int(vz)} | {t[0]:.4e}+-{t[1]:.1e} | "
                                  f"{s:.4e} | - | {ratio:.3f} | {norm:.3f} | "
                                  f"{band:.3f} | {gate} |")

        # 3D contrast gate: gap/deck and block/deck agreement
        print("\n3D contrast gates (anchors cancel in the in-code ratios; "
              "band = 3 combined SE + 5%):")
        print("| contrast | est | sza | vz | twilight | SHDOM | "
              "twi/shdom | band | gate |")
        print("|---|---|---|---|---|---|---|---|---|")
        for cname, oname in [("gap/deck", "gap"), ("block/deck", "center"),
                             ("edge/deck", "edge")]:
            for est in ["hybrid", "multiple"]:
                for sza in SZAS:
                    for vz in VZS:
                        tn = twi.get(("cube", oname, est, sza, vz, wl))
                        td = twi.get(("deck", "center", est, sza, vz, wl))
                        s_n = sh.get((f"cube_{w}", sza, vz, wl, oname))
                        s_d = sh.get((f"deck_{w}", sza, vz, wl, "center"))
                        if not tn or not td or s_n is None or s_d is None:
                            continue
                        if tn[0] <= 0.0 or td[0] <= 0.0:
                            print(f"| {cname} | {est} | {int(sza)} | "
                                  f"{int(vz)} | BAD DATA (non-positive "
                                  f"mean) | | | | FAIL |")
                            continue
                        c_t = tn[0] / td[0]
                        c_s = s_n / s_d
                        r = c_t / c_s
                        se = math.sqrt((tn[1] / tn[0]) ** 2
                                       + (td[1] / td[0]) ** 2)
                        band = 3 * se + 0.05
                        gate = "PASS" if abs(r - 1.0) <= band else "FAIL"
                        print(f"| {cname} | {est} | {int(sza)} | {int(vz)} | "
                              f"{c_t:.4f} | {c_s:.4f} | {r:.3f} | "
                              f"{band:.3f} | {gate} |")
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", default="all",
                    choices=["all", "fields", "shdom", "twilight", "report"])
    ap.add_argument("--shdom-bin", default=str(SHDOM_BIN_DEFAULT))
    ap.add_argument("--photons-hybrid", type=int, default=600,
                    help="hybrid secondary rays per LOS step; scalar "
                         "hybrid uses ALIS (one hero path, all 41 "
                         "wavelengths evaluated), ~0.11 s CPU per ray "
                         "per geometry")
    ap.add_argument("--photons-multiple", type=int, default=800,
                    help="multiple-mode photons PER WAVELENGTH "
                         "(41-wavelength rayon fan-out, ~0.2 s CPU per "
                         "photon per geometry)")
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--force", action="store_true",
                    help="rerun even if cached outputs exist")
    args = ap.parse_args()

    if args.stage in ("all", "fields"):
        write_sidecars()
    if args.stage in ("all", "shdom"):
        stage_shdom(Path(args.shdom_bin), args.force)
    if args.stage in ("all", "twilight"):
        photons = {"hybrid": args.photons_hybrid,
                   "multiple": args.photons_multiple}
        stage_twilight(photons, args.seeds, args.jobs, args.force)
    if args.stage in ("all", "report"):
        stage_report()


if __name__ == "__main__":
    main()
