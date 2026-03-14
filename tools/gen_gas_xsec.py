#!/usr/bin/env python3
"""Generate gas absorption cross-section data for twilight-core.

Reads authoritative spectroscopic data and produces a Rust source file
(gas_absorption_data.rs) with const arrays for all supported absorbers.

Sources:
  O3   -- Serdyuchenko et al. (2014), doi:10.5281/zenodo.5793207, CC-BY 4.0
  O2   -- HITRAN 2020 line list (Gordon et al. 2022), Voigt profiles
  H2O  -- HITRAN 2020 line list (Gordon et al. 2022), Voigt profiles
  NO2  -- HITRAN UV cross-section database (measured, 220K and 294K)
  O4   -- HITRAN 2024 CIA database (O2-O2 collision-induced absorption)
  Atm  -- US Standard Atmosphere 1976 + WMO O3 profile

Usage:
  1. Download Serdyuchenko data:
       curl -L "https://zenodo.org/api/records/5793207/files/SerdyuchenkoGorshelev5digits_latest.dat/content" \
            -o data/xsec/SerdyuchenkoGorshelev5digits_latest.dat
  2. Download HITRAN data to /tmp/hitran_data/:
       - NO2 XSC files (220K, 294K)
       - O2-O2 CIA file (O2-O2_2024.cia)
       - O2.data and H2O.data line parameter files
  3. pip install hitran-api numpy
  4. python tools/gen_gas_xsec.py
"""

import sys, os, math, pathlib
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WL_MIN = 380.0   # nm
WL_MAX = 780.0   # nm
WL_STEP = 1.0    # nm
N_WL = int((WL_MAX - WL_MIN) / WL_STEP) + 1  # 401

WAVELENGTHS_NM = np.arange(WL_MIN, WL_MAX + 0.5 * WL_STEP, WL_STEP)[:N_WL]
WAVENUMBERS_CM = 1e7 / WAVELENGTHS_NM  # cm^-1, descending

# 2D (P, T) grid for O2/H2O cross-sections.
# 5 pressures x 4 temperatures = 20 conditions.
PRESSURES_HPA = [1013.25, 500.0, 300.0, 100.0, 50.0]
TEMPERATURES_K = [296.0, 260.0, 230.0, 210.0]

# H2O line intensity threshold for visible range
H2O_S_THRESHOLD = 1e-25

# Visible wavenumber range
WN_VIS_MIN = 1e7 / 780.0  # ~12821 cm-1
WN_VIS_MAX = 1e7 / 380.0  # ~26316 cm-1

REPO = pathlib.Path(__file__).resolve().parent.parent
SERDYUCHENKO_FILE = REPO / "data" / "xsec" / "SerdyuchenkoGorshelev5digits_latest.dat"
HITRAN_DB = "/tmp/hitran_data"
OUT_RS = REPO / "crates" / "twilight-core" / "src" / "gas_absorption_data.rs"

NO2_XSC_294 = pathlib.Path(HITRAN_DB) / "NO2_294.0_0.0_15002.0-42002.3_00.xsc"
NO2_XSC_220 = pathlib.Path(HITRAN_DB) / "NO2_220.0_0.0_15002.0-42002.3_00.xsc"
CIA_FILE = pathlib.Path(HITRAN_DB) / "O2-O2_2024.cia"
O2_LINE_FILE = pathlib.Path(HITRAN_DB) / "O2.data"
H2O_LINE_FILE = pathlib.Path(HITRAN_DB) / "H2O.data"


# ---------------------------------------------------------------------------
# O3: Serdyuchenko et al. (2014)
# ---------------------------------------------------------------------------
def load_serdyuchenko_o3():
    """Load O3 cross-sections from Serdyuchenko data file.
    Returns dict mapping temperature (K) -> interpolated cross-sections at our grid.
    Temperatures: 293, 283, 273, 263, 253, 243, 233, 223, 213, 203, 193 K
    """
    print("Loading O3 Serdyuchenko data...")
    if not SERDYUCHENKO_FILE.exists():
        print(f"ERROR: {SERDYUCHENKO_FILE} not found. Download it first (see docstring).")
        sys.exit(1)

    # Parse the data file (45 header lines, then wavelength + 11 temperatures)
    wl_raw = []
    xs_raw = {t: [] for t in [293, 283, 273, 263, 253, 243, 233, 223, 213, 203, 193]}
    temps_order = [293, 283, 273, 263, 253, 243, 233, 223, 213, 203, 193]

    with open(SERDYUCHENKO_FILE, 'r') as f:
        for i in range(45):
            f.readline()  # skip header
        for line in f:
            parts = line.split()
            if len(parts) < 12:
                continue
            wl = float(parts[0])
            if wl < WL_MIN - 1 or wl > WL_MAX + 1:
                continue
            wl_raw.append(wl)
            for j, t in enumerate(temps_order):
                xs_raw[t].append(float(parts[j + 1]))

    wl_raw = np.array(wl_raw)
    print(f"  Read {len(wl_raw)} points in [{wl_raw[0]:.3f}, {wl_raw[-1]:.3f}] nm")

    # Interpolate to our 1nm grid
    o3_data = {}
    for t in temps_order:
        xs = np.array(xs_raw[t])
        o3_data[t] = np.interp(WAVELENGTHS_NM, wl_raw, xs)

    # Validation against known optical depths
    xs_293 = o3_data[293]
    idx_550 = int(550 - WL_MIN)
    idx_602 = int(602 - WL_MIN)
    n_col_300du = 300 * 2.687e16  # molecules/cm^2
    tau_550 = xs_293[idx_550] * n_col_300du
    tau_602 = xs_293[idx_602] * n_col_300du
    print(f"  Validation (300 DU):")
    print(f"    sigma(550nm, 293K) = {xs_293[idx_550]:.4e} cm^2  -> tau = {tau_550:.4f} (expect ~0.027)")
    print(f"    sigma(602nm, 293K) = {xs_293[idx_602]:.4e} cm^2  -> tau = {tau_602:.4f} (expect ~0.040)")

    return o3_data, temps_order


# ---------------------------------------------------------------------------
# NO2: Real HITRAN XSC measured cross-sections
# ---------------------------------------------------------------------------
def parse_hitran_xsc(filepath):
    """Parse a HITRAN XSC (cross-section) file.
    Returns (wavenumber_grid_cm1, cross_sections_cm2, temperature).

    HITRAN XSC header is fixed-width (100 chars):
      cols  0-19: molecule name (right-justified in 20 chars)
      cols 20-29: numin (10 chars)
      cols 30-39: numax (10 chars)
      cols 40-46: npnts (7 chars)
      cols 47-53: temperature (7 chars)
      cols 54-59: pressure (6 chars)
      ... rest: sigma_max, resolution, common name, etc.
    """
    with open(filepath, 'r') as f:
        # Header line
        header = f.readline()
        # Fixed-width parsing
        numin = float(header[20:30])
        numax = float(header[30:40])
        npnts = int(header[40:47])
        temp = float(header[47:54])

        # Read data values (10 per line)
        values = []
        for line in f:
            for val_str in line.split():
                values.append(float(val_str))
                if len(values) >= npnts:
                    break
            if len(values) >= npnts:
                break

    values = np.array(values[:npnts])
    wn_grid = np.linspace(numin, numax, npnts)

    return wn_grid, values, temp


def load_no2_xsc():
    """Load real NO2 cross-sections from HITRAN XSC files at 294K and 220K.
    Interpolates to our 1nm wavelength grid.
    Returns (no2_294k, no2_220k) arrays of length N_WL.
    """
    print("Loading NO2 cross-sections from HITRAN XSC files...")

    for path, label in [(NO2_XSC_294, "294K"), (NO2_XSC_220, "220K")]:
        if not path.exists():
            print(f"ERROR: {path} not found.")
            sys.exit(1)

    # Parse both files
    wn_294, xs_294, temp_294 = parse_hitran_xsc(NO2_XSC_294)
    wn_220, xs_220, temp_220 = parse_hitran_xsc(NO2_XSC_220)

    print(f"  294K: {len(xs_294)} points, wn=[{wn_294[0]:.1f}, {wn_294[-1]:.1f}] cm-1, "
          f"wl=[{1e7/wn_294[-1]:.1f}, {1e7/wn_294[0]:.1f}] nm, max={xs_294.max():.4e}")
    print(f"  220K: {len(xs_220)} points, wn=[{wn_220[0]:.1f}, {wn_220[-1]:.1f}] cm-1, "
          f"wl=[{1e7/wn_220[-1]:.1f}, {1e7/wn_220[0]:.1f}] nm, max={xs_220.max():.4e}")

    # Convert our wavelength grid to wavenumber (descending)
    wn_target = WAVENUMBERS_CM  # descending

    # HITRAN XSC wavenumber grids are ascending. Interpolate.
    # np.interp needs ascending x.
    no2_294k = np.interp(wn_target, wn_294, xs_294, left=0.0, right=0.0)
    no2_220k = np.interp(wn_target, wn_220, xs_220, left=0.0, right=0.0)

    # XSC coverage ends at ~667nm (15002 cm-1); above that, values should be zero.
    # The np.interp with left=0.0 handles this: wavenumbers below 15002 get zero.
    # But wavenumber is descending in our grid, so we need to check:
    # wn_target at 667nm = 15002, at 780nm = 12821.
    # Since wn_294 goes from 15002 to 42002, anything below 15002 (i.e. 667-780nm) gets 0.
    # Verify:
    idx_667 = int(667 - WL_MIN)
    idx_670 = int(670 - WL_MIN)
    print(f"  Boundary check: sigma(667nm)={no2_294k[idx_667]:.4e}, sigma(670nm)={no2_294k[idx_670]:.4e}")

    # Ensure non-negative (measurement noise could produce tiny negatives)
    no2_294k = np.maximum(no2_294k, 0.0)
    no2_220k = np.maximum(no2_220k, 0.0)

    # Validation
    idx_400 = int(400 - WL_MIN)
    print(f"  sigma_NO2(400nm, 294K) = {no2_294k[idx_400]:.4e} cm^2 (Vandaele had 6.30e-19)")
    print(f"  sigma_NO2(400nm, 220K) = {no2_220k[idx_400]:.4e} cm^2")

    return no2_294k, no2_220k


# ---------------------------------------------------------------------------
# O2-O2 CIA: Real HITRAN 2024 CIA data
# ---------------------------------------------------------------------------
def parse_cia_file(filepath):
    """Parse HITRAN CIA file. Returns list of band segments:
    [(temp, wn_array, cia_array), ...]
    """
    segments = []
    with open(filepath, 'r') as f:
        while True:
            header = f.readline()
            if not header or not header.strip():
                break

            parts = header.split()
            if len(parts) < 5:
                break

            # Header: O2-O2  nu_min  nu_max  n_points  temperature  ...
            try:
                nu_min = float(parts[1])
                nu_max = float(parts[2])
                n_points = int(parts[3])
                temperature = float(parts[4])
            except (ValueError, IndexError):
                break

            wn_arr = []
            cia_arr = []
            for _ in range(n_points):
                line = f.readline()
                if not line:
                    break
                vals = line.split()
                if len(vals) >= 2:
                    wn_arr.append(float(vals[0]))
                    cia_arr.append(float(vals[1]))

            segments.append((temperature, np.array(wn_arr), np.array(cia_arr)))

    return segments


def load_o4_cia():
    """Load O2-O2 CIA from HITRAN 2024 CIA file.
    Returns composite CIA spectrum at room temperature on our 1nm grid.
    """
    print("Loading O2-O2 CIA from HITRAN 2024...")

    if not CIA_FILE.exists():
        print(f"ERROR: {CIA_FILE} not found.")
        sys.exit(1)

    segments = parse_cia_file(CIA_FILE)
    print(f"  Parsed {len(segments)} band segments total")

    # Filter to visible range and room temperature (T >= 280K)
    # Our visible wavenumber range: 12821 to 26316 cm-1
    visible_segments = []
    for temp, wn, cia in segments:
        # Check if segment overlaps with visible range
        if wn[-1] < WN_VIS_MIN or wn[0] > WN_VIS_MAX:
            continue
        visible_segments.append((temp, wn, cia))

    print(f"  {len(visible_segments)} segments overlap visible range")

    # For each temperature, list segments
    temps_seen = sorted(set(s[0] for s in visible_segments))
    print(f"  Temperatures with visible data: {temps_seen}")

    # We want a composite at the best available near-room-temperature.
    # Strategy: For each wavelength, use the highest-temperature segment
    # that covers it, preferring temps closest to 296K.
    # First, build a composite from all segments, giving priority to
    # higher-temperature data (more relevant for surface conditions).

    # Sort segments by temperature (descending) so room-temp data wins
    visible_segments.sort(key=lambda s: -s[0])

    # Initialize with zeros on our wavenumber grid
    cia_composite = np.zeros(N_WL)

    for temp, wn, cia in visible_segments:
        # Clamp negative CIA values (measurement artifacts)
        cia_clean = np.maximum(cia, 0.0)

        # Interpolate this segment onto our wavelength grid
        # Our WAVENUMBERS_CM is descending; np.interp needs ascending
        # CIA data may be ascending or descending in wavenumber
        if wn[0] > wn[-1]:
            wn = wn[::-1]
            cia_clean = cia_clean[::-1]

        # Interpolate, returning 0 outside segment range
        segment_on_grid = np.interp(WAVENUMBERS_CM, wn, cia_clean, left=0.0, right=0.0)

        # Composite: where we have nonzero data from a higher-temp segment,
        # keep it; otherwise fill with this segment's data.
        mask = (cia_composite == 0.0) & (segment_on_grid > 0.0)
        cia_composite[mask] = segment_on_grid[mask]

    # Validation
    idx_477 = int(477 - WL_MIN)
    idx_577 = int(577 - WL_MIN)
    idx_630 = int(630 - WL_MIN)
    print(f"  Composite CIA:")
    print(f"    477nm: {cia_composite[idx_477]:.4e} cm^5/mol^2 (Gaussian approx had 1.27e-46)")
    print(f"    577nm: {cia_composite[idx_577]:.4e} cm^5/mol^2 (Gaussian approx had 1.58e-46)")
    print(f"    630nm: {cia_composite[idx_630]:.4e} cm^5/mol^2")

    return cia_composite


# ---------------------------------------------------------------------------
# O2 and H2O: HITRAN line-by-line via HAPI (2D P,T grid)
# ---------------------------------------------------------------------------
def compute_hitran_xsec_2d(table_name, molecule_id, isotope_id):
    """Compute absorption cross-sections from HITRAN line data using HAPI
    on a 2D (P, T) grid.
    Returns dict mapping (P_hpa, T_K) -> cross-section array [cm^2/molecule].
    """
    from hapi import db_begin, absorptionCoefficient_Voigt, getColumn
    db_begin(HITRAN_DB)

    n_lines = len(getColumn(table_name, 'nu'))
    print(f"  {table_name}: {n_lines} lines in database")

    # HAPI needs ascending wavenumbers.
    wn_grid = np.sort(WAVENUMBERS_CM)  # ascending

    xsec_data = {}
    for p_hpa in PRESSURES_HPA:
        for t_k in TEMPERATURES_K:
            p_atm = p_hpa / 1013.25
            print(f"    P={p_hpa:.1f} hPa, T={t_k:.0f} K ...", end='', flush=True)
            try:
                nu, coef = absorptionCoefficient_Voigt(
                    SourceTables=table_name,
                    Environment={'p': p_atm, 'T': t_k},
                    WavenumberGrid=wn_grid,
                    HITRAN_units=True,   # cm^2/molecule
                )
                # Map back to our wavelength grid (ascending wavelength = descending wavenumber)
                xsec = np.interp(WAVENUMBERS_CM, nu, coef)
                xsec_data[(p_hpa, t_k)] = xsec
                print(f" max={xsec.max():.4e} cm^2")
            except Exception as e:
                print(f" FAILED: {e}")
                xsec_data[(p_hpa, t_k)] = np.zeros(N_WL)

    return xsec_data


# ---------------------------------------------------------------------------
# O2 and H2O line parameters for runtime Voigt computation
# ---------------------------------------------------------------------------
def parse_hitran_line_params(filepath, wn_min, wn_max, s_threshold=0.0):
    """Parse HITRAN 160-char fixed-width line parameter file.
    Returns list of (nu, sw, gamma_air, gamma_self, elower, n_air) tuples.
    Filters to wavenumber range and intensity threshold.
    """
    lines = []
    with open(filepath, 'r') as f:
        for line in f:
            if len(line) < 60:
                continue
            nu = float(line[3:15])
            sw = float(line[15:25])
            if nu < wn_min or nu > wn_max:
                continue
            if sw < s_threshold:
                continue
            gamma_air = float(line[35:40])
            gamma_self = float(line[40:45])
            elower = float(line[45:55])
            n_air = float(line[55:59])
            lines.append((nu, sw, gamma_air, gamma_self, elower, n_air))

    # Sort by wavenumber
    lines.sort(key=lambda x: x[0])
    return lines


def load_line_params():
    """Load O2 and H2O line parameters for visible range.
    Returns (o2_lines, h2o_lines) where each is a list of
    (nu, sw, gamma_air, gamma_self, elower, n_air) tuples.
    """
    print("\nExtracting line parameters for runtime Voigt computation...")

    # O2: all 481 lines are in visible range
    print("  Parsing O2 line parameters...")
    if not O2_LINE_FILE.exists():
        print(f"ERROR: {O2_LINE_FILE} not found.")
        sys.exit(1)
    o2_lines = parse_hitran_line_params(O2_LINE_FILE, WN_VIS_MIN, WN_VIS_MAX)
    print(f"    O2: {len(o2_lines)} lines in visible range")
    if o2_lines:
        sws = [l[1] for l in o2_lines]
        print(f"    S range: {min(sws):.3e} to {max(sws):.3e}")

    # H2O: filter to S > 1e-25
    print("  Parsing H2O line parameters...")
    if not H2O_LINE_FILE.exists():
        print(f"ERROR: {H2O_LINE_FILE} not found.")
        sys.exit(1)
    h2o_lines = parse_hitran_line_params(
        H2O_LINE_FILE, WN_VIS_MIN, WN_VIS_MAX, s_threshold=H2O_S_THRESHOLD
    )
    print(f"    H2O: {len(h2o_lines)} lines in visible range (S > {H2O_S_THRESHOLD:.0e})")
    if h2o_lines:
        sws = [l[1] for l in h2o_lines]
        print(f"    S range: {min(sws):.3e} to {max(sws):.3e}")
        nbytes = len(h2o_lines) * 6 * 8
        print(f"    Estimated data size: {nbytes/1024:.1f} KB")

    return o2_lines, h2o_lines


# ---------------------------------------------------------------------------
# US Standard Atmosphere 1976 profiles
# ---------------------------------------------------------------------------
def standard_atmosphere():
    """Return standard atmosphere profiles at fixed altitude grid."""
    alts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
    temps = [288.15, 281.65, 275.15, 268.66, 262.17, 255.68, 249.19, 242.70, 236.22, 229.73, 223.25, 216.65, 216.65, 216.65, 221.55, 226.51, 236.51, 250.35, 264.16, 270.65, 247.02, 214.65, 196.65, 186.87, 195.08]
    pressures = [1013.25, 898.76, 795.01, 701.21, 616.60, 540.48, 472.17, 411.05, 356.51, 308.00, 264.99, 193.99, 121.11, 55.293, 25.492, 11.970, 5.746, 2.871, 1.491, 0.798, 0.220, 0.052, 0.010, 0.0016, 0.00032]

    o3_density = [
        5.40e+17, 5.40e+17, 5.40e+17, 5.40e+17, 5.40e+17, 5.50e+17,
        5.60e+17, 6.00e+17, 6.60e+17, 7.50e+17, 8.60e+17, 1.20e+18,
        1.70e+18, 3.50e+18, 5.20e+18, 5.40e+18, 4.60e+18, 3.20e+18,
        1.90e+18, 1.00e+18, 2.80e+17, 6.00e+16, 1.20e+16, 2.00e+15,
        5.00e+14
    ]

    k_B = 1.380649e-23
    air_density = [p * 100 / (k_B * t) for p, t in zip(pressures, temps)]

    no2_density = [
        4.0e+15, 3.0e+15, 2.0e+15, 1.2e+15, 7.0e+14, 4.0e+14,
        2.0e+14, 1.0e+14, 5.0e+13, 2.0e+13, 1.0e+13, 3.0e+12,
        1.0e+12, 2.0e+11, 5.0e+10, 1.0e+10, 2.0e+09, 0, 0, 0,
        0, 0, 0, 0, 0
    ]

    return {
        'n_alts': len(alts),
        'alt_km': alts,
        'temp_k': temps,
        'pressure_hpa': pressures,
        'o3_density': o3_density,
        'air_density': air_density,
        'no2_density': no2_density,
    }


# ---------------------------------------------------------------------------
# Rust source generation
# ---------------------------------------------------------------------------
def fmt_f64(val):
    """Format f64 value for Rust source."""
    if val == 0.0:
        return "0.0"
    if abs(val) < 1e-300:
        return "0.0"
    return f"{val:.6e}"


def write_rust_source(o3_data, o3_temps, o2_xsec, h2o_xsec,
                      no2_294k, no2_220k, o4_cia, std_atm,
                      o2_lines, h2o_lines):
    """Write the gas_absorption_data.rs file."""
    print(f"\nWriting Rust source to {OUT_RS}...")

    lines = []
    lines.append("// Auto-generated by tools/gen_gas_xsec.py -- DO NOT EDIT")
    lines.append("//")
    lines.append("// Molecular gas absorption cross-section data for the visible spectrum.")
    lines.append("//")
    lines.append("// Sources:")
    lines.append("//   O3  : Serdyuchenko et al. (2014), doi:10.5281/zenodo.5793207, CC-BY 4.0")
    lines.append("//   O2  : HITRAN 2020 line list (Gordon et al. 2022), Voigt profiles")
    lines.append("//   H2O : HITRAN 2020 line list (Gordon et al. 2022), Voigt profiles")
    lines.append("//   NO2 : HITRAN cross-section database (measured at 220K and 294K)")
    lines.append("//   O4  : HITRAN 2024 CIA database (O2-O2 collision-induced absorption)")
    lines.append("//   Atm : US Standard Atmosphere 1976 + WMO O3 profile")
    lines.append("//")
    lines.append(f"// Wavelength grid: {WL_MIN}-{WL_MAX} nm at {WL_STEP} nm spacing ({N_WL} points)")
    lines.append(f"// O2/H2O grid: {len(PRESSURES_HPA)} pressures x {len(TEMPERATURES_K)} temperatures = "
                 f"{len(PRESSURES_HPA) * len(TEMPERATURES_K)} conditions")
    lines.append(f"// O2 line params: {len(o2_lines)} lines")
    lines.append(f"// H2O line params: {len(h2o_lines)} lines (S > {H2O_S_THRESHOLD:.0e})")
    lines.append("//")
    lines.append("")
    lines.append(f"/// Number of wavelength points in the reference grid.")
    lines.append(f"pub const GAS_WL_COUNT: usize = {N_WL};")
    lines.append(f"/// Minimum wavelength of the reference grid [nm].")
    lines.append(f"pub const GAS_WL_MIN_NM: f64 = {WL_MIN};")
    lines.append(f"/// Wavelength step of the reference grid [nm].")
    lines.append(f"pub const GAS_WL_STEP_NM: f64 = {WL_STEP};")
    lines.append("")

    # ---- O3 ----
    lines.append(f"/// Number of temperature points for O3 cross-sections.")
    lines.append(f"pub const O3_N_TEMPS: usize = {len(o3_temps)};")
    temps_str = ", ".join(f"{t:.1f}" for t in o3_temps)
    lines.append(f"/// O3 temperature grid [K] (293 to 193 in 10K steps).")
    lines.append(f"pub const O3_TEMPS_K: [f64; {len(o3_temps)}] = [{temps_str}];")
    lines.append(f"/// O3 absorption cross-sections [cm^2/molecule].")
    lines.append(f"/// Indexed: [temp_idx][wl_idx] where wl = {WL_MIN} + wl_idx nm.")
    lines.append(f"pub const O3_XS: [[f64; {N_WL}]; {len(o3_temps)}] = [")
    for t in o3_temps:
        xs = o3_data[t]
        vals = ", ".join(fmt_f64(v) for v in xs)
        lines.append(f"    // {t}K")
        lines.append(f"    [{vals}],")
    lines.append("];")
    lines.append("")

    # ---- O2/H2O 2D (P, T) grid ----
    n_p = len(PRESSURES_HPA)
    n_t = len(TEMPERATURES_K)
    n_cond = n_p * n_t

    lines.append(f"/// Number of pressure points in the O2/H2O grid.")
    lines.append(f"pub const PT_N_PRESSURES: usize = {n_p};")
    lines.append(f"/// Number of temperature points in the O2/H2O grid.")
    lines.append(f"pub const PT_N_TEMPS: usize = {n_t};")
    p_str = ", ".join(f"{p:.2f}" for p in PRESSURES_HPA)
    t_str = ", ".join(f"{t:.1f}" for t in TEMPERATURES_K)
    lines.append(f"/// Pressure grid for O2/H2O [hPa] (descending).")
    lines.append(f"pub const PT_PRESSURES_HPA: [f64; {n_p}] = [{p_str}];")
    lines.append(f"/// Temperature grid for O2/H2O [K] (descending).")
    lines.append(f"pub const PT_TEMPS_K: [f64; {n_t}] = [{t_str}];")
    lines.append("")

    # Emit O2 and H2O as 3D arrays [p_idx][t_idx][wl_idx]
    for name, xsec_dict in [("O2", o2_xsec), ("H2O", h2o_xsec)]:
        lines.append(f"/// {name} absorption cross-sections [cm^2/molecule].")
        lines.append(f"/// Indexed: [{name}_XS[p_idx][t_idx][wl_idx]].")
        lines.append(f"/// Voigt profiles computed at each (P,T) grid point.")
        lines.append(f"pub const {name}_XS: [[[f64; {N_WL}]; {n_t}]; {n_p}] = [")
        for pi, p_hpa in enumerate(PRESSURES_HPA):
            lines.append(f"    // P = {p_hpa:.1f} hPa")
            lines.append(f"    [")
            for ti, t_k in enumerate(TEMPERATURES_K):
                xs = xsec_dict.get((p_hpa, t_k), np.zeros(N_WL))
                vals = ", ".join(fmt_f64(v) for v in xs)
                lines.append(f"        // T = {t_k:.0f} K")
                lines.append(f"        [{vals}],")
            lines.append(f"    ],")
        lines.append("];")
        lines.append("")

    # ---- NO2 ----
    lines.append(f"/// NO2 absorption cross-sections at 294K [cm^2/molecule].")
    lines.append(f"/// Source: HITRAN cross-section database (measured).")
    vals = ", ".join(fmt_f64(v) for v in no2_294k)
    lines.append(f"pub const NO2_XS_294K: [f64; {N_WL}] = [{vals}];")
    lines.append(f"/// NO2 absorption cross-sections at 220K [cm^2/molecule].")
    lines.append(f"/// Source: HITRAN cross-section database (measured).")
    vals = ", ".join(fmt_f64(v) for v in no2_220k)
    lines.append(f"pub const NO2_XS_220K: [f64; {N_WL}] = [{vals}];")
    lines.append("")

    # ---- O4 CIA ----
    lines.append(f"/// O2-O2 collision-induced absorption cross-sections [cm^5/molecule^2].")
    lines.append(f"/// Multiply by [O2]^2 (in molecules/cm^3)^2 to get extinction [cm^-1].")
    lines.append(f"/// Source: HITRAN 2024 CIA database (O2-O2, room temperature composite).")
    vals = ", ".join(fmt_f64(v) for v in o4_cia)
    lines.append(f"pub const O4_CIA_XS: [f64; {N_WL}] = [{vals}];")
    lines.append("")

    # ---- Line parameters for runtime Voigt ----
    def emit_line_params(name, line_list):
        n = len(line_list)
        lines.append(f"/// Number of {name} spectral lines for runtime Voigt computation.")
        lines.append(f"pub const {name}_N_LINES: usize = {n};")

        # Each parameter as separate array for better cache locality
        nus = [l[0] for l in line_list]
        sws = [l[1] for l in line_list]
        gammas_air = [l[2] for l in line_list]
        gammas_self = [l[3] for l in line_list]
        elowers = [l[4] for l in line_list]
        n_airs = [l[5] for l in line_list]

        lines.append(f"/// {name} line centers [cm^-1].")
        lines.append(f"pub const {name}_LINE_NU: [f64; {n}] = [")
        # Write in rows of 8
        for i in range(0, n, 8):
            chunk = nus[i:i+8]
            lines.append("    " + ", ".join(f"{v:.6f}" for v in chunk) + ",")
        lines.append("];")

        lines.append(f"/// {name} line intensities at 296K [cm^-1/(molecule*cm^-2)].")
        lines.append(f"pub const {name}_LINE_SW: [f64; {n}] = [")
        for i in range(0, n, 8):
            chunk = sws[i:i+8]
            lines.append("    " + ", ".join(fmt_f64(v) for v in chunk) + ",")
        lines.append("];")

        lines.append(f"/// {name} air-broadened half-widths at 296K [cm^-1/atm].")
        lines.append(f"pub const {name}_LINE_GAMMA_AIR: [f64; {n}] = [")
        for i in range(0, n, 8):
            chunk = gammas_air[i:i+8]
            lines.append("    " + ", ".join(f"{v:.5f}" for v in chunk) + ",")
        lines.append("];")

        lines.append(f"/// {name} self-broadened half-widths at 296K [cm^-1/atm].")
        lines.append(f"pub const {name}_LINE_GAMMA_SELF: [f64; {n}] = [")
        for i in range(0, n, 8):
            chunk = gammas_self[i:i+8]
            lines.append("    " + ", ".join(f"{v:.4f}" for v in chunk) + ",")
        lines.append("];")

        lines.append(f"/// {name} lower state energies [cm^-1].")
        lines.append(f"pub const {name}_LINE_ELOWER: [f64; {n}] = [")
        for i in range(0, n, 8):
            chunk = elowers[i:i+8]
            lines.append("    " + ", ".join(f"{v:.4f}" for v in chunk) + ",")
        lines.append("];")

        lines.append(f"/// {name} temperature exponent for air-broadened half-width.")
        lines.append(f"pub const {name}_LINE_N_AIR: [f64; {n}] = [")
        for i in range(0, n, 8):
            chunk = n_airs[i:i+8]
            lines.append("    " + ", ".join(f"{v:.2f}" for v in chunk) + ",")
        lines.append("];")
        lines.append("")

    emit_line_params("O2", o2_lines)
    emit_line_params("H2O", h2o_lines)

    # ---- Standard atmosphere ----
    n_alts = std_atm['n_alts']
    lines.append(f"/// Standard atmosphere altitude grid [km].")
    lines.append(f"pub const STD_N_ALTS: usize = {n_alts};")
    vals = ", ".join(f"{v:.1f}" for v in std_atm['alt_km'])
    lines.append(f"pub const STD_ALT_KM: [f64; {n_alts}] = [{vals}];")

    vals = ", ".join(f"{v:.2f}" for v in std_atm['temp_k'])
    lines.append(f"/// Standard atmosphere temperature [K].")
    lines.append(f"pub const STD_TEMP_K: [f64; {n_alts}] = [{vals}];")

    vals = ", ".join(f"{v:.4f}" for v in std_atm['pressure_hpa'])
    lines.append(f"/// Standard atmosphere pressure [hPa].")
    lines.append(f"pub const STD_PRESSURE_HPA: [f64; {n_alts}] = [{vals}];")

    vals = ", ".join(fmt_f64(v) for v in std_atm['air_density'])
    lines.append(f"/// Standard atmosphere air number density [molecules/m^3].")
    lines.append(f"pub const STD_AIR_DENSITY: [f64; {n_alts}] = [{vals}];")

    vals = ", ".join(fmt_f64(v) for v in std_atm['o3_density'])
    lines.append(f"/// Standard atmosphere O3 number density [molecules/m^3].")
    lines.append(f"pub const STD_O3_DENSITY: [f64; {n_alts}] = [{vals}];")

    vals = ", ".join(fmt_f64(v) for v in std_atm['no2_density'])
    lines.append(f"/// Standard atmosphere NO2 number density [molecules/m^3].")
    lines.append(f"pub const STD_NO2_DENSITY: [f64; {n_alts}] = [{vals}];")
    lines.append("")

    OUT_RS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_RS, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    size_kb = os.path.getsize(OUT_RS) / 1024
    print(f"  Written {size_kb:.1f} KB ({len(lines)} lines)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=== Gas Absorption Cross-Section Generator ===")
    print(f"Grid: {WL_MIN}-{WL_MAX} nm, {WL_STEP} nm step, {N_WL} points")
    print(f"O2/H2O grid: {len(PRESSURES_HPA)} pressures x {len(TEMPERATURES_K)} temperatures\n")

    # O3 from Serdyuchenko data file
    o3_data, o3_temps = load_serdyuchenko_o3()

    # NO2 from real HITRAN XSC measured data
    print()
    no2_294k, no2_220k = load_no2_xsc()

    # O2-O2 CIA from real HITRAN 2024 data
    print()
    o4_cia = load_o4_cia()

    # O2 from HITRAN line-by-line (2D grid)
    print("\nComputing O2 cross-sections from HITRAN (2D P,T grid)...")
    o2_xsec = compute_hitran_xsec_2d('O2', 7, 1)

    # H2O from HITRAN line-by-line (2D grid)
    print("\nComputing H2O cross-sections from HITRAN (2D P,T grid)...")
    h2o_xsec = compute_hitran_xsec_2d('H2O', 1, 1)

    # Line parameters for runtime Voigt
    o2_lines, h2o_lines = load_line_params()

    # Standard atmosphere
    print("\nBuilding US Standard Atmosphere 1976 profiles...")
    std_atm = standard_atmosphere()

    # Generate Rust source
    write_rust_source(o3_data, o3_temps, o2_xsec, h2o_xsec,
                      no2_294k, no2_220k, o4_cia, std_atm,
                      o2_lines, h2o_lines)

    print("\nDone! Review the generated file and run `cargo test -p twilight-core`.")


if __name__ == '__main__':
    main()
