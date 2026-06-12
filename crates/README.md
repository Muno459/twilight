# Crates

The twilight workspace is split into 11 crates. Dependencies flow strictly downward.

```
                          twilight-cli
                              |
          +-------------------+-------------------+
          |                   |                   |
    twilight-cpu        twilight-weather    twilight-gpu
          |                   |                   |
    +-----+-----+             |                [Metal]
    |     |     |             |
    |  thresh  solar        data
    |     |                   |
    +-----+-----+-------------+---+
    |           |                 |
  terrain    skyglow            data
    |           |                 |
    |     +-----+-----+           |
    |     |           |           |
    |   core        data        core
    |
    +--- [Copernicus GLO-30]

  twilight-core    (no_std, zero heap, forbid(unsafe_code))
       |
  twilight-ffi     (C FFI: cdylib + staticlib)
```

## Overview

| Crate | `no_std` | Description |
|---|:---:|---|
| **twilight-core** | yes | Physics kernel. Spherical geometry, Rayleigh/HG scattering, full Stokes Mueller matrices, atmosphere model with 64 shells and 64 wavelength slots, Snell's-law refraction machinery (not yet enabled in the production pipeline), molecular gas absorption for five species (O3, O2, H2O, NO2, O4 CIA) via prebaked HITRAN/Serdyuchenko tables, single-scatter LOS integrator, backward MC tracer with NEE, hybrid multi-scatter engine (under rewrite for unbiasedness + variance). All fixed-size arrays, zero heap allocation. |
| **twilight-solar** | no | Solar position. NREL SPA (VSOP87) with nutation. JPL DE440 ephemeris backend with pure Rust DAF/SPK reader, Chebyshev interpolation, IAU precession-nutation. Geometric ICRF positions are mm-level vs Horizons; the delivered topocentric accuracy is arcsecond-level (UT1≈UTC approximation, simplified nutation/precession), which is far more than sufficient for prayer times. |
| **twilight-data** | no | Embedded atmospheric data and builder. US Standard 1976 profile (the only implemented profile). O3 cross-sections (Serdyuchenko 2014, 11 temperatures) and O2/H2O/NO2/O4 data generated from HITRAN by `tools/gen_gas_xsec.py`. TSIS-1 solar spectrum. OPAC-style aerosol climatology (6 types). Cloud optical models (6 types, single-term HG - Mie phase functions pending). `build_full()` assembles a complete atmosphere. |
| **twilight-threshold** | no | CIE vision and prayer time logic. Photopic V(lambda), scotopic V'(lambda), mesopic blend (simplified - not the full CIE 191:2010 MES2 system). Twilight color classification via spectral centroid. Fajr/Isha threshold crossing with log-space interpolation. NOTE: the threshold constants are provisional tuning values, not yet calibrated against published photometry - re-sourcing them is an open task. |
| **twilight-weather** | no | Live weather from Open-Meteo (free, no API key). Fetches AOD at 550nm, cloud cover by altitude, visibility, surface O3 and NO2 from CAMS. Maps AOD to aerosol optical properties and cloud cover to cloud layers. (Surface-O3-to-column and NO2 profile scaling are under revision - surface readings do not determine columns.) |
| **twilight-terrain** | no | Terrain masking. Downloads Copernicus GLO-30 DEM tiles on demand (30m, global). Minimal GeoTIFF reader. 360-point horizon profiles. Currently adjusts sunrise/sunset only - horizon-aware Fajr/Isha is an open task. |
| **twilight-skyglow** | no | Light pollution modeling. Spectral LED/HPS lamp profiles, Bortle class mapping, VIIRS radiance conversion. A Garstang-style RT integration exists but is not yet wired into the pipeline; magnitudes are under validation. |
| **twilight-gpu** | no | Metal GPU backend (the only one). Packed f32 atmosphere buffers with header validation. CPU f64 oracle test suite. Hybrid kernel issues (watchdog, variance) are being fixed alongside the CPU estimator rewrite. |
| **twilight-cpu** | no | Rayon-parallel CPU backend. Simulation driver dispatching single-scatter, MC, and hybrid modes. Two-pass adaptive prayer time pipeline: coarse scan, refine around crossings, threshold analysis, SZA-to-time conversion. Supports terrain masking, skyglow injection, and gas composition overrides. |
| **twilight-ffi** | no | C-compatible FFI. Currently exports a single function (`twilight_solar_zenith`). No prayer-time API is exposed yet. |
| **twilight-cli** | no | Command-line interface: `solar`, `mcrt`, `pray`, `render` subcommands. Supports `--weather`, `--terrain`, `--bortle`, `--de440`, `--fast` (scalar mode), and `--cpu`. |

Test counts are intentionally not listed here; run `cargo test --workspace` for
the current numbers. GPU integration tests skip silently without a Metal device.

## Dependency rules

- **twilight-core** depends on nothing except `libm`. It is the foundation.
- **twilight-solar** depends on nothing except `libm`. Independent of the physics kernel.
- **twilight-data** depends on `twilight-core` (for `AtmosphereModel` and gas absorption types).
- **twilight-threshold** depends on `twilight-core` (for spectral array types).
- **twilight-weather** depends on `twilight-data` (for aerosol/cloud property types). Uses `ureq` + `serde_json`.
- **twilight-terrain** depends on nothing in the workspace. Uses `tiff` + `ureq`.
- **twilight-skyglow** depends on `twilight-core` and `twilight-data`.
- **twilight-gpu** depends on `twilight-core`, `twilight-data`, `twilight-threshold`, `twilight-skyglow`. Metal is behind the `metal` feature flag.
- **twilight-cpu** depends on most crates. This is the orchestration layer.
- **twilight-cli** depends on everything. This is the user-facing binary.
- **twilight-ffi** depends on `twilight-core` and `twilight-solar` only, keeping the FFI surface minimal.

## `no_std` contract

`twilight-core` is `#![no_std]` and `#![forbid(unsafe_code)]`. It uses:

- No `Vec`, `String`, `Box`, or any heap allocation
- No `std::*` imports
- `libm` for transcendental math (`sin`, `cos`, `exp`, `log`, `sqrt`, `atan2`)
- Fixed-size arrays: `[f64; MAX_WAVELENGTHS]`, `[Shell; MAX_SHELLS]` where both limits are 64

This means the same physics code compiles for bare-metal embedded targets and
WASM. The `no_std` constraint is enforced at the crate level and will break
the build if violated.

## Build

From the workspace root:

```bash
# Run the test suite
cargo test --workspace

# Build everything (release)
cargo build --release

# Run the CLI
./target/release/twilight-cli pray \
  --lat 21.4225 --lon 39.8262 --date 2024-03-20 --tz 3.0 --weather

# Build with the Metal GPU backend (Apple Silicon)
cargo build --release -p twilight-cli --features gpu
```
