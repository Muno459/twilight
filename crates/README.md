# Crates

The twilight workspace is split into 11 crates plus one placeholder. Dependencies flow strictly downward.

```
                          twilight-cli
                              |
          +-------------------+-------------------+
          |                   |                   |
    twilight-cpu        twilight-weather    twilight-gpu
          |                   |                   |
    +-----+-----+            |          +---------+---------+
    |     |     |            |          |         |         |
    |  thresh  solar      data     [Metal]  [Vulkan]  [CUDA]  [wgpu]
    |     |                 |
    +-----+-----+-----------+-----+
    |           |                 |
  terrain    skyglow            data
    |           |                 |
    |     +-----+-----+          |
    |     |           |          |
    |   core        data        core
    |
    +--- [Copernicus GLO-30]
    +--- [Danish SDFI LiDAR]

  twilight-core    (no_std, zero heap, forbid(unsafe_code))
       |
  twilight-ffi     (C FFI: cdylib + staticlib)

  twilight-clouds  (placeholder, not yet active)
```

## Overview

| Crate | `no_std` | Tests | Description |
|---|:---:|---:|---|
| **twilight-core** | yes | 311 | Physics kernel. Spherical geometry, Rayleigh/HG scattering, full Stokes Mueller matrices, atmosphere model with 64 shells and 64 wavelength slots, atmospheric refraction via Snell's law, molecular gas absorption for five species (O3, O2, H2O, NO2, O4 CIA) with Voigt line-by-line profiles, single-scatter LOS integrator, backward MC tracer with NEE, hybrid multi-scatter engine. All fixed-size arrays, zero heap allocation. |
| **twilight-solar** | no | 73 | Solar position. NREL SPA (VSOP87, +/-0.0003 degrees) with full 63-term nutation. JPL DE440 ephemeris backend with pure Rust DAF/SPK reader, Chebyshev interpolation, IAU precession-nutation, ICRF-to-topocentric conversion. Validated to 8 meters vs JPL Horizons. Binary search zenith crossing for SZA-to-time conversion. |
| **twilight-data** | no | 153 | Embedded atmospheric data and builder. US Standard 1976 profiles (T, P, n from 0-100 km). O3 cross-sections (Serdyuchenko 2014). TSIS-1 solar spectrum. OPAC aerosol climatology (6 types). Cloud optical models (6 types). `build_full()` assembles a complete atmosphere with Rayleigh, gas absorption, aerosols, and clouds. `build_full_with_gas()` accepts live O3 column and NO2 overrides. |
| **twilight-threshold** | no | 72 | CIE vision and prayer time logic. Photopic V(lambda) (CIE 1924), scotopic V'(lambda) (CIE 1951), CIE 2010 mesopic adaptation. Spectral luminance integration at 683 lm/W and 1700 lm/W. Twilight color classification (blue, white, orange, red, dark) via spectral centroid. Fajr/Isha threshold crossing with log-space interpolation. |
| **twilight-weather** | no | 45 | Live weather from Open-Meteo (free, no API key). Fetches AOD at 550nm, cloud cover by altitude (low/mid/high), visibility, dust concentration, surface O3 and NO2 from CAMS global forecasts. Maps AOD to aerosol optical properties, cloud cover to cloud layers, O3 to total column Dobson Units, NO2 to number density. |
| **twilight-terrain** | no | 47 | Terrain masking. Downloads Copernicus GLO-30 DEM tiles on demand (30m resolution, global). Parses GeoTIFF with a minimal built-in reader. Computes 360-point horizon profiles. Also supports Danish SDFI national LiDAR (0.4m resolution). Adjusts effective SZA per azimuth for prayer time computation. |
| **twilight-skyglow** | no | 69 | Light pollution modeling. Garstang radiative transfer for skyglow brightness as a function of distance and zenith angle. Spectral LED (3000K-5000K) and HPS lamp profiles with Rayleigh/Mie wavelength-dependent scattering. Bortle class (1-9) mapping, VIIRS nighttime radiance conversion, prayer time shift estimation. |
| **twilight-gpu** | no | 99 | Four GPU backends, each hand-tuned for its native API. Metal (.metal shaders, objc2-metal host), Vulkan (GLSL compiled to SPIR-V, ash host), CUDA (.cu shaders, cudarc host), wgpu (.wgsl shaders for WASM). Packed f32 atmosphere buffers with header validation. CPU oracle test suite for cross-backend physics parity. Benchmarks. |
| **twilight-cpu** | no | 82 | Rayon-parallel CPU backend. Simulation driver dispatching single-scatter, MC, and hybrid modes. Two-pass adaptive prayer time pipeline: coarse scan (0.5 degrees), refine around crossings (0.1 degrees), threshold analysis, SZA-to-time conversion. Supports terrain masking, skyglow injection, and gas composition overrides. |
| **twilight-ffi** | no | 0 | C-compatible FFI. Exports `twilight_solar_zenith` as both `cdylib` (shared library) and `staticlib` (static archive). Targets iOS (aarch64-apple-ios), Android (aarch64-linux-android), and Flutter (Dart FFI). |
| **twilight-cli** | no | 0 | Command-line interface. Three subcommands: `solar` (position and conventional times), `mcrt` (raw spectral radiance), `pray` (full MCRT prayer times with comparison to fixed-angle methods). Supports `--weather`, `--terrain`, `--bortle`, `--de440`, `--polarized`, and GPU backend selection. |

## Dependency rules

- **twilight-core** depends on nothing except `libm`. It is the foundation.
- **twilight-solar** depends on nothing except `libm`. Independent of the physics kernel.
- **twilight-data** depends on `twilight-core` (for `AtmosphereModel` and gas absorption types).
- **twilight-threshold** depends on `twilight-core` (for spectral array types).
- **twilight-weather** depends on `twilight-data` (for aerosol/cloud property types). Uses `ureq` + `serde_json`.
- **twilight-terrain** depends on nothing in the workspace. Uses `tiff` + `ureq`.
- **twilight-skyglow** depends on `twilight-core` and `twilight-data`.
- **twilight-gpu** depends on `twilight-core`, `twilight-data`, `twilight-threshold`, `twilight-skyglow`. GPU libraries are behind feature flags.
- **twilight-cpu** depends on most crates. This is the orchestration layer.
- **twilight-cli** depends on everything. This is the user-facing binary.
- **twilight-ffi** depends on `twilight-core` and `twilight-solar` only, keeping the FFI surface minimal.

## `no_std` contract

`twilight-core` is `#![no_std]` and `#![forbid(unsafe_code)]`. It uses:

- No `Vec`, `String`, `Box`, or any heap allocation
- No `std::*` imports
- `libm` for transcendental math (`sin`, `cos`, `exp`, `log`, `sqrt`, `atan2`)
- Fixed-size arrays: `[f64; MAX_WAVELENGTHS]`, `[Shell; MAX_SHELLS]` where both limits are 64

This means the same physics code compiles for bare-metal embedded targets, WASM, and can be translated to GPU shader languages. The `no_std` constraint is enforced at the crate level and will break the build if violated.

## Build

From the workspace root:

```bash
# Run all 938 tests (~14 seconds)
cargo test --workspace

# Build everything (release)
cargo build --release

# Run the CLI
./target/release/twilight-cli pray \
  --lat 21.4225 --lon 39.8262 --date 2024-03-20 --tz 3.0 --weather

# Build with GPU backends (feature-gated)
cargo build --release --features gpu-metal    # Apple Silicon
cargo build --release --features gpu-vulkan   # AMD/Intel/Android
cargo build --release --features gpu-cuda     # NVIDIA
cargo build --release --features gpu-wgpu     # WASM/browsers
```
