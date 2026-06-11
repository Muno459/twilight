<p align="center">
  <img src="assets/banner.svg" alt="twilight" width="100%"/>
</p>

<p align="center">
  <a href="https://github.com/Muno459/twilight"><img src="https://img.shields.io/badge/rust-pure_%23!%5Bno__std%5D_core-orange?style=flat-square" alt="rust"/></a>
  <a href="https://github.com/Muno459/twilight"><img src="https://img.shields.io/badge/GPU-Metal-blueviolet?style=flat-square" alt="gpu"/></a>
  <a href="#license"><img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-blue?style=flat-square" alt="license"/></a>
</p>

<h3 align="center">A physically-based dawn and dusk calculator.</h3>
<p align="center">Computing Fajr and Isha from atmospheric radiative transfer instead of fixed depression angles.</p>

<br/>

`twilight` computes Fajr and Isha prayer times by simulating how sunlight scatters through the atmosphere: Monte Carlo radiative transfer through a 50-shell spherical atmosphere with Rayleigh scattering, five-species molecular gas absorption (O3, O2, H2O, NO2, O4 CIA from HITRAN/Serdyuchenko data), OPAC-style aerosols, cloud layers, full Stokes polarization, terrain masking from Copernicus GLO-30 DEM, a light-pollution estimate, and optional live weather from Open-Meteo. Optional JPL DE440 ephemeris for solar positioning. Metal GPU acceleration on Apple Silicon.

## Project status — read this first

This engine is under an active correctness overhaul (`overhaul` branch). A
full two-model multi-agent review (2026-06) found real physics bugs and
overstated claims in earlier versions of this README; this version states
what the code actually does. Until the items below are closed, treat all
computed prayer times as **experimental** — do not use them for worship
without cross-checking against established calendars.

Progress so far (see git log on this branch):

1. ~~Phase-function scattering angle bug~~ FIXED (all 14 sites + regression test)
2. ~~MS estimator seed bias / chain boundary-kill / SSA ordering~~ FIXED
   (unbiased one-sample-MIS seed; chains traverse shells; suites green).
   Measured deep-twilight CV (500 seeds, hybrid): 0.12-0.30 at SZA 96-102
   with ZERO negative samples (previously negative radiance seeds occurred);
   rare tail outliers remain at SZA >= 104 (~1 seed in 500 at 20-100x) —
   K-seed averaging + crossing-on-fit in the pipeline is the tracked next
   step for sub-minute crossing stability
3. ~~Cloudy-sky collapse~~ FIXED: delta-Eddington scaled cloud optics +
   Eddington diffuse transmission for the cloud portion of eye/sun paths.
   OD-10 stratus now gives Fajr at depression ~13.7° (clear sky ~15.9°)
   instead of degenerating to sunrise.
4. Thresholds re-anchored to published photometry (night-sky background
   2.2e-4 cd/m² per Patat 2008; mesopic/cone-threshold boundaries) with the
   derivation documented — field SQM calibration still pending
5. OPEN: atmosphere ceiling at 100 km truncates deep-twilight radiance;
   the libRadtran tier-1b (MYSTIC spherical) harness quantifies this once
   libRadtran is installed
6. OPEN: Metal hybrid kernel needs the corrected estimator ported
   (watchdog/variance root causes identified)

## Why

Every prayer app hardcodes a solar depression angle. MWL says 18 degrees.
Egypt says 15 degrees. Umm al-Qura uses fixed offsets. They disagree because
"when does twilight end?" depends on the atmosphere, not on a single number.
Aerosols, ozone, clouds, terrain, and light pollution all shift the moment
the sky actually darkens. A radiative-transfer engine can model that — once
its physics is verified. That verification (against libRadtran/DISORT/MYSTIC
and published twilight photometry) is the current focus of this project.

## Quick start

```bash
cargo build --release

# Prayer times (clear sky)
./target/release/twilight-cli pray \
  --lat 21.4225 --lon 39.8262 --date 2024-03-20 --tz 3.0

# With live weather (AOD, cloud cover, O3/NO2 from Open-Meteo)
./target/release/twilight-cli pray \
  --lat 21.4225 --lon 39.8262 --date 2024-03-20 --tz 3.0 --weather

# With terrain masking (Copernicus GLO-30 DEM, auto-downloaded)
# NOTE: terrain currently adjusts sunrise/sunset only, not Fajr/Isha
./target/release/twilight-cli pray \
  --lat 21.4225 --lon 39.8262 --date 2024-03-20 --tz 3.0 --terrain

# With a light-pollution estimate (Bortle class)
./target/release/twilight-cli pray \
  --lat 55.653 --lon 12.412 --date 2026-03-06 --bortle 7

# Manual aerosols and/or clouds (thick clouds use delta-Eddington diffuse
# transmission; see Project status)
./target/release/twilight-cli pray \
  --lat 21.4225 --lon 39.8262 --date 2024-03-20 --tz 3.0 \
  --aerosol urban --cloud thin-cirrus

# Multi-scatter hybrid mode (exact order 1 + MC orders 2+) — the default
./target/release/twilight-cli pray \
  --lat 21.4225 --lon 39.8262 --date 2024-03-20 --tz 3.0 \
  --scattering hybrid --photons 500

# Force CPU (opt out of Metal GPU)
./target/release/twilight-cli pray \
  --lat 21.4225 --lon 39.8262 --date 2024-03-20 --tz 3.0 --cpu

# Scalar mode (skip Stokes polarization for speed)
./target/release/twilight-cli mcrt \
  --lat 21.4225 --lon 39.8262 --sza-start 90 --sza-end 108 \
  --scattering hybrid --fast

# Solar position with JPL DE440 vs SPA comparison
./target/release/twilight-cli solar \
  --lat 21.4225 --lon 39.8262 --date 2024-06-15 --tz 3 \
  --de440 data/de440.bsp

# Raw spectral radiance across twilight
./target/release/twilight-cli mcrt \
  --lat 21.4225 --lon 39.8262 --sza-start 90 --sza-end 108
```

Aerosol types: `continental-clean`, `continental-average`, `urban`, `maritime-clean`, `maritime-polluted`, `desert`.

Cloud types: `thin-cirrus`, `thick-cirrus`, `altostratus`, `stratus`, `stratocumulus`, `cumulus`.

`--weather` fetches current conditions from [Open-Meteo](https://open-meteo.com/) (free, no API key). It maps measured AOD at 550nm to aerosol optical properties and cloud cover by altitude to cloud layers. (The surface-O3/NO2 → column mappings are under revision: a surface reading does not determine a column.)

## How it works

<details>
<summary>Full pipeline</summary>

**1. Solar position.** NREL SPA (VSOP87) as default. Optional JPL DE440 ephemeris backend with a pure Rust DAF/SPK reader, Chebyshev interpolation, and IAU precession-nutation. Geometric ICRF positions are mm-level vs Horizons; the delivered topocentric chain is arcsecond-level (UT1≈UTC, simplified nutation), far more than sufficient for prayer times. Binary search for sunrise/sunset.

**2. Atmosphere.** 50 concentric spherical shells, 0 to 100 km, non-uniform spacing. Rayleigh scattering via Bodhaine (1999). Five-species molecular gas absorption from prebaked tables generated offline from HITRAN/Serdyuchenko data by `tools/gen_gas_xsec.py` (O3 11-temperature, O2/H2O bilinear P-T grids, NO2 two-temperature, O4 CIA). OPAC-style aerosol climatology (6 types) with Henyey-Greenstein phase function. Cloud layers (6 types; single-term HG — a known limitation, Mie phase functions pending). Lambertian ground reflection. Snell's-law refraction at shell boundaries (enabled in production; surface n = 1.000293). Optically thick clouds use delta-Eddington scaled optics with Eddington diffuse transmission for the cloud portion of eye/sun paths.

**3. Radiative transfer.** Three modes: (a) single-scatter LOS integration with analytical shadow rays (deterministic); (b) backward Monte Carlo with next-event estimation; (c) hybrid: exact single-scatter + MC secondary chains for orders 2+. Full Stokes polarized RT (default; `--fast` for scalar). 41 wavelengths, 380-780 nm. The orders-2+ seed is an unbiased one-sample-MIS estimator (balance heuristic over a phase/zenith/terminator mixture); chains traverse shell boundaries with memoryless resampling.

**4. Terrain masking.** Copernicus GLO-30 DEM tiles (auto-downloaded). Computes a 360-point horizon profile. Currently adjusts sunrise/sunset only; horizon-aware Fajr/Isha is an open task.

**5. Light pollution.** Bortle class or VIIRS-style radiance input → spectral LED/HPS skyglow estimate added to the twilight signal. A Garstang RT integration exists in the codebase but is not yet wired in or validated.

**6. Vision model.** CIE photopic/scotopic luminance with a simplified mesopic blend. Spectral centroid classifies twilight color: blue, white (*shafaq al-abyad*), orange, red (*shafaq al-ahmar*), dark.

**7. Threshold search.** Coarse scan then fine scan around crossings. Fajr/Isha thresholds are provisional constants pending calibration against published twilight photometry. SZA converted to clock time via SPA binary search.

**8. GPU acceleration.** Metal backend (Apple Silicon) with hand-written MSL shaders, f32 precision engineering (half-b ray-sphere intersection, boundary snapping, Kahan summation), and a CPU f64 oracle test suite. The hybrid kernel currently has known watchdog/variance issues being fixed.

</details>

## Crates

```
twilight/
  crates/
    twilight-core/       Physics kernel (#![no_std], zero heap)
    twilight-solar/      SPA + DE440 ephemeris
    twilight-data/       Profiles, cross-sections, atmosphere builder
    twilight-threshold/  CIE vision, twilight color, prayer thresholds
    twilight-cpu/        Rayon parallel driver, adaptive pipeline
    twilight-gpu/        Metal backend
    twilight-weather/    Live weather from Open-Meteo
    twilight-terrain/    Copernicus GLO-30 DEM, horizon masking
    twilight-skyglow/    Light pollution estimate
    twilight-ffi/        C FFI (currently a single SPA function)
    twilight-cli/        CLI: solar, mcrt, pray, render
```

See [crates/README.md](crates/README.md) for per-crate detail and the
dependency graph. `twilight-core` is `no_std` with no `Vec`, `String`, or
`Box` — the same physics code can run on a phone or in a browser.

## Tests

`cargo test --workspace` runs the suite (several minutes; the MC convergence
tests are slow). GPU integration tests skip silently when no Metal device is
present. Verified external anchors include the NREL SPA reference case and
US Standard Atmosphere values; a libRadtran comparison harness is being added
as the primary physics anchor.

## Roadmap

- [x] Solar position (SPA + DE440), atmosphere model, single-scatter engine
- [x] CIE vision model, threshold scan pipeline
- [x] Surface albedo, OPAC-style aerosols (6 types), cloud layers (6 types)
- [x] Backward MC + hybrid multi-scatter (correctness rewrite IN PROGRESS)
- [x] Live weather via Open-Meteo (AOD, cloud cover)
- [x] Terrain masking for sunrise/sunset (Copernicus GLO-30)
- [x] Metal GPU backend
- [ ] libRadtran/DISORT/MYSTIC validation harness  ← current focus
- [x] Unbiased multiple-scattering estimator (CPU; Metal port pending)
- [x] Cloud transport that survives optical depth > 1 (delta-Eddington coupling)
- [ ] Atmosphere ceiling above 100 km for deep-twilight radiance
- [x] Refraction enabled in the production pipeline
- [ ] Thresholds: anchored to published photometry (done); field SQM calibration pending
- [ ] Horizon-aware Fajr/Isha (terrain)
- [ ] Garstang skyglow wired into the pipeline and validated
- [ ] Mobile SDKs (iOS/Android), WASM demo

## License

MIT OR Apache-2.0
