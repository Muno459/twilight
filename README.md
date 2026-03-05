<p align="center">
  <img src="assets/banner.svg" alt="twilight" width="100%"/>
</p>

<p align="center">
  <a href="https://github.com/Muno459/twilight/actions"><img src="https://img.shields.io/badge/tests-494_passing-brightgreen?style=flat-square" alt="tests"/></a>
  <a href="https://github.com/Muno459/twilight"><img src="https://img.shields.io/badge/rust-pure_%23!%5Bno__std%5D_core-orange?style=flat-square" alt="rust"/></a>
  <a href="#license"><img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-blue?style=flat-square" alt="license"/></a>
</p>

<h3 align="center">The most accurate dawn and dusk calculator ever (yet) built.</h3>
<p align="center">Your prayer app doesn't know if it's cloudy.</p>

<br/>

`twilight` computes Fajr and Isha prayer times by simulating how sunlight actually scatters through the atmosphere. No lookup tables, no fixed depression angles. Photons in, prayer times out. 8 ms (single-scatter) or 50 s (hybrid multi-scatter). Three scattering modes: deterministic single-scatter, backward Monte Carlo, and hybrid (exact order 1 + MC orders 2+). Optional JPL DE440 ephemeris for sub-meter solar positioning.

## Why

Every prayer app hardcodes a solar depression angle. MWL says 18°. Egypt says 15°. Umm al-Qura says 19.5°. They disagree because "when does twilight end?" depends on the atmosphere, not on a number someone picked in 1986.

```
Mecca, March equinox, clear sky:

  Fajr:   05:23   depression 14.97°    (Egypt 15° says 05:23. Spot on.)
  Isha:   19:32   depression 14.83°    (ISNA 18° says 19:45. Off by 13 min.)
```

Add urban aerosol haze and the answer shifts:

```
Mecca, March equinox, urban aerosol (AOD 0.30):

  Fajr:   05:32   depression 13.03°    (+8 min vs clear sky)
  Isha:   19:25   depression 13.21°    (-7 min vs clear sky)
```

Fixed angles can't account for this. The atmosphere matters.


## Quick start

```bash
cargo build --release

# Prayer times (clear sky)
cargo run --release -- pray --lat 21.4225 --lon 39.8262 --date 2024-03-20 --tz 3.0

# With aerosols and/or clouds
cargo run --release -- pray --lat 21.4225 --lon 39.8262 --date 2024-03-20 --tz 3.0 --aerosol urban
cargo run --release -- pray --lat 21.4225 --lon 39.8262 --date 2024-03-20 --tz 3.0 --cloud thin-cirrus
cargo run --release -- pray --lat 21.4225 --lon 39.8262 --date 2024-03-20 --tz 3.0 --aerosol urban --cloud thin-cirrus

# Solar position with JPL DE440 comparison
cargo run --release -- solar --lat 21.4225 --lon 39.8262 --date 2024-06-15 --tz 3 --de440 data/de440.bsp

# Raw spectral radiance across twilight
cargo run --release -- mcrt --lat 21.4225 --lon 39.8262 --sza-start 90 --sza-end 108

# Hybrid multi-scatter mode (reaches 18° depression)
cargo run --release -- mcrt --lat 21.4225 --lon 39.8262 --scattering hybrid --photons 1000
cargo run --release -- pray --lat 21.4225 --lon 39.8262 --date 2024-03-20 --tz 3.0 --scattering hybrid --photons 500
```

Aerosol types: `continental-clean`, `continental-average`, `urban`, `maritime-clean`, `maritime-polluted`, `desert`.
Cloud types: `thin-cirrus`, `thick-cirrus`, `altostratus`, `stratus`, `stratocumulus`, `cumulus`.


## How it works

SPA computes the sun's position (VSOP87, ±0.0003°). The atmosphere builder creates 50 spherical shells with Rayleigh scattering, ozone absorption, optional aerosol extinction, and optional cloud layers at 41 wavelengths. For each solar zenith angle, the integrator traces a line of sight toward the horizon. In single-scatter mode, analytical shadow rays are computed through each shell (deterministic, fast). In hybrid mode, each LOS step also launches MC secondary chains that scatter upward to sunlit altitudes and back down, capturing orders 2+ of scattering. CIE mesopic vision converts spectral radiance to perceived luminance. A two-pass adaptive scan finds where brightness drops below the perceptual threshold.

Single-scatter mode is deterministic. Same input, same output, bit-for-bit. Hybrid mode uses MC for orders 2+ and converges with ~500-1000 secondary rays per step.

<details>
<summary>Pipeline details</summary>

1. **Solar position.** NREL SPA (VSOP87, ±0.0003°) as default. Optional JPL DE440 ephemeris backend with pure Rust DAF/SPK reader, Chebyshev interpolation, IAU precession-nutation, and ICRF-to-topocentric conversion. DE440 validated to 8 meters vs JPL Horizons. Binary search for sunrise/sunset. Persistent twilight detection at high latitudes.

2. **Atmosphere.** 50 shells, 0 to 100 km. Rayleigh via Bodhaine (1999) with exact Lorentz-Lorenz. O₃ via Serdyuchenko (2014). OPAC aerosol climatology (6 types) with Angstrom extinction and Henyey-Greenstein phase function. Cloud layers (6 types: cirrus to cumulus). Lambertian ground reflection.

3. **Radiative transfer.** Three modes: (a) single-scatter LOS integration with analytical shadow rays (8 ms, deterministic); (b) backward Monte Carlo with next-event estimation (all orders, noisy); (c) hybrid -- exact single-scatter + MC secondary chains with upward-biased importance sampling for orders 2+ (reaches 18° depression). 41 wavelengths, 380 to 780 nm.

4. **Vision model.** CIE photopic/scotopic/mesopic luminance. Spectral centroid classifies twilight color: blue, white (*shafaq al-abyad*), orange, red (*shafaq al-ahmar*), dark.

5. **Threshold search.** Coarse scan (0.5°) then fine scan (0.1°) around crossings. Fajr = mesopic threshold. Isha = red-band threshold. SZA converted to clock time via SPA.

</details>


## Crates

| Crate | What |
|---|---|
| `twilight-core` | Physics kernel. `#![no_std]`, `#![forbid(unsafe_code)]`, zero heap. Geometry, scattering, atmosphere, single-scatter integrator, backward MC tracer, hybrid multi-scatter engine. |
| `twilight-solar` | NREL SPA (±0.0003°) + JPL DE440 ephemeris backend (±0.001"). Pure Rust DAF/SPK reader. |
| `twilight-data` | Embedded data. US Std 1976, TSIS-1 solar spectrum, O₃ cross-sections, OPAC aerosols, cloud types, builder. |
| `twilight-threshold` | CIE vision, mesopic luminance, twilight color classification, prayer time thresholds. |
| `twilight-cpu` | Rayon parallel backend. Simulation driver (single/MC/hybrid dispatch), adaptive pipeline. |
| `twilight-ffi` | C FFI. `cdylib` + `staticlib` for iOS/Android/Flutter. |
| `twilight-cli` | CLI. `solar`, `mcrt`, `pray`. |

`twilight-core` is `no_std` with no `Vec`, `String`, or `Box`. Everything is `[f64; 64]`. Same physics code runs on phone, browser, GPU.


## What's missing

- **Thick cloud multi-scatter convergence.** Hybrid mode reaches 18° in clear sky and thin clouds, but thick clouds (stratus OD=10) still attenuate the signal below threshold. Needs more secondary rays or smarter importance sampling in the cloud layer.
- **Terrain, light pollution.** Planned.
- **One atmosphere profile.** US Standard 1976 everywhere, for now.
- **GPU acceleration.** Hybrid mode takes ~50s on Apple Silicon. CUDA/WGSL backend would bring this under 1s.


## Tests

494 tests, ~11 seconds. `cargo test --workspace`

| Crate | Tests |
|---|---|
| `twilight-core` | 147 |
| `twilight-data` | 139 |
| `twilight-threshold` | 72 |
| `twilight-solar` | 63 (+10 DE440 integration) |
| `twilight-cpu` | 73 |


## Roadmap

- [x] Solar position, atmosphere model, single-scatter engine, vision model, prayer pipeline, ground reflection, aerosols, cloud layers, C FFI
- [x] JPL DE440 ephemeris (pure Rust DAF/SPK reader, validated to 8 m vs Horizons)
- [x] Multiple scattering: backward MC with NEE, hybrid single-scatter + MC orders 2+
- [x] Upward-biased importance sampling for deep twilight connectivity
- [ ] Real-time weather, satellite cloud fields (GOES/Himawari/Meteosat + ML)
- [ ] Light pollution, terrain masking
- [ ] GPU backend, neural surrogate, mobile SDKs, WASM


## License

MIT OR Apache-2.0
