<p align="center">
  <img src="assets/banner.svg" alt="twilight" width="100%"/>
</p>

<p align="center">
  <a href="https://github.com/Muno459/twilight/actions"><img src="https://img.shields.io/badge/tests-446_passing-brightgreen?style=flat-square" alt="tests"/></a>
  <a href="https://github.com/Muno459/twilight"><img src="https://img.shields.io/badge/rust-pure_%23!%5Bno__std%5D_core-orange?style=flat-square" alt="rust"/></a>
  <a href="#license"><img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-blue?style=flat-square" alt="license"/></a>
</p>

<h3 align="center">The most accurate dawn and dusk calculator ever (yet) built.</h3>
<p align="center">Your prayer app doesn't know if it's cloudy.</p>

<br/>

`twilight` computes Fajr and Isha prayer times by simulating how sunlight actually scatters through the atmosphere. No lookup tables, no fixed depression angles. Photons in, prayer times out. 8 ms.

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

# Raw spectral radiance across twilight
cargo run --release -- mcrt --lat 21.4225 --lon 39.8262 --sza-start 90 --sza-end 108
```

Aerosol types: `continental-clean`, `continental-average`, `urban`, `maritime-clean`, `maritime-polluted`, `desert`.
Cloud types: `thin-cirrus`, `thick-cirrus`, `altostratus`, `stratus`, `stratocumulus`, `cumulus`.


## How it works

SPA computes the sun's position (VSOP87, ±0.0003°). The atmosphere builder creates 50 spherical shells with Rayleigh scattering, ozone absorption, optional aerosol extinction, and optional cloud layers at 41 wavelengths. For each solar zenith angle, the single-scatter integrator traces a line of sight toward the horizon, computing analytical shadow rays to the sun through each shell. CIE mesopic vision converts spectral radiance to perceived luminance. A two-pass adaptive scan finds where brightness drops below the perceptual threshold.

Deterministic. No Monte Carlo noise. Same input, same output, bit-for-bit.

<details>
<summary>Pipeline details</summary>

1. **Solar position.** NREL SPA with full VSOP87 + 63-term nutation. Binary search for sunrise/sunset. Persistent twilight detection at high latitudes.

2. **Atmosphere.** 50 shells, 0 to 100 km. Rayleigh via Bodhaine (1999) with exact Lorentz-Lorenz. O₃ via Serdyuchenko (2014). OPAC aerosol climatology (6 types) with Angstrom extinction and Henyey-Greenstein phase function. Cloud layers (6 types: cirrus to cumulus). Lambertian ground reflection.

3. **Radiative transfer.** LOS integration with analytical shell-by-shell shadow rays. 41 wavelengths, 380 to 780 nm. Produces spectral radiance vectors at each SZA.

4. **Vision model.** CIE photopic/scotopic/mesopic luminance. Spectral centroid classifies twilight color: blue, white (*shafaq al-abyad*), orange, red (*shafaq al-ahmar*), dark.

5. **Threshold search.** Coarse scan (0.5°) then fine scan (0.1°) around crossings. Fajr = mesopic threshold. Isha = red-band threshold. SZA converted to clock time via SPA.

</details>


## Crates

| Crate | What |
|---|---|
| `twilight-core` | Physics kernel. `#![no_std]`, `#![forbid(unsafe_code)]`, zero heap. Geometry, scattering, atmosphere, single-scatter integrator, MC tracer. |
| `twilight-solar` | NREL SPA. ±0.0003° for years -2000 to 6000. |
| `twilight-data` | Embedded data. US Std 1976, TSIS-1 solar spectrum, O₃ cross-sections, OPAC aerosols, cloud types, builder. |
| `twilight-threshold` | CIE vision, mesopic luminance, twilight color classification, prayer time thresholds. |
| `twilight-cpu` | Rayon parallel backend. Simulation driver, adaptive pipeline. |
| `twilight-ffi` | C FFI. `cdylib` + `staticlib` for iOS/Android/Flutter. |
| `twilight-cli` | CLI. `solar`, `mcrt`, `pray`. |

`twilight-core` is `no_std` with no `Vec`, `String`, or `Box`. Everything is `[f64; 64]`. Same physics code runs on phone, browser, GPU.


## What's missing

- **Multiple scattering.** Single-scatter underestimates deep twilight brightness. This is why we get ~15° instead of 18°. Also limits accuracy for thick clouds (OD > ~5).
- **Terrain, light pollution.** Planned.
- **One atmosphere profile.** US Standard 1976 everywhere, for now.


## Tests

446 tests, 0.2 seconds. `cargo test --workspace`

| Crate | Tests |
|---|---|
| `twilight-core` | 135 |
| `twilight-data` | 139 |
| `twilight-threshold` | 72 |
| `twilight-cpu` | 52 |
| `twilight-solar` | 47 |


## Roadmap

- [x] Solar position, atmosphere model, single-scatter engine, vision model, prayer pipeline, ground reflection, aerosols, cloud layers, C FFI
- [ ] Multiple scattering (backward MC with next-event estimation)
- [ ] Real-time weather, satellite cloud fields (GOES/Himawari/Meteosat + ML)
- [ ] Light pollution, terrain masking
- [ ] GPU backend, neural surrogate, mobile SDKs, WASM


## License

MIT OR Apache-2.0
