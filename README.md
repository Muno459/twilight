# twilight

A Monte Carlo Radiative Transfer engine in pure Rust that computes Fajr and Isha prayer times from first principles. Instead of using fixed solar depression angles (the standard approach since the 1970s), `twilight` simulates photon transport through a spherical atmosphere during twilight and determines when sky luminance crosses perceptual thresholds.

The engine currently models Rayleigh scattering, ozone absorption, and 41-band spectral resolution (380--780 nm) through a 50-shell spherical atmosphere based on the US Standard 1976 profile. It produces physically-based prayer times in under 10 ms on consumer hardware.

## Why this exists

Every Islamic prayer time app on the planet uses the same method: pick a solar depression angle (15, 18, 19.5 degrees depending on convention), and call it a day. This works reasonably well at mid-latitudes, but the actual onset of dawn and disappearance of twilight depend on atmospheric conditions --- aerosols, clouds, ozone, surface albedo, light pollution, altitude --- none of which a single angle captures.

The disagreement between conventions (MWL uses 18, Egyptian authority uses 15, Umm al-Qura uses 19.5) exists precisely because no single angle is universally correct. The physics varies by location, season, and weather.

This project replaces the angle with the physics.


## Quick start

```
cargo build --release

# Compute prayer times for Mecca on the spring equinox
cargo run --release -- pray --lat 21.4225 --lon 39.8262 --date 2024-03-20 --tz 3.0

# Solar position and conventional twilight times
cargo run --release -- solar --lat 21.4225 --lon 39.8262 --date 2024-03-20 --tz 3.0

# Run the MCRT engine directly across twilight SZA range
cargo run --release -- mcrt --lat 21.4225 --lon 39.8262 --sza-start 90 --sza-end 108
```


## Sample output

```
Twilight MCRT Prayer Time Calculator
====================================
Date:       2024-03-20
Location:   21.4225°N, 39.8262°E
Timezone:   UTC+3.0
Atmosphere: US Standard 1976 (clear sky)
Method:     Single-scatter MCRT + CIE mesopic vision

Done in 8 ms

Prayer Times (MCRT-derived):
  Sunrise:              06:24:31
  Sunset:               18:31:51

  Fajr (true dawn):     05:23:43  (SZA 104.97°, depression 14.97°)
  Isha (al-abyad):      19:30:43  (SZA 104.50°, depression 14.50°)
    └ Hanafi school — white twilight disappears
  Isha (al-ahmar):      19:32:08  (SZA 104.83°, depression 14.83°)
    └ Shafi'i/Maliki/Hanbali — red glow disappears

Comparison with conventional fixed-angle methods:
  Fajr 18° (MWL/ISNA)          05:10:36  (diff: +13.1 min)
  Fajr 15° (Egypt)             05:23:34  (diff: +0.1 min)
  Fajr 19.5° (Umm al-Qura)     05:04:06  (diff: +19.6 min)
  Isha 17° (MWL)               19:41:31  (diff: -9.4 min)
  Isha 17.5° (Egypt)           19:43:41  (diff: -11.5 min)
  Isha 18° (ISNA)              19:45:51  (diff: -13.7 min)
```

The ~15° depression angle the engine produces for clear sky matches the Egyptian General Authority convention. Reaching the 18° conventions requires multiple scattering, which is Phase 6+.


## How it works

The pipeline runs in five stages:

**1. Solar position (NREL SPA).** Computes topocentric solar zenith and azimuth to ±0.0003° for any date between -2000 and 6000 CE. Uses the full VSOP87 planetary theory with nutation corrections.

**2. Atmosphere construction.** Builds a 50-shell spherical atmosphere from 0--100 km using the US Standard 1976 temperature/pressure/density profiles. Computes Rayleigh scattering cross-sections via the Bodhaine et al. (1999) formula with the exact Lorentz-Lorenz factor and Bates (1984) King correction. Includes Serdyuchenko (2014) ozone absorption cross-sections.

**3. Spectral radiance computation.** For each solar zenith angle in the twilight range (90--108°), performs single-scattering line-of-sight integration along the observer's viewing ray. At each point along the ray, an analytical shadow ray computes the optical depth to the sun through each atmospheric shell. The Rayleigh phase function weights the scattered contribution. This runs at 41 wavelengths from 380--780 nm.

**4. Luminance and color classification.** Converts spectral radiance to CIE photopic, scotopic, and mesopic luminance using the standard V(lambda) and V'(lambda) curves. Computes the spectral centroid to classify twilight color: Blue (early), White (shafaq al-abyad), Orange (transition), Red (shafaq al-ahmar), Dark.

**5. Threshold crossing.** Finds the solar zenith angle where mesopic luminance drops below perceptual thresholds, using a two-pass adaptive scan (coarse 0.5° then fine 0.1°). Converts the threshold SZA back to clock time via binary search on the SPA.

The engine also detects persistent twilight at high latitudes in summer (e.g., London in June, Reykjavik in June).


## Architecture

```
twilight/
  crates/
    twilight-core/       #![no_std] — Vec3, ray-sphere, atmosphere model,
                         Rayleigh/HG phase functions, single-scatter integrator,
                         backward MC photon tracer, spectral cross-sections
    twilight-solar/      NREL SPA implementation (±0.0003° accuracy)
    twilight-data/       US Standard 1976 profiles, TSIS-1 solar spectrum,
                         Serdyuchenko O₃ cross-sections, atmosphere builder
    twilight-threshold/  CIE V(λ)/V'(λ) vision functions, photopic/scotopic/mesopic
                         luminance, spectral color classification, prayer time thresholds
    twilight-cpu/        Rayon parallel backend — simulation driver, end-to-end pipeline
    twilight-ffi/        C FFI for iOS/Android/Flutter
    twilight-cli/        Command-line interface (solar, mcrt, pray subcommands)
```

`twilight-core` is `#![no_std]` and `#![forbid(unsafe_code)]`. No heap allocation, no `String`, no `Vec`. Everything uses fixed-size arrays (`[f64; 64]`). This is non-negotiable for portability to WASM, embedded, and GPU backends.


## Physics

Scattering model:
- Rayleigh scattering with wavelength-dependent cross-sections (Bodhaine 1999)
- Exact Lorentz-Lorenz factor `((n²-1)/(n²+2))²` with Peck & Reeder (1972) refractive index
- Bates (1984) wavelength-dependent King correction factor
- Ozone absorption (Serdyuchenko 2014 cross-sections, Chappuis + Huggins bands)
- TSIS-1 Hybrid Solar Reference Spectrum for solar irradiance weighting
- Validated against published Rayleigh optical depths: τ(550nm)=0.097 (ref: 0.098), τ(400nm)=0.359 (ref: 0.36)

Vision model:
- CIE 1931 photopic V(λ) and CIE 1951 scotopic V'(λ)
- CIE 2010 mesopic luminance (adapts between rod and cone vision as sky darkens)
- Separate red-band (>600nm) and blue-band (<500nm) luminance tracking
- Spectral centroid for twilight color classification (blue → white → orange → red → dark)


## Test suite

362 tests across the workspace. Run them:

```
cargo test --workspace
```

Breakdown by crate:

| Crate | Tests | Coverage |
|-------|------:|----------|
| twilight-core | 127 | Vec3, ray-sphere, atmosphere, Rayleigh/HG, single-scatter, photon MC, spectral |
| twilight-data | 64 | US Std 1976 profiles, O₃ cross-sections, solar spectrum, builder |
| twilight-threshold | 72 | Vision functions, luminance, color classification, prayer thresholds |
| twilight-cpu | 52 | Simulation driver, pipeline, parallel tracer |
| twilight-solar | 47 | NREL SPA, Julian day, zenith crossing, declination, EoT, nutation |

All tests run in ~0.2 seconds on Apple Silicon.


## Roadmap

Completed:
- Phase 0: Solar position (NREL SPA) + project scaffold
- Phase 1: Atmosphere model (clear sky, US Standard 1976)
- Phase 2: Core MCRT (spherical geometry, single-scatter + Rayleigh + O₃)
- Phase 3: Spectral luminance + mesopic vision + threshold model

Next:
- Phase 4: Surface albedo contribution in single-scatter integrator
- Phase 5: Tropospheric aerosols (OPAC climatology)
- Phase 6: Cloud layers (1D, Henyey-Greenstein)
- Phase 7: Multiple scattering (backward MC with next-event estimation)
- Phase 8: Online weather data (Open-Meteo cloud/temperature/pressure)
- Phase 9: Satellite-derived cloud fields (MERRA-2, Sentinel)
- Phase 10: Light pollution (Falchi atlas)
- Phase 11: Terrain horizon masking
- Phase 12: GPU backend (wgpu/CUDA)
- Phase 13: Neural surrogate model for real-time inference
- Phase 14: Mobile SDKs (iOS/Android via FFI)
- Phase 15: Web interface (WASM)


## Performance

Full prayer time computation (two-pass adaptive scan, 41 wavelengths, 50 atmospheric shells):
- ~8 ms on Apple Silicon (release build)
- ~1--8 ms depending on SZA step resolution

The single-scatter integrator is deterministic (no Monte Carlo noise), so results are reproducible to the last bit.


## Accuracy tiers (planned)

| Tier | Data source | Target |
|------|-------------|--------|
| Offline | Embedded climatology (~500 KB) | ±5 min |
| Weather-aware | Open-Meteo cloud/snow/T/P API | ±2--3 min |
| Location-specific | MERRA-2 aerosols + Falchi light pollution | ±1--2 min |
| Real-time 3D clouds | Satellite imagery → ML → MCRT | ±0.5--1 min |
| Fused | Tier 3 + Tier 4 blended | Best available |

Currently only Tier 1 (offline, clear sky) is implemented.


## License

MIT OR Apache-2.0
