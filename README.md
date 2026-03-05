<p align="center">
  <img src="assets/banner.svg" alt="twilight — Monte Carlo Radiative Transfer engine" width="100%"/>
</p>

<p align="center">
  <a href="#quick-start"><code>quick start</code></a>&ensp;·&ensp;<a href="#how-it-works"><code>how it works</code></a>&ensp;·&ensp;<a href="#architecture"><code>architecture</code></a>&ensp;·&ensp;<a href="#physics"><code>physics</code></a>&ensp;·&ensp;<a href="#roadmap"><code>roadmap</code></a>
</p>

<br/>

Every Islamic prayer time app on the planet uses the same method: pick a solar depression angle (15, 18, 19.5 degrees depending on convention), and call it a day.

The disagreement between conventions exists because no single angle is universally correct. The actual onset of dawn and disappearance of twilight depend on atmospheric conditions --- Rayleigh scattering, ozone absorption, aerosols, clouds, surface albedo, light pollution, altitude --- none of which a fixed angle captures.

`twilight` replaces the angle with the physics. It simulates photon transport through a spherical atmosphere during twilight and determines when sky luminance crosses perceptual thresholds, producing prayer times from first principles.

The engine currently models 41-band spectral resolution (380--780 nm) through a 50-shell spherical atmosphere based on the US Standard 1976 profile. Full prayer time computation runs in **8 ms** on consumer hardware.

<br/>

## Quick start

```bash
cargo build --release

# Compute prayer times for Mecca, spring equinox
cargo run --release -- pray --lat 21.4225 --lon 39.8262 --date 2024-03-20 --tz 3.0

# Solar position and conventional twilight times
cargo run --release -- solar --lat 21.4225 --lon 39.8262 --date 2024-03-20 --tz 3.0

# Run the MCRT engine directly across twilight SZA range
cargo run --release -- mcrt --lat 21.4225 --lon 39.8262 --sza-start 90 --sza-end 108
```

<br/>

## Sample output

<table><tr><td>

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

  Fajr (true dawn):     05:23:43  (depression 14.97°)
  Isha (al-abyad):      19:30:43  (depression 14.50°)
    └ Hanafi — white twilight disappears
  Isha (al-ahmar):      19:32:08  (depression 14.83°)
    └ Shafi'i/Maliki/Hanbali — red glow disappears

Comparison with conventional fixed-angle methods:
  Fajr 18° (MWL/ISNA)       05:10:36  (diff: +13.1 min)
  Fajr 15° (Egypt)           05:23:34  (diff: +0.1 min)
  Fajr 19.5° (Umm al-Qura)  05:04:06  (diff: +19.6 min)
  Isha 17° (MWL)             19:41:31  (diff: -9.4 min)
  Isha 17.5° (Egypt)         19:43:41  (diff: -11.5 min)
  Isha 18° (ISNA)            19:45:51  (diff: -13.7 min)
```

</td></tr></table>

The ~15 degree depression angle the engine produces for clear sky matches the Egyptian General Authority convention. The 18 degree conventions require multiple scattering (Phase 7).

<br/>

## How it works

The pipeline runs in five stages.

**Solar position** --- NREL SPA computes topocentric solar zenith and azimuth to ±0.0003 degrees for any date between -2000 and 6000 CE. Full VSOP87 planetary theory with nutation corrections.

**Atmosphere construction** --- Builds a 50-shell spherical atmosphere from 0--100 km using US Standard 1976 temperature/pressure/density profiles. Rayleigh cross-sections via Bodhaine et al. (1999) with exact Lorentz-Lorenz factor. Serdyuchenko (2014) ozone absorption.

**Spectral radiance** --- For each solar zenith angle in the twilight range (90--108 degrees), single-scattering line-of-sight integration along the observer's viewing ray. Analytical shadow ray computes optical depth to the sun through each shell. Rayleigh phase function weights the scattered contribution. 41 wavelengths, 380--780 nm.

**Luminance and color** --- Converts spectral radiance to CIE photopic, scotopic, and mesopic luminance. Spectral centroid classifies twilight color through the progression: blue, white (*shafaq al-abyad*), orange, red (*shafaq al-ahmar*), dark.

**Threshold crossing** --- Two-pass adaptive scan (coarse 0.5 degrees, fine 0.1 degrees) finds the SZA where mesopic luminance drops below perceptual thresholds. Binary search on the SPA converts threshold SZA back to clock time.

The engine detects persistent twilight at high latitudes in summer.

<br/>

## Architecture

```
twilight/
├── twilight-core          #![no_std] #![forbid(unsafe_code)]
│   ├── geometry.rs        Vec3, ray-sphere intersection, ECEF, solar direction
│   ├── atmosphere.rs      50-shell spherical atmosphere model
│   ├── spectrum.rs        Rayleigh cross-sections (Bodhaine 1999)
│   ├── scattering.rs      Rayleigh + Henyey-Greenstein phase functions
│   ├── single_scatter.rs  Line-of-sight integrator with analytical shadow ray
│   └── photon.rs          Backward MC photon tracer, xorshift RNG
│
├── twilight-solar         NREL SPA (±0.0003°), VSOP87, nutation, zenith crossing
├── twilight-data          US Std 1976, TSIS-1 solar spectrum, O₃ Serdyuchenko 2014
├── twilight-threshold     CIE V(λ)/V'(λ), mesopic luminance, color classification
├── twilight-cpu           Rayon parallel backend, simulation driver, full pipeline
├── twilight-ffi           C FFI for iOS / Android / Flutter
└── twilight-cli           Command-line interface (solar, mcrt, pray)
```

`twilight-core` has zero heap allocation. No `String`, no `Vec`, no `std`. Fixed-size arrays only (`[f64; 64]`). This is non-negotiable for portability to WASM, embedded, and GPU backends.

<br/>

## Physics

**Scattering**

Rayleigh scattering with wavelength-dependent cross-sections per Bodhaine et al. (1999). Exact Lorentz-Lorenz factor `((n²-1)/(n²+2))²` with Peck & Reeder (1972) refractive index and Bates (1984) King correction. Ozone absorption from Serdyuchenko (2014), covering both Chappuis and Huggins bands. Solar irradiance weighting from the TSIS-1 Hybrid Solar Reference Spectrum.

Validated against published Rayleigh optical depths:

```
τ(550nm) = 0.097    reference: 0.098
τ(400nm) = 0.359    reference: 0.36
```

**Vision**

CIE 1931 photopic V(λ) and CIE 1951 scotopic V'(λ). Mesopic luminance per CIE 2010 adapts between rod and cone vision as the sky darkens through twilight. Separate red-band (>600 nm) and blue-band (<500 nm) luminance tracking. Spectral centroid drives color classification through the twilight progression.

<br/>

## Tests

362 tests. 0.2 seconds on Apple Silicon.

```bash
cargo test --workspace
```

```
twilight-core        127 tests    geometry, atmosphere, Rayleigh, phase functions,
                                  single-scatter, photon MC, spectral cross-sections

twilight-threshold    72 tests    vision curves, luminance, color classification,
                                  prayer time thresholds

twilight-data         64 tests    US Std 1976 profiles, ozone, solar spectrum, builder

twilight-cpu          52 tests    simulation driver, pipeline, parallel tracer

twilight-solar        47 tests    NREL SPA, Julian day, declination, EoT, nutation,
                                  zenith crossing, input validation
```

<br/>

## Roadmap

```
done    Phase 0    Solar position (NREL SPA) + project scaffold
done    Phase 1    Atmosphere model (clear sky, US Standard 1976)
done    Phase 2    Core MCRT engine (spherical, single-scatter, Rayleigh, O₃)
done    Phase 3    Spectral luminance + mesopic vision + threshold model
─────────────────────────────────────────────────────────────────────
next    Phase 4    Surface albedo in single-scatter integrator
        Phase 5    Tropospheric aerosols (OPAC climatology)
        Phase 6    Cloud layers (1D, Henyey-Greenstein)
        Phase 7    Multiple scattering (backward MC, next-event estimation)
        Phase 8    Online weather data (Open-Meteo API)
        Phase 9    Satellite cloud fields (MERRA-2, Sentinel)
        Phase 10   Light pollution (Falchi 2016 atlas)
        Phase 11   Terrain horizon masking
        Phase 12   GPU backend (wgpu / CUDA)
        Phase 13   Neural surrogate for real-time inference
        Phase 14   Mobile SDKs (iOS/Android via C FFI)
        Phase 15   Web interface (WASM)
```

<br/>

## Accuracy tiers (planned)

```
Tier 1   Offline             Embedded climatology (~500 KB)           ±5 min
Tier 2   Weather-aware       Open-Meteo cloud/snow/T/P               ±2-3 min
Tier 3   Location-specific   MERRA-2 aerosols + Falchi LP            ±1-2 min
Tier 4   Real-time clouds    Satellite imagery → ML → MCRT           ±0.5-1 min
Tier 5   Fused               Tier 3 + Tier 4 blended                 best available
```

Currently Tier 1 only.

<br/>

## Performance

Full prayer time computation (two-pass adaptive scan, 41 wavelengths, 50 shells):

```
Apple Silicon (release)     ~8 ms
```

The single-scatter integrator is deterministic. No Monte Carlo noise. Results are reproducible to the last bit.

<br/>

## License

MIT OR Apache-2.0
