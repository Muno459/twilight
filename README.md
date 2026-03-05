<p align="center">
  <img src="assets/banner.svg" alt="twilight" width="100%"/>
</p>

<p align="center">
  <a href="https://github.com/Muno459/twilight/actions"><img src="https://img.shields.io/badge/tests-362_passing-brightgreen?style=flat-square" alt="tests"/></a>
  <a href="https://github.com/Muno459/twilight"><img src="https://img.shields.io/badge/rust-pure_%23!%5Bno__std%5D_core-orange?style=flat-square" alt="rust"/></a>
  <a href="#license"><img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-blue?style=flat-square" alt="license"/></a>
</p>

<br/>

`twilight` computes Fajr and Isha prayer times by simulating how sunlight actually scatters through the atmosphere during twilight. No lookup tables, no fixed depression angles. Photons in, prayer times out.

The engine traces sunlight through a 50-shell spherical atmosphere at 41 wavelengths (380–780 nm), accounting for Rayleigh scattering and ozone absorption, then determines when the sky becomes too dim to see using CIE mesopic vision models. The whole thing runs in 8 ms.

## Why

Every prayer time app uses the same trick from the 1970s: hardcode a solar depression angle and call it done. The Muslim World League says 18°. The Egyptian authority says 15°. Umm al-Qura says 19.5°. They disagree because the question "when does twilight end?" doesn't have a single answer. It depends on what's in the atmosphere.

At 21°N in clear desert air, twilight ends earlier than in humid coastal air with aerosols. A fresh snowfall brightens the sky. Volcanic ash darkens it. The fixed-angle approach can't see any of this. We thought: what if you just simulated the light?

```
Mecca, March equinox — clear sky, single-scatter MCRT:

  Fajr (true dawn):     05:23:43   depression 14.97°
  Isha (al-abyad):      19:30:43   depression 14.50°
  Isha (al-ahmar):      19:32:08   depression 14.83°

  vs. Fajr 18° (MWL):   05:10:36   diff: +13.1 min
  vs. Fajr 15° (Egypt):  05:23:34   diff: +0.1 min
  vs. Isha 18° (ISNA):   19:45:51   diff: -13.7 min
```

The ~15° result under clear sky matches the Egyptian convention almost exactly. The 18° conventions implicitly assume multiple scattering and aerosol contributions that we haven't added yet (Phase 7).


## Quick start

```bash
cargo build --release

# Full prayer time computation
cargo run --release -- pray --lat 21.4225 --lon 39.8262 --date 2024-03-20 --tz 3.0

# Solar position + conventional times for comparison
cargo run --release -- solar --lat 21.4225 --lon 39.8262 --date 2024-03-20 --tz 3.0

# Raw MCRT spectral output across twilight
cargo run --release -- mcrt --lat 21.4225 --lon 39.8262 --sza-start 90 --sza-end 108
```


## What actually happens when you run `pray`

1. NREL SPA computes the sun's position to ±0.0003° using the full VSOP87 planetary theory. Finds sunrise and sunset by binary-searching for the 90.8333° zenith crossing. Checks if persistent twilight occurs (high latitudes in summer where the sun never dips far enough).

2. The atmosphere builder constructs 50 concentric spherical shells from sea level to 100 km using US Standard 1976 profiles. Each shell gets Rayleigh scattering coefficients from Bodhaine et al. (1999) with the exact Lorentz-Lorenz factor, plus ozone absorption from Serdyuchenko (2014). All 41 wavelength bands, 380–780 nm.

3. For each solar zenith angle from 90° to 108°, the single-scatter integrator marches along the observer's line of sight (viewing 85° from zenith, toward the horizon). At each sample point, an analytical shadow ray computes the optical depth to the sun through every shell it crosses. The Rayleigh phase function weights how much light scatters toward the observer. This produces a 41-element spectral radiance vector at each SZA.

4. The spectral radiance gets converted to luminance through CIE vision curves. During twilight your eyes transition from cone vision (photopic, peaks at 555 nm) to rod vision (scotopic, peaks at 507 nm). The mesopic model blends both. The spectral centroid classifies the twilight color: blue → white (*shafaq al-abyad*) → orange → red (*shafaq al-ahmar*) → dark.

5. A two-pass adaptive scan (coarse 0.5°, then fine 0.1° around crossings) finds where luminance drops below perceptual thresholds. Fajr = mesopic luminance threshold. Isha al-abyad = same threshold, evening side. Isha al-ahmar = red-band luminance threshold. The threshold SZA gets converted back to clock time through another SPA binary search.

The entire pipeline is deterministic. No Monte Carlo noise. Same input, same output, bit-for-bit.


## Crates

```
twilight-core          the physics kernel. #![no_std], #![forbid(unsafe_code)],
                       zero heap allocation. Vec3, ray-sphere, atmosphere model,
                       Rayleigh + HG phase functions, single-scatter integrator,
                       backward MC photon tracer with next-event estimation.

twilight-solar         NREL SPA. VSOP87, 63-term nutation, topocentric parallax.
                       ±0.0003° for years -2000 to 6000.

twilight-data          embedded data. US Std 1976 profiles, TSIS-1 solar spectrum,
                       Serdyuchenko O₃ cross-sections, atmosphere builder.

twilight-threshold     CIE V(λ)/V'(λ), mesopic luminance, spectral centroid,
                       twilight color classification, prayer time thresholds.

twilight-cpu           rayon parallel backend. simulation driver, two-pass
                       adaptive pipeline, parallel MC tracer.

twilight-ffi           C FFI. builds as cdylib + staticlib for iOS/Android/Flutter.

twilight-cli           CLI with three subcommands: solar, mcrt, pray.
```

The `no_std` constraint on `twilight-core` is the important one. No `Vec`, no `String`, no `Box`. Everything is `[f64; 64]`. This is what makes it portable to WASM, GPU compute shaders, and embedded targets. The same physics code will run on a phone, in a browser, and eventually on a GPU without any rewrite.


## Validation

Rayleigh optical depths match published values:
```
λ = 550 nm    τ = 0.097    ref: 0.098    (Bucholtz 1995, Bodhaine 1999)
λ = 400 nm    τ = 0.359    ref: 0.36
```

SPA validated against the NREL reference case (Oct 17, 2003, Golden CO): zenith 50.111° (ref: 50.112°), azimuth 194.340° (ref: 194.340°).


## What's missing (honest version)

This is a single-scattering, clear-sky, sea-level engine. That's a solid foundation but it has known limitations:

- **No multiple scattering.** Real twilight has photons that scatter 2, 3, 10 times. This matters most at deep twilight (SZA > 100°) where the single-scatter approximation underestimates sky brightness. This is why we get ~15° instead of 18°.
- **No aerosols.** Dust, smoke, pollution, and sea salt all scatter and absorb. Saharan dust makes twilight redder and longer. Urban haze brightens the sky.
- **No clouds.** A cloud layer at 5 km altitude changes everything.
- **No terrain.** Mountains mask the horizon. Snow-covered ground reflects light back up.
- **No light pollution.** City glow washes out astronomical twilight.
- **One atmosphere profile.** We use US Standard 1976 everywhere. A tropical atmosphere has a different temperature/density structure than a subarctic one.

Each of these is a planned phase. The architecture supports all of them. The `AtmosphereModel` already has cloud layer insertion, aerosol asymmetry parameters, and surface albedo per wavelength. We just haven't populated them with real data yet.


## Tests

362 tests, 0.2 seconds.

```bash
cargo test --workspace
```

```
twilight-core        127    ray-sphere intersection, Rayleigh cross-sections validated
                            against Bodhaine 1999, phase function normalization integrals,
                            single-scatter radiance properties, MC photon tracer bounds,
                            xorshift RNG uniformity (chi-squared)

twilight-threshold    72    CIE V(λ) peak/range/symmetry, Purkinje shift, mesopic
                            blending, spectral centroid, color classification,
                            threshold crossing interpolation

twilight-data         64    US Std 1976 reference values (sea level T/P/n, tropopause,
                            lapse rate), O₃ Chappuis peak, solar spectrum integration,
                            builder shell contiguity, cloud layer effects

twilight-cpu          52    spectral result properties, twilight scan monotonicity,
                            trapezoidal integration, end-to-end Mecca/London results,
                            persistent twilight detection, parallel tracer determinism

twilight-solar        47    NREL reference case, Julian day edge cases, solstice
                            declinations, perihelion/aphelion distances, EoT bounds,
                            midnight sun, polar night, input validation
```


## Roadmap

What's done, what's next. Checked boxes are in this repo with tests.

- [x] NREL SPA solar position
- [x] 50-shell spherical atmosphere (US Standard 1976)
- [x] Single-scatter Rayleigh + O₃ MCRT engine
- [x] CIE mesopic vision model + spectral color classification
- [x] Two-pass adaptive prayer time pipeline (8 ms)
- [x] C FFI for mobile targets
- [ ] Ground reflection (surface albedo in the integrator)
- [ ] Tropospheric aerosols (OPAC climatology)
- [ ] Cloud layers (Henyey-Greenstein, 1D)
- [ ] Multiple scattering (backward MC with next-event estimation)
- [ ] Real-time weather (Open-Meteo API)
- [ ] Satellite cloud fields (MERRA-2, Sentinel)
- [ ] Light pollution (Falchi 2016 atlas)
- [ ] Terrain masking
- [ ] GPU backend (wgpu/CUDA)
- [ ] Neural surrogate for <1 ms inference
- [ ] iOS and Android SDKs
- [ ] WASM web interface


## License

MIT OR Apache-2.0
