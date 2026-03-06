# twilight-core

`#![no_std]` `#![forbid(unsafe_code)]`

The physics kernel. Everything in this crate compiles without the standard library and uses zero heap allocation. All data structures use fixed-size arrays (`[f64; 64]`, `[Shell; 64]`). This makes the code portable to WASM, embedded targets, and GPU shader translation.

## Modules

**`geometry`**. `Vec3` type with dot/cross/normalize, ray-sphere intersection (quadratic solver), shell boundary crossing, geographic-to-ECEF coordinate transforms, solar direction computation from zenith angle and azimuth, and Snell's law refraction at shell boundaries for true ray curvature through the atmosphere.

**`atmosphere`**. `AtmosphereModel` representing the atmosphere as concentric spherical shells around the Earth. Each shell stores optical properties (extinction, single-scattering albedo, asymmetry parameter, Rayleigh fraction) at each wavelength, plus a refractive index for atmospheric refraction. Shell indexing by radius, optical depth computation. `MAX_SHELLS=64`, `MAX_WAVELENGTHS=64`.

**`spectrum`**. Rayleigh scattering cross-section following Bodhaine et al. (1999). Peck & Reeder (1972) refractive index of air, exact Lorentz-Lorenz factor `((n^2-1)/(n^2+2))^2`, Bates (1984) King correction. Scattering coefficient computation from cross-section and number density.

**`scattering`**. Rayleigh phase function `P(mu) = 3/(16pi)(1+mu^2)` and Henyey-Greenstein phase function for aerosol/cloud forward scattering. CDF inversion sampling for both. `scatter_direction` rotates a photon direction by a sampled scattering angle. Full Stokes vector type, Mueller matrix type, Rayleigh and HG Mueller matrices, rotation Mueller matrices for the scattering plane. Supports both scalar and polarized RT modes.

**`gas_absorption`**. Molecular gas absorption for five species: O3 (Serdyuchenko 2014, 11-temperature grid), O2 (HITRAN line-by-line with Voigt profiles via Humlicek/Laplace continued fraction, A-band at 762nm), H2O (HITRAN LBL Voigt, 720nm region), NO2 (HITRAN XSC at 220K and 294K with temperature interpolation), O4 CIA (HITRAN 2024 collision-induced absorption). O2 and H2O use bilinear (pressure, temperature) interpolation on 3D grids. `GasProfile` and `ShellGas` types for per-shell gas densities. `apply_gas_absorption` adds absorption extinction to existing shell optics while preserving the scattering coefficient. O3 column scaling in Dobson Units.

**`gas_absorption_data`**. Auto-generated data tables (~364 KB). O3 cross-sections at 11 temperatures, O2/H2O cross-sections on [5 pressures][4 temperatures][401 wavelengths] grids from real HITRAN data, NO2 XSC at two temperatures, O4 CIA coefficients, O2/H2O spectral line parameters for Voigt LBL, US Standard Atmosphere reference profiles.

**`single_scatter`**. Deterministic single-scattering line-of-sight integrator. Marches along the observer's viewing ray with optional Snell's law refraction at each shell boundary. At each sample point, computes an analytical shadow ray to the sun through each atmospheric shell. Returns spectral radiance as a 64-element array. No Monte Carlo noise.

**`photon`**. Backward Monte Carlo photon tracer with next-event estimation. Traces photons from observer into the atmosphere, sampling free paths from Beer-Lambert, scattering via phase function sampling, and accumulating direct solar contribution at each scatter event. Russian roulette termination. Hybrid mode: exact single-scatter (order 1) + MC secondary chains (orders 2+) with upward-biased importance sampling for deep twilight connectivity. Optional full Stokes vector tracking with Mueller matrix scattering. Minimal xorshift64 RNG for `no_std` compatibility.

## Constraints

- No `std` imports. Uses `libm` for transcendental functions.
- No `unsafe`. `#![forbid(unsafe_code)]` is set at crate level.
- No `Vec`, `String`, `Box`, or any heap allocation.
- All arrays are `MAX_SHELLS=64` and `MAX_WAVELENGTHS=64`.

## Tests

311 tests covering geometry (Vec3 operations, ray-sphere intersection, Snell's law refraction, ECEF conversion), atmosphere construction, Rayleigh cross-sections (validated against published optical depths), phase function normalization, Mueller matrix algebra, Stokes vector operations, gas absorption (74 tests: cross-section values at reference conditions, temperature interpolation, Voigt profiles, O3 column scaling, vertical optical depths, apply_gas_absorption preservation properties), single-scatter radiance, ground reflection, MC photon tracer, and hybrid multi-scatter engine.
