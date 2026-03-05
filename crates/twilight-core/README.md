# twilight-core

`#![no_std]` `#![forbid(unsafe_code)]`

The physics kernel. Everything in this crate compiles without the standard library and uses zero heap allocation. All data structures use fixed-size arrays (`[f64; 64]`, `[Shell; 64]`). This makes the code portable to WASM, embedded targets, and future GPU backends (WGSL, CUDA PTX).

## Modules

**`geometry`**. `Vec3` type with dot/cross/normalize, ray-sphere intersection (quadratic solver), shell boundary crossing, geographic-to-ECEF coordinate transforms, and solar direction computation from zenith angle and azimuth.

**`atmosphere`**. `AtmosphereModel` representing the atmosphere as concentric spherical shells around the Earth. Each shell stores optical properties (extinction, single-scattering albedo, asymmetry parameter, Rayleigh fraction) at each wavelength. Shell indexing by radius, optical depth computation.

**`spectrum`**. Rayleigh scattering cross-section following Bodhaine et al. (1999). Peck & Reeder (1972) refractive index of air, exact Lorentz-Lorenz factor `((n²-1)/(n²+2))²`, Bates (1984) King correction. Scattering coefficient computation from cross-section and number density.

**`scattering`**. Rayleigh phase function `P(μ) = 3/(16π)(1+μ²)` and Henyey-Greenstein phase function for aerosol/cloud forward scattering. CDF inversion sampling for both. `scatter_direction` rotates a photon direction by a sampled scattering angle around an arbitrary axis.

**`single_scatter`**. Deterministic single-scattering line-of-sight integrator. Marches along the observer's viewing ray, and at each sample point computes an analytical shadow ray to the sun through each atmospheric shell. Returns spectral radiance as a 64-element array. This is the primary engine. no Monte Carlo noise.

**`photon`**. Backward Monte Carlo photon tracer with next-event estimation. Traces photons from observer into the atmosphere, sampling free paths from Beer-Lambert, scattering via phase function sampling, and accumulating direct solar contribution at each scatter event. Russian roulette termination. Includes a minimal xorshift64 RNG for `no_std` compatibility.

## Constraints

- No `std` imports. Uses `libm` for transcendental functions.
- No `unsafe`. `#![forbid(unsafe_code)]` is set at crate level.
- No `Vec`, `String`, `Box`, or any heap allocation.
- All arrays are `MAX_SHELLS=64` and `MAX_WAVELENGTHS=64`.

## Tests

127 tests covering geometry, atmosphere construction, Rayleigh cross-sections (validated against published optical depths), phase function normalization, single-scatter radiance properties, and MC photon tracer behavior.
