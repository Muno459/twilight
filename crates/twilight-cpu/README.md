# twilight-cpu

Rayon-based CPU backend. Ties together the physics (`twilight-core`), solar position (`twilight-solar`), atmospheric data (`twilight-data`), vision model (`twilight-threshold`), terrain masking (`twilight-terrain`), and light pollution (`twilight-skyglow`) into a complete prayer time computation pipeline.

## Modules

**`simulation`**. High-level simulation driver. `simulate_at_sza` computes spectral sky radiance at a single solar zenith angle by calling the single-scatter integrator across all 41 wavelength bands with optional TSIS-1 solar irradiance weighting. Dispatches to single-scatter, MC, or hybrid mode based on `ScatteringMode`. `simulate_twilight_scan` sweeps across a range of SZA values. `total_radiance` integrates the spectral result via trapezoidal rule.

**`pipeline`**. End-to-end prayer time computation. Given a `PrayerTimeInput` (location, date, timezone, albedo, step size, aerosol/cloud config, O3/NO2 overrides, terrain, skyglow), runs the full pipeline:

1. Build atmosphere model (Rayleigh + gas absorption + aerosol + cloud)
2. Find sunrise/sunset via SPA or DE440 zenith crossing
3. Check maximum SZA for persistent twilight detection
4. Coarse MCRT scan (0.5 degree steps) to locate threshold regions
5. Fine MCRT scan (0.1 degree steps) around each crossing
6. Combine and deduplicate spectral results
7. Analyze twilight luminance and color at each SZA
8. Find Fajr, Isha al-abyad, and Isha al-ahmar threshold crossings
9. Convert threshold SZA to clock time via binary search

Supports terrain-adjusted effective SZA, skyglow brightness injection, and gas composition overrides from live weather. Returns `PrayerTimeOutput` with prayer times, depression angles, persistent twilight flag, and full diagnostic data.

**`tracer`**. Rayon parallel wrapper around the backward MC photon tracer in `twilight-core`. Distributes photons across threads with deterministic per-photon seeding.

**`gpu_dispatch`**. Converts atmosphere models and simulation parameters into GPU-ready packed buffers for `twilight-gpu`. Handles the CPU/GPU dispatch decision.

## Performance

~30 ms for full prayer time computation on Apple Silicon (release build, with gas absorption). The two-pass adaptive scan typically evaluates 40-60 SZA points, each computing 41 wavelength channels through 50 atmospheric shells.

## Tests

82 tests. Simulation config defaults, spectral result properties, twilight scan ordering, trapezoidal integration, pipeline end-to-end results (Mecca equinox, London winter), persistent twilight detection, depression angle validation, time formatting, parallel tracer determinism/convergence, GPU dispatch buffer conversion.
