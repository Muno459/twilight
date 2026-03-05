# twilight-cpu

Rayon-based CPU backend. Ties together the physics (`twilight-core`), solar position (`twilight-solar`), atmospheric data (`twilight-data`), and vision model (`twilight-threshold`) into a complete prayer time computation pipeline.

## Modules

**`simulation`**. High-level simulation driver. `simulate_at_sza` computes spectral sky radiance at a single solar zenith angle by calling the single-scatter integrator across all 41 wavelength bands with optional TSIS-1 solar irradiance weighting. `simulate_twilight_scan` sweeps across a range of SZA values. `total_radiance` integrates the spectral result via trapezoidal rule.

**`pipeline`**. End-to-end prayer time computation. Given a `PrayerTimeInput` (latitude, longitude, date, timezone, albedo, step size), runs the full pipeline:

1. Build atmosphere model
2. Find sunrise/sunset via SPA zenith crossing
3. Check maximum SZA for persistent twilight detection
4. Coarse MCRT scan (0.5 degree steps) to locate threshold regions
5. Fine MCRT scan (0.1 degree steps) around each crossing
6. Combine and deduplicate spectral results
7. Analyze twilight luminance and color at each SZA
8. Find Fajr, Isha al-abyad, and Isha al-ahmar threshold crossings
9. Convert threshold SZA to clock time via SPA binary search

Returns `PrayerTimeOutput` with prayer times, depression angles, persistent twilight flag, and full diagnostic data (spectral results + twilight analyses).

**`tracer`**. Rayon parallel wrapper around the backward MC photon tracer in `twilight-core`. Distributes photons across threads with deterministic per-photon seeding. Currently used for validation; the primary engine is the deterministic single-scatter integrator.

## Performance

~8 ms for full prayer time computation on Apple Silicon (release build). The two-pass adaptive scan typically evaluates 40–60 SZA points, each computing 41 wavelength channels through 50 atmospheric shells.

## Tests

52 tests covering simulation config defaults, spectral result properties, twilight scan ordering, trapezoidal integration, pipeline end-to-end results (Mecca equinox, London winter), persistent twilight detection, time formatting, and parallel tracer determinism/convergence.
