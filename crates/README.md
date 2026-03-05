# crates/

The twilight workspace is split into seven crates. Dependencies flow strictly downward in this diagram:

```
twilight-cli         (binary, user-facing)
    |
twilight-cpu         (rayon parallelism, pipeline orchestration)
    |
    +--- twilight-threshold   (CIE vision model, luminance, prayer time logic)
    |
    +--- twilight-data        (embedded atmosphere profiles, ozone, solar spectrum, aerosols, builder)
    |        |
    |        +--- twilight-core
    |
    +--- twilight-solar       (NREL SPA, zenith crossing search)
    |
twilight-core        (no_std physics kernel: geometry, scattering, atmosphere, single-scatter engine)
    |
twilight-ffi         (C FFI for iOS/Android/Flutter)
    |
    +--- twilight-solar
```

## Crate summary

| Crate | Purpose | `no_std` | Tests |
|---|---|---|---|
| `twilight-core` | Physics kernel. Spherical geometry, Rayleigh/HG phase functions, atmosphere model, single-scatter LOS integrator, MC photon tracer. All fixed-size arrays, zero heap allocation. | yes | 135 |
| `twilight-solar` | NREL Solar Position Algorithm (Reda & Andreas 2003). VSOP87 + 63-term nutation. Zenith, azimuth, declination, EoT. Binary search zenith crossing for time conversion. | no | 47 |
| `twilight-data` | Embedded data: US Std 1976 atmosphere profiles, ozone cross-sections (Serdyuchenko 2014), TSIS-1 solar spectrum, OPAC aerosol climatology (6 types). Atmosphere builder that assembles shell optics from all of these. | no | 117 |
| `twilight-threshold` | CIE photopic/scotopic/mesopic vision model. Spectral luminance integration. Twilight color classification (blue, white, orange, red, dark). Fajr/Isha threshold crossing detection. | no | 72 |
| `twilight-cpu` | CPU backend. Simulation driver, two-pass adaptive prayer time pipeline, rayon parallel MC tracer. | no | 52 |
| `twilight-cli` | CLI binary. `solar`, `mcrt`, and `pray` subcommands. | no | 0 |
| `twilight-ffi` | C-compatible FFI. `twilight_solar_zenith` exported as `cdylib`/`staticlib`. | no | 0 |

Two additional crate directories exist as placeholders for future work:

- `twilight-clouds` -- ML cloud model (Phase 6+)
- `twilight-gpu` -- wgpu/CUDA compute backend (Phase 8+)

## Build

From the workspace root:

```bash
cargo test --workspace          # run all 423 tests
cargo build --release           # build everything
cargo run --release -p twilight-cli -- pray --lat 21.4225 --lon 39.8262 --date 2024-03-20 --tz 3.0
```
