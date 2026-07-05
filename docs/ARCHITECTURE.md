# Architecture: the simulator and the application

`twilight` is a general Monte Carlo radiative-transfer simulator of the
twilight sky with one flagship application (Quranic dawn/dusk visibility
timing) layered on top. The two layers are separated by the crate graph,
and the separation is enforced by what each crate is allowed to depend on.

## Layer 1: the simulator

Pure physics. Given an atmosphere, an optional 3D cloud field, a geometry
(observer, view direction, sun direction), and a wavelength grid, compute
absolute spectral sky radiance.

| Crate | Role | Constraints |
|---|---|---|
| `twilight-core` | The transport kernel: photon chains, forced flights, 3D cloud DDA traversal, the estimator stack, the importance machinery. | `#![no_std]`, `#![forbid(unsafe_code)]`, only dependency is `libm`. Builds for `wasm32-unknown-unknown` and embedded targets. |
| `twilight-data` | Atmosphere construction: USSA-76 + thermosphere shells to 150 km, prebaked molecular cross-sections (HITRAN/Serdyuchenko), OPAC aerosol and cloud optics, 3D cloud field building. | Data tables are embedded and regenerable (`tools/gen_*.py`). |
| `twilight-solar` | Solar/lunar state: NREL SPA, a pure-Rust JPL DE440 SPK reader, earth rotation. | No network. |
| `twilight-gpu` | Metal and portable wgpu (Vulkan/DX12/Metal) ports of the full estimator. | Every kernel is parity-gated against the CPU implementation; buffer layouts are pinned by offset parse tests; the two shader languages are kept semantically twinned. |
| `twilight-threshold` | Human vision: CIE photopic/scotopic/mesopic luminance, contrast thresholds (Blackwell/TVI), night-sky background from measured tables. | Measured tables over formulas, always. |

The simulator answers general questions with no prayer-time code involved:
twilight brightness decay curves, zodiacal light visibility, skyglow
propagation, high-latitude persistent twilight, 3D cloud shadowing.

## Layer 2: the application

Given the simulator, model a dark-adapted observer watching the horizon
band and decide WHEN the Quranic events occur, using measured inputs for
tonight's actual sky.

| Crate | Role |
|---|---|
| `twilight-cpu` | The pipeline: K-seed depression scans, the khayt patch fan (5 band + 2 reference patches, per-side sun azimuths), crossing solvers with adaptive refinement, uncertainty propagation (MC + skyglow calibration + duty cycle + aerosol input, RSS-folded and reported per event). |
| `twilight-weather` | Live inputs: Open-Meteo forecast at the prayer hour, CAMS aerosol optical depth (as excess over a measured baseline), NASA GIBS cloud products, the cloud3d neural 3D reconstruction, NOAA F10.7. Typed errors, caches, no fabricated defaults. |
| `twilight-skyglow` | Artificial light: Lorenz 2024 atlas, VIIRS Black Marble, Garstang propagation, source-spectrum-aware mesopic veil. |
| `twilight-terrain` | Horizon profiles from the Copernicus GLO-30 DEM. |
| `twilight-cli` | The user-facing binary (`pray`, `mcrt`, `solar`, `sqm` subcommands). |
| `twilight-ffi` | Minimal C FFI for mobile embedding. |

## The dependency rule

Application crates depend on simulator crates; never the reverse.
`twilight-core` knows nothing about prayer, weather, or files; it cannot
even allocate beyond `alloc`. This is what makes the simulator citable and
reusable on its own, and what makes the application auditable: every
religious-criterion decision lives in `twilight-cpu`'s khayt module and is
regression-pinned by synthetic gates with analytic ground truth.

## Verification structure

Three concentric rings, each documented in `validation/`:

1. **Internal exactness**: unit gates with analytic truths (forced-flight
   first-collision law, majorant invariance, telescoping normalization,
   DDA vs analytic slab tau), bit-identity harnesses across refactors,
   and four-way cross-implementation parity (CPU scalar, CPU polarized,
   Metal, WGSL) with statistical bands derived from measured seed CVs.
2. **External referees**: DISORT (absolute scale), MYSTIC (spherical MC to
   SZA 103 at 1e9 photons), SHDOM (true-3D cubes), measured twilight sky
   decay (Patat, Koomen).
3. **Human ground truth**: sixteen published dawn-observation campaigns,
   the OpenFajr Birmingham panel, and a per-site inversion program showing
   the criterion's one calibrated constant is cross-site invariant.

The adversarial review process (independent re-derivation of estimator
math, cancelling-error hunts, gate-vacuousness audits) is part of the
methodology: findings and their fixes are recorded in commit messages and
the RESULTS documents.
