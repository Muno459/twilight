# twilight-gpu

GPU compute backend for the MCRT engine. One backend: **Metal** (Apple GPUs,
macOS/iOS), behind the `metal` Cargo feature. Host code via `objc2-metal`;
hand-written shaders in `shaders/twilight.metal`, compiled at runtime.

Former Vulkan/CUDA/wgpu host modules were deleted: their shader sources were
never written, so those features could not compile. Re-adding a backend means
writing its shaders, a CI build for the feature, and parity tests that
actually execute.

## Modules

**`buffers`**. Packed f32 atmosphere representation for GPU upload.
`PackedAtmosphere` serializes the `AtmosphereModel` (shell geometry, optics at
each wavelength, surface albedo, wavelength grid) into a flat f32 buffer with
a magic/version header. `DispatchParams` encodes observer position, sun
direction, view direction, photon count, and RNG seed.

**`oracle`** (test-only). CPU reference implementation that generates test
cases for GPU validation: ray-sphere intersection, RNG sequences, phase
functions (Rayleigh + HG), shadow ray transmittance, single-scatter radiance,
and full spectral sweeps. The Metal backend must match within f32 tolerance.

**`tests`**. Metal-vs-CPU-oracle parity tests plus buffer roundtrip tests.
Note: GPU integration tests skip silently when no Metal device is present
(CI without GPU runs them as no-ops).

## Architecture

Gas absorption is CPU-prebaked into `ShellOptics.extinction` and
`ShellOptics.ssa` before GPU upload. The GPU shaders only handle scattering,
ray marching, and phase-function sampling.

## Known issues (tracked on the overhaul branch)

- The hybrid v2 kernel can exceed the macOS GPU watchdog at production ray
  counts and shows high variance at deep-twilight SZA; root causes
  (unbounded importance weights, seed truncation, O(steps^2) tau recompute)
  are being fixed alongside the CPU estimator rewrite.
