# twilight-gpu

GPU compute backends for the MCRT engine. Four backends, each hand-tuned for its native GPU API. All backends produce identical physics output, verified by a shared CPU oracle test suite.

## Backends

**Metal** (`metal.rs`). Apple GPUs (macOS, iOS). Hand-written `.metal` shaders compiled at runtime. Host code via `objc2-metal`. Tested on Apple Silicon.

**Vulkan** (`vulkan.rs`). AMD, Intel, NVIDIA, Android, Linux. GLSL shaders compiled to SPIR-V. Host code via `ash`. GPU memory management via `gpu-allocator`.

**CUDA** (`cuda.rs`). NVIDIA GPUs. Hand-written `.cu` shaders. Host code via `cudarc`. Targets compute capability 7.0+.

**wgpu** (`wgpu_backend.rs`). WASM and browsers. Hand-written `.wgsl` shaders. Runs on any `wgpu`-supported backend. Primary target is the web via WebGPU.

All four backends are behind Cargo feature flags and compiled independently.

## Modules

**`buffers`**. Packed f32 atmosphere representation for GPU upload. `PackedAtmosphere` serializes the `AtmosphereModel` (shell geometry, optics at each wavelength, surface albedo, wavelength grid) into a flat f32 buffer with a magic/version header. `DispatchParams` encodes observer position, sun direction, view direction, photon count, and RNG seed. Also packs solar irradiance and CIE vision functions into GPU-friendly buffers.

**`oracle`**. CPU reference implementation that generates test cases for GPU validation. Produces expected results for ray-sphere intersection, RNG sequences, phase functions (Rayleigh + HG), shadow ray transmittance, single-scatter radiance, and full spectral sweeps. The GPU backends must match these within f32 tolerance.

**`tests`**. Cross-backend parity tests. Verifies that Metal, Vulkan, CUDA, and wgpu all agree on: single-scatter radiance monotonicity, deep night giving zero, MCRT sign agreement, and absolute values matching the CPU oracle. Also includes latency benchmarks.

## Architecture

Gas absorption is CPU-prebaked into `ShellOptics.extinction` and `ShellOptics.ssa` before GPU upload. The GPU shaders only need to handle scattering, ray marching, and phase function sampling. This keeps the shaders simple and avoids shipping cross-section lookup tables to the GPU.

## Tests

99 tests. Buffer packing/unpacking roundtrips, oracle determinism, physics validation (radiance monotonicity, non-negativity, transmittance bounds, spectral red-dominance at twilight), per-backend integration tests (init, upload, single-scatter, MCRT, hybrid), cross-backend parity, and GPU latency benchmarks.
