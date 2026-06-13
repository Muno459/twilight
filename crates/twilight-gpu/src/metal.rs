//! Metal GPU backend for Apple Silicon (macOS, iOS, iPadOS).
//!
//! Compiles the MSL shader at runtime via `newLibraryWithSource`, creates
//! active compute pipeline states, and dispatches work using
//! shared (zero-copy) buffers on Apple unified memory.

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};

use crate::buffers::{
    dispatch_groups, PackedAtmosphere, PackedDispatchParams, PackedLightSource,
    PackedSolarSpectrum, PackedVisionLuts,
};
use crate::{
    BackendKind, BatchKernel, BatchRequest, GpuBackend, GpuConfig, GpuDeviceInfo, GpuError,
    GpuSpectralResult,
};

// Required for MTLCreateSystemDefaultDevice to link correctly.
#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {}

/// Embedded MSL shader source. In release builds we use include_str! to embed
/// the shader at compile time. In debug builds we load from disk for faster
/// iteration (if the file exists), falling back to the embedded source.
const SHADER_SOURCE: &str = include_str!("../shaders/twilight.metal");

/// Kernels write this to output[0] and abort when the atmosphere buffer
/// fails the magic/version gate (HEADER_SENTINEL in twilight.metal).
/// Radiance and per-photon weights are never negative, so the exact bit
/// pattern of -1.0 cannot occur in valid output.
const HEADER_SENTINEL: f32 = -1.0;

/// Detect the shader-side header gate sentinel in a readback slice.
fn check_header_sentinel(output: &[f32]) -> Result<(), GpuError> {
    if output.first().map(|v| v.to_bits()) == Some(HEADER_SENTINEL.to_bits()) {
        return Err(GpuError::BufferVersionMismatch);
    }
    Ok(())
}

/// Metal backend implementing the [`GpuBackend`] trait.
///
/// # Safety
///
/// Metal device, command queue, pipeline states, and shared buffers are all
/// thread-safe in Apple's Metal API. The device and command queue are
/// explicitly documented as thread-safe, and shared-mode buffers on unified
/// memory can be safely accessed from any thread (as long as GPU work has
/// completed before CPU readback, which we guarantee via `waitUntilCompleted`).
pub struct MetalBackend {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,

    // Active compute pipeline states.
    pso_single_scatter: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pso_mcrt_trace: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pso_hybrid_v2: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pso_garstang: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    // Test-only: device-side field DDA probe (G-DDA-PARITY). Only read by
    // the #[cfg(test)] field_tau_probe helper, so it is dead in lib builds.
    #[cfg_attr(not(test), allow(dead_code))]
    pso_field_tau_probe: Retained<ProtocolObject<dyn MTLComputePipelineState>>,

    // Uploaded atmosphere + constant buffers (persisted between dispatches).
    buf_atm: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    // Uploaded 3D cloud field (v4). None => 1D shell-cloud path.
    buf_field: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    // Always-bound stub so the kernel's buffer(3) argument is never null
    // when no field is present (the shader gates on field_present anyway).
    buf_field_stub: Retained<ProtocolObject<dyn MTLBuffer>>,
    // Solar and vision LUTs uploaded at init. Reserved for spectral weighting
    // kernels in Phase 11f pipeline integration.
    #[allow(dead_code)]
    buf_solar: Retained<ProtocolObject<dyn MTLBuffer>>,
    #[allow(dead_code)]
    buf_vision: Retained<ProtocolObject<dyn MTLBuffer>>,

    info: GpuDeviceInfo,
    config: GpuConfig,
    num_wavelengths: u32,
}

// Safety: Metal objects (device, queue, pipeline states, shared buffers) are
// thread-safe per Apple documentation. We synchronize GPU/CPU access via
// waitUntilCompleted before any buffer readback.
unsafe impl Send for MetalBackend {}

// ── Probe ───────────────────────────────────────────────────────────────

/// Lightweight probe: can we get a Metal device?
pub fn probe() -> bool {
    MTLCreateSystemDefaultDevice().is_some()
}

// ── Init ────────────────────────────────────────────────────────────────

/// Initialize the Metal backend: get device, compile shaders, create pipelines.
pub fn init(config: &GpuConfig) -> Result<Box<dyn GpuBackend>, GpuError> {
    Ok(Box::new(init_backend(config)?))
}

/// Concrete-typed init. Tests need the concrete `MetalBackend` to reach
/// test-only helpers (e.g. corrupting the uploaded buffer header).
pub(crate) fn init_backend(config: &GpuConfig) -> Result<MetalBackend, GpuError> {
    // 1. Get default Metal device
    let device = MTLCreateSystemDefaultDevice().ok_or(GpuError::NoDevice)?;

    // 2. Create command queue
    let queue = device
        .newCommandQueue()
        .ok_or_else(|| GpuError::Platform("failed to create command queue".into()))?;

    // 3. Compile MSL source at runtime
    let shader_source = load_shader_source();
    let ns_source = NSString::from_str(&shader_source);

    let library = device
        .newLibraryWithSource_options_error(&ns_source, None)
        .map_err(|e| GpuError::ShaderCompilation(format!("{}", e)))?;

    // 4. Create pipeline states for active kernels
    let pso_single_scatter = make_pipeline(&device, &library, "single_scatter_spectrum")?;
    let pso_mcrt_trace = make_pipeline(&device, &library, "mcrt_trace_photon")?;
    let pso_hybrid_v2 = make_pipeline(&device, &library, "hybrid_scatter_v2")?;
    let pso_garstang = make_pipeline(&device, &library, "garstang_zenith")?;
    let pso_field_tau_probe = make_pipeline(&device, &library, "field_tau_probe")?;

    // 5. Pack and upload constant buffers (solar spectrum, vision LUTs)
    let solar = PackedSolarSpectrum::pack();
    let vision = PackedVisionLuts::pack();

    let buf_solar = create_buffer_from_f32(&device, &solar.data)?;
    let buf_vision = create_buffer_from_f32(&device, &vision.data)?;
    // Stub field buffer (16 f32): bound at index 3 whenever no real field
    // is present. The shader never reads it (field_present == 0).
    let buf_field_stub = create_empty_buffer(&device, 16)?;

    // 6. Build device info
    let name = device.name().to_string();
    let info = GpuDeviceInfo {
        name,
        backend: BackendKind::Metal,
        memory_bytes: 0,         // Apple doesn't expose this directly
        max_workgroup_size: 256, // Conservative default for Apple GPUs
    };

    Ok(MetalBackend {
        device,
        queue,
        pso_single_scatter,
        pso_mcrt_trace,
        pso_hybrid_v2,
        pso_garstang,
        pso_field_tau_probe,
        buf_atm: None,
        buf_field: None,
        buf_field_stub,
        buf_solar,
        buf_vision,
        info,
        config: config.clone(),
        num_wavelengths: 0,
    })
}

// ── GpuBackend implementation ───────────────────────────────────────────

impl GpuBackend for MetalBackend {
    fn device_info(&self) -> &GpuDeviceInfo {
        &self.info
    }

    fn upload_atmosphere(
        &mut self,
        atm: &twilight_core::atmosphere::AtmosphereModel,
    ) -> Result<(), GpuError> {
        let packed = PackedAtmosphere::pack(atm);
        self.num_wavelengths = packed.num_wavelengths;
        self.buf_atm = Some(create_buffer_from_f32(&self.device, &packed.data)?);
        Ok(())
    }

    fn upload_field(
        &mut self,
        field: Option<&twilight_core::cloud_field::Cloud3DField>,
    ) -> Result<(), GpuError> {
        match field {
            None => {
                self.buf_field = None;
                Ok(())
            }
            Some(f) => {
                let packed = crate::buffers::PackedCloudField::pack(f);
                self.buf_field = Some(create_buffer_from_f32(&self.device, &packed.data)?);
                Ok(())
            }
        }
    }

    fn single_scatter(
        &self,
        observer_pos: [f64; 3],
        view_dir: [f64; 3],
        sun_dir: [f64; 3],
    ) -> Result<GpuSpectralResult, GpuError> {
        let buf_atm = self
            .buf_atm
            .as_ref()
            .ok_or_else(|| GpuError::Dispatch("atmosphere not uploaded".into()))?;

        let params = PackedDispatchParams::new(observer_pos, view_dir, sun_dir, 0, 0, 0);
        let buf_params = create_buffer_from_f32(&self.device, &params.data)?;

        let nw = self.num_wavelengths as usize;
        let buf_output = create_empty_buffer(&self.device, nw)?;

        self.dispatch_kernel(
            &self.pso_single_scatter,
            &[buf_atm, &buf_params, &buf_output],
            nw as u32,
        )?;

        let radiance = f32_buffer_slice(&buf_output, nw);
        check_header_sentinel(radiance)?;
        Ok(GpuSpectralResult {
            radiance: radiance.iter().map(|&v| v as f64).collect(),
            num_wavelengths: nw,
        })
    }

    fn mcrt_trace(
        &self,
        observer_pos: [f64; 3],
        view_dir: [f64; 3],
        sun_dir: [f64; 3],
        photons_per_wavelength: u32,
        seed: u64,
    ) -> Result<GpuSpectralResult, GpuError> {
        let buf_atm = self
            .buf_atm
            .as_ref()
            .ok_or_else(|| GpuError::Dispatch("atmosphere not uploaded".into()))?;

        let params = PackedDispatchParams::new(
            observer_pos,
            view_dir,
            sun_dir,
            photons_per_wavelength,
            0,
            seed,
        );
        let buf_params = create_buffer_from_f32(&self.device, &params.data)?;

        let nw = self.num_wavelengths as usize;
        let total_threads = nw * photons_per_wavelength as usize;
        let buf_output = create_empty_buffer(&self.device, total_threads)?;

        self.dispatch_kernel(
            &self.pso_mcrt_trace,
            &[buf_atm, &buf_params, &buf_output],
            total_threads as u32,
        )?;

        // CPU reduce: average per-photon weights for each wavelength
        let raw = f32_buffer_slice(&buf_output, total_threads);
        check_header_sentinel(raw)?;
        let ppw = photons_per_wavelength as usize;
        let mut radiance = Vec::with_capacity(nw);
        for w in 0..nw {
            let start = w * ppw;
            let end = start + ppw;
            let sum: f32 = raw[start..end].iter().sum();
            radiance.push((sum / ppw as f32) as f64);
        }

        Ok(GpuSpectralResult {
            radiance,
            num_wavelengths: nw,
        })
    }

    fn hybrid_scatter(
        &self,
        observer_pos: [f64; 3],
        view_dir: [f64; 3],
        sun_dir: [f64; 3],
        secondary_rays: u32,
        seed: u64,
    ) -> Result<GpuSpectralResult, GpuError> {
        if secondary_rays == 0 {
            return self.single_scatter(observer_pos, view_dir, sun_dir);
        }

        let buf_atm = self
            .buf_atm
            .as_ref()
            .ok_or_else(|| GpuError::Dispatch("atmosphere not uploaded".into()))?;

        let nw = self.num_wavelengths as usize;
        const MAX_LOS_STEPS: usize = 200;
        let output_len = nw * MAX_LOS_STEPS;

        // Each dispatch traces RAYS_PER_DISPATCH rays across all (wl, step)
        // threadgroups. The GPU watchdog allows ~2s per dispatch; each 2500-ray
        // dispatch finishes in ~1-5ms, well under the limit. Raising from 256
        // to 2500 cuts command buffer submissions by ~10x.
        // Sized so one command buffer stays safely under the macOS GPU
        // watchdog (~2 s) even for deep-twilight chains that traverse many
        // shells: 250 rays/buffer empirically completes in well under a
        // second on Apple Silicon; the extra commit/wait round-trips cost
        // only ~1 ms each. (The old 2500 x 4 = 10000-ray buffers were
        // killed with kIOGPUCommandBufferCallbackErrorImpactingInteractivity.)
        // With a 3D field bound, each ray's chain walks the device-side DDA
        // (NEE shadow legs + the cloud-channel race), so a 250-ray buffer can
        // exceed the macOS GPU watchdog (~2 s) at deep twilight. Measured cost
        // on a DENSE uniform deck (worst case: no empty-tile DDA skips, every
        // NEE shadow ray crosses thousands of occupied cells) is ~0.15-0.2 s
        // per ray at SZA 96 on an M2 Pro, so 8 rays/buffer = ~1.4 s already
        // brushes the watchdog and 16 trips it. Drop to 4 rays/buffer when a
        // field is present (~0.9 s worst case, comfortable margin); broken
        // real fields are far cheaper per ray (empty-tile skips), so this is
        // a floor, not the typical cost. More commit/wait round-trips (~1 ms
        // each) but every command buffer stays safely under the watchdog.
        let rays_per_dispatch: u32 = if self.buf_field.is_some() { 4 } else { 250 };
        const DISPATCHES_PER_COMMAND_BUFFER: usize = 1;
        const PARAMS_STRIDE: usize = 16;
        let num_dispatches = secondary_rays.div_ceil(rays_per_dispatch).max(1);

        // Pre-allocate a single reusable output buffer for the largest chunk.
        // This eliminates per-chunk Metal buffer allocation + zeroing (~77 MB
        // of memset per prayer at the old 256-ray setting).
        let max_chunk = DISPATCHES_PER_COMMAND_BUFFER.min(num_dispatches as usize);
        let buf_output = create_empty_buffer(&self.device, max_chunk * output_len)?;
        let mut accum = vec![0.0f64; output_len];

        for chunk_start in (0..num_dispatches as usize).step_by(DISPATCHES_PER_COMMAND_BUFFER) {
            let chunk_len =
                (num_dispatches as usize - chunk_start).min(DISPATCHES_PER_COMMAND_BUFFER);
            let mut all_params = Vec::with_capacity(chunk_len * PARAMS_STRIDE);

            for chunk_idx in 0..chunk_len {
                let d = (chunk_start + chunk_idx) as u32;
                let ray_start = d * rays_per_dispatch;
                let rays_this = (secondary_rays - ray_start).min(rays_per_dispatch);

                // Fold the FULL 64-bit seed into the low 32 bits before
                // packing (splitmix64 finalizer). The previous
                // `seed & 0xFFFF_FFFF` discarded the high word - and
                // `sza_deg.to_bits()` for SZAs on a 0.5-degree grid has
                // all-zero low bits, so every SZA in a prayer scan ran
                // with base_seed = 0 (identical RNG streams).
                let folded = {
                    let mut z = seed;
                    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                    z ^ (z >> 31)
                };
                let ray_seed = (folded & 0xFFFF_FFFF) | ((ray_start as u64) << 32);
                let params = PackedDispatchParams::new_with_field(
                    observer_pos,
                    view_dir,
                    sun_dir,
                    secondary_rays,
                    rays_this,
                    ray_seed,
                    self.buf_field.is_some(),
                );
                all_params.extend_from_slice(&params.data);
            }

            let buf_params = create_buffer_from_f32(&self.device, &all_params)?;

            // Zero only the portion we'll use (reuse same buffer across chunks)
            let zero_bytes = chunk_len * output_len * std::mem::size_of::<f32>();
            unsafe {
                std::ptr::write_bytes(buf_output.contents().as_ptr() as *mut u8, 0, zero_bytes);
            }

            let field_buf: &ProtocolObject<dyn MTLBuffer> = self
                .buf_field
                .as_deref()
                .unwrap_or(&self.buf_field_stub);
            self.dispatch_hybrid_v2_chunk(
                &self.pso_hybrid_v2,
                buf_atm,
                &buf_params,
                &buf_output,
                field_buf,
                nw as u32,
                MAX_LOS_STEPS as u32,
                chunk_len,
                output_len,
                PARAMS_STRIDE,
            )?;

            let raw = f32_buffer_slice(&buf_output, chunk_len * output_len);
            check_header_sentinel(raw)?;
            for chunk_idx in 0..chunk_len {
                let base = chunk_idx * output_len;
                for (i, &v) in raw[base..base + output_len].iter().enumerate() {
                    if v.is_finite() {
                        accum[i] += v as f64;
                    }
                }
            }
        }

        let inv_rays = 1.0 / secondary_rays.max(1) as f64;
        let mut radiance = Vec::with_capacity(nw);
        for w in 0..nw {
            let base = w * MAX_LOS_STEPS;
            let sum: f64 = accum[base..base + MAX_LOS_STEPS].iter().sum();
            radiance.push(sum * inv_rays);
        }

        Ok(GpuSpectralResult {
            radiance,
            num_wavelengths: nw,
        })
    }

    fn garstang_zenith(
        &self,
        _observer_pos: [f64; 3],
        sources: &[PackedLightSource],
    ) -> Result<f64, GpuError> {
        if sources.is_empty() {
            return Ok(0.0);
        }

        let num_sources = sources.len();

        // Pack sources into flat f32 buffer (8 f32 per source)
        let mut source_data = Vec::with_capacity(num_sources * 8);
        for s in sources {
            source_data.push(s.distance_m);
            source_data.push(s.zenith_angle_rad);
            source_data.push(s.radiance);
            source_data.push(s.spectrum_type);
            source_data.push(s.height_m);
            source_data.push(s._pad1);
            source_data.push(s._pad2);
            source_data.push(s._pad3);
        }
        let buf_sources = create_buffer_from_f32(&self.device, &source_data)?;

        // Config buffer: 8 f32
        let config_data: [f32; 8] = [
            0.0,                // observer_elevation
            0.15,               // aod_550
            0.10,               // uplight_fraction
            0.15,               // ground_reflectance
            550.0,              // wavelength_nm
            50.0,               // altitude_steps
            30000.0,            // max_altitude
            num_sources as f32, // num_sources
        ];
        let buf_config = create_buffer_from_f32(&self.device, &config_data)?;

        let buf_output = create_empty_buffer(&self.device, num_sources)?;

        self.dispatch_kernel(
            &self.pso_garstang,
            &[&buf_sources, &buf_config, &buf_output],
            num_sources as u32,
        )?;

        // Sum all source contributions on CPU
        let results = f32_buffer_slice(&buf_output, num_sources);
        let total: f64 = results.iter().map(|&v| v as f64).sum();
        Ok(total)
    }

    fn scan_batch(&self, requests: &[BatchRequest]) -> Result<Vec<GpuSpectralResult>, GpuError> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        if requests
            .iter()
            .any(|req| matches!(req.kernel, BatchKernel::Hybrid { .. }))
        {
            let mut results: Vec<Option<GpuSpectralResult>> =
                (0..requests.len()).map(|_| None).collect();
            let mut non_hybrid_indices = Vec::new();
            let mut non_hybrid_requests = Vec::new();

            for (idx, req) in requests.iter().enumerate() {
                match req.kernel {
                    BatchKernel::Hybrid {
                        secondary_rays,
                        seed,
                    } => {
                        results[idx] = Some(self.hybrid_scatter(
                            req.observer_pos,
                            req.view_dir,
                            req.sun_dir,
                            secondary_rays,
                            seed,
                        )?);
                    }
                    _ => {
                        non_hybrid_indices.push(idx);
                        non_hybrid_requests.push(req.clone());
                    }
                }
            }

            if !non_hybrid_requests.is_empty() {
                let batched = self.scan_batch_non_hybrid(&non_hybrid_requests)?;
                for (idx, result) in non_hybrid_indices.into_iter().zip(batched) {
                    results[idx] = Some(result);
                }
            }

            return Ok(results
                .into_iter()
                .map(|result| result.expect("scan_batch result slot should be filled"))
                .collect());
        }

        self.scan_batch_non_hybrid(requests)
    }
}

// ── Internal helpers ────────────────────────────────────────────────────

impl MetalBackend {
    /// Overwrite the packed version word of the uploaded atmosphere buffer
    /// in place (shared-mode buffers are CPU-visible). Simulates a stale
    /// or misversioned upload for the header-gate test.
    #[cfg(test)]
    pub(crate) fn corrupt_atm_version_word(&self) {
        use crate::buffers::{atm_offsets, BUFFER_VERSION};
        let buf = self.buf_atm.as_ref().expect("atmosphere not uploaded");
        let ptr = buf.contents().as_ptr() as *mut f32;
        unsafe {
            *ptr.add(atm_offsets::HEADER_VERSION) = f32::from_bits(BUFFER_VERSION + 1);
        }
    }

    /// Overwrite the packed version word of the uploaded field buffer to a
    /// specific value (e.g. the old v3) for the field header-gate test.
    #[cfg(test)]
    pub(crate) fn set_field_version_word(&self, version: u32) {
        use crate::buffers::field_offsets;
        let buf = self.buf_field.as_ref().expect("field not uploaded");
        let ptr = buf.contents().as_ptr() as *mut f32;
        unsafe {
            *ptr.add(field_offsets::HEADER_VERSION) = f32::from_bits(version);
        }
    }

    /// Dispatch the device-side field DDA probe (G-DDA-PARITY). Each ray is
    /// (p0, t_max, dir): 8 f32. Returns one cloud optical depth per ray.
    /// Requires a field uploaded via `upload_field`.
    #[cfg(test)]
    pub(crate) fn field_tau_probe(&self, rays: &[[f64; 7]]) -> Result<Vec<f64>, GpuError> {
        let fld = self
            .buf_field
            .as_ref()
            .ok_or_else(|| GpuError::Dispatch("field not uploaded".into()))?;

        let n = rays.len();
        let mut packed = vec![0.0f32; n * 8];
        for (i, ray) in rays.iter().enumerate() {
            let b = i * 8;
            packed[b] = ray[0] as f32; // p0.x
            packed[b + 1] = ray[1] as f32; // p0.y
            packed[b + 2] = ray[2] as f32; // p0.z
            packed[b + 3] = ray[3] as f32; // t_max
            packed[b + 4] = ray[4] as f32; // dir.x
            packed[b + 5] = ray[5] as f32; // dir.y
            packed[b + 6] = ray[6] as f32; // dir.z
        }
        let buf_rays = create_buffer_from_f32(&self.device, &packed)?;
        let buf_out = create_empty_buffer(&self.device, n)?;

        let cmd_buf = self
            .queue
            .commandBuffer()
            .ok_or_else(|| GpuError::Dispatch("failed to create command buffer".into()))?;
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| GpuError::Dispatch("failed to create compute encoder".into()))?;
        encoder.setComputePipelineState(&self.pso_field_tau_probe);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(fld), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&buf_rays), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&buf_out), 0, 2);
        }
        let tg = MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };
        let grid = MTLSize {
            width: n.div_ceil(64),
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let raw = f32_buffer_slice(&buf_out, n);
        Ok(raw.iter().map(|&v| v as f64).collect())
    }

    fn scan_batch_non_hybrid(
        &self,
        requests: &[BatchRequest],
    ) -> Result<Vec<GpuSpectralResult>, GpuError> {
        debug_assert!(requests
            .iter()
            .all(|req| !matches!(req.kernel, BatchKernel::Hybrid { .. })));

        let buf_atm = self
            .buf_atm
            .as_ref()
            .ok_or_else(|| GpuError::Dispatch("atmosphere not uploaded".into()))?;

        let nw = self.num_wavelengths as usize;
        let n = requests.len();

        // ── Unified memory optimization ─────────────────────────────────
        //
        // Apple Silicon has unified memory: CPU and GPU share the same
        // physical DRAM. Creating many small Metal buffers wastes time on
        // per-buffer bookkeeping (the Metal runtime tracks each buffer for
        // residency, reference counting, and hazard tracking). Instead we
        // pack ALL N params into one contiguous shared buffer and ALL N
        // outputs into one contiguous shared buffer, then use byte offsets
        // at bind time via setBuffer:offset:atIndex:. This gives:
        //
        //   - 2 buffer allocations total instead of 2N
        //   - Zero CPU->GPU copies (StorageModeShared, unified memory)
        //   - Better cache locality for sequential GPU access
        //   - Single readback pointer cast (no DMA, no copy)

        // PackedDispatchParams is 16 f32 = 64 bytes, already 16-byte aligned.
        const PARAMS_STRIDE: usize = 16; // f32 count per dispatch

        // Pack all N params contiguously into one flat f32 array.
        let mut all_params = Vec::with_capacity(n * PARAMS_STRIDE);
        for req in requests {
            let (ppw, sec, seed) = match req.kernel {
                BatchKernel::SingleScatter => (0u32, 0u32, 0u64),
                BatchKernel::McrtTrace {
                    photons_per_wavelength,
                    seed,
                } => (photons_per_wavelength, 0, seed),
                BatchKernel::Hybrid {
                    secondary_rays,
                    seed,
                } => (0, secondary_rays, seed),
            };
            let p = PackedDispatchParams::new(
                req.observer_pos,
                req.view_dir,
                req.sun_dir,
                ppw,
                sec,
                seed,
            );
            all_params.extend_from_slice(&p.data);
        }
        let buf_all_params = create_buffer_from_f32(&self.device, &all_params)?;

        // Compute output layout: each dispatch's f32 count, padded to
        // 16-byte (4 f32) alignment so Metal buffer offsets stay valid.
        const ALIGN_F32: usize = 4; // 16 bytes / sizeof(f32)

        struct SliceInfo {
            offset_f32: usize,
            raw_len: usize,
            kernel: BatchKernel,
        }

        let mut slices = Vec::with_capacity(n);
        let mut cursor: usize = 0;

        for req in requests {
            let raw_len = match req.kernel {
                BatchKernel::McrtTrace {
                    photons_per_wavelength,
                    ..
                } => nw * photons_per_wavelength as usize,
                _ => nw,
            };
            slices.push(SliceInfo {
                offset_f32: cursor,
                raw_len,
                kernel: req.kernel,
            });
            // Advance cursor, padded to 16-byte alignment.
            let padded = (raw_len + ALIGN_F32 - 1) & !(ALIGN_F32 - 1);
            cursor += padded;
        }

        let total_output_f32 = cursor.max(1); // avoid zero-length buffer
        let buf_all_output = create_empty_buffer(&self.device, total_output_f32)?;

        // ── Encode all N dispatches into ONE command buffer ─────────────

        let cmd_buf = self
            .queue
            .commandBuffer()
            .ok_or_else(|| GpuError::Dispatch("failed to create command buffer".into()))?;

        let wg_size = self.config.workgroup_size;

        // ── ONE encoder for ALL dispatches ──────────────────────────────
        //
        // Metal allows multiple setComputePipelineState + setBuffer +
        // dispatchThreadgroups calls within a single compute encoder.
        // Creating one encoder per dispatch (the old code) adds ~0.5ms
        // of Metal runtime overhead per encoder, which dominates for
        // lightweight kernels like single_scatter (50 dispatches of ~10
        // threads each). One encoder eliminates this overhead entirely.
        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| GpuError::Dispatch("failed to create compute encoder".into()))?;

        for (i, s) in slices.iter().enumerate() {
            let pipeline = match s.kernel {
                BatchKernel::SingleScatter => &self.pso_single_scatter,
                BatchKernel::McrtTrace { .. } => &self.pso_mcrt_trace,
                BatchKernel::Hybrid { .. } => unreachable!("hybrid handled by early fallback"),
            };

            if s.raw_len == 0 {
                continue;
            }

            encoder.setComputePipelineState(pipeline);

            let params_byte_offset = i * PARAMS_STRIDE * std::mem::size_of::<f32>();
            let output_byte_offset = s.offset_f32 * std::mem::size_of::<f32>();

            unsafe {
                encoder.setBuffer_offset_atIndex(Some(buf_atm), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&buf_all_params), params_byte_offset, 1);
                encoder.setBuffer_offset_atIndex(Some(&buf_all_output), output_byte_offset, 2);
            }

            let total_threads = s.raw_len as u32;
            let num_groups = dispatch_groups(total_threads, wg_size);
            let grid_size = MTLSize {
                width: num_groups as usize,
                height: 1,
                depth: 1,
            };
            let threadgroup_size = MTLSize {
                width: wg_size as usize,
                height: 1,
                depth: 1,
            };

            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        }

        encoder.endEncoding();

        // ONE commit, ONE wait -- the whole point of batching.
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        // ── Readback from unified memory ────────────────────────────────
        //
        // StorageModeShared on Apple Silicon means buf_all_output.contents()
        // points directly into unified DRAM -- this is a pointer cast, not
        // a DMA transfer. We read once into a Vec then slice per-dispatch.
        let all_output = f32_buffer_slice(&buf_all_output, total_output_f32);

        let mut results = Vec::with_capacity(n);

        for s in &slices {
            check_header_sentinel(&all_output[s.offset_f32..s.offset_f32 + s.raw_len])?;
            match s.kernel {
                BatchKernel::McrtTrace {
                    photons_per_wavelength,
                    ..
                } => {
                    let ppw = photons_per_wavelength as usize;
                    let base = s.offset_f32;
                    let mut radiance = Vec::with_capacity(nw);
                    for w in 0..nw {
                        let start = base + w * ppw;
                        let end = start + ppw;
                        let sum: f32 = all_output[start..end].iter().sum();
                        radiance.push((sum / ppw as f32) as f64);
                    }
                    results.push(GpuSpectralResult {
                        radiance,
                        num_wavelengths: nw,
                    });
                }
                _ => {
                    let base = s.offset_f32;
                    let raw = &all_output[base..base + nw];
                    results.push(GpuSpectralResult {
                        radiance: raw.iter().map(|&v| v as f64).collect(),
                        num_wavelengths: nw,
                    });
                }
            }
        }

        Ok(results)
    }

    /// Encode and dispatch a compute kernel with the given buffers.
    ///
    /// Used for single_scatter, mcrt_trace, and garstang kernels where each
    /// thread is independent and the workgroup size is configurable.
    fn dispatch_kernel(
        &self,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        buffers: &[&ProtocolObject<dyn MTLBuffer>],
        total_threads: u32,
    ) -> Result<(), GpuError> {
        if total_threads == 0 {
            return Ok(());
        }

        let cmd_buf = self
            .queue
            .commandBuffer()
            .ok_or_else(|| GpuError::Dispatch("failed to create command buffer".into()))?;

        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| GpuError::Dispatch("failed to create compute encoder".into()))?;

        encoder.setComputePipelineState(pipeline);

        for (i, buf) in buffers.iter().enumerate() {
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(*buf), 0, i);
            }
        }

        let wg_size = self.config.workgroup_size;
        let num_groups = dispatch_groups(total_threads, wg_size);

        let threadgroup_size = MTLSize {
            width: wg_size as usize,
            height: 1,
            depth: 1,
        };
        let grid_size = MTLSize {
            width: num_groups as usize,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        encoder.endEncoding();

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        Ok(())
    }

    #[allow(clippy::too_many_arguments)] // GPU dispatch plumbing: buffers + geometry are independent
    fn dispatch_hybrid_v2_chunk(
        &self,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        atm_buffer: &ProtocolObject<dyn MTLBuffer>,
        params_buffer: &ProtocolObject<dyn MTLBuffer>,
        output_buffer: &ProtocolObject<dyn MTLBuffer>,
        field_buffer: &ProtocolObject<dyn MTLBuffer>,
        num_wavelengths: u32,
        num_steps: u32,
        dispatch_count: usize,
        output_stride_f32: usize,
        params_stride_f32: usize,
    ) -> Result<(), GpuError> {
        if num_wavelengths == 0 || num_steps == 0 || dispatch_count == 0 {
            return Ok(());
        }

        let cmd_buf = self
            .queue
            .commandBuffer()
            .ok_or_else(|| GpuError::Dispatch("failed to create command buffer".into()))?;

        let encoder = cmd_buf
            .computeCommandEncoder()
            .ok_or_else(|| GpuError::Dispatch("failed to create compute encoder".into()))?;

        encoder.setComputePipelineState(pipeline);

        let threadgroup_size = MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        };
        let grid_size = MTLSize {
            width: num_wavelengths as usize,
            height: num_steps as usize,
            depth: 1,
        };

        for dispatch_idx in 0..dispatch_count {
            let params_byte_offset = dispatch_idx * params_stride_f32 * std::mem::size_of::<f32>();
            let output_byte_offset = dispatch_idx * output_stride_f32 * std::mem::size_of::<f32>();

            unsafe {
                encoder.setBuffer_offset_atIndex(Some(atm_buffer), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(params_buffer), params_byte_offset, 1);
                encoder.setBuffer_offset_atIndex(Some(output_buffer), output_byte_offset, 2);
                encoder.setBuffer_offset_atIndex(Some(field_buffer), 0, 3);
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, threadgroup_size);
        }

        encoder.endEncoding();
        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let status = cmd_buf.status();
        if status == objc2_metal::MTLCommandBufferStatus::Error {
            let err_msg = cmd_buf
                .error()
                .map(|e| format!("{}", e))
                .unwrap_or_else(|| "unknown GPU error".into());
            return Err(GpuError::Dispatch(format!(
                "Metal command buffer error in hybrid_scatter_v2 chunk: {}",
                err_msg
            )));
        }

        Ok(())
    }
}

/// Load MSL shader source. In debug builds, try to load from disk first
/// for faster shader iteration. Fall back to the embedded source.
fn load_shader_source() -> String {
    #[cfg(debug_assertions)]
    {
        let disk_path = concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/twilight.metal");
        if let Ok(source) = std::fs::read_to_string(disk_path) {
            return source;
        }
    }
    SHADER_SOURCE.to_string()
}

/// Create a compute pipeline state from a named kernel function.
fn make_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, GpuError> {
    let ns_name = NSString::from_str(name);
    let function = library
        .newFunctionWithName(&ns_name)
        .ok_or_else(|| GpuError::ShaderCompilation(format!("function '{}' not found", name)))?;

    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| GpuError::ShaderCompilation(format!("pipeline '{}': {}", name, e)))
}

/// Create a shared Metal buffer from a slice of f32 values.
fn create_buffer_from_f32(
    device: &ProtocolObject<dyn MTLDevice>,
    data: &[f32],
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, GpuError> {
    let byte_len = std::mem::size_of_val(data);
    if byte_len == 0 {
        return Err(GpuError::BufferAllocation("zero-length buffer".into()));
    }

    let ptr = NonNull::new(data.as_ptr() as *mut c_void)
        .ok_or_else(|| GpuError::BufferAllocation("null data pointer".into()))?;

    let buf = unsafe {
        device.newBufferWithBytes_length_options(
            ptr,
            byte_len,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or_else(|| GpuError::BufferAllocation("Metal buffer allocation failed".into()))?;

    Ok(buf)
}

/// Create an empty shared Metal buffer for `n` f32 output elements, zeroed.
fn create_empty_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    n: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, GpuError> {
    let byte_len = n * std::mem::size_of::<f32>();
    if byte_len == 0 {
        return Err(GpuError::BufferAllocation(
            "zero-length output buffer".into(),
        ));
    }

    let buf = device
        .newBufferWithLength_options(byte_len, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| {
            GpuError::BufferAllocation("Metal output buffer allocation failed".into())
        })?;

    // Zero the buffer -- Metal does not guarantee zeroed contents for
    // StorageModeShared allocations. Required for correctness because
    // invalid threadgroup slots (step >= num_steps) may not be written.
    let ptr = buf.contents();
    unsafe {
        std::ptr::write_bytes(ptr.as_ptr() as *mut u8, 0, byte_len);
    }

    Ok(buf)
}

/// Borrow f32 values from a shared Metal buffer.
fn f32_buffer_slice(buffer: &ProtocolObject<dyn MTLBuffer>, n: usize) -> &[f32] {
    let ptr = buffer.contents();
    unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const f32, n) }
}
