//! Metal GPU backend for Apple Silicon (macOS, iOS, iPadOS).
//!
//! Compiles the MSL shader at runtime via `newLibraryWithSource`, creates
//! four compute pipeline states (one per kernel), and dispatches work using
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
use crate::{BackendKind, GpuBackend, GpuConfig, GpuDeviceInfo, GpuError, GpuSpectralResult};

// Required for MTLCreateSystemDefaultDevice to link correctly.
#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {}

/// Embedded MSL shader source. In release builds we use include_str! to embed
/// the shader at compile time. In debug builds we load from disk for faster
/// iteration (if the file exists), falling back to the embedded source.
const SHADER_SOURCE: &str = include_str!("../shaders/twilight.metal");

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

    // Four compute pipeline states, one per kernel.
    pso_single_scatter: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pso_mcrt_trace: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pso_hybrid: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pso_garstang: Retained<ProtocolObject<dyn MTLComputePipelineState>>,

    // Uploaded atmosphere + constant buffers (persisted between dispatches).
    buf_atm: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
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
    let ptr = unsafe { MTLCreateSystemDefaultDevice() };
    !ptr.is_null()
}

// ── Init ────────────────────────────────────────────────────────────────

/// Initialize the Metal backend: get device, compile shaders, create pipelines.
pub fn init(config: &GpuConfig) -> Result<Box<dyn GpuBackend>, GpuError> {
    // 1. Get default Metal device
    let device = {
        let ptr = unsafe { MTLCreateSystemDefaultDevice() };
        unsafe { Retained::retain(ptr) }.ok_or(GpuError::NoDevice)?
    };

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

    // 4. Create pipeline states for all four kernels
    let pso_single_scatter = make_pipeline(&device, &library, "single_scatter_spectrum")?;
    let pso_mcrt_trace = make_pipeline(&device, &library, "mcrt_trace_photon")?;
    let pso_hybrid = make_pipeline(&device, &library, "hybrid_scatter")?;
    let pso_garstang = make_pipeline(&device, &library, "garstang_zenith")?;

    // 5. Pack and upload constant buffers (solar spectrum, vision LUTs)
    let solar = PackedSolarSpectrum::pack();
    let vision = PackedVisionLuts::pack();

    let buf_solar = create_buffer_from_f32(&device, &solar.data)?;
    let buf_vision = create_buffer_from_f32(&device, &vision.data)?;

    // 6. Build device info
    let name = device.name().to_string();
    let info = GpuDeviceInfo {
        name,
        backend: BackendKind::Metal,
        memory_bytes: 0,         // Apple doesn't expose this directly
        max_workgroup_size: 256, // Conservative default for Apple GPUs
    };

    Ok(Box::new(MetalBackend {
        device,
        queue,
        pso_single_scatter,
        pso_mcrt_trace,
        pso_hybrid,
        pso_garstang,
        buf_atm: None,
        buf_solar,
        buf_vision,
        info,
        config: config.clone(),
        num_wavelengths: 0,
    }))
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

        let radiance = read_f32_buffer(&buf_output, nw);
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
        let raw = read_f32_buffer(&buf_output, total_threads);
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
        let buf_atm = self
            .buf_atm
            .as_ref()
            .ok_or_else(|| GpuError::Dispatch("atmosphere not uploaded".into()))?;

        let params =
            PackedDispatchParams::new(observer_pos, view_dir, sun_dir, 0, secondary_rays, seed);
        let buf_params = create_buffer_from_f32(&self.device, &params.data)?;

        let nw = self.num_wavelengths as usize;
        let buf_output = create_empty_buffer(&self.device, nw)?;

        self.dispatch_kernel(
            &self.pso_hybrid,
            &[buf_atm, &buf_params, &buf_output],
            nw as u32,
        )?;

        let radiance = read_f32_buffer(&buf_output, nw);
        Ok(GpuSpectralResult {
            radiance: radiance.iter().map(|&v| v as f64).collect(),
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
        let results = read_f32_buffer(&buf_output, num_sources);
        let total: f64 = results.iter().map(|&v| v as f64).sum();
        Ok(total)
    }
}

// ── Internal helpers ────────────────────────────────────────────────────

impl MetalBackend {
    /// Encode and dispatch a compute kernel with the given buffers.
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
                encoder.setBuffer_offset_atIndex(Some(*buf), 0, i as usize);
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
        unsafe { cmd_buf.waitUntilCompleted() };

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
    let byte_len = data.len() * std::mem::size_of::<f32>();
    if byte_len == 0 {
        return Err(GpuError::BufferAllocation("zero-length buffer".into()));
    }

    let ptr = NonNull::new(data.as_ptr() as *mut c_void)
        .ok_or_else(|| GpuError::BufferAllocation("null data pointer".into()))?;

    let buf = unsafe {
        device.newBufferWithBytes_length_options(
            ptr,
            byte_len,
            MTLResourceOptions::MTLResourceStorageModeShared,
        )
    }
    .ok_or_else(|| GpuError::BufferAllocation("Metal buffer allocation failed".into()))?;

    Ok(buf)
}

/// Create an empty shared Metal buffer for `n` f32 output elements.
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

    device
        .newBufferWithLength_options(byte_len, MTLResourceOptions::MTLResourceStorageModeShared)
        .ok_or_else(|| GpuError::BufferAllocation("Metal output buffer allocation failed".into()))
}

/// Read f32 values from a shared Metal buffer.
fn read_f32_buffer(buffer: &ProtocolObject<dyn MTLBuffer>, n: usize) -> Vec<f32> {
    let ptr = buffer.contents();
    let slice = unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const f32, n) };
    slice.to_vec()
}
