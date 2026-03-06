//! CUDA GPU backend for NVIDIA GPUs.
//!
//! Compiles the CUDA C source at runtime via nvrtc (CUDA C -> PTX), then
//! JIT-compiles to native SASS on the target GPU. Creates four kernel
//! function handles and dispatches work using device-allocated buffers.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

use crate::buffers::{
    dispatch_groups, PackedAtmosphere, PackedDispatchParams, PackedLightSource,
    PackedSolarSpectrum, PackedVisionLuts,
};
use crate::{BackendKind, GpuBackend, GpuConfig, GpuDeviceInfo, GpuError, GpuSpectralResult};

/// Block size for the hybrid kernel. Must match HYBRID_BLOCK_SIZE in
/// twilight.cu (256 threads = 8 warps of 32).
const HYBRID_BLOCK_SIZE: u32 = 256;

/// Embedded CUDA C shader source.
const CUDA_SOURCE: &str = include_str!("../shaders/twilight.cu");

/// CUDA backend implementing the [`GpuBackend`] trait.
///
/// Uses `cudarc` for device management, nvrtc for runtime PTX compilation,
/// and the CUDA driver API for kernel dispatch.
pub struct CudaBackend {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,

    // Four kernel function handles
    fn_single_scatter: CudaFunction,
    fn_mcrt_trace: CudaFunction,
    fn_hybrid: CudaFunction,
    fn_garstang: CudaFunction,

    // Uploaded atmosphere buffer (persisted between dispatches)
    buf_atm: Option<CudaSlice<f32>>,
    // Solar and vision LUTs (reserved for Phase 11f pipeline integration)
    #[allow(dead_code)]
    buf_solar: CudaSlice<f32>,
    #[allow(dead_code)]
    buf_vision: CudaSlice<f32>,

    info: GpuDeviceInfo,
    config: GpuConfig,
    num_wavelengths: u32,
}

// Safety: CudaContext, CudaStream, CudaFunction, and CudaSlice are all
// Send + Sync in cudarc 0.19. The Arc wrappers ensure proper lifetime
// management across threads.
unsafe impl Send for CudaBackend {}

// ── Probe ───────────────────────────────────────────────────────────────

/// Lightweight probe: can we initialize CUDA and find a device?
pub fn probe() -> bool {
    CudaContext::new(0).is_ok()
}

// ── Init ────────────────────────────────────────────────────────────────

/// Initialize the CUDA backend: create context, compile PTX, load kernels.
pub fn init(config: &GpuConfig) -> Result<Box<dyn GpuBackend>, GpuError> {
    // 1. Create CUDA context on device 0
    let ctx = CudaContext::new(0)
        .map_err(|e| GpuError::Platform(format!("CUDA context creation failed: {}", e)))?;

    // 2. Get device info
    let name = ctx
        .name()
        .map_err(|e| GpuError::Platform(format!("failed to get device name: {}", e)))?;

    let (major, minor) = ctx
        .compute_capability()
        .map_err(|e| GpuError::Platform(format!("failed to get compute capability: {}", e)))?;

    let total_mem = cudarc::driver::result::mem_get_info()
        .map(|(_, total)| total as u64)
        .unwrap_or(0);

    // 3. Compile CUDA C source to PTX via nvrtc
    let arch = format!("compute_{}{}", major, minor);
    let opts = CompileOptions {
        arch: None, // Let nvrtc pick based on current device
        options: vec![
            format!("--gpu-architecture={}", arch),
            "--use_fast_math".to_string(),
        ],
        ..Default::default()
    };

    let ptx = compile_ptx_with_opts(CUDA_SOURCE, opts)
        .map_err(|e| GpuError::ShaderCompilation(format!("nvrtc compilation failed: {}", e)))?;

    // 4. Load PTX module (JIT compiles to native SASS)
    let module = ctx
        .load_module(ptx)
        .map_err(|e| GpuError::ShaderCompilation(format!("PTX JIT load failed: {}", e)))?;

    // 5. Get kernel function handles
    let fn_single_scatter = module
        .load_function("single_scatter_spectrum")
        .map_err(|e| GpuError::ShaderCompilation(format!("single_scatter_spectrum: {}", e)))?;

    let fn_mcrt_trace = module
        .load_function("mcrt_trace_photon")
        .map_err(|e| GpuError::ShaderCompilation(format!("mcrt_trace_photon: {}", e)))?;

    let fn_hybrid = module
        .load_function("hybrid_scatter")
        .map_err(|e| GpuError::ShaderCompilation(format!("hybrid_scatter: {}", e)))?;

    let fn_garstang = module
        .load_function("garstang_zenith")
        .map_err(|e| GpuError::ShaderCompilation(format!("garstang_zenith: {}", e)))?;

    // 6. Create stream
    let stream = ctx.default_stream();

    // 7. Pack and upload constant buffers
    let solar = PackedSolarSpectrum::pack();
    let vision = PackedVisionLuts::pack();

    let buf_solar = stream
        .clone_htod(&solar.data)
        .map_err(|e| GpuError::BufferAllocation(format!("solar spectrum: {}", e)))?;
    let buf_vision = stream
        .clone_htod(&vision.data)
        .map_err(|e| GpuError::BufferAllocation(format!("vision LUTs: {}", e)))?;

    // 8. Build device info
    let info = GpuDeviceInfo {
        name,
        backend: BackendKind::Cuda,
        memory_bytes: total_mem,
        max_workgroup_size: 1024, // Standard CUDA max threads per block
    };

    Ok(Box::new(CudaBackend {
        ctx,
        stream,
        fn_single_scatter,
        fn_mcrt_trace,
        fn_hybrid,
        fn_garstang,
        buf_atm: None,
        buf_solar,
        buf_vision,
        info,
        config: config.clone(),
        num_wavelengths: 0,
    }))
}

// ── GpuBackend implementation ───────────────────────────────────────────

impl GpuBackend for CudaBackend {
    fn device_info(&self) -> &GpuDeviceInfo {
        &self.info
    }

    fn upload_atmosphere(
        &mut self,
        atm: &twilight_core::atmosphere::AtmosphereModel,
    ) -> Result<(), GpuError> {
        let packed = PackedAtmosphere::pack(atm);
        self.num_wavelengths = packed.num_wavelengths;
        self.buf_atm = Some(
            self.stream
                .clone_htod(&packed.data)
                .map_err(|e| GpuError::BufferAllocation(format!("atmosphere: {}", e)))?,
        );
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
        let buf_params = self
            .stream
            .clone_htod(&params.data)
            .map_err(|e| GpuError::BufferAllocation(format!("params: {}", e)))?;

        let nw = self.num_wavelengths as usize;
        let mut buf_output: CudaSlice<f32> = self
            .stream
            .alloc_zeros(nw)
            .map_err(|e| GpuError::BufferAllocation(format!("output: {}", e)))?;

        let num_threads = nw as u32;
        let cfg = launch_config(num_threads, self.config.workgroup_size);

        unsafe {
            self.stream
                .launch_builder(&self.fn_single_scatter)
                .arg(buf_atm)
                .arg(&buf_params)
                .arg(&mut buf_output)
                .arg(&num_threads)
                .launch(cfg)
                .map_err(|e| GpuError::Dispatch(format!("single_scatter launch: {}", e)))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| GpuError::Dispatch(format!("synchronize: {}", e)))?;

        let radiance = self
            .stream
            .clone_dtoh(&buf_output)
            .map_err(|e| GpuError::Readback(format!("single_scatter readback: {}", e)))?;

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
        let buf_params = self
            .stream
            .clone_htod(&params.data)
            .map_err(|e| GpuError::BufferAllocation(format!("params: {}", e)))?;

        let nw = self.num_wavelengths as usize;
        let total_threads = (nw * photons_per_wavelength as usize) as u32;
        let mut buf_output: CudaSlice<f32> = self
            .stream
            .alloc_zeros(total_threads as usize)
            .map_err(|e| GpuError::BufferAllocation(format!("output: {}", e)))?;

        let cfg = launch_config(total_threads, self.config.workgroup_size);

        unsafe {
            self.stream
                .launch_builder(&self.fn_mcrt_trace)
                .arg(buf_atm)
                .arg(&buf_params)
                .arg(&mut buf_output)
                .arg(&total_threads)
                .launch(cfg)
                .map_err(|e| GpuError::Dispatch(format!("mcrt_trace launch: {}", e)))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| GpuError::Dispatch(format!("synchronize: {}", e)))?;

        let raw = self
            .stream
            .clone_dtoh(&buf_output)
            .map_err(|e| GpuError::Readback(format!("mcrt_trace readback: {}", e)))?;

        // CPU reduce: average per-photon weights for each wavelength
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
        let buf_params = self
            .stream
            .clone_htod(&params.data)
            .map_err(|e| GpuError::BufferAllocation(format!("params: {}", e)))?;

        let nw = self.num_wavelengths as usize;
        let mut buf_output: CudaSlice<f32> = self
            .stream
            .alloc_zeros(nw)
            .map_err(|e| GpuError::BufferAllocation(format!("output: {}", e)))?;

        // Hybrid v2: dispatch nw blocks of HYBRID_BLOCK_SIZE threads.
        // Each block handles one wavelength; threads within the block each
        // handle one LOS step with secondary chain tracing, then reduce via
        // __shfl_down_sync + __shared__ memory.
        let num_threads = nw as u32;
        let cfg = LaunchConfig {
            grid_dim: (nw as u32, 1, 1),
            block_dim: (HYBRID_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.fn_hybrid)
                .arg(buf_atm)
                .arg(&buf_params)
                .arg(&mut buf_output)
                .arg(&num_threads)
                .launch(cfg)
                .map_err(|e| GpuError::Dispatch(format!("hybrid_scatter launch: {}", e)))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| GpuError::Dispatch(format!("synchronize: {}", e)))?;

        let radiance = self
            .stream
            .clone_dtoh(&buf_output)
            .map_err(|e| GpuError::Readback(format!("hybrid_scatter readback: {}", e)))?;

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
        let buf_sources = self
            .stream
            .clone_htod(&source_data)
            .map_err(|e| GpuError::BufferAllocation(format!("sources: {}", e)))?;

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
        let buf_config = self
            .stream
            .clone_htod(&config_data)
            .map_err(|e| GpuError::BufferAllocation(format!("config: {}", e)))?;

        let mut buf_output: CudaSlice<f32> = self
            .stream
            .alloc_zeros(num_sources)
            .map_err(|e| GpuError::BufferAllocation(format!("output: {}", e)))?;

        let num_threads = num_sources as u32;
        let cfg = launch_config(num_threads, self.config.workgroup_size);

        unsafe {
            self.stream
                .launch_builder(&self.fn_garstang)
                .arg(&buf_sources)
                .arg(&buf_config)
                .arg(&mut buf_output)
                .arg(&num_threads)
                .launch(cfg)
                .map_err(|e| GpuError::Dispatch(format!("garstang launch: {}", e)))?;
        }

        self.stream
            .synchronize()
            .map_err(|e| GpuError::Dispatch(format!("synchronize: {}", e)))?;

        let results = self
            .stream
            .clone_dtoh(&buf_output)
            .map_err(|e| GpuError::Readback(format!("garstang readback: {}", e)))?;

        let total: f64 = results.iter().map(|&v| v as f64).sum();
        Ok(total)
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Build a 1D launch configuration from total threads and block size.
fn launch_config(total_threads: u32, block_size: u32) -> LaunchConfig {
    let grid_x = dispatch_groups(total_threads, block_size);
    LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    }
}
