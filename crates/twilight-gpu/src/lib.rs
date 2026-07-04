//! GPU compute backend for the Twilight MCRT engine.
//!
//! Two backends exist, both implementing the [`GpuBackend`] trait over the
//! shared buffer packing in [`buffers`]:
//!
//! - **Metal** (`shaders/twilight.metal`, `objc2-metal` host): the original
//!   reference implementation for Apple GPUs.
//! - **wgpu** (`shaders/twilight.wgsl`, `wgpu` host): a portable WGSL
//!   translation of the same kernels that runs on any wgpu-supported
//!   adapter (Vulkan, DX12, Metal-via-wgpu, GL), unlocking headless Linux
//!   servers and NVIDIA hardware.
//!
//! Both backends pack the CPU reference engine's `f64` atmosphere model
//! into identical f32 layouts (byte-for-byte the same buffers), so the two
//! GPU stacks are directly comparable against the one CPU specification.
//!
//! # Feature gates
//!
//! ```toml
//! twilight-gpu = { version = "0.1", features = ["metal"] }          # Apple
//! twilight-gpu = { version = "0.1", features = ["wgpu"] }           # portable
//! twilight-gpu = { version = "0.1", features = ["metal", "wgpu"] }  # both
//! ```
//!
//! Without either feature this crate only provides the buffer-packing
//! layer and the [`GpuBackend`] trait.
//!
//! # Backend selection
//!
//! [`try_init`] prefers Metal when both backends are compiled in and a
//! Metal device is present (macOS behavior unchanged). The environment
//! variable `TWILIGHT_GPU_BACKEND=wgpu|metal` overrides both the built-in
//! order and `GpuConfig::preferred_backend` -- its purpose is testing the
//! wgpu backend on machines where Metal would otherwise win.

pub mod buffers;

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "wgpu")]
pub mod wgpu_backend;

#[cfg(test)]
mod oracle;

#[cfg(test)]
pub(crate) mod parity;

#[cfg(test)]
mod tests;

// ── Error types ─────────────────────────────────────────────────────────

/// Errors that can occur during GPU backend initialization or dispatch.
#[derive(Debug)]
pub enum GpuError {
    /// No suitable GPU device was found.
    NoDevice,
    /// The requested backend is not available (feature not compiled or driver missing).
    BackendUnavailable(BackendKind),
    /// Shader compilation failed.
    ShaderCompilation(String),
    /// Buffer allocation failed (e.g., out of GPU memory).
    BufferAllocation(String),
    /// Kernel dispatch / command submission failed.
    Dispatch(String),
    /// Data readback from GPU failed.
    Readback(String),
    /// A kernel rejected a buffer whose header magic/version did not match
    /// the layout this host was compiled against (stale or corrupt upload).
    BufferVersionMismatch,
    /// Timeout waiting for GPU results.
    Timeout,
    /// Generic platform-specific error.
    Platform(String),
}

impl core::fmt::Display for GpuError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GpuError::NoDevice => write!(f, "no suitable GPU device found"),
            GpuError::BackendUnavailable(kind) => {
                write!(f, "GPU backend not available: {:?}", kind)
            }
            GpuError::ShaderCompilation(msg) => write!(f, "shader compilation failed: {}", msg),
            GpuError::BufferAllocation(msg) => write!(f, "GPU buffer allocation failed: {}", msg),
            GpuError::Dispatch(msg) => write!(f, "GPU dispatch failed: {}", msg),
            GpuError::Readback(msg) => write!(f, "GPU readback failed: {}", msg),
            GpuError::BufferVersionMismatch => write!(
                f,
                "GPU buffer magic/version mismatch: kernel rejected a stale or \
                 corrupt buffer (expected version {})",
                buffers::BUFFER_VERSION,
            ),
            GpuError::Timeout => write!(f, "GPU operation timed out"),
            GpuError::Platform(msg) => write!(f, "GPU platform error: {}", msg),
        }
    }
}

impl std::error::Error for GpuError {}

// ── Backend identification ──────────────────────────────────────────────

/// Which GPU backend is being used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    Metal,
    /// Portable WebGPU backend (Vulkan / DX12 / Metal-via-wgpu / GL).
    Wgpu,
}

impl core::fmt::Display for BackendKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BackendKind::Metal => write!(f, "Metal"),
            BackendKind::Wgpu => write!(f, "wgpu"),
        }
    }
}

// ── GPU device info ─────────────────────────────────────────────────────

/// Summary information about the selected GPU device.
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    /// Human-readable device name (e.g., "NVIDIA GeForce RTX 4090").
    pub name: String,
    /// Backend used to access this device.
    pub backend: BackendKind,
    /// Approximate total device memory in bytes (0 if unknown).
    pub memory_bytes: u64,
    /// Maximum workgroup (threadgroup / block) size in the X dimension.
    pub max_workgroup_size: u32,
}

// ── Configuration ───────────────────────────────────────────────────────

/// Configuration for GPU dispatch.
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Preferred backend. If `None`, auto-detect the best available.
    pub preferred_backend: Option<BackendKind>,
    /// Workgroup size (threads per group). Default: 256.
    pub workgroup_size: u32,
    /// Number of photons per wavelength for MC kernels.
    pub photons_per_wavelength: u32,
    /// Number of secondary rays per LOS step for hybrid kernel.
    pub secondary_rays_per_step: u32,
    /// RNG base seed.
    pub rng_seed: u64,
    /// Enable debug buffer output (shader printf / debug values).
    pub debug: bool,
    /// Enable full Stokes [I,Q,U,V] polarization tracking (default: true).
    ///
    /// When true (the default), the GPU shaders propagate full 4-component
    /// Stokes vectors through scattering events, capturing polarization-
    /// intensity coupling (Rayleigh + aerosol Mueller matrices).
    ///
    /// When false (`--fast` mode), the shaders use scalar radiance tracking
    /// (P11 phase function only). This is slightly faster but loses the
    /// ~0.5-2% polarization correction to intensity.
    ///
    /// NOTE: GPU shaders currently always run Stokes internally. This flag
    /// is reserved for a future scalar shader path optimization.
    pub polarized: bool,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            preferred_backend: None,
            workgroup_size: 256,
            photons_per_wavelength: 10_000,
            secondary_rays_per_step: 100,
            rng_seed: 42,
            debug: false,
            polarized: true,
        }
    }
}

// ── GPU backend trait ───────────────────────────────────────────────────

/// Result of a single-scatter or hybrid spectral computation on GPU.
///
/// Radiance values are in the same units as the CPU reference engine
/// (proportional to W/m^2/sr/nm before solar irradiance weighting).
#[derive(Debug, Clone)]
pub struct GpuSpectralResult {
    /// Spectral radiance per wavelength. Length = `num_wavelengths`.
    pub radiance: Vec<f64>,
    /// Number of active wavelengths.
    pub num_wavelengths: usize,
}

/// Which kernel to dispatch for a batched scan request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchKernel {
    /// Deterministic single-scatter kernel.
    SingleScatter,
    /// Full backward MC photon tracing kernel.
    McrtTrace {
        photons_per_wavelength: u32,
        seed: u64,
    },
    /// Hybrid single-scatter + MC secondary chain kernel.
    Hybrid {
        secondary_rays: u32,
        seed: u64,
    },
}

/// A single request within a batched GPU scan.
///
/// Each request represents one SZA point with its pre-computed geometry.
#[derive(Debug, Clone)]
pub struct BatchRequest {
    pub observer_pos: [f64; 3],
    pub view_dir: [f64; 3],
    pub sun_dir: [f64; 3],
    pub kernel: BatchKernel,
}

/// Trait implemented by GPU backends (currently only Metal).
///
/// The lifecycle is:
/// 1. `try_init()` -- probe for hardware, compile shaders, allocate pipeline
/// 2. `upload_atmosphere()` -- pack and upload the atmosphere model
/// 3. `single_scatter()` / `mcrt_trace()` / `hybrid_scatter()` -- dispatch
/// 4. Drop -- release all GPU resources
///
/// All methods that can fail return `Result<_, GpuError>`. If `try_init()`
/// fails, the caller falls back to the CPU engine.
pub trait GpuBackend: Send {
    /// Return information about the selected device.
    fn device_info(&self) -> &GpuDeviceInfo;

    /// Upload an atmosphere model to GPU memory.
    ///
    /// This packs the f64 `AtmosphereModel` into f32 GPU buffers using
    /// the layout defined in [`buffers`]. Subsequent kernel dispatches
    /// use these buffers until `upload_atmosphere` is called again.
    fn upload_atmosphere(
        &mut self,
        atm: &twilight_core::atmosphere::AtmosphereModel,
    ) -> Result<(), GpuError>;

    /// Upload (or clear) the 3D cloud field.
    ///
    /// `Some(field)` packs and binds the voxel field so subsequent hybrid
    /// dispatches take the gray cloud channel (Beer-Lambert, explicit
    /// in-cloud scattering, Stage 3). `None` clears any bound field and
    /// restores the legacy 1D shell-cloud path (Eddington diffuse
    /// transmittance, unchanged).
    ///
    /// The default implementation is a no-op (`None` accepted, `Some`
    /// rejected) for backends without field support.
    fn upload_field(
        &mut self,
        field: Option<&twilight_core::cloud_field::Cloud3DField>,
    ) -> Result<(), GpuError> {
        match field {
            None => Ok(()),
            Some(_) => Err(GpuError::Dispatch(
                "this backend does not support 3D cloud fields".into(),
            )),
        }
    }

    /// Run the deterministic single-scatter spectrum kernel.
    ///
    /// Equivalent to `twilight_core::single_scatter::single_scatter_spectrum`.
    fn single_scatter(
        &self,
        observer_pos: [f64; 3],
        view_dir: [f64; 3],
        sun_dir: [f64; 3],
    ) -> Result<GpuSpectralResult, GpuError>;

    /// Run the backward MC photon tracing kernel.
    ///
    /// Equivalent to `twilight_core::photon::mc_scatter_spectrum`.
    fn mcrt_trace(
        &self,
        observer_pos: [f64; 3],
        view_dir: [f64; 3],
        sun_dir: [f64; 3],
        photons_per_wavelength: u32,
        seed: u64,
    ) -> Result<GpuSpectralResult, GpuError>;

    /// Run the hybrid single-scatter + MC secondary chain kernel.
    ///
    /// Equivalent to `twilight_core::photon::hybrid_scatter_spectrum`.
    fn hybrid_scatter(
        &self,
        observer_pos: [f64; 3],
        view_dir: [f64; 3],
        sun_dir: [f64; 3],
        secondary_rays: u32,
        seed: u64,
    ) -> Result<GpuSpectralResult, GpuError>;

    /// Run the Garstang zenith skyglow kernel (light pollution).
    ///
    /// Returns artificial zenith brightness in cd/m^2.
    fn garstang_zenith(
        &self,
        observer_pos: [f64; 3],
        sources: &[buffers::PackedLightSource],
    ) -> Result<f64, GpuError>;

    /// Dispatch multiple SZA points in a single GPU submission.
    ///
    /// Encodes all N dispatches into one command buffer, avoiding the
    /// per-dispatch synchronization overhead of serial dispatch for prayer
    /// pipeline scans (~50 SZA points).
    ///
    /// The default implementation falls back to serial dispatch for backends
    /// that haven't implemented batching yet.
    fn scan_batch(
        &self,
        requests: &[BatchRequest],
    ) -> Result<Vec<GpuSpectralResult>, GpuError> {
        // Default: serial fallback -- call per-SZA methods in a loop.
        let mut results = Vec::with_capacity(requests.len());
        for req in requests {
            let r = match req.kernel {
                BatchKernel::SingleScatter => {
                    self.single_scatter(req.observer_pos, req.view_dir, req.sun_dir)?
                }
                BatchKernel::McrtTrace {
                    photons_per_wavelength,
                    seed,
                } => self.mcrt_trace(
                    req.observer_pos,
                    req.view_dir,
                    req.sun_dir,
                    photons_per_wavelength,
                    seed,
                )?,
                BatchKernel::Hybrid {
                    secondary_rays,
                    seed,
                } => self.hybrid_scatter(
                    req.observer_pos,
                    req.view_dir,
                    req.sun_dir,
                    secondary_rays,
                    seed,
                )?,
            };
            results.push(r);
        }
        Ok(results)
    }
}

// ── Backend auto-detection ──────────────────────────────────────────────

/// Detect which GPU backends are available at runtime.
///
/// Metal is listed FIRST when available (Apple platforms, `metal` feature):
/// the pre-existing preference order is unchanged by the wgpu port. The
/// wgpu backend is listed when its feature is compiled in and an adapter
/// responds to a lightweight probe. Does not compile shaders or allocate
/// buffers.
pub fn detect_backends() -> Vec<BackendKind> {
    #[allow(unused_mut)]
    let mut available = Vec::new();

    // Metal: check if feature is compiled (always works on macOS/iOS)
    #[cfg(feature = "metal")]
    {
        if probe_metal() {
            available.push(BackendKind::Metal);
        }
    }

    // wgpu: any adapter on any backend (Vulkan / DX12 / Metal / GL).
    #[cfg(feature = "wgpu")]
    {
        if probe_wgpu() {
            available.push(BackendKind::Wgpu);
        }
    }

    available
}

/// Backend preference from the `TWILIGHT_GPU_BACKEND` environment variable
/// (`wgpu` or `metal`, case-insensitive). Unset or unrecognized => None.
/// This override exists so the wgpu backend can be exercised on machines
/// where Metal would otherwise be selected (wgpu then runs over its own
/// Metal driver: two independent GPU stacks against one CPU spec).
fn env_backend_override() -> Option<BackendKind> {
    let val = std::env::var("TWILIGHT_GPU_BACKEND").ok()?;
    match val.to_ascii_lowercase().as_str() {
        "metal" => Some(BackendKind::Metal),
        "wgpu" => Some(BackendKind::Wgpu),
        other => {
            eprintln!(
                "Warning: TWILIGHT_GPU_BACKEND={other:?} not recognized \
                 (expected 'metal' or 'wgpu'); ignoring"
            );
            None
        }
    }
}

/// Select the best available backend, respecting user preference.
///
/// Precedence: `TWILIGHT_GPU_BACKEND` env override, then `preferred`,
/// then the first available from [`detect_backends`] (Metal first on
/// Apple platforms when both are compiled in).
pub fn select_backend(preferred: Option<BackendKind>) -> Option<BackendKind> {
    let available = detect_backends();

    if let Some(pref) = env_backend_override().or(preferred) {
        if available.contains(&pref) {
            return Some(pref);
        }
    }

    available.into_iter().next()
}

/// Try to initialize the best available GPU backend.
///
/// Returns `Ok(Box<dyn GpuBackend>)` on success, or `Err(GpuError)` if
/// no backend could be initialized. The caller should fall back to the
/// CPU engine on error.
pub fn try_init(config: &GpuConfig) -> Result<Box<dyn GpuBackend>, GpuError> {
    let kind = match select_backend(config.preferred_backend) {
        Some(k) => k,
        None => return Err(GpuError::NoDevice),
    };

    match kind {
        #[cfg(feature = "metal")]
        BackendKind::Metal => init_metal(config),
        #[cfg(feature = "wgpu")]
        BackendKind::Wgpu => init_wgpu(config),
        #[allow(unreachable_patterns)]
        _ => Err(GpuError::BackendUnavailable(kind)),
    }
}

// ── Probe / init (lightweight device checks) ────────────────────────────

#[cfg(feature = "metal")]
fn probe_metal() -> bool {
    metal::probe()
}

#[cfg(feature = "metal")]
fn init_metal(config: &GpuConfig) -> Result<Box<dyn GpuBackend>, GpuError> {
    metal::init(config)
}

#[cfg(feature = "wgpu")]
fn probe_wgpu() -> bool {
    wgpu_backend::probe()
}

#[cfg(feature = "wgpu")]
fn init_wgpu(config: &GpuConfig) -> Result<Box<dyn GpuBackend>, GpuError> {
    wgpu_backend::init(config)
}
