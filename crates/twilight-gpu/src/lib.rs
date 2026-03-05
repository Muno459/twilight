//! GPU compute backends for the Twilight MCRT engine.
//!
//! Provides four native GPU backends for maximum performance on every platform:
//!
//! - **CUDA** (`.cu` shaders, `cudarc` host) -- NVIDIA GPUs
//! - **Metal** (`.metal` shaders, `objc2-metal` host) -- Apple GPUs
//! - **Vulkan** (GLSL->SPIR-V, `ash` host) -- AMD/Intel/Android/Linux
//! - **wgpu** (`.wgsl` shaders) -- WASM/browsers only
//!
//! All backends implement the [`GpuBackend`] trait and share the same buffer
//! packing code ([`buffers`]), so the CPU reference engine's `f64` atmosphere
//! model is converted to GPU-friendly `f32` layouts exactly once.
//!
//! # Feature gates
//!
//! Each backend is behind a cargo feature:
//!
//! ```toml
//! twilight-gpu = { version = "0.1", features = ["metal"] }
//! ```
//!
//! Enable multiple backends to get automatic fallback via [`detect_backends`].

pub mod buffers;

#[cfg(feature = "metal")]
pub mod metal;

#[cfg(feature = "vulkan")]
pub mod vulkan;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(test)]
mod oracle;

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
    Cuda,
    Metal,
    Vulkan,
    Wgpu,
}

impl core::fmt::Display for BackendKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            BackendKind::Cuda => write!(f, "CUDA"),
            BackendKind::Metal => write!(f, "Metal"),
            BackendKind::Vulkan => write!(f, "Vulkan"),
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

/// Trait implemented by each GPU backend (CUDA, Metal, Vulkan, wgpu).
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
}

// ── Backend auto-detection ──────────────────────────────────────────────

/// Detect which GPU backends are available at runtime.
///
/// Returns a list of `BackendKind` values sorted by preference:
/// - On macOS/iOS: `[Metal]`
/// - On Windows with NVIDIA: `[Cuda, Vulkan]`
/// - On Windows with AMD/Intel: `[Vulkan]`
/// - On Linux: `[Cuda, Vulkan]` or `[Vulkan]`
/// - On WASM: `[Wgpu]`
///
/// This only checks whether the feature was compiled in and makes a
/// lightweight probe (e.g., can we open a device). It does not compile
/// shaders or allocate buffers.
pub fn detect_backends() -> Vec<BackendKind> {
    #[allow(unused_mut)]
    let mut available = Vec::new();

    // CUDA: check if feature is compiled and driver is accessible
    #[cfg(feature = "cuda")]
    {
        if probe_cuda() {
            available.push(BackendKind::Cuda);
        }
    }

    // Metal: check if feature is compiled (always works on macOS/iOS)
    #[cfg(feature = "metal")]
    {
        if probe_metal() {
            available.push(BackendKind::Metal);
        }
    }

    // Vulkan: check if feature is compiled and a device is available
    #[cfg(feature = "vulkan")]
    {
        if probe_vulkan() {
            available.push(BackendKind::Vulkan);
        }
    }

    // wgpu: check if feature is compiled
    #[cfg(feature = "webgpu")]
    {
        available.push(BackendKind::Wgpu);
    }

    available
}

/// Select the best available backend, respecting user preference.
///
/// If `preferred` is `Some` and that backend is available, use it.
/// Otherwise, pick the first available from [`detect_backends`].
pub fn select_backend(preferred: Option<BackendKind>) -> Option<BackendKind> {
    let available = detect_backends();

    if let Some(pref) = preferred {
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
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => init_cuda(config),
        #[cfg(feature = "metal")]
        BackendKind::Metal => init_metal(config),
        #[cfg(feature = "vulkan")]
        BackendKind::Vulkan => init_vulkan(config),
        #[cfg(feature = "webgpu")]
        BackendKind::Wgpu => init_wgpu(config),
        #[allow(unreachable_patterns)]
        _ => Err(GpuError::BackendUnavailable(kind)),
    }
}

// ── Probe functions (lightweight device checks) ─────────────────────────

#[cfg(feature = "cuda")]
fn probe_cuda() -> bool {
    cuda::probe()
}

#[cfg(feature = "metal")]
fn probe_metal() -> bool {
    metal::probe()
}

#[cfg(feature = "vulkan")]
fn probe_vulkan() -> bool {
    vulkan::probe()
}

// ── Backend init functions (stubs for Phase 11a) ────────────────────────

#[cfg(feature = "cuda")]
fn init_cuda(config: &GpuConfig) -> Result<Box<dyn GpuBackend>, GpuError> {
    cuda::init(config)
}

#[cfg(feature = "metal")]
fn init_metal(config: &GpuConfig) -> Result<Box<dyn GpuBackend>, GpuError> {
    metal::init(config)
}

#[cfg(feature = "vulkan")]
fn init_vulkan(config: &GpuConfig) -> Result<Box<dyn GpuBackend>, GpuError> {
    vulkan::init(config)
}

#[cfg(feature = "webgpu")]
fn init_wgpu(_config: &GpuConfig) -> Result<Box<dyn GpuBackend>, GpuError> {
    Err(GpuError::BackendUnavailable(BackendKind::Wgpu))
}
