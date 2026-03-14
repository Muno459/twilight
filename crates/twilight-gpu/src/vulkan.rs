//! Vulkan GPU backend for AMD, Intel, Android, Linux, and macOS (via MoltenVK).
//!
//! Loads pre-compiled SPIR-V shaders (built from GLSL at compile time via
//! `include_bytes!`), creates four compute pipelines, and dispatches work
//! using host-visible storage buffers for zero-copy data transfer.

use std::cell::UnsafeCell;
use std::ffi::CStr;
use std::mem::ManuallyDrop;
use std::sync::OnceLock;

use ash::vk;
use gpu_allocator::vulkan::{
    Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc,
};
use gpu_allocator::MemoryLocation;

// ── Cached Vulkan entry point ───────────────────────────────────────────
//
// Loading the Vulkan library with `ash::Entry::load()` calls `dlopen`.
// Dropping the Entry calls `dlclose`. On macOS, MoltenVK does not survive
// being dlclose'd and dlopen'd in the same process (SIGSEGV). We cache
// the Entry in a OnceLock so it's loaded once and never dropped.

static VK_ENTRY: OnceLock<Result<ash::Entry, String>> = OnceLock::new();

fn load_entry() -> Result<&'static ash::Entry, String> {
    VK_ENTRY
        .get_or_init(|| unsafe { ash::Entry::load() }.map_err(|e| format!("{}", e)))
        .as_ref()
        .map_err(|e| e.clone())
}

use crate::buffers::{
    dispatch_groups, PackedAtmosphere, PackedDispatchParams, PackedLightSource,
    PackedSolarSpectrum, PackedVisionLuts,
};
use crate::{
    BackendKind, BatchKernel, BatchRequest, GpuBackend, GpuConfig, GpuDeviceInfo, GpuError,
    GpuSpectralResult,
};

/// Workgroup size for the hybrid kernel. Must match HYBRID_THREADGROUP_SIZE
/// in twilight.comp (256 threads per workgroup).
#[allow(dead_code)]
const HYBRID_WORKGROUP_SIZE: u32 = 256;

// ── Embedded SPIR-V shaders ────────────────────────────────────────────

const SPV_SINGLE_SCATTER: &[u8] = include_bytes!("../shaders/single_scatter.spv");
const SPV_MCRT_TRACE: &[u8] = include_bytes!("../shaders/mcrt_trace.spv");
const SPV_HYBRID: &[u8] = include_bytes!("../shaders/hybrid_scatter.spv");
const SPV_GARSTANG: &[u8] = include_bytes!("../shaders/garstang_zenith.spv");

// ── Helper: bytes to u32 words for SPIR-V ──────────────────────────────

fn spv_to_words(bytes: &[u8]) -> Vec<u32> {
    assert!(bytes.len().is_multiple_of(4), "SPIR-V must be 4-byte aligned");
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// ── Vulkan buffer with allocation ──────────────────────────────────────

struct VkBuffer {
    buffer: vk::Buffer,
    allocation: Option<Allocation>,
}

// ── Vulkan backend ─────────────────────────────────────────────────────

/// Vulkan backend implementing the [`GpuBackend`] trait.
///
/// Uses `ash` for the Vulkan API and `gpu-allocator` for memory management.
/// All four compute kernels are loaded from pre-compiled SPIR-V at init time.
pub struct VulkanBackend {
    // Entry is now cached in VK_ENTRY OnceLock (never dropped).
    // We keep a reference lifetime tied to 'static.
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,

    // Descriptor set layout shared by all kernels (3 storage buffers)
    desc_set_layout: vk::DescriptorSetLayout,
    // Descriptor pool for per-dispatch descriptor sets
    desc_pool: vk::DescriptorPool,

    // Four compute pipelines
    pipeline_layout: vk::PipelineLayout,
    pipeline_single_scatter: vk::Pipeline,
    pipeline_mcrt_trace: vk::Pipeline,
    pipeline_hybrid: vk::Pipeline,
    pipeline_garstang: vk::Pipeline,

    // Command pool for compute dispatches
    cmd_pool: vk::CommandPool,
    // Reusable fence for synchronization
    fence: vk::Fence,

    // Memory allocator (UnsafeCell because gpu-allocator requires &mut
    // for allocate/free, but our dispatch methods take &self. Safety is
    // ensured by single-threaded dispatch with fence synchronization.)
    //
    // ManuallyDrop because the Allocator must be dropped BEFORE
    // destroy_device() -- its Drop accesses the Vulkan device internally.
    allocator: UnsafeCell<ManuallyDrop<Allocator>>,

    // Uploaded atmosphere buffer (persisted between dispatches)
    buf_atm: Option<VkBuffer>,
    // Solar and vision LUTs (reserved for Phase 11f pipeline integration)
    #[allow(dead_code)]
    buf_solar: VkBuffer,
    #[allow(dead_code)]
    buf_vision: VkBuffer,

    info: GpuDeviceInfo,
    config: GpuConfig,
    num_wavelengths: u32,
}

// Safety: Vulkan handles and the gpu-allocator are thread-safe when
// externally synchronized (we use fences and wait-idle before readback).
unsafe impl Send for VulkanBackend {}

// ── Probe ───────────────────────────────────────────────────────────────

/// Lightweight probe: can we load the Vulkan library and find a compute device?
pub fn probe() -> bool {
    let entry = match load_entry() {
        Ok(e) => e,
        Err(_) => return false,
    };

    let app_info = vk::ApplicationInfo::default()
        .api_version(vk::make_api_version(0, 1, 1, 0));

    // macOS needs VK_KHR_portability_enumeration for MoltenVK
    let mut create_flags = vk::InstanceCreateFlags::empty();
    let mut extensions: Vec<*const i8> = Vec::new();

    #[cfg(target_os = "macos")]
    {
        create_flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        extensions.push(
            c"VK_KHR_portability_enumeration".as_ptr(),
        );
    }

    let instance_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .flags(create_flags)
        .enabled_extension_names(&extensions);

    let instance = match unsafe { entry.create_instance(&instance_info, None) } {
        Ok(i) => i,
        Err(_) => return false,
    };

    let has_compute = unsafe { instance.enumerate_physical_devices() }
        .unwrap_or_default()
        .iter()
        .any(|&pd| {
            unsafe { instance.get_physical_device_queue_family_properties(pd) }
                .iter()
                .any(|qf| qf.queue_flags.contains(vk::QueueFlags::COMPUTE))
        });

    unsafe { instance.destroy_instance(None) };
    has_compute
}

// ── Init ────────────────────────────────────────────────────────────────

/// Initialize the Vulkan backend: load library, create device, compile
/// pipelines from embedded SPIR-V, allocate command pool and fence.
pub fn init(config: &GpuConfig) -> Result<Box<dyn GpuBackend>, GpuError> {
    // 1. Load Vulkan entry (cached -- never dropped to avoid MoltenVK dlclose crash)
    let entry = load_entry()
        .map_err(|e| GpuError::Platform(format!("failed to load Vulkan library: {}", e)))?;

    // 2. Create instance
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"twilight-mcrt")
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(c"twilight-gpu")
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::make_api_version(0, 1, 1, 0));

    let mut create_flags = vk::InstanceCreateFlags::empty();
    let mut instance_extensions: Vec<*const i8> = Vec::new();

    #[cfg(target_os = "macos")]
    {
        create_flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        instance_extensions.push(
            c"VK_KHR_portability_enumeration".as_ptr(),
        );
    }

    let instance_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .flags(create_flags)
        .enabled_extension_names(&instance_extensions);

    let instance = unsafe { entry.create_instance(&instance_info, None) }
        .map_err(|e| GpuError::Platform(format!("failed to create Vulkan instance: {:?}", e)))?;

    // 3. Select physical device with compute capability
    let physical_devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(|e| GpuError::Platform(format!("failed to enumerate devices: {:?}", e)))?;

    let (physical_device, queue_family_index) = physical_devices
        .iter()
        .find_map(|&pd| {
            let qf_props =
                unsafe { instance.get_physical_device_queue_family_properties(pd) };
            qf_props
                .iter()
                .enumerate()
                .find(|(_, qf)| qf.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|(idx, _)| (pd, idx as u32))
        })
        .ok_or(GpuError::NoDevice)?;

    let device_props = unsafe { instance.get_physical_device_properties(physical_device) };
    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let device_name = unsafe { CStr::from_ptr(device_props.device_name.as_ptr()) }
        .to_string_lossy()
        .to_string();

    // Compute total device-local memory
    let device_memory: u64 = (0..mem_props.memory_heap_count as usize)
        .filter(|&i| {
            mem_props.memory_heaps[i]
                .flags
                .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
        })
        .map(|i| mem_props.memory_heaps[i].size)
        .sum();

    // 4. Create logical device with compute queue
    let queue_priorities = [1.0f32];
    let queue_create_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&queue_priorities);

    // macOS MoltenVK needs VK_KHR_portability_subset
    let mut device_extensions: Vec<*const i8> = Vec::new();

    #[cfg(target_os = "macos")]
    {
        let ext_props = unsafe {
            instance
                .enumerate_device_extension_properties(physical_device)
                .unwrap_or_default()
        };
        let has_portability = ext_props.iter().any(|e| {
            let name = unsafe { CStr::from_ptr(e.extension_name.as_ptr()) };
            name.to_bytes() == b"VK_KHR_portability_subset"
        });
        if has_portability {
            device_extensions.push(
                c"VK_KHR_portability_subset".as_ptr(),
            );
        }
    }

    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(std::slice::from_ref(&queue_create_info))
        .enabled_extension_names(&device_extensions);

    let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }
        .map_err(|e| GpuError::Platform(format!("failed to create Vulkan device: {:?}", e)))?;

    let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

    // 5. Create memory allocator
    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance: instance.clone(),
        device: device.clone(),
        physical_device,
        debug_settings: Default::default(),
        buffer_device_address: false,
        allocation_sizes: Default::default(),
    })
    .map_err(|e| GpuError::Platform(format!("failed to create GPU allocator: {}", e)))?;

    // 6. Create descriptor set layout (3 storage buffers: input, params, output)
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];

    let desc_layout_info =
        vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    let desc_set_layout = unsafe { device.create_descriptor_set_layout(&desc_layout_info, None) }
        .map_err(|e| GpuError::Platform(format!("descriptor set layout: {:?}", e)))?;

    // 7. Create descriptor pool (enough for many dispatches)
    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::STORAGE_BUFFER,
        descriptor_count: 3 * 64, // 3 buffers x up to 64 dispatches
    }];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
        .max_sets(64)
        .pool_sizes(&pool_sizes);
    let desc_pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
        .map_err(|e| GpuError::Platform(format!("descriptor pool: {:?}", e)))?;

    // 8. Create pipeline layout
    let layouts = [desc_set_layout];
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
        .map_err(|e| GpuError::Platform(format!("pipeline layout: {:?}", e)))?;

    // 9. Create compute pipelines from SPIR-V
    let pipeline_single_scatter =
        create_compute_pipeline(&device, pipeline_layout, SPV_SINGLE_SCATTER, "single_scatter")?;
    let pipeline_mcrt_trace =
        create_compute_pipeline(&device, pipeline_layout, SPV_MCRT_TRACE, "mcrt_trace")?;
    let pipeline_hybrid =
        create_compute_pipeline(&device, pipeline_layout, SPV_HYBRID, "hybrid_scatter")?;
    let pipeline_garstang =
        create_compute_pipeline(&device, pipeline_layout, SPV_GARSTANG, "garstang_zenith")?;

    // 10. Create command pool and fence
    let cmd_pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let cmd_pool = unsafe { device.create_command_pool(&cmd_pool_info, None) }
        .map_err(|e| GpuError::Platform(format!("command pool: {:?}", e)))?;

    let fence_info = vk::FenceCreateInfo::default();
    let fence = unsafe { device.create_fence(&fence_info, None) }
        .map_err(|e| GpuError::Platform(format!("fence: {:?}", e)))?;

    // 11. Pack and upload constant buffers
    let solar = PackedSolarSpectrum::pack();
    let vision = PackedVisionLuts::pack();

    let buf_solar = create_buffer_from_f32(&device, &mut allocator, &solar.data, "solar_spectrum")?;
    let buf_vision = create_buffer_from_f32(&device, &mut allocator, &vision.data, "vision_luts")?;

    // 12. Build device info
    let max_wg = device_props.limits.max_compute_work_group_size[0];
    let info = GpuDeviceInfo {
        name: device_name,
        backend: BackendKind::Vulkan,
        memory_bytes: device_memory,
        max_workgroup_size: max_wg,
    };

    Ok(Box::new(VulkanBackend {
        instance,
        device,
        queue,
        desc_set_layout,
        desc_pool,
        pipeline_layout,
        pipeline_single_scatter,
        pipeline_mcrt_trace,
        pipeline_hybrid,
        pipeline_garstang,
        cmd_pool,
        fence,
        allocator: UnsafeCell::new(ManuallyDrop::new(allocator)),
        buf_atm: None,
        buf_solar,
        buf_vision,
        info,
        config: config.clone(),
        num_wavelengths: 0,
    }))
}

// ── GpuBackend implementation ───────────────────────────────────────────

impl GpuBackend for VulkanBackend {
    fn device_info(&self) -> &GpuDeviceInfo {
        &self.info
    }

    fn upload_atmosphere(
        &mut self,
        atm: &twilight_core::atmosphere::AtmosphereModel,
    ) -> Result<(), GpuError> {
        // Free previous atmosphere buffer
        if let Some(old) = self.buf_atm.take() {
            free_buffer(&self.device, self.allocator.get_mut(), old);
        }

        let packed = PackedAtmosphere::pack(atm);
        self.num_wavelengths = packed.num_wavelengths;
        self.buf_atm = Some(create_buffer_from_f32(
            &self.device,
            self.allocator.get_mut(),
            &packed.data,
            "atmosphere",
        )?);
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
        let buf_params =
            create_buffer_from_f32(&self.device, self.allocator_mut(), &params.data, "params")?;

        let nw = self.num_wavelengths as usize;
        let buf_output = create_empty_buffer(&self.device, self.allocator_mut(), nw, "output")?;

        self.dispatch_kernel(
            self.pipeline_single_scatter,
            &[buf_atm, &buf_params, &buf_output],
            nw as u32,
        )?;

        let radiance = read_f32_buffer(&buf_output, nw);

        // Clean up transient buffers (cast away shared ref -- see safety note below)
        self.free_transient_buffers(vec![buf_params, buf_output]);

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
        let buf_params =
            create_buffer_from_f32(&self.device, self.allocator_mut(), &params.data, "params")?;

        let nw = self.num_wavelengths as usize;
        let total_threads = nw * photons_per_wavelength as usize;
        let buf_output =
            create_empty_buffer(&self.device, self.allocator_mut(), total_threads, "output")?;

        self.dispatch_kernel(
            self.pipeline_mcrt_trace,
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

        self.free_transient_buffers(vec![buf_params, buf_output]);

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
        let buf_params =
            create_buffer_from_f32(&self.device, self.allocator_mut(), &params.data, "params")?;

        let nw = self.num_wavelengths as usize;
        let buf_output = create_empty_buffer(&self.device, self.allocator_mut(), nw, "output")?;

        // Hybrid v2: dispatch nw workgroups of HYBRID_WORKGROUP_SIZE threads.
        // Each workgroup handles one wavelength; threads within the group each
        // handle one LOS step with secondary chain tracing, then reduce via
        // subgroupAdd() + shared memory.
        self.dispatch_hybrid(
            self.pipeline_hybrid,
            &[buf_atm, &buf_params, &buf_output],
            nw as u32,
        )?;

        let radiance = read_f32_buffer(&buf_output, nw);
        self.free_transient_buffers(vec![buf_params, buf_output]);

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
        let buf_sources =
            create_buffer_from_f32(&self.device, self.allocator_mut(), &source_data, "sources")?;

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
        let buf_config =
            create_buffer_from_f32(&self.device, self.allocator_mut(), &config_data, "config")?;

        let buf_output =
            create_empty_buffer(&self.device, self.allocator_mut(), num_sources, "output")?;

        self.dispatch_kernel(
            self.pipeline_garstang,
            &[&buf_sources, &buf_config, &buf_output],
            num_sources as u32,
        )?;

        // Sum all source contributions on CPU
        let results = read_f32_buffer(&buf_output, num_sources);
        let total: f64 = results.iter().map(|&v| v as f64).sum();

        self.free_transient_buffers(vec![buf_sources, buf_config, buf_output]);

        Ok(total)
    }

    fn scan_batch(&self, requests: &[BatchRequest]) -> Result<Vec<GpuSpectralResult>, GpuError> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }

        let buf_atm = self
            .buf_atm
            .as_ref()
            .ok_or_else(|| GpuError::Dispatch("atmosphere not uploaded".into()))?;

        let nw = self.num_wavelengths as usize;
        let n = requests.len();
        let wg_size = self.config.workgroup_size;

        // ── Pre-allocate all per-dispatch buffers ───────────────────────
        //
        // Vulkan descriptor sets reference (buffer, offset, range) tuples.
        // We create per-dispatch params and output buffers, then record ALL
        // dispatches into a single command buffer with a single fence wait.
        // This eliminates the per-SZA submit+wait round trip that was the
        // bottleneck (50 round trips -> 1).

        struct DispatchSlice {
            params_buf: VkBuffer,
            output_buf: VkBuffer,
            output_len: usize,
            desc_set: vk::DescriptorSet,
            kernel: BatchKernel,
        }

        let mut slices = Vec::with_capacity(n);

        // Allocate N descriptor sets in one call.
        let layouts: Vec<vk::DescriptorSetLayout> = vec![self.desc_set_layout; n];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.desc_pool)
            .set_layouts(&layouts);
        let desc_sets = unsafe { self.device.allocate_descriptor_sets(&alloc_info) }
            .map_err(|e| GpuError::Dispatch(format!("batch descriptor set allocation: {:?}", e)))?;

        for (i, req) in requests.iter().enumerate() {
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

            let params = PackedDispatchParams::new(
                req.observer_pos,
                req.view_dir,
                req.sun_dir,
                ppw,
                sec,
                seed,
            );
            let params_buf = create_buffer_from_f32(
                &self.device,
                self.allocator_mut(),
                &params.data,
                "batch_params",
            )?;

            let output_len = match req.kernel {
                BatchKernel::McrtTrace {
                    photons_per_wavelength,
                    ..
                } => nw * photons_per_wavelength as usize,
                _ => nw,
            };
            let output_buf = create_empty_buffer(
                &self.device,
                self.allocator_mut(),
                output_len,
                "batch_output",
            )?;

            // Write descriptor set bindings for this dispatch.
            let buffer_infos = [
                vk::DescriptorBufferInfo::default()
                    .buffer(buf_atm.buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE),
                vk::DescriptorBufferInfo::default()
                    .buffer(params_buf.buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE),
                vk::DescriptorBufferInfo::default()
                    .buffer(output_buf.buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE),
            ];

            let writes: Vec<vk::WriteDescriptorSet> = buffer_infos
                .iter()
                .enumerate()
                .map(|(j, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(desc_sets[i])
                        .dst_binding(j as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info))
                })
                .collect();

            unsafe { self.device.update_descriptor_sets(&writes, &[]) };

            slices.push(DispatchSlice {
                params_buf,
                output_buf,
                output_len,
                desc_set: desc_sets[i],
                kernel: req.kernel,
            });
        }

        // ── Record all N dispatches into ONE command buffer ─────────────

        let cmd_alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_bufs = unsafe { self.device.allocate_command_buffers(&cmd_alloc_info) }
            .map_err(|e| GpuError::Dispatch(format!("batch command buffer: {:?}", e)))?;
        let cmd = cmd_bufs[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { self.device.begin_command_buffer(cmd, &begin_info) }
            .map_err(|e| GpuError::Dispatch(format!("begin batch command buffer: {:?}", e)))?;

        for s in &slices {
            let pipeline = match s.kernel {
                BatchKernel::SingleScatter => self.pipeline_single_scatter,
                BatchKernel::McrtTrace { .. } => self.pipeline_mcrt_trace,
                BatchKernel::Hybrid { .. } => self.pipeline_hybrid,
            };

            let is_hybrid = matches!(s.kernel, BatchKernel::Hybrid { .. });

            if !is_hybrid && s.output_len == 0 {
                continue;
            }

            // Hybrid: dispatch nw workgroups of HYBRID_WORKGROUP_SIZE threads.
            // Non-hybrid: flat dispatch with configurable workgroup size.
            let num_groups = if is_hybrid {
                nw as u32
            } else {
                let total_threads = s.output_len as u32;
                if total_threads == 0 {
                    continue;
                }
                dispatch_groups(total_threads, wg_size)
            };

            unsafe {
                self.device
                    .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
                self.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_layout,
                    0,
                    &[s.desc_set],
                    &[],
                );
                self.device.cmd_dispatch(cmd, num_groups, 1, 1);
            }

            // Memory barrier between dispatches: ensure writes from this
            // dispatch are visible to the next (they share the atm buffer
            // read-only, but the barrier ensures correct ordering).
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);

            unsafe {
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                );
            }
        }

        // Final barrier: make all compute writes visible to host reads.
        let host_barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);

        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[host_barrier],
                &[],
                &[],
            );
        }

        unsafe { self.device.end_command_buffer(cmd) }
            .map_err(|e| GpuError::Dispatch(format!("end batch command buffer: {:?}", e)))?;

        // ONE submit, ONE fence wait.
        let cmd_bufs_submit = [cmd];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_bufs_submit);

        unsafe { self.device.reset_fences(&[self.fence]) }
            .map_err(|e| GpuError::Dispatch(format!("reset fence: {:?}", e)))?;

        unsafe {
            self.device
                .queue_submit(self.queue, &[submit_info], self.fence)
        }
        .map_err(|e| GpuError::Dispatch(format!("batch queue submit: {:?}", e)))?;

        unsafe { self.device.wait_for_fences(&[self.fence], true, u64::MAX) }
            .map_err(|e| GpuError::Dispatch(format!("batch wait for fence: {:?}", e)))?;

        // ── Readback and reduce ─────────────────────────────────────────

        let mut results = Vec::with_capacity(n);

        for s in &slices {
            match s.kernel {
                BatchKernel::McrtTrace {
                    photons_per_wavelength,
                    ..
                } => {
                    let ppw = photons_per_wavelength as usize;
                    let raw = read_f32_buffer(&s.output_buf, s.output_len);
                    let mut radiance = Vec::with_capacity(nw);
                    for w in 0..nw {
                        let start = w * ppw;
                        let end = start + ppw;
                        let sum: f32 = raw[start..end].iter().sum();
                        radiance.push((sum / ppw as f32) as f64);
                    }
                    results.push(GpuSpectralResult {
                        radiance,
                        num_wavelengths: nw,
                    });
                }
                _ => {
                    let raw = read_f32_buffer(&s.output_buf, nw);
                    results.push(GpuSpectralResult {
                        radiance: raw.iter().map(|&v| v as f64).collect(),
                        num_wavelengths: nw,
                    });
                }
            }
        }

        // Clean up: free transient buffers, descriptor sets, command buffer.
        let mut transient_bufs = Vec::with_capacity(2 * n);
        let mut desc_set_list = Vec::with_capacity(n);
        for s in slices {
            desc_set_list.push(s.desc_set);
            transient_bufs.push(s.params_buf);
            transient_bufs.push(s.output_buf);
        }

        unsafe {
            self.device.free_command_buffers(self.cmd_pool, &[cmd]);
            let _ = self
                .device
                .free_descriptor_sets(self.desc_pool, &desc_set_list);
        }

        self.free_transient_buffers(transient_bufs);

        Ok(results)
    }
}

// ── Internal helpers ────────────────────────────────────────────────────

impl VulkanBackend {
    /// Get a mutable reference to the allocator via UnsafeCell.
    ///
    /// # Safety
    ///
    /// This is safe because:
    /// 1. We never have concurrent allocations (dispatch_kernel is synchronous)
    /// 2. VulkanBackend is not Sync (only Send), so no concurrent &self access
    /// 3. The allocator is only accessed through this method
    #[allow(clippy::mut_from_ref)] // Interior mutability via UnsafeCell is intentional
    fn allocator_mut(&self) -> &mut Allocator {
        unsafe { &mut *self.allocator.get() }
    }

    /// Dispatch a compute kernel with the given storage buffers.
    fn dispatch_kernel(
        &self,
        pipeline: vk::Pipeline,
        buffers: &[&VkBuffer],
        total_threads: u32,
    ) -> Result<(), GpuError> {
        if total_threads == 0 {
            return Ok(());
        }

        let wg_size = self.config.workgroup_size;
        let num_groups = dispatch_groups(total_threads, wg_size);

        // Allocate a descriptor set for this dispatch
        let layouts = [self.desc_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.desc_pool)
            .set_layouts(&layouts);
        let desc_sets = unsafe { self.device.allocate_descriptor_sets(&alloc_info) }
            .map_err(|e| GpuError::Dispatch(format!("descriptor set allocation: {:?}", e)))?;
        let desc_set = desc_sets[0];

        // Update descriptor set with buffer bindings
        let buffer_infos: Vec<vk::DescriptorBufferInfo> = buffers
            .iter()
            .map(|b| {
                vk::DescriptorBufferInfo::default()
                    .buffer(b.buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            })
            .collect();

        let writes: Vec<vk::WriteDescriptorSet> = buffer_infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();

        unsafe { self.device.update_descriptor_sets(&writes, &[]) };

        // Allocate and record command buffer
        let cmd_alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_bufs = unsafe { self.device.allocate_command_buffers(&cmd_alloc_info) }
            .map_err(|e| GpuError::Dispatch(format!("command buffer allocation: {:?}", e)))?;
        let cmd = cmd_bufs[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { self.device.begin_command_buffer(cmd, &begin_info) }
            .map_err(|e| GpuError::Dispatch(format!("begin command buffer: {:?}", e)))?;

        unsafe {
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[desc_set],
                &[],
            );
            self.device.cmd_dispatch(cmd, num_groups, 1, 1);
        }

        // Memory barrier: ensure compute writes are visible to host reads
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);

        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }

        unsafe { self.device.end_command_buffer(cmd) }
            .map_err(|e| GpuError::Dispatch(format!("end command buffer: {:?}", e)))?;

        // Submit and wait
        let cmd_bufs_submit = [cmd];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_bufs_submit);

        unsafe { self.device.reset_fences(&[self.fence]) }
            .map_err(|e| GpuError::Dispatch(format!("reset fence: {:?}", e)))?;

        unsafe { self.device.queue_submit(self.queue, &[submit_info], self.fence) }
            .map_err(|e| GpuError::Dispatch(format!("queue submit: {:?}", e)))?;

        unsafe { self.device.wait_for_fences(&[self.fence], true, u64::MAX) }
            .map_err(|e| GpuError::Dispatch(format!("wait for fence: {:?}", e)))?;

        // Free command buffer and descriptor set
        unsafe {
            self.device.free_command_buffers(self.cmd_pool, &[cmd]);
            self.device
                .free_descriptor_sets(self.desc_pool, &[desc_set])
                .ok();
        }

        Ok(())
    }

    /// Dispatch the hybrid kernel with fixed workgroup size.
    ///
    /// The hybrid kernel uses exactly `num_workgroups` workgroups of
    /// HYBRID_WORKGROUP_SIZE threads each. Each workgroup handles one
    /// wavelength; threads within the group each handle one LOS step and
    /// reduce via subgroupAdd() + shared memory.
    fn dispatch_hybrid(
        &self,
        pipeline: vk::Pipeline,
        buffers: &[&VkBuffer],
        num_workgroups: u32,
    ) -> Result<(), GpuError> {
        if num_workgroups == 0 {
            return Ok(());
        }

        // Allocate a descriptor set for this dispatch
        let layouts = [self.desc_set_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.desc_pool)
            .set_layouts(&layouts);
        let desc_sets = unsafe { self.device.allocate_descriptor_sets(&alloc_info) }
            .map_err(|e| GpuError::Dispatch(format!("descriptor set allocation: {:?}", e)))?;
        let desc_set = desc_sets[0];

        // Update descriptor set with buffer bindings
        let buffer_infos: Vec<vk::DescriptorBufferInfo> = buffers
            .iter()
            .map(|b| {
                vk::DescriptorBufferInfo::default()
                    .buffer(b.buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            })
            .collect();

        let writes: Vec<vk::WriteDescriptorSet> = buffer_infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect();

        unsafe { self.device.update_descriptor_sets(&writes, &[]) };

        // Allocate and record command buffer
        let cmd_alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_bufs = unsafe { self.device.allocate_command_buffers(&cmd_alloc_info) }
            .map_err(|e| GpuError::Dispatch(format!("command buffer allocation: {:?}", e)))?;
        let cmd = cmd_bufs[0];

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { self.device.begin_command_buffer(cmd, &begin_info) }
            .map_err(|e| GpuError::Dispatch(format!("begin command buffer: {:?}", e)))?;

        unsafe {
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[desc_set],
                &[],
            );
            // Dispatch num_workgroups workgroups -- the shader's local_size_x
            // is HYBRID_WORKGROUP_SIZE (256), so each workgroup gets 256 threads.
            self.device.cmd_dispatch(cmd, num_workgroups, 1, 1);
        }

        // Memory barrier: ensure compute writes are visible to host reads
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ);

        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }

        unsafe { self.device.end_command_buffer(cmd) }
            .map_err(|e| GpuError::Dispatch(format!("end command buffer: {:?}", e)))?;

        // Submit and wait
        let cmd_bufs_submit = [cmd];
        let submit_info = vk::SubmitInfo::default().command_buffers(&cmd_bufs_submit);

        unsafe { self.device.reset_fences(&[self.fence]) }
            .map_err(|e| GpuError::Dispatch(format!("reset fence: {:?}", e)))?;

        unsafe { self.device.queue_submit(self.queue, &[submit_info], self.fence) }
            .map_err(|e| GpuError::Dispatch(format!("queue submit: {:?}", e)))?;

        unsafe { self.device.wait_for_fences(&[self.fence], true, u64::MAX) }
            .map_err(|e| GpuError::Dispatch(format!("wait for fence: {:?}", e)))?;

        // Free command buffer and descriptor set
        unsafe {
            self.device.free_command_buffers(self.cmd_pool, &[cmd]);
            self.device
                .free_descriptor_sets(self.desc_pool, &[desc_set])
                .ok();
        }

        Ok(())
    }

    /// Free transient buffers after dispatch is complete.
    fn free_transient_buffers(&self, buffers: Vec<VkBuffer>) {
        let allocator = self.allocator_mut();
        for buf in buffers {
            free_buffer(&self.device, allocator, buf);
        }
    }
}

impl Drop for VulkanBackend {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();

            let alloc = self.allocator.get_mut();

            // Free atmosphere buffer
            if let Some(atm) = self.buf_atm.take() {
                free_buffer(&self.device, alloc, atm);
            }

            // Free constant buffers
            let solar = std::mem::replace(
                &mut self.buf_solar,
                VkBuffer {
                    buffer: vk::Buffer::null(),
                    allocation: None,
                },
            );
            let vision = std::mem::replace(
                &mut self.buf_vision,
                VkBuffer {
                    buffer: vk::Buffer::null(),
                    allocation: None,
                },
            );
            free_buffer(&self.device, alloc, solar);
            free_buffer(&self.device, alloc, vision);

            // Drop the allocator BEFORE destroying the device.
            // gpu-allocator's Allocator::drop() accesses the Vulkan device
            // internally, so it must run while the device is still valid.
            ManuallyDrop::drop(self.allocator.get_mut());

            // Destroy Vulkan objects
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device
                .destroy_pipeline(self.pipeline_single_scatter, None);
            self.device.destroy_pipeline(self.pipeline_mcrt_trace, None);
            self.device.destroy_pipeline(self.pipeline_hybrid, None);
            self.device.destroy_pipeline(self.pipeline_garstang, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_pool(self.desc_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.desc_set_layout, None);

            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// ── Free-standing helpers ───────────────────────────────────────────────

/// Create a compute pipeline from embedded SPIR-V bytes.
fn create_compute_pipeline(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    spv_bytes: &[u8],
    name: &str,
) -> Result<vk::Pipeline, GpuError> {
    let words = spv_to_words(spv_bytes);

    let module_info = vk::ShaderModuleCreateInfo::default().code(&words);
    let module = unsafe { device.create_shader_module(&module_info, None) }
        .map_err(|e| GpuError::ShaderCompilation(format!("{}: {:?}", name, e)))?;

    let stage_info = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(c"main");

    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(stage_info)
        .layout(layout);

    let result = unsafe {
        device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
    };

    // Clean up shader module regardless of pipeline creation result
    unsafe { device.destroy_shader_module(module, None) };

    match result {
        Ok(pipelines) => Ok(pipelines[0]),
        Err((_, e)) => Err(GpuError::ShaderCompilation(format!(
            "{} pipeline: {:?}",
            name, e
        ))),
    }
}

/// Create a host-visible storage buffer from f32 data using gpu-allocator.
fn create_buffer_from_f32(
    device: &ash::Device,
    allocator: &mut Allocator,
    data: &[f32],
    name: &str,
) -> Result<VkBuffer, GpuError> {
    let byte_len = std::mem::size_of_val(data);
    if byte_len == 0 {
        return Err(GpuError::BufferAllocation("zero-length buffer".into()));
    }

    let buffer_info = vk::BufferCreateInfo::default()
        .size(byte_len as vk::DeviceSize)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe { device.create_buffer(&buffer_info, None) }
        .map_err(|e| GpuError::BufferAllocation(format!("{}: {:?}", name, e)))?;

    let mem_req = unsafe { device.get_buffer_memory_requirements(buffer) };

    let allocation = allocator
        .allocate(&AllocationCreateDesc {
            name,
            requirements: mem_req,
            location: MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(|e| {
            unsafe { device.destroy_buffer(buffer, None) };
            GpuError::BufferAllocation(format!("{}: {}", name, e))
        })?;

    unsafe {
        device
            .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
            .map_err(|e| {
                GpuError::BufferAllocation(format!("{} bind: {:?}", name, e))
            })?;
    }

    // Copy data to mapped memory
    if let Some(ptr) = allocation.mapped_ptr() {
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                ptr.as_ptr() as *mut u8,
                byte_len,
            );
        }
    } else {
        return Err(GpuError::BufferAllocation(format!(
            "{}: memory not host-mapped",
            name
        )));
    }

    Ok(VkBuffer {
        buffer,
        allocation: Some(allocation),
    })
}

/// Create an empty host-visible storage buffer for `n` f32 output values.
fn create_empty_buffer(
    device: &ash::Device,
    allocator: &mut Allocator,
    n: usize,
    name: &str,
) -> Result<VkBuffer, GpuError> {
    let byte_len = n * std::mem::size_of::<f32>();
    if byte_len == 0 {
        return Err(GpuError::BufferAllocation(
            "zero-length output buffer".into(),
        ));
    }

    let buffer_info = vk::BufferCreateInfo::default()
        .size(byte_len as vk::DeviceSize)
        .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe { device.create_buffer(&buffer_info, None) }
        .map_err(|e| GpuError::BufferAllocation(format!("{}: {:?}", name, e)))?;

    let mem_req = unsafe { device.get_buffer_memory_requirements(buffer) };

    let allocation = allocator
        .allocate(&AllocationCreateDesc {
            name,
            requirements: mem_req,
            location: MemoryLocation::GpuToCpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .map_err(|e| {
            unsafe { device.destroy_buffer(buffer, None) };
            GpuError::BufferAllocation(format!("{}: {}", name, e))
        })?;

    unsafe {
        device
            .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
            .map_err(|e| {
                GpuError::BufferAllocation(format!("{} bind: {:?}", name, e))
            })?;
    }

    // Zero-initialize the output buffer
    if let Some(ptr) = allocation.mapped_ptr() {
        unsafe {
            std::ptr::write_bytes(ptr.as_ptr() as *mut u8, 0, byte_len);
        }
    }

    Ok(VkBuffer {
        buffer,
        allocation: Some(allocation),
    })
}

/// Read f32 values from a host-visible buffer.
fn read_f32_buffer(buf: &VkBuffer, n: usize) -> Vec<f32> {
    if let Some(ref alloc) = buf.allocation {
        if let Some(ptr) = alloc.mapped_ptr() {
            let slice =
                unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const f32, n) };
            return slice.to_vec();
        }
    }
    vec![0.0; n]
}

/// Free a Vulkan buffer and its allocation.
fn free_buffer(device: &ash::Device, allocator: &mut Allocator, mut buf: VkBuffer) {
    if let Some(alloc) = buf.allocation.take() {
        let _ = allocator.free(alloc);
    }
    if buf.buffer != vk::Buffer::null() {
        unsafe { device.destroy_buffer(buf.buffer, None) };
    }
}
