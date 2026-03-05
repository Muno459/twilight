//! wgpu/WebGPU backend for WASM browsers and native fallback.
//!
//! Uses WGSL compute shaders with the wgpu crate. All four kernels are
//! separate entry points in a single WGSL module. Dispatch uses storage
//! buffers with the same packed layout as the other backends.

use wgpu::util::DeviceExt;

use crate::buffers::{
    dispatch_groups, PackedAtmosphere, PackedDispatchParams, PackedLightSource,
    PackedSolarSpectrum, PackedVisionLuts,
};
use crate::{BackendKind, GpuBackend, GpuConfig, GpuDeviceInfo, GpuError, GpuSpectralResult};

/// Embedded WGSL shader source.
const WGSL_SOURCE: &str = include_str!("../shaders/twilight.wgsl");

/// wgpu backend implementing the [`GpuBackend`] trait.
pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,

    // Four compute pipelines (one per kernel entry point)
    pipeline_single_scatter: wgpu::ComputePipeline,
    pipeline_mcrt_trace: wgpu::ComputePipeline,
    pipeline_hybrid: wgpu::ComputePipeline,
    pipeline_garstang: wgpu::ComputePipeline,

    // Bind group layout (shared by all pipelines)
    bind_group_layout: wgpu::BindGroupLayout,

    // Uploaded atmosphere buffer (persisted between dispatches)
    buf_atm: Option<wgpu::Buffer>,
    // Solar and vision LUTs (reserved for Phase 11f pipeline integration)
    #[allow(dead_code)]
    buf_solar: wgpu::Buffer,
    #[allow(dead_code)]
    buf_vision: wgpu::Buffer,

    info: GpuDeviceInfo,
    config: GpuConfig,
    num_wavelengths: u32,
}

// Safety: wgpu Device, Queue, Buffer, ComputePipeline etc. are Send+Sync.
unsafe impl Send for WgpuBackend {}

// ── Probe ───────────────────────────────────────────────────────────────

/// Lightweight probe: can we initialize wgpu and find a device?
pub fn probe() -> bool {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }));
    adapter.is_ok()
}

// ── Init ────────────────────────────────────────────────────────────────

/// Initialize the wgpu backend: request device, compile shaders, create pipelines.
pub fn init(config: &GpuConfig) -> Result<Box<dyn GpuBackend>, GpuError> {
    let instance = wgpu::Instance::default();

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .map_err(|e| GpuError::Platform(format!("wgpu adapter request failed: {}", e)))?;

    let adapter_info = adapter.get_info();

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("twilight-mcrt"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        ..Default::default()
    }))
    .map_err(|e| GpuError::Platform(format!("wgpu device request failed: {}", e)))?;

    // Compile WGSL shader module
    let shader_module: wgpu::ShaderModule =
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("twilight.wgsl"),
            source: wgpu::ShaderSource::Wgsl(WGSL_SOURCE.into()),
        });

    // Create bind group layout (shared by all 4 kernels)
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("twilight_bgl"),
        entries: &[
            // Binding 0: input buffer (atmosphere or sources)
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 1: dispatch parameters / config
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Binding 2: output buffer
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("twilight_pl"),
        bind_group_layouts: &[&bind_group_layout],
        immediate_size: 0,
    });

    // Create 4 compute pipelines, one per entry point
    let create_pipeline = |entry: &str, label: &str| -> wgpu::ComputePipeline {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };

    let pipeline_single_scatter = create_pipeline("single_scatter_spectrum", "single_scatter");
    let pipeline_mcrt_trace = create_pipeline("mcrt_trace_photon", "mcrt_trace");
    let pipeline_hybrid = create_pipeline("hybrid_scatter", "hybrid");
    let pipeline_garstang = create_pipeline("garstang_zenith", "garstang");

    // Pack and upload constant buffers
    let solar = PackedSolarSpectrum::pack();
    let vision = PackedVisionLuts::pack();

    let buf_solar = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("solar_spectrum"),
        contents: bytemuck_cast_f32_slice(&solar.data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let buf_vision = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("vision_luts"),
        contents: bytemuck_cast_f32_slice(&vision.data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let info = GpuDeviceInfo {
        name: adapter_info.name.clone(),
        backend: BackendKind::Wgpu,
        memory_bytes: 0, // wgpu doesn't expose total memory
        max_workgroup_size: device.limits().max_compute_workgroup_size_x,
    };

    Ok(Box::new(WgpuBackend {
        device,
        queue,
        pipeline_single_scatter,
        pipeline_mcrt_trace,
        pipeline_hybrid,
        pipeline_garstang,
        bind_group_layout,
        buf_atm: None,
        buf_solar,
        buf_vision,
        info,
        config: config.clone(),
        num_wavelengths: 0,
    }))
}

// ── GpuBackend implementation ───────────────────────────────────────────

impl GpuBackend for WgpuBackend {
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
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("atmosphere"),
                    contents: bytemuck_cast_f32_slice(&packed.data),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
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
        let nw = self.num_wavelengths as usize;

        let output =
            self.dispatch_compute(&self.pipeline_single_scatter, buf_atm, &params.data, nw)?;

        Ok(GpuSpectralResult {
            radiance: output.iter().map(|&v| v as f64).collect(),
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

        let nw = self.num_wavelengths as usize;
        let total_threads = nw * photons_per_wavelength as usize;

        let raw = self.dispatch_compute(
            &self.pipeline_mcrt_trace,
            buf_atm,
            &params.data,
            total_threads,
        )?;

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
        let nw = self.num_wavelengths as usize;

        let output = self.dispatch_compute(&self.pipeline_hybrid, buf_atm, &params.data, nw)?;

        Ok(GpuSpectralResult {
            radiance: output.iter().map(|&v| v as f64).collect(),
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
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("garstang_sources"),
                contents: bytemuck_cast_f32_slice(&source_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

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

        let output = self.dispatch_compute_raw(
            &self.pipeline_garstang,
            &buf_sources,
            &config_data,
            num_sources,
        )?;

        let total: f64 = output.iter().map(|&v| v as f64).sum();
        Ok(total)
    }
}

// ── Dispatch helpers ────────────────────────────────────────────────────

impl WgpuBackend {
    /// Generic dispatch: upload params, create output buffer, run pipeline, readback.
    fn dispatch_compute(
        &self,
        pipeline: &wgpu::ComputePipeline,
        input_buf: &wgpu::Buffer,
        params_data: &[f32],
        output_count: usize,
    ) -> Result<Vec<f32>, GpuError> {
        self.dispatch_compute_raw(pipeline, input_buf, params_data, output_count)
    }

    /// Core dispatch with raw buffers.
    fn dispatch_compute_raw(
        &self,
        pipeline: &wgpu::ComputePipeline,
        input_buf: &wgpu::Buffer,
        params_data: &[f32],
        output_count: usize,
    ) -> Result<Vec<f32>, GpuError> {
        let buf_params = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("params"),
                contents: bytemuck_cast_f32_slice(params_data),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_size = (output_count * std::mem::size_of::<f32>()) as u64;
        let buf_output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let buf_staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dispatch_bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_output.as_entire_binding(),
                },
            ],
        });

        let num_groups = dispatch_groups(output_count as u32, self.config.workgroup_size);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dispatch"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_groups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&buf_output, 0, &buf_staging, 0, output_size);

        self.queue.submit(std::iter::once(encoder.finish()));

        // Map the staging buffer and read back
        let buffer_slice = buf_staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });

        receiver
            .recv()
            .map_err(|e| GpuError::Readback(format!("channel recv failed: {}", e)))?
            .map_err(|e| GpuError::Readback(format!("buffer map failed: {}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        drop(data);
        buf_staging.unmap();

        Ok(result)
    }
}

// ── Utility ─────────────────────────────────────────────────────────────

/// Cast an f32 slice to a byte slice for wgpu buffer creation.
fn bytemuck_cast_f32_slice(data: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) }
}
