//! Portable wgpu backend (Vulkan / DX12 / Metal-via-wgpu / GL).
//!
//! Runs the WGSL translation of the transport kernels
//! (`shaders/twilight.wgsl`) on any wgpu-supported adapter, which is
//! what takes the validated engine to headless Linux servers and NVIDIA
//! hardware, where there is no interactive GPU watchdog and the large
//! batch shapes become legal.
//!
//! Contract with the shader: ONE bind group of eight storage buffers
//! (atm, params, out, field, hctx, garstang sources, garstang config,
//! probe rays; unused slots get a small zero stub), byte-identical
//! packed layouts from [`crate::buffers`] (shared with the Metal
//! backend, pinned by the offset parse test), and the same dispatch
//! geometry as `metal.rs`: gid grids for the flat kernels, a
//! (wavelength x LOS-step) workgroup grid with 64 ray lanes and a
//! params-carried step-window offset for the hybrid chain kernel.
//!
//! Readback goes through an explicit staging buffer + map_async: the
//! portable path (no unified-memory `contents()` shortcut), correct on
//! discrete GPUs.
//!
//! HEADLESS DEPLOYMENT NOTE: the step-window split and inter-window
//! yields exist for interactive-compositor watchdogs (macOS). On a
//! headless Vulkan/DX12 box set TWILIGHT_WGPU_WINDOWS=1 (the default
//! here) and raise TWILIGHT_WGPU_RAYS; nothing else changes.

use crate::buffers::{
    dispatch_groups, PackedAtmosphere, PackedCloudField, PackedDispatchParams, PackedLightSource,
};
use crate::{BackendKind, GpuBackend, GpuConfig, GpuDeviceInfo, GpuError, GpuSpectralResult};

/// Embedded WGSL shader (single source of truth for this backend).
const SHADER_SOURCE: &str = include_str!("../shaders/twilight.wgsl");

/// Mirrors HEADER_SENTINEL in the shader: kernels write this to
/// out_buf[0] and abort when a packed buffer fails its header gate.
const HEADER_SENTINEL: f32 = -1.0;

/// LOS steps of the hybrid kernel (mirrors the shader constant).
const MAX_LOS_STEPS: usize = 200;

/// Context-prefix stride (4 header slots + 3 x 64 lane slots), mirrors
/// HCTX_STRIDE in the shader and the Metal host.
const HCTX_STRIDE: usize = 4 + 3 * 64;

fn check_header_sentinel(output: &[f32]) -> Result<(), GpuError> {
    if output.first().map(|v| v.to_bits()) == Some(HEADER_SENTINEL.to_bits()) {
        return Err(GpuError::BufferVersionMismatch);
    }
    Ok(())
}

/// Kernel entry points, in bind-slot-compatible order.
struct Pipelines {
    single_scatter: wgpu::ComputePipeline,
    mcrt_trace: wgpu::ComputePipeline,
    hybrid_ctx: wgpu::ComputePipeline,
    hybrid_v2: wgpu::ComputePipeline,
    field_tau_probe: wgpu::ComputePipeline,
    garstang: wgpu::ComputePipeline,
}

pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    pipelines: Pipelines,
    /// Zero stub bound to any slot a kernel does not read.
    stub: wgpu::Buffer,
    buf_atm: Option<wgpu::Buffer>,
    buf_field: Option<wgpu::Buffer>,
    num_wavelengths: u32,
    info: GpuDeviceInfo,
}

// wgpu resources are Send + Sync; the trait needs Send.
// (No unsafe impl required: wgpu types are already Send.)

/// Probe: is any wgpu adapter present?
pub fn probe() -> bool {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .is_ok()
}

pub fn init(config: &GpuConfig) -> Result<Box<dyn GpuBackend>, GpuError> {
    Ok(Box::new(init_concrete(config)?))
}

/// Concrete-typed init: tests need `WgpuBackend` to reach the probe
/// entry point (mirrors `metal::init_backend`).
pub(crate) fn init_concrete(_config: &GpuConfig) -> Result<WgpuBackend, GpuError> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .map_err(|_| GpuError::NoDevice)?;

    let ainfo = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("twilight-wgpu"),
        // Default limits guarantee 8 storage buffers per stage, which
        // is exactly the shader's bind group.
        ..Default::default()
    }))
    .map_err(|e| GpuError::Platform(format!("wgpu device request failed: {e}")))?;

    // Compile the WGSL; capture validation errors as ShaderCompilation.
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("twilight.wgsl"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
    });
    if let Some(e) = pollster::block_on(scope.pop()) {
        return Err(GpuError::ShaderCompilation(format!("{e}")));
    }

    // One layout for every kernel: binding 2 is the only writable slot.
    let entries: Vec<wgpu::BindGroupLayoutEntry> = (0..8u32)
        .map(|i| wgpu::BindGroupLayoutEntry {
            binding: i,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: i != 2 },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("twilight-bind-layout"),
        entries: &entries,
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("twilight-pipeline-layout"),
        bind_group_layouts: &[&layout],
        immediate_size: 0,
    });

    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mk = |entry: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let pipelines = Pipelines {
        single_scatter: mk("single_scatter_spectrum"),
        mcrt_trace: mk("mcrt_trace_photon"),
        hybrid_ctx: mk("hybrid_context_prefix"),
        hybrid_v2: mk("hybrid_scatter_v2"),
        field_tau_probe: mk("field_tau_probe"),
        garstang: mk("garstang_zenith"),
    };
    if let Some(e) = pollster::block_on(scope.pop()) {
        return Err(GpuError::ShaderCompilation(format!(
            "pipeline creation failed: {e}"
        )));
    }

    let stub = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("stub"),
        size: 64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let info = GpuDeviceInfo {
        name: format!("{} ({:?})", ainfo.name, ainfo.backend),
        backend: BackendKind::Wgpu,
        memory_bytes: 0,
        max_workgroup_size: 256,
    };

    Ok(WgpuBackend {
        device,
        queue,
        layout,
        pipelines,
        stub,
        buf_atm: None,
        buf_field: None,
        num_wavelengths: 0,
        info,
    })
}

impl WgpuBackend {
    fn storage_from_f32(&self, data: &[f32], label: &str) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
    }

    fn empty_storage(&self, n_f32: usize, label: &str) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (n_f32 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// One dispatch: bind the eight slots (stub where None), run
    /// `groups`, and block until the queue drains.
    #[allow(clippy::too_many_arguments)]
    fn dispatch(
        &self,
        pipeline: &wgpu::ComputePipeline,
        atm: Option<&wgpu::Buffer>,
        params: Option<&wgpu::Buffer>,
        out: &wgpu::Buffer,
        field: Option<&wgpu::Buffer>,
        hctx: Option<&wgpu::Buffer>,
        gsrc: Option<&wgpu::Buffer>,
        gcfg: Option<&wgpu::Buffer>,
        rays: Option<&wgpu::Buffer>,
        groups: (u32, u32),
    ) -> Result<(), GpuError> {
    let bind_entries = [
            (0u32, atm.unwrap_or(&self.stub)),
            (1, params.unwrap_or(&self.stub)),
            (2, out),
            (3, field.unwrap_or(&self.stub)),
            (4, hctx.unwrap_or(&self.stub)),
            (5, gsrc.unwrap_or(&self.stub)),
            (6, gcfg.unwrap_or(&self.stub)),
            (7, rays.unwrap_or(&self.stub)),
        ]
        .map(|(i, buf)| wgpu::BindGroupEntry {
            binding: i,
            resource: buf.as_entire_binding(),
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &bind_entries,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, 1);
        }
        self.queue.submit([encoder.finish()]);
        // Bounded wait: an interactive-host watchdog (macOS compositor)
        // can kill a long submission out from under us, and an unbounded
        // poll then hangs forever (measured: 1h42m blocked in
        // Device::maintain on a full-grid field dispatch). Surface it as
        // a timeout so the caller can fall back or re-window.
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(180)),
            })
            .map_err(|e| match e {
                wgpu::PollError::Timeout => GpuError::Timeout,
                other => GpuError::Dispatch(format!("wgpu poll: {other:?}")),
            })?;
        Ok(())
    }

    /// Copy a storage buffer back to the CPU (staging + map_async).
    fn read_back(&self, src: &wgpu::Buffer, n_f32: usize) -> Result<Vec<f32>, GpuError> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (n_f32 * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(src, 0, &staging, 0, (n_f32 * 4) as u64);
        self.queue.submit([encoder.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(180)),
            })
            .map_err(|e| GpuError::Readback(format!("wgpu poll: {e:?}")))?;
        rx.recv()
            .map_err(|_| GpuError::Readback("map_async channel closed".into()))?
            .map_err(|e| GpuError::Readback(format!("map_async: {e:?}")))?;
        let data: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
        staging.unmap();
        Ok(data)
    }

    /// Device-side field tau probe (parity gates). Same 8-f32-header
    /// rays layout as the Metal probe.
    pub fn field_tau_probe(&self, rays: &[[f64; 7]]) -> Result<Vec<f64>, GpuError> {
        let field = self
            .buf_field
            .as_ref()
            .ok_or_else(|| GpuError::Dispatch("field not uploaded".into()))?;
        let mut data = vec![0.0f32; 8];
        data[0] = f32::from_bits(rays.len() as u32);
        for r in rays {
            for (k, v) in r.iter().enumerate() {
                let _ = k;
                data.push(*v as f32);
            }
            data.push(0.0); // pad to 8 per ray
        }
        let buf_rays = self.storage_from_f32(&data, "probe-rays");
        let out = self.empty_storage(rays.len().max(1), "probe-out");
        self.dispatch(
            &self.pipelines.field_tau_probe,
            None,
            None,
            &out,
            Some(field),
            None,
            None,
            None,
            Some(&buf_rays),
            (dispatch_groups(rays.len() as u32, 64), 1),
        )?;
        let raw = self.read_back(&out, rays.len())?;
        check_header_sentinel(&raw)?;
        Ok(raw.iter().map(|&v| v as f64).collect())
    }
}

impl GpuBackend for WgpuBackend {
    fn device_info(&self) -> &GpuDeviceInfo {
        &self.info
    }

    fn upload_atmosphere(
        &mut self,
        atm: &twilight_core::atmosphere::AtmosphereModel,
    ) -> Result<(), GpuError> {
        let packed = PackedAtmosphere::pack(atm);
        self.num_wavelengths = atm.num_wavelengths as u32;
        self.buf_atm = Some(self.storage_from_f32(&packed.data, "atm"));
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
                let packed = PackedCloudField::pack(f);
                self.buf_field = Some(self.storage_from_f32(&packed.data, "field"));
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
        let atm = self
            .buf_atm
            .as_ref()
            .ok_or_else(|| GpuError::Dispatch("atmosphere not uploaded".into()))?;
        let params = PackedDispatchParams::new(observer_pos, view_dir, sun_dir, 0, 0, 0);
        let buf_params = self.storage_from_f32(&params.data, "params");
        let nw = self.num_wavelengths as usize;
        let out = self.empty_storage(nw, "ss-out");
        self.dispatch(
            &self.pipelines.single_scatter,
            Some(atm),
            Some(&buf_params),
            &out,
            None,
            None,
            None,
            None,
            None,
            (dispatch_groups(nw as u32, 256), 1),
        )?;
        let radiance = self.read_back(&out, nw)?;
        check_header_sentinel(&radiance)?;
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
        let atm = self
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
        let buf_params = self.storage_from_f32(&params.data, "params");
        let nw = self.num_wavelengths as usize;
        let total = nw * photons_per_wavelength as usize;
        let out = self.empty_storage(total, "mcrt-out");
        self.dispatch(
            &self.pipelines.mcrt_trace,
            Some(atm),
            Some(&buf_params),
            &out,
            None,
            None,
            None,
            None,
            None,
            (dispatch_groups(total as u32, 256), 1),
        )?;
        let raw = self.read_back(&out, total)?;
        check_header_sentinel(&raw)?;
        let ppw = photons_per_wavelength as usize;
        let radiance = (0..nw)
            .map(|w| {
                let s: f32 = raw[w * ppw..(w + 1) * ppw].iter().sum();
                (s / ppw as f32) as f64
            })
            .collect();
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
        let atm = self
            .buf_atm
            .as_ref()
            .ok_or_else(|| GpuError::Dispatch("atmosphere not uploaded".into()))?;
        let nw = self.num_wavelengths as usize;
        let output_len = nw * MAX_LOS_STEPS;
        let has_field = self.buf_field.is_some();

        // Headless default: one window, big ray chunks. Interactive
        // wgpu-over-Metal hosts can shrink via the env knobs (the same
        // watchdog physiology the Metal backend documents).
        let env_u32 = |k: &str, d: u32| {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d)
        };
        let step_windows = env_u32("TWILIGHT_WGPU_WINDOWS", 1).max(1);
        let rays_per_dispatch = env_u32(
            "TWILIGHT_WGPU_RAYS",
            if has_field { 64 } else { 250 },
        )
        .max(1);
        let window_len = (MAX_LOS_STEPS as u32).div_ceil(step_windows);

        // Hoisted per-(wl, step) eye context, computed once per call
        // (identical to the Metal host).
        let buf_ctx = self.empty_storage(nw * MAX_LOS_STEPS * HCTX_STRIDE, "hctx");
        {
            let params_ctx = PackedDispatchParams::new_with_field(
                observer_pos,
                view_dir,
                sun_dir,
                secondary_rays, // global budget: the substep allocation domain
                secondary_rays,
                0,
                has_field,
            );
            let buf_params_ctx = self.storage_from_f32(&params_ctx.data, "ctx-params");
            self.dispatch(
                &self.pipelines.hybrid_ctx,
                Some(atm),
                Some(&buf_params_ctx),
                &buf_ctx,
                self.buf_field.as_ref(),
                None,
                None,
                None,
                None,
                (dispatch_groups((nw * MAX_LOS_STEPS) as u32, 256), 1),
            )?;
        }

        // NOTE on the writable-slot layout: the ctx kernel writes hctx
        // through binding 2 (out slot) while the chain kernel READS it
        // at binding 4; the dispatch above therefore passed buf_ctx as
        // `out` and the chain dispatches below pass it as `hctx`.
        let out = self.empty_storage(output_len, "hybrid-out");
        let mut accum = vec![0.0f64; output_len];

        let num_chunks = secondary_rays.div_ceil(rays_per_dispatch).max(1);
        for chunk in 0..num_chunks {
            let ray_start = chunk * rays_per_dispatch;
            let rays_this = (secondary_rays - ray_start).min(rays_per_dispatch);
            let folded = {
                let mut z = seed;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^ (z >> 31)
            };
            let ray_seed = (folded & 0xFFFF_FFFF) | ((ray_start as u64) << 32);

            // Zero the output for this chunk (fresh buffer per chunk is
            // simplest and cheap relative to the chain work).
            let out_chunk = self.empty_storage(output_len, "hybrid-out-chunk");
            let _ = &out; // keep the first allocation shape documented

            for w in 0..step_windows {
                let step_off = w * window_len;
                let steps_this = window_len.min(MAX_LOS_STEPS as u32 - step_off);
                if steps_this == 0 {
                    break;
                }
                let params = PackedDispatchParams::new_with_field(
                    observer_pos,
                    view_dir,
                    sun_dir,
                    secondary_rays,
                    rays_this,
                    ray_seed,
                    has_field,
                )
                .with_step_offset(step_off);
                let buf_params = self.storage_from_f32(&params.data, "hybrid-params");
                self.dispatch(
                    &self.pipelines.hybrid_v2,
                    Some(atm),
                    Some(&buf_params),
                    &out_chunk,
                    self.buf_field.as_ref(),
                    Some(&buf_ctx),
                    None,
                    None,
                    None,
                    (nw as u32, steps_this),
                )?;
                if step_windows > 1 {
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
            }

            let raw = self.read_back(&out_chunk, output_len)?;
            check_header_sentinel(&raw)?;
            for (i, &v) in raw.iter().enumerate() {
                if v.is_finite() {
                    accum[i] += v as f64;
                }
            }
        }

        let inv_rays = 1.0 / secondary_rays.max(1) as f64;
        let radiance = (0..nw)
            .map(|w| {
                let base = w * MAX_LOS_STEPS;
                accum[base..base + MAX_LOS_STEPS].iter().sum::<f64>() * inv_rays
            })
            .collect();
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
        let n = sources.len();
        let mut src = Vec::with_capacity(n * 8);
        for s in sources {
            src.extend_from_slice(&[
                s.distance_m,
                s.zenith_angle_rad,
                s.radiance,
                s.spectrum_type,
                s.height_m,
                s._pad1,
                s._pad2,
                s._pad3,
            ]);
        }
        let buf_src = self.storage_from_f32(&src, "garstang-src");
        let cfg: [f32; 8] = [0.0, 0.15, 0.10, 0.15, 550.0, 50.0, 30000.0, n as f32];
        let buf_cfg = self.storage_from_f32(&cfg, "garstang-cfg");
        let out = self.empty_storage(n, "garstang-out");
        self.dispatch(
            &self.pipelines.garstang,
            None,
            None,
            &out,
            None,
            None,
            Some(&buf_src),
            Some(&buf_cfg),
            None,
            (dispatch_groups(n as u32, 256), 1),
        )?;
        let raw = self.read_back(&out, n)?;
        Ok(raw.iter().map(|&v| v as f64).sum())
    }
}
