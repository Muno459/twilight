//! GPU backend adapter for the CPU pipeline.
//!
//! Bridges `twilight_gpu::GpuBackend` to produce [`SpectralResult`] values
//! compatible with the existing simulation and pipeline modules. This allows
//! the pipeline to seamlessly swap between CPU (rayon) and GPU compute for
//! the MCRT simulation passes.
//!
//! # Lifecycle
//!
//! 1. Caller initializes a GPU backend via `twilight_gpu::try_init()`
//! 2. Caller uploads the atmosphere via `gpu.upload_atmosphere(&atm)`
//! 3. This module's scan/simulate functions dispatch to the GPU
//! 4. Results are converted to `SpectralResult` with solar irradiance weighting
//!
//! The threshold analysis and SZA-to-time conversion remain on CPU.

use twilight_core::atmosphere::AtmosphereModel;
use twilight_data::solar_spectrum::SOLAR_IRRADIANCE;
use twilight_gpu::{BatchKernel, BatchRequest, GpuBackend, GpuError, GpuSpectralResult};

use crate::simulation::{compute_geometry, ScatteringMode, SimulationConfig, SpectralResult};

/// Simulate at a single solar zenith angle using the GPU backend.
///
/// Dispatches to the appropriate GPU kernel based on `config.scattering_mode`,
/// then converts the GPU result to `SpectralResult` with optional solar
/// irradiance weighting. Produces results equivalent to
/// [`simulation::simulate_at_sza`] but on GPU hardware.
pub fn simulate_at_sza_gpu(
    gpu: &dyn GpuBackend,
    atm: &AtmosphereModel,
    config: &SimulationConfig,
    sza_deg: f64,
) -> Result<SpectralResult, GpuError> {
    let (observer_pos, sun_dir, view_dir) = compute_geometry(config, sza_deg);

    let obs = [observer_pos.x, observer_pos.y, observer_pos.z];
    let view = [view_dir.x, view_dir.y, view_dir.z];
    let sun = [sun_dir.x, sun_dir.y, sun_dir.z];

    let gpu_result = match config.scattering_mode {
        ScatteringMode::Single => gpu.single_scatter(obs, view, sun)?,
        ScatteringMode::Multiple => {
            let seed = sza_deg.to_bits();
            gpu.mcrt_trace(obs, view, sun, config.photons_per_wavelength as u32, seed)?
        }
        ScatteringMode::Hybrid => {
            let seed = sza_deg.to_bits();
            gpu.hybrid_scatter(obs, view, sun, config.photons_per_wavelength as u32, seed)?
        }
    };

    Ok(gpu_result_to_spectral(
        atm,
        &gpu_result,
        sza_deg,
        config.apply_solar_irradiance,
    ))
}

/// Scan a range of SZA values using GPU.
///
/// Pre-computes geometry for all SZA points, then dispatches them as a
/// single batch via [`GpuBackend::scan_batch`]. This encodes all N SZA
/// dispatches into one GPU command submission (one commit, one wait)
/// instead of N serial round trips, which is ~25x faster on Metal.
///
/// The atmosphere must already be uploaded via `gpu.upload_atmosphere()`.
pub fn simulate_twilight_scan_gpu(
    gpu: &dyn GpuBackend,
    atm: &AtmosphereModel,
    config: &SimulationConfig,
    sza_start: f64,
    sza_end: f64,
    sza_step: f64,
) -> Result<Vec<SpectralResult>, GpuError> {
    // Collect all SZA values upfront.
    let mut sza_values = Vec::new();
    let mut sza = sza_start;
    while sza <= sza_end + 1e-6 {
        sza_values.push(sza);
        sza += sza_step;
    }

    if sza_values.is_empty() {
        return Ok(Vec::new());
    }

    // Build one BatchRequest per SZA with pre-computed geometry.
    let requests: Vec<BatchRequest> = sza_values
        .iter()
        .map(|&sza_deg| {
            let (observer_pos, sun_dir, view_dir) = compute_geometry(config, sza_deg);
            let kernel = match config.scattering_mode {
                ScatteringMode::Single => BatchKernel::SingleScatter,
                ScatteringMode::Multiple => BatchKernel::McrtTrace {
                    photons_per_wavelength: config.photons_per_wavelength as u32,
                    seed: sza_deg.to_bits(),
                },
                ScatteringMode::Hybrid => BatchKernel::Hybrid {
                    secondary_rays: config.photons_per_wavelength as u32,
                    seed: sza_deg.to_bits(),
                },
            };
            BatchRequest {
                observer_pos: [observer_pos.x, observer_pos.y, observer_pos.z],
                view_dir: [view_dir.x, view_dir.y, view_dir.z],
                sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z],
                kernel,
            }
        })
        .collect();

    // Single batched GPU submission.
    let gpu_results = gpu.scan_batch(&requests)?;

    // Convert GPU results to SpectralResult with solar irradiance weighting.
    let results: Vec<SpectralResult> = gpu_results
        .iter()
        .zip(sza_values.iter())
        .map(|(gr, &sza_deg)| {
            gpu_result_to_spectral(atm, gr, sza_deg, config.apply_solar_irradiance)
        })
        .collect();

    Ok(results)
}

/// Convert GPU spectral result to CPU-compatible `SpectralResult`.
///
/// The GPU backend returns raw radiance in the same units as the core
/// physics functions (proportional to W/m^2/sr/nm before solar irradiance
/// weighting). Solar irradiance is applied here if `apply_solar_irradiance`
/// is true, matching the behavior of `simulation::build_spectral_result`.
fn gpu_result_to_spectral(
    atm: &AtmosphereModel,
    gpu_result: &GpuSpectralResult,
    sza_deg: f64,
    apply_solar_irradiance: bool,
) -> SpectralResult {
    let nw = gpu_result.num_wavelengths.min(atm.num_wavelengths);
    let mut wavelengths = Vec::with_capacity(nw);
    let mut radiance = Vec::with_capacity(nw);

    for (w, solar_irr) in SOLAR_IRRADIANCE.iter().enumerate().take(nw) {
        wavelengths.push(atm.wavelengths_nm[w]);
        let r = if apply_solar_irradiance {
            gpu_result.radiance[w] * solar_irr
        } else {
            gpu_result.radiance[w]
        };
        radiance.push(r);
    }

    SpectralResult {
        wavelengths_nm: wavelengths,
        radiance,
        sza_deg,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_atm() -> AtmosphereModel {
        use twilight_data::atmosphere_profiles::AtmosphereType;
        use twilight_data::builder;
        builder::build_clear_sky(AtmosphereType::UsStandard, 0.15)
    }

    // ── gpu_result_to_spectral ──────────────────────────────────────────

    #[test]
    fn conversion_basic_no_irradiance() {
        let atm = make_test_atm();
        let nw = atm.num_wavelengths;

        let gpu_result = GpuSpectralResult {
            radiance: vec![1.0; nw],
            num_wavelengths: nw,
        };

        let sr = gpu_result_to_spectral(&atm, &gpu_result, 96.0, false);
        assert_eq!(sr.wavelengths_nm.len(), nw);
        assert_eq!(sr.radiance.len(), nw);
        assert!((sr.sza_deg - 96.0).abs() < 1e-10);
        // Without irradiance weighting, raw values pass through
        for &r in &sr.radiance {
            assert!((r - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn conversion_with_solar_irradiance() {
        let atm = make_test_atm();
        let nw = atm.num_wavelengths;

        let gpu_result = GpuSpectralResult {
            radiance: vec![1.0; nw],
            num_wavelengths: nw,
        };

        let sr = gpu_result_to_spectral(&atm, &gpu_result, 96.0, true);
        // Solar irradiance weighting should multiply each channel
        for (w, &r) in sr.radiance.iter().enumerate() {
            if w < SOLAR_IRRADIANCE.len() {
                let expected = SOLAR_IRRADIANCE[w];
                assert!(
                    (r - expected).abs() < 1e-5,
                    "wl[{}]: got {}, expected {}",
                    w,
                    r,
                    expected,
                );
            }
        }
    }

    #[test]
    fn conversion_preserves_wavelengths() {
        let atm = make_test_atm();
        let nw = atm.num_wavelengths;

        let gpu_result = GpuSpectralResult {
            radiance: vec![0.5; nw],
            num_wavelengths: nw,
        };

        let sr = gpu_result_to_spectral(&atm, &gpu_result, 96.0, false);
        for w in 0..nw {
            assert!(
                (sr.wavelengths_nm[w] - atm.wavelengths_nm[w]).abs() < 1e-10,
                "wl[{}]: got {}, expected {}",
                w,
                sr.wavelengths_nm[w],
                atm.wavelengths_nm[w],
            );
        }
    }

    #[test]
    fn conversion_fewer_gpu_wavelengths() {
        let atm = make_test_atm();

        // GPU returns fewer wavelengths than the atmosphere model
        let gpu_result = GpuSpectralResult {
            radiance: vec![2.0; 10],
            num_wavelengths: 10,
        };

        let sr = gpu_result_to_spectral(&atm, &gpu_result, 93.0, false);
        assert_eq!(sr.wavelengths_nm.len(), 10);
        assert_eq!(sr.radiance.len(), 10);
        assert!((sr.sza_deg - 93.0).abs() < 1e-10);
    }

    #[test]
    fn conversion_zero_radiance() {
        let atm = make_test_atm();
        let nw = atm.num_wavelengths;

        let gpu_result = GpuSpectralResult {
            radiance: vec![0.0; nw],
            num_wavelengths: nw,
        };

        let sr = gpu_result_to_spectral(&atm, &gpu_result, 108.0, true);
        for &r in &sr.radiance {
            assert!(r.abs() < 1e-20, "Zero input should give zero output");
        }
    }

    #[test]
    fn conversion_stores_sza() {
        let atm = make_test_atm();
        let gpu_result = GpuSpectralResult {
            radiance: vec![1.0; atm.num_wavelengths],
            num_wavelengths: atm.num_wavelengths,
        };

        for sza in &[90.0, 96.0, 100.5, 108.0] {
            let sr = gpu_result_to_spectral(&atm, &gpu_result, *sza, false);
            assert!(
                (sr.sza_deg - *sza).abs() < 1e-10,
                "Expected SZA {}, got {}",
                sza,
                sr.sza_deg,
            );
        }
    }

    #[test]
    fn conversion_irradiance_positive() {
        let atm = make_test_atm();
        let nw = atm.num_wavelengths;

        let gpu_result = GpuSpectralResult {
            radiance: vec![1.0; nw],
            num_wavelengths: nw,
        };

        let sr = gpu_result_to_spectral(&atm, &gpu_result, 96.0, true);
        for (w, &r) in sr.radiance.iter().enumerate() {
            assert!(
                r > 0.0,
                "Irradiance-weighted wl[{}] should be positive, got {}",
                w,
                r,
            );
        }
    }

    // ── simulate_at_sza_gpu (requires actual GPU, so just test geometry) ──

    #[test]
    fn compute_geometry_returns_unit_vectors() {
        let config = SimulationConfig::default();
        let (obs, sun, view) = compute_geometry(&config, 96.0);

        // Observer should be at roughly Earth radius distance from origin
        let r = (obs.x * obs.x + obs.y * obs.y + obs.z * obs.z).sqrt();
        assert!(
            r > 6.3e6 && r < 6.4e6,
            "Observer radius {} should be ~6.37e6",
            r
        );

        // Sun and view should be roughly unit vectors
        let sun_mag = (sun.x * sun.x + sun.y * sun.y + sun.z * sun.z).sqrt();
        let view_mag = (view.x * view.x + view.y * view.y + view.z * view.z).sqrt();
        assert!(
            (sun_mag - 1.0).abs() < 1e-10,
            "Sun dir magnitude {} should be 1",
            sun_mag
        );
        assert!(
            (view_mag - 1.0).abs() < 1e-10,
            "View dir magnitude {} should be 1",
            view_mag
        );
    }

    #[test]
    fn scattering_mode_dispatch_selection() {
        // Verify the match arms cover all modes (compile-time check)
        let modes = [
            ScatteringMode::Single,
            ScatteringMode::Multiple,
            ScatteringMode::Hybrid,
        ];
        for mode in &modes {
            let config = SimulationConfig {
                scattering_mode: *mode,
                ..SimulationConfig::default()
            };
            // Just verify the config accepts each mode
            assert_eq!(config.scattering_mode, *mode);
        }
    }
}
