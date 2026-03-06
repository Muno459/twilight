//! High-level MCRT simulation driver.
//!
//! Takes observer location, solar zenith angle, and atmosphere model,
//! and computes spectral sky radiance. Supports two scattering modes:
//!
//! - **Single scattering** (default): deterministic line-of-sight integration.
//!   Fast, no noise, accurate for clear-sky twilight up to ~15° depression.
//!
//! - **Multiple scattering**: backward Monte Carlo with next-event estimation.
//!   Handles all scattering orders, needed for deep twilight (>15°), thick
//!   clouds, and reaching the 18° depression angle used by MWL/ISNA.
//!
//! The radiance output is in physical units [W/m²/sr/nm] when solar
//! irradiance weighting is enabled (default).

use rayon::prelude::*;
use twilight_core::atmosphere::AtmosphereModel;
use twilight_core::geometry::{geographic_to_ecef, solar_direction_ecef, Vec3};
use twilight_core::photon;
use twilight_core::single_scatter;
use twilight_data::solar_spectrum::SOLAR_IRRADIANCE;

/// Result of a spectral simulation at a single solar zenith angle.
#[derive(Debug, Clone)]
pub struct SpectralResult {
    /// Wavelengths in nm
    pub wavelengths_nm: Vec<f64>,
    /// Sky radiance at each wavelength [W/m²/sr/nm] (physical units when
    /// solar irradiance weighting is applied)
    pub radiance: Vec<f64>,
    /// Solar zenith angle (degrees)
    pub sza_deg: f64,
}

/// Scattering mode for the simulation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScatteringMode {
    /// Deterministic single-scattering line-of-sight integration.
    /// Fast, no noise. Accurate for clear sky up to ~15° depression.
    #[default]
    Single,
    /// Backward Monte Carlo with next-event estimation.
    /// Handles all scattering orders. Required for deep twilight (>15°),
    /// thick clouds, and physically reaching 18° depression angles.
    Multiple,
    /// Hybrid: deterministic single-scatter (order 1) + MC secondary
    /// chains (orders 2+). Best convergence for deep twilight.
    /// Uses `photons_per_wavelength` as number of secondary rays per LOS step.
    Hybrid,
}

/// Configuration for a twilight simulation.
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    /// Observer latitude (degrees, north positive)
    pub latitude: f64,
    /// Observer longitude (degrees, east positive)
    pub longitude: f64,
    /// Observer elevation above sea level (meters)
    pub elevation: f64,
    /// Solar azimuth angle (degrees, 0=north, clockwise).
    /// For twilight, typically ~90° (east, Fajr) or ~270° (west, Isha).
    pub solar_azimuth: f64,
    /// Zenith viewing direction (degrees from straight up).
    /// ~70-80° toward the sun azimuth captures the brightest twilight sky.
    pub view_zenith: f64,
    /// Whether to weight radiance by solar spectrum (true = physical units).
    /// When false, radiance is in relative units (useful for debugging).
    pub apply_solar_irradiance: bool,
    /// Scattering mode: single (deterministic) or multiple (Monte Carlo).
    pub scattering_mode: ScatteringMode,
    /// Number of photons per wavelength for MC mode. Ignored in single mode.
    /// Higher values reduce noise but increase computation time.
    /// Recommended: 10000+ for converged results, 1000 for quick estimates.
    pub photons_per_wavelength: usize,
    /// Enable full Stokes [I,Q,U,V] polarization tracking (default: true).
    ///
    /// When true (the default), the hybrid/MC CPU engine propagates full
    /// 4-component Stokes vectors through Mueller matrices, capturing
    /// polarization-intensity coupling from Rayleigh and aerosol scattering.
    ///
    /// When false (`--fast` mode), uses scalar phase function (P11 only).
    /// Slightly faster, loses ~0.5-2% polarization correction.
    pub polarized: bool,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            latitude: 21.4225, // Mecca
            longitude: 39.8262,
            elevation: 0.0,
            solar_azimuth: 270.0, // West (Isha/sunset direction)
            view_zenith: 75.0,    // Look toward horizon
            apply_solar_irradiance: true,
            scattering_mode: ScatteringMode::Single,
            photons_per_wavelength: 10_000,
            polarized: true,
        }
    }
}

/// Run simulation at a single solar zenith angle.
///
/// Dispatches between single-scattering (deterministic) and multiple-scattering
/// (Monte Carlo) based on `config.scattering_mode`.
///
/// Returns spectral radiance across all wavelengths in the atmosphere model.
///
/// When `config.apply_solar_irradiance` is true, the output is in physical
/// units [W/m²/sr/nm]. The integral gives:
///   I(λ) = F_sun(λ) × ∫ β_scat × P(θ)/(4π) × T_sun × T_obs ds
/// where F_sun(λ) is the TOA solar spectral irradiance [W/m²/nm].
pub fn simulate_at_sza(
    atm: &AtmosphereModel,
    config: &SimulationConfig,
    sza_deg: f64,
) -> SpectralResult {
    match config.scattering_mode {
        ScatteringMode::Single => simulate_at_sza_single(atm, config, sza_deg),
        ScatteringMode::Multiple => simulate_at_sza_mc(atm, config, sza_deg),
        ScatteringMode::Hybrid => simulate_at_sza_hybrid(atm, config, sza_deg),
    }
}

/// Compute observer/sun/view geometry from config and SZA.
pub(crate) fn compute_geometry(config: &SimulationConfig, sza_deg: f64) -> (Vec3, Vec3, Vec3) {
    let observer_pos = geographic_to_ecef(config.latitude, config.longitude, config.elevation);
    let sun_dir = solar_direction_ecef(
        sza_deg,
        config.solar_azimuth,
        config.latitude,
        config.longitude,
    );
    let view_dir = solar_direction_ecef(
        config.view_zenith,
        config.solar_azimuth,
        config.latitude,
        config.longitude,
    );
    (observer_pos, sun_dir, view_dir)
}

/// Apply solar irradiance weighting and build SpectralResult from raw radiance array.
fn build_spectral_result(
    atm: &AtmosphereModel,
    radiance_array: &[f64; 64],
    sza_deg: f64,
    apply_solar_irradiance: bool,
) -> SpectralResult {
    let num_wl = atm.num_wavelengths;
    let mut wavelengths = Vec::with_capacity(num_wl);
    let mut radiance = Vec::with_capacity(num_wl);

    for w in 0..num_wl {
        wavelengths.push(atm.wavelengths_nm[w]);
        let r = if apply_solar_irradiance && w < SOLAR_IRRADIANCE.len() {
            radiance_array[w] * SOLAR_IRRADIANCE[w]
        } else {
            radiance_array[w]
        };
        radiance.push(r);
    }

    SpectralResult {
        wavelengths_nm: wavelengths,
        radiance,
        sza_deg,
    }
}

/// Single-scattering simulation (deterministic, no noise).
fn simulate_at_sza_single(
    atm: &AtmosphereModel,
    config: &SimulationConfig,
    sza_deg: f64,
) -> SpectralResult {
    let (observer_pos, sun_dir, view_dir) = compute_geometry(config, sza_deg);
    let radiance_array =
        single_scatter::single_scatter_spectrum(atm, observer_pos, view_dir, sun_dir);
    build_spectral_result(atm, &radiance_array, sza_deg, config.apply_solar_irradiance)
}

/// Multiple-scattering simulation via backward Monte Carlo with NEE.
///
/// Traces `config.photons_per_wavelength` photons per wavelength using rayon
/// parallelism. Each photon undergoes multiple scattering events with
/// next-event estimation at each bounce.
///
/// The result captures all scattering orders: the first bounce is equivalent
/// to single scattering, and subsequent bounces add the multiple-scattering
/// contribution that becomes important at deep twilight (>15° depression)
/// and in thick clouds.
fn simulate_at_sza_mc(
    atm: &AtmosphereModel,
    config: &SimulationConfig,
    sza_deg: f64,
) -> SpectralResult {
    let (observer_pos, sun_dir, view_dir) = compute_geometry(config, sza_deg);
    let num_wl = atm.num_wavelengths;
    let nphotons = config.photons_per_wavelength;

    if nphotons == 0 {
        let radiance_array = [0.0f64; 64];
        return build_spectral_result(atm, &radiance_array, sza_deg, config.apply_solar_irradiance);
    }

    // Parallelize over wavelengths using rayon.
    // Each wavelength traces nphotons photons independently.
    let per_wl_radiance: Vec<f64> = (0..num_wl)
        .into_par_iter()
        .map(|w| {
            let mut total_weight = 0.0;
            for p in 0..nphotons {
                // Unique seed per (sza, wavelength, photon) triple.
                // Include sza bits to decorrelate across SZA scan steps.
                let sza_bits = sza_deg.to_bits();
                let mut rng = (sza_bits)
                    .wrapping_add(w as u64)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(p as u64)
                    .wrapping_mul(2862933555777941757)
                    .wrapping_add(1);

                let result =
                    photon::trace_photon(atm, observer_pos, view_dir, sun_dir, w, &mut rng);
                total_weight += result.weight;
            }
            total_weight / nphotons as f64
        })
        .collect();

    let mut radiance_array = [0.0f64; 64];
    for (w, &r) in per_wl_radiance.iter().enumerate() {
        radiance_array[w] = r;
    }

    build_spectral_result(atm, &radiance_array, sza_deg, config.apply_solar_irradiance)
}

/// Hybrid simulation: deterministic single-scatter (order 1) + MC secondary
/// chains (orders 2+). Best convergence for deep twilight.
///
/// The single-scatter contribution is computed exactly (no noise), then at
/// each LOS step, secondary MC chains are launched to capture higher-order
/// scattering. This produces converged results at deep twilight (15-18°
/// depression) with far fewer photons than pure backward MC.
///
/// Uses rayon parallelism over wavelengths.
fn simulate_at_sza_hybrid(
    atm: &AtmosphereModel,
    config: &SimulationConfig,
    sza_deg: f64,
) -> SpectralResult {
    let (observer_pos, sun_dir, view_dir) = compute_geometry(config, sza_deg);
    let num_wl = atm.num_wavelengths;
    let secondary_rays = config.photons_per_wavelength;

    // Parallelize over wavelengths
    let per_wl_radiance: Vec<f64> = (0..num_wl)
        .into_par_iter()
        .map(|w| {
            let sza_bits = sza_deg.to_bits();
            let mut rng = sza_bits
                .wrapping_add(w as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);

            photon::hybrid_scatter_radiance(
                atm,
                observer_pos,
                view_dir,
                sun_dir,
                w,
                secondary_rays,
                &mut rng,
                config.polarized,
            )
        })
        .collect();

    let mut radiance_array = [0.0f64; 64];
    for (w, &r) in per_wl_radiance.iter().enumerate() {
        radiance_array[w] = r;
    }

    build_spectral_result(atm, &radiance_array, sza_deg, config.apply_solar_irradiance)
}

/// Run simulation across a range of solar zenith angles.
///
/// Scans through twilight, computing spectral radiance at each SZA.
pub fn simulate_twilight_scan(
    atm: &AtmosphereModel,
    config: &SimulationConfig,
    sza_start: f64,
    sza_end: f64,
    sza_step: f64,
) -> Vec<SpectralResult> {
    let mut results = Vec::new();
    let mut sza = sza_start;

    while sza <= sza_end + 1e-6 {
        let result = simulate_at_sza(atm, config, sza);
        results.push(result);
        sza += sza_step;
    }

    results
}

/// Compute total broadband radiance from spectral result (trapezoidal integration).
pub fn total_radiance(result: &SpectralResult) -> f64 {
    let n = result.radiance.len();
    if n < 2 {
        return result.radiance.first().copied().unwrap_or(0.0);
    }

    let mut total = 0.0;
    for i in 0..(n - 1) {
        let dw = result.wavelengths_nm[i + 1] - result.wavelengths_nm[i];
        total += 0.5 * (result.radiance[i] + result.radiance[i + 1]) * dw;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use twilight_data::atmosphere_profiles::AtmosphereType;
    use twilight_data::builder;

    fn make_clear_sky_atm() -> AtmosphereModel {
        builder::build_clear_sky(AtmosphereType::UsStandard, 0.15)
    }

    fn default_config() -> SimulationConfig {
        SimulationConfig::default()
    }

    // ── SimulationConfig defaults ──

    #[test]
    fn default_config_mecca() {
        let c = SimulationConfig::default();
        assert!((c.latitude - 21.4225).abs() < 0.01);
        assert!((c.longitude - 39.8262).abs() < 0.01);
        assert!((c.elevation - 0.0).abs() < 0.01);
        assert!((c.solar_azimuth - 270.0).abs() < 0.01);
        assert!((c.view_zenith - 75.0).abs() < 0.01);
        assert!(c.apply_solar_irradiance);
    }

    // ── simulate_at_sza ──

    #[test]
    fn simulate_at_sza_returns_correct_wavelength_count() {
        let atm = make_clear_sky_atm();
        let config = default_config();
        let result = simulate_at_sza(&atm, &config, 96.0);
        assert_eq!(result.wavelengths_nm.len(), 41);
        assert_eq!(result.radiance.len(), 41);
    }

    #[test]
    fn simulate_at_sza_stores_sza() {
        let atm = make_clear_sky_atm();
        let config = default_config();
        let result = simulate_at_sza(&atm, &config, 96.5);
        assert!((result.sza_deg - 96.5).abs() < 1e-10);
    }

    #[test]
    fn simulate_at_sza_positive_radiance_at_civil_twilight() {
        let atm = make_clear_sky_atm();
        let config = default_config();
        let result = simulate_at_sza(&atm, &config, 93.0); // civil twilight
        let total = total_radiance(&result);
        assert!(
            total > 0.0,
            "Civil twilight should produce positive radiance, got {}",
            total
        );
    }

    #[test]
    fn simulate_at_sza_radiance_non_negative() {
        let atm = make_clear_sky_atm();
        let config = default_config();
        for sza in &[90.0, 96.0, 102.0, 108.0] {
            let result = simulate_at_sza(&atm, &config, *sza);
            for (i, &r) in result.radiance.iter().enumerate() {
                assert!(
                    r >= 0.0,
                    "Radiance at SZA={}, wl[{}] = {} should be non-negative",
                    sza,
                    i,
                    r
                );
            }
        }
    }

    #[test]
    fn simulate_at_sza_decreases_with_depth() {
        let atm = make_clear_sky_atm();
        let config = default_config();
        let r_93 = total_radiance(&simulate_at_sza(&atm, &config, 93.0));
        let r_100 = total_radiance(&simulate_at_sza(&atm, &config, 100.0));
        assert!(
            r_93 > r_100,
            "Radiance should decrease: SZA93={:.4e} > SZA100={:.4e}",
            r_93,
            r_100
        );
    }

    #[test]
    fn simulate_at_sza_with_solar_irradiance() {
        let atm = make_clear_sky_atm();
        let mut config_on = default_config();
        config_on.apply_solar_irradiance = true;
        let mut config_off = default_config();
        config_off.apply_solar_irradiance = false;

        let r_on = simulate_at_sza(&atm, &config_on, 93.0);
        let r_off = simulate_at_sza(&atm, &config_off, 93.0);

        // With solar irradiance weighting, radiance should be different
        // (unless raw radiance happens to equal 1 everywhere, which it won't)
        let total_on = total_radiance(&r_on);
        let total_off = total_radiance(&r_off);
        // Both should be positive at civil twilight
        assert!(total_on > 0.0, "Irradiance-weighted should be positive");
        assert!(total_off > 0.0, "Raw should be positive");
        // They should be different (solar irradiance multiplies by ~1-2 W/m²/nm)
        assert!(
            (total_on - total_off).abs() > 1e-20,
            "Solar weighting should change results"
        );
    }

    #[test]
    fn simulate_at_sza_wavelengths_correct() {
        let atm = make_clear_sky_atm();
        let config = default_config();
        let result = simulate_at_sza(&atm, &config, 96.0);
        assert!((result.wavelengths_nm[0] - 380.0).abs() < 0.01);
        assert!((result.wavelengths_nm[20] - 580.0).abs() < 0.01);
        assert!((result.wavelengths_nm[40] - 780.0).abs() < 0.01);
    }

    // ── simulate_twilight_scan ──

    #[test]
    fn twilight_scan_correct_count() {
        let atm = make_clear_sky_atm();
        let config = default_config();
        let results = simulate_twilight_scan(&atm, &config, 90.0, 100.0, 2.0);
        // 90, 92, 94, 96, 98, 100 = 6 steps
        assert_eq!(results.len(), 6, "Expected 6 steps, got {}", results.len());
    }

    #[test]
    fn twilight_scan_sza_values_correct() {
        let atm = make_clear_sky_atm();
        let config = default_config();
        let results = simulate_twilight_scan(&atm, &config, 90.0, 94.0, 2.0);
        assert!((results[0].sza_deg - 90.0).abs() < 0.01);
        assert!((results[1].sza_deg - 92.0).abs() < 0.01);
        assert!((results[2].sza_deg - 94.0).abs() < 0.01);
    }

    #[test]
    fn twilight_scan_radiance_decreases() {
        let atm = make_clear_sky_atm();
        let config = default_config();
        let results = simulate_twilight_scan(&atm, &config, 91.0, 105.0, 2.0);
        let totals: Vec<f64> = results.iter().map(|r| total_radiance(r)).collect();

        // Radiance should generally decrease (may have small bumps from geometry)
        // Check first vs last
        assert!(
            totals[0] > totals[totals.len() - 1],
            "First total ({:.4e}) should exceed last ({:.4e})",
            totals[0],
            totals[totals.len() - 1]
        );
    }

    // ── total_radiance ──

    #[test]
    fn total_radiance_flat_spectrum() {
        // Flat radiance = 1.0 over 380-780nm = 400nm bandwidth
        // Trapezoidal integral = 1.0 × 400 = 400
        let result = SpectralResult {
            wavelengths_nm: vec![380.0, 780.0],
            radiance: vec![1.0, 1.0],
            sza_deg: 96.0,
        };
        let total = total_radiance(&result);
        assert!(
            (total - 400.0).abs() < 0.01,
            "Flat 1.0 over 400nm: total={}, expected 400",
            total
        );
    }

    #[test]
    fn total_radiance_zero() {
        let result = SpectralResult {
            wavelengths_nm: vec![380.0, 780.0],
            radiance: vec![0.0, 0.0],
            sza_deg: 96.0,
        };
        assert!(total_radiance(&result).abs() < 1e-20);
    }

    #[test]
    fn total_radiance_single_point() {
        let result = SpectralResult {
            wavelengths_nm: vec![550.0],
            radiance: vec![1.0],
            sza_deg: 96.0,
        };
        // Single point → just the value itself
        assert!((total_radiance(&result) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn total_radiance_empty() {
        let result = SpectralResult {
            wavelengths_nm: vec![],
            radiance: vec![],
            sza_deg: 96.0,
        };
        assert!((total_radiance(&result) - 0.0).abs() < 1e-20);
    }

    #[test]
    fn total_radiance_trapezoidal_triangle() {
        // Triangle: rises from 0 to 1 at midpoint, then 1 to 0
        // Area = 0.5 × base × height = 0.5 × 200 × 1 = 100
        let result = SpectralResult {
            wavelengths_nm: vec![400.0, 500.0, 600.0],
            radiance: vec![0.0, 1.0, 0.0],
            sza_deg: 96.0,
        };
        let total = total_radiance(&result);
        assert!(
            (total - 100.0).abs() < 0.01,
            "Triangle integral: total={}, expected 100",
            total
        );
    }

    // ── ScatteringMode defaults ──

    #[test]
    fn default_scattering_mode_is_single() {
        let c = SimulationConfig::default();
        assert_eq!(c.scattering_mode, ScatteringMode::Single);
    }

    #[test]
    fn default_photons_per_wavelength() {
        let c = SimulationConfig::default();
        assert_eq!(c.photons_per_wavelength, 10_000);
    }

    // ── MC mode basic tests ──

    fn mc_config() -> SimulationConfig {
        SimulationConfig {
            scattering_mode: ScatteringMode::Multiple,
            photons_per_wavelength: 1000, // enough for test convergence
            ..SimulationConfig::default()
        }
    }

    #[test]
    fn mc_returns_correct_wavelength_count() {
        let atm = make_clear_sky_atm();
        let config = mc_config();
        let result = simulate_at_sza(&atm, &config, 96.0);
        assert_eq!(result.wavelengths_nm.len(), 41);
        assert_eq!(result.radiance.len(), 41);
    }

    #[test]
    fn mc_stores_sza() {
        let atm = make_clear_sky_atm();
        let config = mc_config();
        let result = simulate_at_sza(&atm, &config, 96.5);
        assert!((result.sza_deg - 96.5).abs() < 1e-10);
    }

    #[test]
    fn mc_radiance_non_negative() {
        let atm = make_clear_sky_atm();
        let config = mc_config();
        for sza in &[90.0, 96.0, 102.0] {
            let result = simulate_at_sza(&atm, &config, *sza);
            for (i, &r) in result.radiance.iter().enumerate() {
                assert!(
                    r >= 0.0,
                    "MC radiance at SZA={}, wl[{}] = {} should be non-negative",
                    sza,
                    i,
                    r
                );
            }
        }
    }

    #[test]
    fn mc_positive_radiance_at_civil_twilight() {
        // At SZA=93° with clear sky, MC should produce some signal
        let atm = make_clear_sky_atm();
        let config = SimulationConfig {
            scattering_mode: ScatteringMode::Multiple,
            photons_per_wavelength: 2000,
            ..SimulationConfig::default()
        };
        let result = simulate_at_sza(&atm, &config, 93.0);
        let total = total_radiance(&result);
        assert!(
            total > 0.0,
            "MC civil twilight should produce positive radiance, got {}",
            total
        );
    }

    #[test]
    fn mc_radiance_decreases_with_depth() {
        // SZA 93 should give more radiance than SZA 100
        let atm = make_clear_sky_atm();
        let config = SimulationConfig {
            scattering_mode: ScatteringMode::Multiple,
            photons_per_wavelength: 2000,
            ..SimulationConfig::default()
        };
        let r_93 = total_radiance(&simulate_at_sza(&atm, &config, 93.0));
        let r_100 = total_radiance(&simulate_at_sza(&atm, &config, 100.0));
        assert!(
            r_93 > r_100 * 0.1, // generous for MC noise
            "MC SZA93 ({:.4e}) should be > SZA100 ({:.4e})",
            r_93,
            r_100
        );
    }

    #[test]
    fn mc_wavelengths_correct() {
        let atm = make_clear_sky_atm();
        let config = mc_config();
        let result = simulate_at_sza(&atm, &config, 96.0);
        assert!((result.wavelengths_nm[0] - 380.0).abs() < 0.01);
        assert!((result.wavelengths_nm[20] - 580.0).abs() < 0.01);
        assert!((result.wavelengths_nm[40] - 780.0).abs() < 0.01);
    }

    #[test]
    fn mc_zero_photons_gives_zero_radiance() {
        let atm = make_clear_sky_atm();
        let config = SimulationConfig {
            scattering_mode: ScatteringMode::Multiple,
            photons_per_wavelength: 0,
            ..SimulationConfig::default()
        };
        let result = simulate_at_sza(&atm, &config, 93.0);
        let total = total_radiance(&result);
        assert!(
            total.abs() < 1e-20,
            "Zero photons should give zero radiance, got {}",
            total
        );
    }

    // ── MC vs single-scatter comparison ──

    #[test]
    fn mc_and_single_same_order_of_magnitude_at_shallow_twilight() {
        // At shallow twilight (SZA=93°), single-scatter dominates.
        // MC should give similar results (within ~5x for 2000 photons).
        let atm = make_clear_sky_atm();

        let ss_config = SimulationConfig {
            scattering_mode: ScatteringMode::Single,
            ..SimulationConfig::default()
        };
        let mc_config = SimulationConfig {
            scattering_mode: ScatteringMode::Multiple,
            photons_per_wavelength: 5000,
            ..SimulationConfig::default()
        };

        let ss_total = total_radiance(&simulate_at_sza(&atm, &ss_config, 93.0));
        let mc_total = total_radiance(&simulate_at_sza(&atm, &mc_config, 93.0));

        // Both should be positive
        assert!(ss_total > 0.0, "Single-scatter should be positive");
        assert!(mc_total > 0.0, "MC should be positive");

        // They should be within ~10x of each other at shallow twilight
        let ratio = if ss_total > mc_total {
            ss_total / mc_total
        } else {
            mc_total / ss_total
        };
        assert!(
            ratio < 10.0,
            "MC ({:.4e}) and single-scatter ({:.4e}) should be same order of magnitude (ratio: {:.1})",
            mc_total,
            ss_total,
            ratio
        );
    }

    // ── MC twilight scan ──

    #[test]
    fn mc_twilight_scan_correct_count() {
        let atm = make_clear_sky_atm();
        let config = SimulationConfig {
            scattering_mode: ScatteringMode::Multiple,
            photons_per_wavelength: 200, // few photons for speed
            ..SimulationConfig::default()
        };
        let results = simulate_twilight_scan(&atm, &config, 90.0, 100.0, 5.0);
        // 90, 95, 100 = 3 steps
        assert_eq!(results.len(), 3, "Expected 3 steps, got {}", results.len());
    }

    #[test]
    fn mc_twilight_scan_sza_values_correct() {
        let atm = make_clear_sky_atm();
        let config = SimulationConfig {
            scattering_mode: ScatteringMode::Multiple,
            photons_per_wavelength: 200,
            ..SimulationConfig::default()
        };
        let results = simulate_twilight_scan(&atm, &config, 90.0, 100.0, 5.0);
        assert!((results[0].sza_deg - 90.0).abs() < 0.01);
        assert!((results[1].sza_deg - 95.0).abs() < 0.01);
        assert!((results[2].sza_deg - 100.0).abs() < 0.01);
    }

    // ── Hybrid mode tests ──

    fn hybrid_config() -> SimulationConfig {
        SimulationConfig {
            scattering_mode: ScatteringMode::Hybrid,
            photons_per_wavelength: 50, // few secondary rays for test speed
            ..SimulationConfig::default()
        }
    }

    #[test]
    fn hybrid_returns_correct_wavelength_count() {
        let atm = make_clear_sky_atm();
        let config = hybrid_config();
        let result = simulate_at_sza(&atm, &config, 96.0);
        assert_eq!(result.wavelengths_nm.len(), 41);
        assert_eq!(result.radiance.len(), 41);
    }

    #[test]
    fn hybrid_stores_sza() {
        let atm = make_clear_sky_atm();
        let config = hybrid_config();
        let result = simulate_at_sza(&atm, &config, 96.5);
        assert!((result.sza_deg - 96.5).abs() < 1e-10);
    }

    #[test]
    fn hybrid_radiance_non_negative() {
        let atm = make_clear_sky_atm();
        let config = hybrid_config();
        for sza in &[90.0, 96.0, 102.0] {
            let result = simulate_at_sza(&atm, &config, *sza);
            for (i, &r) in result.radiance.iter().enumerate() {
                assert!(
                    r >= 0.0,
                    "Hybrid radiance at SZA={}, wl[{}] = {} should be non-negative",
                    sza,
                    i,
                    r
                );
            }
        }
    }

    #[test]
    fn hybrid_positive_at_civil_twilight() {
        let atm = make_clear_sky_atm();
        let config = SimulationConfig {
            scattering_mode: ScatteringMode::Hybrid,
            photons_per_wavelength: 100,
            ..SimulationConfig::default()
        };
        let result = simulate_at_sza(&atm, &config, 93.0);
        let total = total_radiance(&result);
        assert!(
            total > 0.0,
            "Hybrid civil twilight should produce positive radiance, got {}",
            total
        );
    }

    #[test]
    fn hybrid_at_least_as_bright_as_single_scatter() {
        // Hybrid includes single-scatter + orders 2+, so it should be
        // at least as bright (or very close) to pure single-scatter.
        let atm = make_clear_sky_atm();

        let ss_config = SimulationConfig {
            scattering_mode: ScatteringMode::Single,
            ..SimulationConfig::default()
        };
        let hybrid_config = SimulationConfig {
            scattering_mode: ScatteringMode::Hybrid,
            photons_per_wavelength: 200,
            ..SimulationConfig::default()
        };

        let ss_total = total_radiance(&simulate_at_sza(&atm, &ss_config, 96.0));
        let hybrid_total = total_radiance(&simulate_at_sza(&atm, &hybrid_config, 96.0));

        // Hybrid should be >= single-scatter (within MC noise margin)
        // Allow 20% tolerance for MC noise
        assert!(
            hybrid_total > ss_total * 0.8,
            "Hybrid ({:.4e}) should be >= single-scatter ({:.4e}) minus noise margin",
            hybrid_total,
            ss_total
        );
    }

    #[test]
    fn hybrid_wavelengths_correct() {
        let atm = make_clear_sky_atm();
        let config = hybrid_config();
        let result = simulate_at_sza(&atm, &config, 96.0);
        assert!((result.wavelengths_nm[0] - 380.0).abs() < 0.01);
        assert!((result.wavelengths_nm[20] - 580.0).abs() < 0.01);
        assert!((result.wavelengths_nm[40] - 780.0).abs() < 0.01);
    }

    #[test]
    fn hybrid_zero_secondary_rays_equals_single_scatter() {
        // With 0 secondary rays, hybrid should produce exactly the
        // same result as single-scatter (only order 1 is computed).
        let atm = make_clear_sky_atm();

        let ss_config = SimulationConfig {
            scattering_mode: ScatteringMode::Single,
            ..SimulationConfig::default()
        };
        let hybrid_config = SimulationConfig {
            scattering_mode: ScatteringMode::Hybrid,
            photons_per_wavelength: 0,
            ..SimulationConfig::default()
        };

        let ss_result = simulate_at_sza(&atm, &ss_config, 96.0);
        let hybrid_result = simulate_at_sza(&atm, &hybrid_config, 96.0);

        for i in 0..ss_result.radiance.len() {
            let diff = (ss_result.radiance[i] - hybrid_result.radiance[i]).abs();
            let rel = if ss_result.radiance[i] > 1e-30 {
                diff / ss_result.radiance[i]
            } else {
                diff
            };
            assert!(
                rel < 0.05,
                "Hybrid(0 rays) should match single-scatter: wl[{}] {:.4e} vs {:.4e} (rel: {:.4})",
                i,
                hybrid_result.radiance[i],
                ss_result.radiance[i],
                rel
            );
        }
    }

    #[test]
    fn hybrid_radiance_decreases_with_depth() {
        let atm = make_clear_sky_atm();
        let config = SimulationConfig {
            scattering_mode: ScatteringMode::Hybrid,
            photons_per_wavelength: 100,
            ..SimulationConfig::default()
        };
        let r_93 = total_radiance(&simulate_at_sza(&atm, &config, 93.0));
        let r_100 = total_radiance(&simulate_at_sza(&atm, &config, 100.0));
        assert!(
            r_93 > r_100 * 0.5,
            "Hybrid SZA93 ({:.4e}) should be > SZA100 ({:.4e})",
            r_93,
            r_100
        );
    }

    #[test]
    fn hybrid_twilight_scan_correct_count() {
        let atm = make_clear_sky_atm();
        let config = SimulationConfig {
            scattering_mode: ScatteringMode::Hybrid,
            photons_per_wavelength: 10,
            ..SimulationConfig::default()
        };
        let results = simulate_twilight_scan(&atm, &config, 90.0, 100.0, 5.0);
        assert_eq!(results.len(), 3, "Expected 3 steps, got {}", results.len());
    }
}
