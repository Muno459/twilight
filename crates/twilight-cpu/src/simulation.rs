//! High-level MCRT simulation driver.
//!
//! Takes observer location, solar zenith angle, and atmosphere model,
//! and computes spectral sky radiance. Uses single-scattering line-of-sight
//! integration as the primary method (deterministic, no noise), with
//! Monte Carlo multiple scattering as a future enhancement.
//!
//! The radiance output is in physical units [W/m²/sr/nm] when solar
//! irradiance weighting is enabled (default).

use twilight_core::atmosphere::AtmosphereModel;
use twilight_core::geometry::{geographic_to_ecef, solar_direction_ecef};
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
        }
    }
}

/// Run single-scattering simulation at a single solar zenith angle.
///
/// Returns spectral radiance across all wavelengths in the atmosphere model.
/// This is deterministic (no Monte Carlo noise).
///
/// When `config.apply_solar_irradiance` is true, the output is in physical
/// units [W/m²/sr/nm]. The single-scatter integral gives:
///   I(λ) = F_sun(λ) × ∫ β_scat × P(θ)/(4π) × T_sun × T_obs ds
/// where F_sun(λ) is the TOA solar spectral irradiance [W/m²/nm].
pub fn simulate_at_sza(
    atm: &AtmosphereModel,
    config: &SimulationConfig,
    sza_deg: f64,
) -> SpectralResult {
    // Observer position in ECEF
    let observer_pos = geographic_to_ecef(config.latitude, config.longitude, config.elevation);

    // Sun direction in ECEF
    let sun_dir = solar_direction_ecef(
        sza_deg,
        config.solar_azimuth,
        config.latitude,
        config.longitude,
    );

    // View direction: look toward the sun azimuth at the specified zenith angle
    let view_dir = solar_direction_ecef(
        config.view_zenith,
        config.solar_azimuth,
        config.latitude,
        config.longitude,
    );

    // Compute single-scattering spectrum (all wavelengths at once)
    let radiance_array =
        single_scatter::single_scatter_spectrum(atm, observer_pos, view_dir, sun_dir);

    let num_wl = atm.num_wavelengths;
    let mut wavelengths = Vec::with_capacity(num_wl);
    let mut radiance = Vec::with_capacity(num_wl);

    for w in 0..num_wl {
        wavelengths.push(atm.wavelengths_nm[w]);
        // Apply solar irradiance weighting: multiply by F_sun(λ) [W/m²/nm]
        // The SOLAR_IRRADIANCE array is aligned with the same wavelength grid
        // (both use 380-780nm at 10nm steps, 41 values).
        let r = if config.apply_solar_irradiance && w < SOLAR_IRRADIANCE.len() {
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
}
