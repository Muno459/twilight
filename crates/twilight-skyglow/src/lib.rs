#![allow(clippy::manual_clamp)]
//! Light pollution skyglow model for twilight prayer times.
//!
//! This crate models the contribution of artificial light (skyglow) to the
//! twilight sky brightness. Light pollution shifts prayer times:
//! - **Isha**: Sky never gets as dark as it would naturally, so the "red/white
//!   glow disappears" threshold is reached at a deeper solar depression (later).
//! - **Fajr**: Dawn must outshine the artificial background, so true dawn is
//!   perceived later.
//!
//! # Architecture
//!
//! Two complementary models compute the artificial sky brightness:
//!
//! 1. **Garstang analytical model** (distant sources, >30km): Single-scatter
//!    radiative transfer integration using the same atmosphere/aerosol parameters
//!    as the MCRT engine. Fast and accurate for distant cities.
//!
//! 2. **MCRT ground-source tracer** (nearby sources, <30km): Full photon tracing
//!    from ground-level light sources through the atmosphere. Captures cloud
//!    reflection, terrain blocking, and multiple scattering effects.
//!
//! # Data Sources
//!
//! Light source intensities come from VIIRS satellite nighttime radiance data:
//! - **Embedded lookup** (~10km resolution, always available offline)
//! - **World Bank S3** (~750m COG tiles, free, no auth)
//! - **Manual input** (Bortle class or direct radiance value)
//!
//! # Spectral Model
//!
//! Artificial light has a distinct spectrum that affects twilight color perception:
//! - **HPS sodium**: Narrow emission around 589nm (orange)
//! - **White LED**: Broad spectrum with blue peak ~450nm and phosphor peak ~580nm
//! - Mix ratio is configurable (pre-2015 cities mostly HPS, post-2020 mostly LED)

pub mod angular;
pub mod bortle;
pub mod garstang;
pub mod spectrum;

/// Skyglow computation configuration.
#[derive(Debug, Clone)]
pub struct SkyglowConfig {
    /// Observer latitude (degrees).
    pub latitude: f64,
    /// Observer longitude (degrees).
    pub longitude: f64,
    /// Observer elevation above sea level (meters).
    pub elevation: f64,
    /// Maximum radius to integrate light sources (km). Default 200.
    pub radius_km: f64,
    /// Radius within which to use full MCRT (km). Default 30.
    pub mcrt_radius_km: f64,
    /// LED fraction of total lighting (0.0 = all HPS, 1.0 = all LED). Default 0.5.
    pub led_fraction: f64,
    /// Fraction of light emitted directly upward (0.0 = fully shielded, 0.15 = typical).
    pub uplight_fraction: f64,
    /// Ground reflectance for indirect uplight (concrete ~0.25, asphalt ~0.07).
    pub ground_reflectance: f64,
}

impl Default for SkyglowConfig {
    fn default() -> Self {
        SkyglowConfig {
            latitude: 0.0,
            longitude: 0.0,
            elevation: 0.0,
            radius_km: 200.0,
            mcrt_radius_km: 30.0,
            led_fraction: 0.5,
            uplight_fraction: 0.10,
            ground_reflectance: 0.15,
        }
    }
}

/// Spectral skyglow result at a particular viewing direction.
#[derive(Debug, Clone)]
pub struct SkyglowResult {
    /// Artificial sky brightness at zenith (cd/m^2).
    pub zenith_luminance: f64,
    /// Artificial sky brightness at the specified viewing elevation (cd/m^2).
    pub directional_luminance: f64,
    /// Viewing elevation angle above horizon (degrees) used for directional result.
    pub view_elevation_deg: f64,
    /// Spectral radiance at each wavelength (W/m^2/sr/nm), matching the
    /// MCRT wavelength grid (380-780nm, 10nm steps, 41 bands).
    pub spectral_radiance: [f64; 64],
    /// Number of active wavelengths in spectral_radiance (typically 41).
    pub num_wavelengths: usize,
    /// Effective Bortle class (1-9) corresponding to the computed zenith luminance.
    pub bortle_class: u8,
    /// Total VIIRS-equivalent radiance integrated within the scan radius (nW/cm^2/sr).
    pub integrated_radiance: f64,
    /// Number of light source bins used in the computation.
    pub num_sources: usize,
}

/// Light source radiance provider.
///
/// Implementations provide VIIRS-like upward radiance values at geographic points.
pub trait RadianceSource {
    /// Return the upward radiance at (lat, lon) in nW/cm^2/sr.
    /// Returns None if no data is available at this location.
    fn radiance_at(&self, lat: f64, lon: f64) -> Option<f64>;

    /// Resolution of the underlying data in meters.
    fn resolution_m(&self) -> f64;

    /// Human-readable name of this source.
    fn name(&self) -> &str;
}

/// A manual radiance source that returns a constant value everywhere.
///
/// Useful for quick estimates when VIIRS data is not available.
#[derive(Debug, Clone)]
pub struct ConstantRadiance {
    /// Radiance in nW/cm^2/sr.
    pub radiance: f64,
}

impl RadianceSource for ConstantRadiance {
    fn radiance_at(&self, _lat: f64, _lon: f64) -> Option<f64> {
        Some(self.radiance)
    }
    fn resolution_m(&self) -> f64 {
        10_000.0
    }
    fn name(&self) -> &str {
        "Constant (user-provided)"
    }
}

/// Compute the zenith artificial sky brightness from a single radiance value
/// using the simplified Garstang model.
///
/// This is the quick-estimate function for when you just have a Bortle class
/// or a single VIIRS radiance reading at the observer's location.
///
/// For the full spatially-resolved computation, use [`garstang::compute_skyglow`].
pub fn quick_estimate(radiance_nw: f64, led_fraction: f64) -> SkyglowResult {
    let zenith_lum = bortle::radiance_to_zenith_luminance(radiance_nw);
    let bortle = bortle::luminance_to_bortle(zenith_lum);

    // Generate spectral radiance for the given LED fraction
    let (spectral, num_wl) = spectrum::mixed_spectrum(radiance_nw, led_fraction);

    // Angular model: at zenith
    let dir_lum = zenith_lum; // no angular correction for zenith

    SkyglowResult {
        zenith_luminance: zenith_lum,
        directional_luminance: dir_lum,
        view_elevation_deg: 90.0,
        spectral_radiance: spectral,
        num_wavelengths: num_wl,
        bortle_class: bortle,
        integrated_radiance: radiance_nw,
        num_sources: 1,
    }
}

/// Compute skyglow at a specified viewing angle above the horizon.
///
/// Light pollution is brighter toward the horizon and in the direction of
/// nearby cities. This function applies the angular model to a zenith estimate.
pub fn quick_estimate_at_angle(
    radiance_nw: f64,
    led_fraction: f64,
    view_elevation_deg: f64,
) -> SkyglowResult {
    let mut result = quick_estimate(radiance_nw, led_fraction);

    // Apply angular enhancement factor
    let factor = angular::enhancement_factor(view_elevation_deg);
    result.directional_luminance = result.zenith_luminance * factor;
    result.view_elevation_deg = view_elevation_deg;

    // Scale spectral radiance by the same factor
    for i in 0..result.num_wavelengths {
        result.spectral_radiance[i] *= factor;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
        let cfg = SkyglowConfig::default();
        assert!((cfg.radius_km - 200.0).abs() < 1e-10);
        assert!((cfg.mcrt_radius_km - 30.0).abs() < 1e-10);
        assert!((cfg.led_fraction - 0.5).abs() < 1e-10);
        assert!((cfg.uplight_fraction - 0.10).abs() < 1e-10);
        assert!((cfg.ground_reflectance - 0.15).abs() < 1e-10);
    }

    #[test]
    fn constant_radiance_source() {
        let src = ConstantRadiance { radiance: 42.0 };
        assert_eq!(src.radiance_at(0.0, 0.0), Some(42.0));
        assert_eq!(src.radiance_at(90.0, 180.0), Some(42.0));
        assert_eq!(src.name(), "Constant (user-provided)");
    }

    #[test]
    fn quick_estimate_dark_site() {
        // Bortle 1 site: radiance ~0.2 nW/cm^2/sr
        let result = quick_estimate(0.2, 0.0);
        assert!(result.zenith_luminance > 0.0);
        assert!(result.bortle_class <= 2);
    }

    #[test]
    fn quick_estimate_city() {
        // Major city: radiance ~100 nW/cm^2/sr
        let result = quick_estimate(100.0, 0.7);
        assert!(result.zenith_luminance > 1.0);
        assert!(result.bortle_class >= 7);
    }

    #[test]
    fn quick_estimate_at_horizon_brighter_than_zenith() {
        let result = quick_estimate_at_angle(50.0, 0.5, 10.0);
        let zenith = quick_estimate(50.0, 0.5);
        assert!(
            result.directional_luminance > zenith.zenith_luminance,
            "Sky should be brighter toward horizon: dir={} > zen={}",
            result.directional_luminance,
            zenith.zenith_luminance
        );
    }

    #[test]
    fn quick_estimate_spectral_has_data() {
        let result = quick_estimate(50.0, 0.5);
        assert_eq!(result.num_wavelengths, 41);
        // At least some wavelengths should have positive radiance
        let sum: f64 = result.spectral_radiance[..41].iter().sum();
        assert!(sum > 0.0, "Spectral radiance should be positive");
    }
}
