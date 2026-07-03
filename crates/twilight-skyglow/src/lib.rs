//! Light pollution skyglow model for twilight prayer times.
//!
//! This crate models the contribution of artificial light (skyglow) to the
//! twilight sky brightness. Light pollution shifts prayer times:
//! - **Isha**: Sky never gets as dark as it would naturally, so the "red/white
//!   glow disappears" threshold is reached at a deeper solar depression (later).
//! - **Fajr**: Dawn must outshine the artificial background, so true dawn is
//!   perceived later.
//!
//! # Architecture (current reality)
//!
//! Two automatic satellite input paths exist alongside the manual one:
//! - [`atlas`]: the David Lorenz 2024 light-pollution atlas (VIIRS-based,
//!   PROPAGATED artificial zenith sky brightness - already the observer-sky
//!   quantity the engine needs), fetched as remote binary tiles and cached
//!   on disk.
//! - [`dnb`]: daily VIIRS Black Marble nighttime-lights radiance (GIBS,
//!   gap-filled and BRDF-corrected), used to scale the atlas from its 2024
//!   epoch to the present, or standing alone where the atlas has no data.
//!
//! The **manual** path (caller supplies a Bortle class 1-9 or a VIIRS-style
//! radiance in nW/cm^2/sr) remains as the override and offline fallback.
//! There is still no ground-source photon tracer.
//!
//! A Garstang (1986)-style single-scatter integration exists in [`garstang`]
//! but is NOT currently wired into the prayer-time pipeline; the pipeline
//! uses the simpler zenith-luminance estimate from [`spectrum`]. Wiring the
//! Garstang propagation in (and validating its magnitudes) is an open task.
//!
//! # Spectral Model
//!
//! Artificial light has a distinct spectrum that affects twilight color perception:
//! - **HPS sodium**: Narrow emission around 589nm (orange)
//! - **White LED**: Broad spectrum with blue peak ~450nm and phosphor peak ~580nm
//! - Mix ratio is configurable (pre-2015 cities mostly HPS, post-2020 mostly LED)

#![allow(clippy::needless_range_loop)] // parallel spectral arrays

pub mod angular;
pub mod atlas;
pub mod bortle;
pub mod dnb;
pub mod dnb_colormap;
pub mod error;
pub mod garstang;
pub mod spectrum;

pub use error::SkyglowError;

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
    /// Artificial sky brightness at zenith (mcd/m^2: the unit the
    /// Falchi-fit producer `bortle::radiance_to_zenith_luminance` and
    /// its named inverse emit; this doc previously said cd/m^2, which
    /// caused a 1000x veil bug in the khayt consumer).
    pub zenith_luminance: f64,
    /// Artificial sky brightness at the viewing elevation (mcd/m^2).
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
/// The spatially-resolved Garstang integration lives in [`garstang`] but is
/// not yet wired into the prayer pipeline.
pub fn quick_estimate(radiance_nw: f64, led_fraction: f64) -> SkyglowResult {
    let zenith_lum = bortle::radiance_to_zenith_luminance(radiance_nw);
    let bortle = bortle::luminance_to_bortle(zenith_lum);

    // Generate spectral radiance for the given LED fraction. The mixed
    // spectrum carries the source SHAPE (HPS/LED); its raw amplitude is
    // the full upward VIIRS radiance, which is NOT the sky brightness:
    // only a small scattered fraction returns as skyglow, and the
    // calibrated amplitude is exactly what the Falchi fit measures.
    // Scale the spectrum so its photopic luminance equals the Falchi
    // zenith value (one photometric rail; before this, the legacy
    // spectral-injection path was ~2 orders of magnitude too bright).
    let (mut spectral, num_wl) = spectrum::mixed_spectrum(radiance_nw, led_fraction);
    let wl: Vec<f64> = (0..num_wl).map(|i| 380.0 + 10.0 * i as f64).collect();
    let phot = twilight_threshold::luminance::photopic_luminance(&wl, &spectral[..num_wl]);
    if phot > 1e-30 {
        let scale = (zenith_lum * 1e-3) / phot;
        for v in spectral.iter_mut().take(num_wl) {
            *v *= scale;
        }
    }

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

    /// The spectral injection must sit on the same photometric rail as
    /// the Falchi zenith luminance: photopic(spectral) == zenith mcd
    /// converted to cd/m^2. Regression for the ~2-orders-bright legacy
    /// inject_skyglow path.
    #[test]
    fn spectral_radiance_calibrated_to_falchi_zenith() {
        for &r in &[1.0f64, 15.0, 60.0] {
            let sg = quick_estimate(r, 0.3);
            let wl: Vec<f64> = (0..sg.num_wavelengths).map(|i| 380.0 + 10.0 * i as f64).collect();
            let phot = twilight_threshold::luminance::photopic_luminance(
                &wl,
                &sg.spectral_radiance[..sg.num_wavelengths],
            );
            let expect = sg.zenith_luminance * 1e-3;
            assert!(
                (phot - expect).abs() <= 1e-9 + 1e-6 * expect,
                "radiance {r}: photopic {phot:.3e} != zenith {expect:.3e} cd/m^2"
            );
        }
    }

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
