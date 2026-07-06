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
//! A Garstang (1986)-style single-scatter integration lives in [`garstang`].
//! Its ABSOLUTE magnitudes are still not what the pipeline consumes (the
//! amplitude rail stays on the Falchi/atlas calibration via [`spectrum`]),
//! but its slant-LOS generalization ([`garstang::slant_brightness`]) now
//! supplies the AZIMUTHAL STRUCTURE of the khayt veil through
//! [`DirectionalSkyglow`] / [`directional_veils`]: per-patch veils whose
//! all-azimuth mean is pinned to the isotropic atlas value (structure from
//! Garstang+VIIRS, amplitude from the atlas rail).
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

/// Azimuthally-resolved artificial skyglow: VIIRS-binned ground light
/// sources plus the Garstang integration parameters, carried by callers
/// (the prayer pipeline, the CLI) that want per-direction veils instead
/// of one isotropic value.
///
/// # NORMALIZATION DECISION (read before touching the amplitudes)
///
/// The Garstang slant integral supplies only the azimuthal STRUCTURE of
/// the veil; the absolute AMPLITUDE stays on the validated Falchi/atlas
/// photometric rail that `quick_estimate` and the khayt veil already
/// use. Concretely, [`DirectionalSkyglow::azimuth_ratios`] normalizes
/// the slant brightnesses so their ALL-AZIMUTH AVERAGE at the patch
/// elevation is exactly 1.0, and a per-patch veil is
/// `(isotropic veil at that elevation) x ratio`. The atlas-equivalent
/// isotropic mean of the directional veils therefore matches the old
/// isotropic path BY CONSTRUCTION:
///
///   structure from Garstang + VIIRS, amplitude from the atlas rail.
///
/// This keeps the absolute calibration anchored to the validated atlas
/// (no tuned constant enters), while an observer south of a city
/// correctly sees a brighter northern sky and a darker southern dawn
/// horizon.
#[derive(Debug, Clone)]
pub struct DirectionalSkyglow {
    /// Binned ground sources (distance/azimuth from the observer).
    pub sources: Vec<garstang::LightSource>,
    /// Garstang integration parameters (observer elevation, AOD, ...).
    pub config: garstang::GarstangConfig,
}

/// Azimuth samples of the all-azimuth normalization mean (5-degree
/// steps; the source binning itself is 10-degree at its finest).
const MEAN_AZ_SAMPLES: usize = 72;

impl DirectionalSkyglow {
    /// Bin a radiance grid around the observer into ground sources.
    ///
    /// Returns `None` when the source yields no usable light bins
    /// (missing tile, all-dark, nodata) so the caller can fall back to
    /// the isotropic path LOUDLY instead of propagating a silent guess.
    pub fn from_radiance_source(
        source: &dyn RadianceSource,
        observer_lat: f64,
        observer_lon: f64,
        config: garstang::GarstangConfig,
        radius_km: f64,
    ) -> Option<Self> {
        let sources = garstang::bin_sources(source, observer_lat, observer_lon, radius_km);
        if sources.is_empty() {
            return None;
        }
        Some(DirectionalSkyglow { sources, config })
    }

    /// Slant brightness at each requested azimuth, DIVIDED by the
    /// all-azimuth mean slant brightness at the same elevation (the
    /// normalization that pins the amplitude to the atlas rail; see the
    /// type-level docs). Returns `None` when the structure is
    /// undefined (no sources, or a degenerate zero/non-finite mean) so
    /// callers fall back to the isotropic veil.
    pub fn azimuth_ratios(&self, azimuths_deg: &[f64], elevation_deg: f64) -> Option<Vec<f64>> {
        if self.sources.is_empty() {
            return None;
        }
        let mean = (0..MEAN_AZ_SAMPLES)
            .map(|i| {
                garstang::slant_brightness(
                    &self.sources,
                    &self.config,
                    i as f64 * 360.0 / MEAN_AZ_SAMPLES as f64,
                    elevation_deg,
                )
            })
            .sum::<f64>()
            / MEAN_AZ_SAMPLES as f64;
        if !mean.is_finite() || mean <= 0.0 {
            return None;
        }
        Some(
            azimuths_deg
                .iter()
                .map(|&az| {
                    garstang::slant_brightness(&self.sources, &self.config, az, elevation_deg)
                        / mean
                })
                .collect(),
        )
    }
}

/// Per-patch artificial veils `(mesopic cd/m^2, red-band cd/m^2)` at the
/// given absolute azimuths and patch elevation, on the SAME photometric
/// rail as [`quick_estimate`]: the spectrum is calibrated to the Falchi
/// zenith luminance, its mesopic/red bands are lifted by the Duriscoe
/// elevation factor (exactly the isotropic khayt veil), and each patch
/// is then scaled by the normalized Garstang slant ratio - so the
/// all-azimuth average of these veils equals the isotropic veil (see
/// [`DirectionalSkyglow`] for the normalization decision).
///
/// `radiance_nw`/`led_fraction` must be the SAME values the isotropic
/// path was fed (the observer's VIIRS radiance and lighting mix).
///
/// Returns `None` when the directional structure is undefined (no
/// sources / degenerate mean): callers MUST fall back to the isotropic
/// path and say so, not guess.
pub fn directional_veils(
    directional: &DirectionalSkyglow,
    radiance_nw: f64,
    led_fraction: f64,
    patch_azimuths_deg: &[f64],
    patch_elevation_deg: f64,
) -> Option<Vec<(f64, f64)>> {
    let ratios = directional.azimuth_ratios(patch_azimuths_deg, patch_elevation_deg)?;
    let sg = quick_estimate(radiance_nw, led_fraction);
    let n = sg.num_wavelengths.min(sg.spectral_radiance.len());
    let wl: Vec<f64> = (0..n).map(|i| 380.0 + 10.0 * i as f64).collect();
    let mes = twilight_threshold::luminance::mesopic_luminance(&wl, &sg.spectral_radiance[..n]);
    let red = twilight_threshold::luminance::red_band_luminance(&wl, &sg.spectral_radiance[..n]);
    let lift = angular::enhancement_factor(patch_elevation_deg);
    if mes <= 1e-30 {
        // Degenerate spectrum (zero/negative radiance): zero veil
        // everywhere, same convention as the isotropic khayt veil.
        return Some(vec![(0.0, 0.0); patch_azimuths_deg.len()]);
    }
    Some(
        ratios
            .iter()
            .map(|r| (mes * lift * r, red * lift * r))
            .collect(),
    )
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
    if phot > 1e-30 && phot.is_finite() && zenith_lum.is_finite() && zenith_lum > 0.0 {
        let scale = (zenith_lum * 1e-3) / phot;
        for v in spectral.iter_mut().take(num_wl) {
            *v *= scale;
        }
    } else {
        // Degenerate input (zero, negative, or non-finite radiance):
        // ZERO the spectrum rather than passing the uncalibrated
        // template through (which is ~2 orders bright, and negative
        // input would inject a huge negative veil: review round 2).
        for v in spectral.iter_mut().take(num_wl) {
            *v = 0.0;
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

    /// Degenerate radiance (zero, negative, NaN) must yield a ZERO
    /// spectrum, never the uncalibrated template (review round 2).
    #[test]
    fn degenerate_radiance_zeroes_spectrum() {
        for &r in &[0.0f64, -5.0, f64::NAN] {
            let sg = quick_estimate(r, 0.3);
            let max = sg.spectral_radiance[..sg.num_wavelengths]
                .iter()
                .cloned()
                .fold(0.0f64, f64::max);
            assert!(
                max == 0.0,
                "radiance {r}: spectrum must be zeroed, got max {max:e}"
            );
        }
    }

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

    /// The all-azimuth average of the directional veils must equal the
    /// isotropic veil (spectrum mesopic x elevation lift) EXACTLY (up
    /// to FP roundoff): this is the normalization contract that keeps
    /// the amplitude on the atlas rail while the slant integral only
    /// contributes structure.
    #[test]
    fn directional_veils_average_to_isotropic() {
        let d = DirectionalSkyglow {
            sources: vec![garstang::LightSource {
                distance_m: 6_400.0, // OpenFajr-like: city 4 miles away
                azimuth_deg: 0.0,
                upward_flux: 5e6,
            }],
            config: garstang::GarstangConfig::default(),
        };
        let elev = 3.0;
        let (radiance, led) = (162.55, 0.15);
        // Probe on the same uniform grid the normalization mean uses.
        let azs: Vec<f64> = (0..MEAN_AZ_SAMPLES)
            .map(|i| i as f64 * 360.0 / MEAN_AZ_SAMPLES as f64)
            .collect();
        let veils = directional_veils(&d, radiance, led, &azs, elev).expect("structure");
        let mean_mes = veils.iter().map(|v| v.0).sum::<f64>() / veils.len() as f64;

        let sg = quick_estimate(radiance, led);
        let wl: Vec<f64> = (0..sg.num_wavelengths).map(|i| 380.0 + 10.0 * i as f64).collect();
        let iso_mes = twilight_threshold::luminance::mesopic_luminance(
            &wl,
            &sg.spectral_radiance[..sg.num_wavelengths],
        ) * angular::enhancement_factor(elev);
        assert!(
            (mean_mes - iso_mes).abs() / iso_mes < 1e-9,
            "all-azimuth mean {mean_mes:e} must equal the isotropic veil {iso_mes:e}"
        );
        // And the structure is real: toward the city beats away from it.
        assert!(
            veils[0].0 > veils[MEAN_AZ_SAMPLES / 2].0,
            "veil toward the city ({:e}) must exceed the anti-city veil ({:e})",
            veils[0].0,
            veils[MEAN_AZ_SAMPLES / 2].0
        );
    }

    /// No usable sources -> None (loud fallback at the caller), never a
    /// silent isotropic guess dressed up as directional data.
    #[test]
    fn directional_from_dark_source_is_none() {
        let dark = ConstantRadiance { radiance: 0.0 };
        let d = DirectionalSkyglow::from_radiance_source(
            &dark,
            52.44,
            -1.95,
            garstang::GarstangConfig::default(),
            200.0,
        );
        assert!(d.is_none(), "all-dark grid must yield no directional model");
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
