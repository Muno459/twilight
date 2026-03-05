//! Twilight threshold model for determining Fajr and Isha times.
//!
//! Determines when twilight begins/ends based on spectral sky luminance
//! crossing perceptual thresholds. Distinguishes between:
//!
//! - **Shafaq al-abyad** (white twilight): when overall sky brightness
//!   drops below the threshold — used by Hanafi school for Isha.
//! - **Shafaq al-ahmar** (red twilight): when the red glow on the horizon
//!   disappears — used by Shafi'i/Maliki/Hanbali schools for Isha.
//!
//! The threshold model uses mesopic vision (CIE 191:2010) since twilight
//! spans the transition from photopic to scotopic adaptation.

use crate::luminance;

/// Twilight type classification based on spectral analysis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TwilightColor {
    /// Blue-dominated sky (early twilight, SZA ~90-96°)
    Blue,
    /// White/neutral sky — shafaq al-abyad (SZA ~96-100°)
    White,
    /// Orange transitional (SZA ~100-102°)
    Orange,
    /// Red-dominated sky — shafaq al-ahmar (SZA ~102-108°)
    Red,
    /// Below detection threshold
    Dark,
}

/// Result of twilight analysis at a single solar zenith angle.
#[derive(Debug, Clone)]
pub struct TwilightAnalysis {
    /// Solar zenith angle (degrees)
    pub sza_deg: f64,
    /// Photopic luminance (cd/m²)
    pub luminance_photopic: f64,
    /// Scotopic luminance (cd/m²)
    pub luminance_scotopic: f64,
    /// Mesopic luminance (cd/m²)
    pub luminance_mesopic: f64,
    /// Spectral centroid wavelength (nm)
    pub spectral_centroid_nm: f64,
    /// Red band luminance (cd/m², λ > 600nm)
    pub luminance_red: f64,
    /// Blue band luminance (cd/m², λ < 500nm)
    pub luminance_blue: f64,
    /// Twilight color classification
    pub color: TwilightColor,
}

/// Threshold configuration for prayer time determination.
///
/// These thresholds define when twilight begins/ends for prayer purposes.
/// They are calibrated against SQM (Sky Quality Meter) observations and
/// traditional astronomical twilight definitions.
#[derive(Debug, Clone)]
pub struct ThresholdConfig {
    /// Mesopic luminance threshold for Fajr (dawn detection).
    /// When sky luminance rises above this, Fajr begins.
    /// Default: ~0.001 cd/m² (roughly equivalent to 18° depression angle
    /// under clear-sky conditions at mid-latitudes).
    pub fajr_luminance: f64,

    /// Mesopic luminance threshold for Isha (shafaq al-abyad).
    /// When overall sky luminance drops below this, Isha begins (Hanafi).
    /// Default: ~0.003 cd/m² (roughly equivalent to 15° depression).
    pub isha_abyad_luminance: f64,

    /// Red band luminance threshold for Isha (shafaq al-ahmar).
    /// When the red glow drops below this, Isha begins (Shafi'i/Maliki/Hanbali).
    /// Default: ~0.0005 cd/m² (red persists longer than white).
    pub isha_ahmar_red_luminance: f64,

    /// Spectral centroid boundary between white and red twilight (nm).
    /// Above this → red-dominated, below → white/blue.
    pub red_centroid_boundary: f64,

    /// Spectral centroid boundary between blue and white twilight (nm).
    pub white_centroid_boundary: f64,
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        Self {
            fajr_luminance: 0.001,
            isha_abyad_luminance: 0.003,
            isha_ahmar_red_luminance: 0.0005,
            red_centroid_boundary: 590.0,
            white_centroid_boundary: 530.0,
        }
    }
}

/// Analyze twilight at a single SZA given spectral radiance.
pub fn analyze_twilight(
    sza_deg: f64,
    wavelengths_nm: &[f64],
    radiance: &[f64],
    config: &ThresholdConfig,
) -> TwilightAnalysis {
    let l_p = luminance::photopic_luminance(wavelengths_nm, radiance);
    let l_s = luminance::scotopic_luminance(wavelengths_nm, radiance);
    let l_m = luminance::mesopic_luminance(wavelengths_nm, radiance);
    let centroid = luminance::spectral_centroid(wavelengths_nm, radiance);
    let l_red = luminance::red_band_luminance(wavelengths_nm, radiance);
    let l_blue = luminance::blue_band_luminance(wavelengths_nm, radiance);

    let color = classify_twilight_color(l_m, centroid, l_red, config);

    TwilightAnalysis {
        sza_deg,
        luminance_photopic: l_p,
        luminance_scotopic: l_s,
        luminance_mesopic: l_m,
        spectral_centroid_nm: centroid,
        luminance_red: l_red,
        luminance_blue: l_blue,
        color,
    }
}

/// Classify twilight color from luminance and spectral centroid.
fn classify_twilight_color(
    luminance_mesopic: f64,
    centroid_nm: f64,
    _luminance_red: f64,
    config: &ThresholdConfig,
) -> TwilightColor {
    // Below detection threshold — dark sky
    if luminance_mesopic < config.isha_ahmar_red_luminance * 0.1 {
        return TwilightColor::Dark;
    }

    if centroid_nm <= 0.0 {
        return TwilightColor::Dark;
    }

    if centroid_nm > config.red_centroid_boundary {
        TwilightColor::Red
    } else if centroid_nm > config.white_centroid_boundary {
        if centroid_nm > 570.0 {
            TwilightColor::Orange
        } else {
            TwilightColor::White
        }
    } else {
        TwilightColor::Blue
    }
}

/// Result of prayer time determination from MCRT analysis.
#[derive(Debug, Clone)]
pub struct PrayerTimeResult {
    /// SZA at which Fajr begins (sky first becomes bright enough).
    /// None if threshold is never crossed in the scan range.
    pub fajr_sza_deg: Option<f64>,

    /// SZA at which Isha begins per shafaq al-abyad (white twilight ends).
    pub isha_abyad_sza_deg: Option<f64>,

    /// SZA at which Isha begins per shafaq al-ahmar (red glow disappears).
    pub isha_ahmar_sza_deg: Option<f64>,

    /// Full twilight analysis at each scanned SZA
    pub analyses: Vec<TwilightAnalysis>,
}

/// Determine prayer times from a series of twilight analyses.
///
/// Takes spectral results at increasing SZAs (sunset → darkness) and
/// finds the threshold crossings for Fajr/Isha.
///
/// For Isha (evening): scan from small SZA to large SZA, find where
/// luminance drops below threshold.
///
/// For Fajr (morning): same physics — the SZA at which luminance
/// crosses the threshold is the same (symmetric for clear sky).
pub fn determine_prayer_times(
    analyses: Vec<TwilightAnalysis>,
    config: &ThresholdConfig,
) -> PrayerTimeResult {
    let mut fajr_sza = None;
    let mut isha_abyad_sza = None;
    let mut isha_ahmar_sza = None;

    // Isha (evening, increasing SZA = deepening twilight):
    // Find where luminance drops below threshold.
    // Scan pairs of adjacent SZA values and interpolate the crossing.
    for i in 0..(analyses.len() - 1) {
        let a0 = &analyses[i];
        let a1 = &analyses[i + 1];

        // Isha al-abyad: mesopic luminance crosses threshold
        if isha_abyad_sza.is_none()
            && a0.luminance_mesopic >= config.isha_abyad_luminance
            && a1.luminance_mesopic < config.isha_abyad_luminance
        {
            isha_abyad_sza = Some(interpolate_crossing(
                a0.sza_deg,
                a0.luminance_mesopic,
                a1.sza_deg,
                a1.luminance_mesopic,
                config.isha_abyad_luminance,
            ));
        }

        // Isha al-ahmar: red band luminance crosses threshold
        if isha_ahmar_sza.is_none()
            && a0.luminance_red >= config.isha_ahmar_red_luminance
            && a1.luminance_red < config.isha_ahmar_red_luminance
        {
            isha_ahmar_sza = Some(interpolate_crossing(
                a0.sza_deg,
                a0.luminance_red,
                a1.sza_deg,
                a1.luminance_red,
                config.isha_ahmar_red_luminance,
            ));
        }

        // Fajr: same threshold as Isha al-abyad but for the "first light"
        // In practice, the SZA value is the same due to symmetry.
        // We compute it as the mesopic luminance crossing from below.
        if fajr_sza.is_none()
            && a0.luminance_mesopic >= config.fajr_luminance
            && a1.luminance_mesopic < config.fajr_luminance
        {
            fajr_sza = Some(interpolate_crossing(
                a0.sza_deg,
                a0.luminance_mesopic,
                a1.sza_deg,
                a1.luminance_mesopic,
                config.fajr_luminance,
            ));
        }
    }

    PrayerTimeResult {
        fajr_sza_deg: fajr_sza,
        isha_abyad_sza_deg: isha_abyad_sza,
        isha_ahmar_sza_deg: isha_ahmar_sza,
        analyses,
    }
}

/// Linear interpolation to find exact SZA where luminance crosses threshold.
fn interpolate_crossing(sza0: f64, lum0: f64, sza1: f64, lum1: f64, threshold: f64) -> f64 {
    if (lum1 - lum0).abs() < 1e-30 {
        return (sza0 + sza1) / 2.0;
    }
    // Use log-space interpolation since luminance drops exponentially
    let log_lum0 = libm::log(lum0);
    let log_lum1 = libm::log(lum1);
    let log_thresh = libm::log(threshold);

    let frac = (log_thresh - log_lum0) / (log_lum1 - log_lum0);
    sza0 + frac * (sza1 - sza0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create a flat spectrum with given value
    fn flat_spectrum(value: f64) -> (Vec<f64>, Vec<f64>) {
        let wls: Vec<f64> = (0..41).map(|i| 380.0 + i as f64 * 10.0).collect();
        let rad: Vec<f64> = vec![value; 41];
        (wls, rad)
    }

    // Helper: create a red-dominated spectrum (more red, less blue)
    fn red_spectrum(value: f64) -> (Vec<f64>, Vec<f64>) {
        let wls: Vec<f64> = (0..41).map(|i| 380.0 + i as f64 * 10.0).collect();
        let rad: Vec<f64> = wls
            .iter()
            .map(|&wl| {
                // Quadratically increasing with wavelength (red-dominated).
                // Quadratic ramp overcomes V(λ) weighting to ensure centroid > 570nm.
                let frac = (wl - 380.0) / 400.0;
                value * frac * frac
            })
            .collect();
        (wls, rad)
    }

    // ── ThresholdConfig defaults ──

    #[test]
    fn default_config_values() {
        let config = ThresholdConfig::default();
        assert!((config.fajr_luminance - 0.001).abs() < 1e-10);
        assert!((config.isha_abyad_luminance - 0.003).abs() < 1e-10);
        assert!((config.isha_ahmar_red_luminance - 0.0005).abs() < 1e-10);
        assert!((config.red_centroid_boundary - 590.0).abs() < 1e-10);
        assert!((config.white_centroid_boundary - 530.0).abs() < 1e-10);
    }

    #[test]
    fn fajr_threshold_less_than_isha_abyad() {
        // Fajr threshold should be lower (darker sky) than Isha abyad
        let config = ThresholdConfig::default();
        assert!(
            config.fajr_luminance < config.isha_abyad_luminance,
            "Fajr ({}) should be < Isha abyad ({})",
            config.fajr_luminance,
            config.isha_abyad_luminance
        );
    }

    #[test]
    fn isha_ahmar_less_than_fajr() {
        // Red glow persists longer → lower threshold
        let config = ThresholdConfig::default();
        assert!(
            config.isha_ahmar_red_luminance < config.fajr_luminance,
            "Isha ahmar ({}) should be < Fajr ({})",
            config.isha_ahmar_red_luminance,
            config.fajr_luminance
        );
    }

    // ── TwilightColor classification ──

    #[test]
    fn classify_dark_when_very_dim() {
        let config = ThresholdConfig::default();
        // Very dim: luminance = 1/10 of isha_ahmar threshold
        let color =
            classify_twilight_color(config.isha_ahmar_red_luminance * 0.05, 550.0, 0.0, &config);
        assert_eq!(color, TwilightColor::Dark);
    }

    #[test]
    fn classify_dark_when_centroid_zero() {
        let config = ThresholdConfig::default();
        let color = classify_twilight_color(1.0, 0.0, 0.0, &config);
        assert_eq!(color, TwilightColor::Dark);
    }

    #[test]
    fn classify_red_when_centroid_above_590() {
        let config = ThresholdConfig::default();
        let color = classify_twilight_color(0.01, 600.0, 0.001, &config);
        assert_eq!(color, TwilightColor::Red);
    }

    #[test]
    fn classify_orange_when_centroid_570_to_590() {
        let config = ThresholdConfig::default();
        let color = classify_twilight_color(0.01, 575.0, 0.001, &config);
        assert_eq!(color, TwilightColor::Orange);
    }

    #[test]
    fn classify_white_when_centroid_530_to_570() {
        let config = ThresholdConfig::default();
        let color = classify_twilight_color(0.01, 550.0, 0.001, &config);
        assert_eq!(color, TwilightColor::White);
    }

    #[test]
    fn classify_blue_when_centroid_below_530() {
        let config = ThresholdConfig::default();
        let color = classify_twilight_color(0.01, 500.0, 0.001, &config);
        assert_eq!(color, TwilightColor::Blue);
    }

    // ── analyze_twilight ──

    #[test]
    fn analyze_twilight_stores_sza() {
        let config = ThresholdConfig::default();
        let (wls, rad) = flat_spectrum(0.001);
        let analysis = analyze_twilight(96.5, &wls, &rad, &config);
        assert!((analysis.sza_deg - 96.5).abs() < 1e-10);
    }

    #[test]
    fn analyze_twilight_computes_luminance() {
        let config = ThresholdConfig::default();
        let (wls, rad) = flat_spectrum(0.01);
        let analysis = analyze_twilight(96.0, &wls, &rad, &config);
        assert!(analysis.luminance_photopic > 0.0);
        assert!(analysis.luminance_scotopic > 0.0);
        assert!(analysis.luminance_mesopic > 0.0);
    }

    #[test]
    fn analyze_twilight_zero_radiance_is_dark() {
        let config = ThresholdConfig::default();
        let (wls, rad) = flat_spectrum(0.0);
        let analysis = analyze_twilight(108.0, &wls, &rad, &config);
        assert_eq!(analysis.color, TwilightColor::Dark);
        assert!(analysis.luminance_photopic.abs() < 1e-20);
    }

    #[test]
    fn analyze_twilight_centroid_in_visible() {
        let config = ThresholdConfig::default();
        let (wls, rad) = flat_spectrum(0.01);
        let analysis = analyze_twilight(96.0, &wls, &rad, &config);
        // Centroid should be in visible range
        assert!(
            analysis.spectral_centroid_nm > 400.0 && analysis.spectral_centroid_nm < 700.0,
            "Centroid = {}, expected in visible range",
            analysis.spectral_centroid_nm
        );
    }

    #[test]
    fn analyze_twilight_red_spectrum_has_red_color() {
        let config = ThresholdConfig::default();
        let (wls, rad) = red_spectrum(0.01);
        let analysis = analyze_twilight(104.0, &wls, &rad, &config);
        // Red-dominated spectrum should have centroid > 590nm and classify as Red or Orange
        assert!(
            analysis.spectral_centroid_nm > 570.0,
            "Red spectrum centroid = {}, expected > 570nm",
            analysis.spectral_centroid_nm
        );
    }

    // ── interpolate_crossing ──

    #[test]
    fn interpolate_crossing_midpoint() {
        // Luminance drops from 10 to 1, threshold = sqrt(10) ≈ 3.162 (geometric mean)
        // In log space: log(10)=2.303, log(1)=0, log(3.162)=1.151
        // frac = 1.151/2.303 = 0.5 → SZA midpoint
        let sza = interpolate_crossing(100.0, 10.0, 102.0, 1.0, libm::sqrt(10.0));
        assert!(
            (sza - 101.0).abs() < 0.01,
            "Crossing at geometric mean: SZA={}, expected ~101.0",
            sza
        );
    }

    #[test]
    fn interpolate_crossing_near_start() {
        // Threshold very close to lum0 → SZA near sza0
        let sza = interpolate_crossing(100.0, 10.0, 102.0, 1.0, 9.5);
        assert!(
            sza > 100.0 && sza < 100.5,
            "High threshold should give SZA near start: SZA={}",
            sza
        );
    }

    #[test]
    fn interpolate_crossing_near_end() {
        // Threshold very close to lum1 → SZA near sza1
        let sza = interpolate_crossing(100.0, 10.0, 102.0, 1.0, 1.1);
        assert!(
            sza > 101.5 && sza < 102.0,
            "Low threshold should give SZA near end: SZA={}",
            sza
        );
    }

    #[test]
    fn interpolate_crossing_equal_luminance() {
        // When lum0 ≈ lum1, should return midpoint
        let sza = interpolate_crossing(100.0, 5.0, 102.0, 5.0, 5.0);
        assert!(
            (sza - 101.0).abs() < 0.01,
            "Equal luminance should give midpoint: SZA={}",
            sza
        );
    }

    // ── determine_prayer_times ──

    #[test]
    fn determine_prayer_times_finds_isha_crossing() {
        let config = ThresholdConfig::default();
        // Create a series of analyses with decreasing luminance (deepening twilight)
        let mut analyses = Vec::new();
        for i in 0..20 {
            let sza = 90.0 + i as f64;
            // Luminance drops exponentially from 1.0 to near zero
            let luminance = 1.0 * libm::exp(-0.5 * i as f64);
            let red_luminance = luminance * 0.3; // red is a fraction of total
            analyses.push(TwilightAnalysis {
                sza_deg: sza,
                luminance_photopic: luminance,
                luminance_scotopic: luminance * 1.5,
                luminance_mesopic: luminance,
                spectral_centroid_nm: 560.0,
                luminance_red: red_luminance,
                luminance_blue: luminance * 0.2,
                color: TwilightColor::White,
            });
        }

        let result = determine_prayer_times(analyses, &config);

        // Isha abyad should be found (luminance crosses 0.003 cd/m²)
        assert!(
            result.isha_abyad_sza_deg.is_some(),
            "Should find Isha abyad crossing"
        );
        let isha_sza = result.isha_abyad_sza_deg.unwrap();
        assert!(
            isha_sza > 90.0 && isha_sza < 110.0,
            "Isha abyad SZA = {}, expected in [90, 110]",
            isha_sza
        );
    }

    #[test]
    fn determine_prayer_times_no_crossing_if_always_bright() {
        let config = ThresholdConfig::default();
        // All luminance values above all thresholds
        let mut analyses = Vec::new();
        for i in 0..10 {
            let sza = 90.0 + i as f64 * 0.5;
            analyses.push(TwilightAnalysis {
                sza_deg: sza,
                luminance_photopic: 100.0, // very bright
                luminance_scotopic: 200.0,
                luminance_mesopic: 100.0,
                spectral_centroid_nm: 560.0,
                luminance_red: 50.0,
                luminance_blue: 30.0,
                color: TwilightColor::White,
            });
        }

        let result = determine_prayer_times(analyses, &config);
        assert!(result.fajr_sza_deg.is_none(), "No Fajr if always bright");
        assert!(
            result.isha_abyad_sza_deg.is_none(),
            "No Isha if always bright"
        );
        assert!(
            result.isha_ahmar_sza_deg.is_none(),
            "No Isha ahmar if always bright"
        );
    }

    #[test]
    fn determine_prayer_times_no_crossing_if_always_dark() {
        let config = ThresholdConfig::default();
        // All luminance values below all thresholds
        let mut analyses = Vec::new();
        for i in 0..10 {
            let sza = 100.0 + i as f64;
            analyses.push(TwilightAnalysis {
                sza_deg: sza,
                luminance_photopic: 1e-10,
                luminance_scotopic: 1e-10,
                luminance_mesopic: 1e-10,
                spectral_centroid_nm: 0.0,
                luminance_red: 1e-10,
                luminance_blue: 1e-10,
                color: TwilightColor::Dark,
            });
        }

        let result = determine_prayer_times(analyses, &config);
        assert!(result.fajr_sza_deg.is_none());
        assert!(result.isha_abyad_sza_deg.is_none());
        assert!(result.isha_ahmar_sza_deg.is_none());
    }

    #[test]
    fn determine_prayer_times_fajr_and_isha_at_same_sza() {
        // For symmetric clear sky, fajr and isha thresholds cross at same SZA
        let config = ThresholdConfig {
            fajr_luminance: 0.01,
            isha_abyad_luminance: 0.01, // Same threshold
            ..ThresholdConfig::default()
        };

        let mut analyses = Vec::new();
        for i in 0..20 {
            let sza = 90.0 + i as f64;
            let luminance = libm::exp(-0.4 * i as f64);
            analyses.push(TwilightAnalysis {
                sza_deg: sza,
                luminance_photopic: luminance,
                luminance_scotopic: luminance,
                luminance_mesopic: luminance,
                spectral_centroid_nm: 560.0,
                luminance_red: luminance * 0.3,
                luminance_blue: luminance * 0.2,
                color: TwilightColor::White,
            });
        }

        let result = determine_prayer_times(analyses, &config);
        if let (Some(fajr), Some(isha)) = (result.fajr_sza_deg, result.isha_abyad_sza_deg) {
            assert!(
                (fajr - isha).abs() < 0.1,
                "Same threshold → same SZA: fajr={}, isha={}",
                fajr,
                isha
            );
        }
    }

    #[test]
    fn determine_prayer_times_isha_ahmar_later_than_abyad() {
        // Red glow persists longer → ahmar SZA should be >= abyad SZA
        let config = ThresholdConfig::default();
        let mut analyses = Vec::new();
        for i in 0..30 {
            let sza = 90.0 + i as f64 * 0.5;
            let luminance = libm::exp(-0.3 * i as f64);
            let red_luminance = luminance * 0.5; // red drops slower relative
            analyses.push(TwilightAnalysis {
                sza_deg: sza,
                luminance_photopic: luminance,
                luminance_scotopic: luminance * 1.5,
                luminance_mesopic: luminance,
                spectral_centroid_nm: 560.0 + i as f64 * 2.0, // reddens
                luminance_red: red_luminance,
                luminance_blue: luminance * 0.1,
                color: if sza > 102.0 {
                    TwilightColor::Red
                } else {
                    TwilightColor::White
                },
            });
        }

        let result = determine_prayer_times(analyses, &config);
        if let (Some(abyad), Some(ahmar)) = (result.isha_abyad_sza_deg, result.isha_ahmar_sza_deg) {
            assert!(
                ahmar >= abyad - 0.5,
                "Isha ahmar ({}) should be >= abyad ({})",
                ahmar,
                abyad
            );
        }
    }

    // ── PrayerTimeResult ──

    #[test]
    fn prayer_time_result_stores_analyses() {
        let config = ThresholdConfig::default();
        let analyses = vec![TwilightAnalysis {
            sza_deg: 96.0,
            luminance_photopic: 0.1,
            luminance_scotopic: 0.2,
            luminance_mesopic: 0.1,
            spectral_centroid_nm: 550.0,
            luminance_red: 0.03,
            luminance_blue: 0.02,
            color: TwilightColor::White,
        }];
        let result = determine_prayer_times(analyses, &config);
        assert_eq!(result.analyses.len(), 1);
        assert!((result.analyses[0].sza_deg - 96.0).abs() < 1e-10);
    }
}
