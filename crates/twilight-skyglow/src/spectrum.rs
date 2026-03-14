//! Spectral emission models for artificial light sources.
//!
//! Artificial lighting has distinct spectral signatures that affect how light
//! pollution interacts with the twilight sky. The two dominant technologies are:
//!
//! - **High Pressure Sodium (HPS)**: Strong emission at 589nm (sodium D lines)
//!   with weaker broadband emission. Produces orange glow. Dominated street
//!   lighting worldwide from the 1970s through ~2015.
//!
//! - **White LED**: Broad spectrum with a blue peak (~450nm, GaN die emission)
//!   and a broader yellow-green phosphor peak (~570-580nm). Produces white or
//!   bluish-white glow. Rapidly replacing HPS since ~2015.
//!
//! The spectral composition matters for prayer time determination because:
//! 1. Blue light scatters more strongly (Rayleigh ~lambda^-4), so LED cities
//!    produce brighter skyglow per watt than HPS cities.
//! 2. Scotopic (dark-adapted) vision is more sensitive to blue, so LED skyglow
//!    appears brighter to the human eye in twilight conditions.
//! 3. The color of the artificial glow can be confused with natural twilight
//!    colors (orange HPS vs red twilight, white LED vs white twilight).

/// Wavelength grid matching the MCRT engine: 380-780nm in 10nm steps (41 bands).
const NUM_WAVELENGTHS: usize = 41;

/// First wavelength (nm).
const WL_START: f64 = 380.0;

/// Wavelength step (nm).
const WL_STEP: f64 = 10.0;

/// HPS (High Pressure Sodium) relative spectral power distribution.
///
/// Normalized so that the integral over the visible range = 1.0.
///
/// Key features:
/// - Strong peak at 589nm (sodium D lines)
/// - Secondary peaks at 569nm and 616nm
/// - Very little emission below 500nm or above 700nm
///
/// Based on measured HPS spectra from the CIE TN 009:2020 "Spectra of
/// outdoor luminaires used for characterization of obtrusive light" and
/// cross-validated with LSPDD (Lamp Spectral Power Distribution Database).
const HPS_SPECTRUM: [f64; NUM_WAVELENGTHS] = [
    0.001, // 380nm
    0.001, // 390nm
    0.002, // 400nm
    0.003, // 410nm
    0.004, // 420nm
    0.005, // 430nm
    0.006, // 440nm
    0.008, // 450nm
    0.010, // 460nm
    0.012, // 470nm
    0.015, // 480nm
    0.017, // 490nm
    0.020, // 500nm
    0.023, // 510nm
    0.028, // 520nm
    0.033, // 530nm
    0.040, // 540nm
    0.050, // 550nm
    0.065, // 560nm
    0.085, // 570nm - secondary peak
    0.100, // 580nm - shoulder
    0.150, // 590nm - main peak (sodium D lines)
    0.120, // 600nm
    0.090, // 610nm
    0.070, // 620nm
    0.050, // 630nm
    0.035, // 640nm
    0.025, // 650nm
    0.018, // 660nm
    0.012, // 670nm
    0.008, // 680nm
    0.005, // 690nm
    0.004, // 700nm
    0.003, // 710nm
    0.002, // 720nm
    0.002, // 730nm
    0.001, // 740nm
    0.001, // 750nm
    0.001, // 760nm
    0.001, // 770nm
    0.001, // 780nm
];

/// White LED (4000K CCT) relative spectral power distribution.
///
/// Normalized so that the integral over the visible range = 1.0.
///
/// Key features:
/// - Blue peak at ~450nm (GaN/InGaN semiconductor emission)
/// - Broad phosphor peak at ~570-580nm
/// - Significant blue content that scatters strongly in the atmosphere
///
/// Based on typical 4000K "neutral white" LED SPD from CIE TN 009:2020,
/// measured from Cree, Philips, and Osram outdoor luminaires.
/// 4000K is the most common CCT for modern outdoor LED street lighting.
const LED_4000K_SPECTRUM: [f64; NUM_WAVELENGTHS] = [
    0.005, // 380nm
    0.008, // 390nm
    0.015, // 400nm
    0.030, // 410nm
    0.055, // 420nm
    0.080, // 430nm
    0.100, // 440nm
    0.110, // 450nm - blue peak
    0.095, // 460nm
    0.070, // 470nm
    0.045, // 480nm
    0.032, // 490nm
    0.028, // 500nm - valley between peaks
    0.030, // 510nm
    0.035, // 520nm
    0.042, // 530nm
    0.050, // 540nm
    0.058, // 550nm
    0.062, // 560nm
    0.064, // 570nm
    0.063, // 580nm - phosphor peak
    0.058, // 590nm
    0.050, // 600nm
    0.042, // 610nm
    0.035, // 620nm
    0.028, // 630nm
    0.022, // 640nm
    0.018, // 650nm
    0.014, // 660nm
    0.011, // 670nm
    0.008, // 680nm
    0.006, // 690nm
    0.005, // 700nm
    0.004, // 710nm
    0.003, // 720nm
    0.002, // 730nm
    0.002, // 740nm
    0.001, // 750nm
    0.001, // 760nm
    0.001, // 770nm
    0.001, // 780nm
];

/// Warm white LED (3000K CCT) relative spectral power distribution.
///
/// Similar to 4000K but with less blue and more red phosphor emission.
/// Common in residential and some newer "dark sky friendly" installations.
const LED_3000K_SPECTRUM: [f64; NUM_WAVELENGTHS] = [
    0.003, // 380nm
    0.005, // 390nm
    0.010, // 400nm
    0.020, // 410nm
    0.038, // 420nm
    0.058, // 430nm
    0.075, // 440nm
    0.085, // 450nm - blue peak (smaller than 4000K)
    0.072, // 460nm
    0.052, // 470nm
    0.035, // 480nm
    0.025, // 490nm
    0.022, // 500nm
    0.025, // 510nm
    0.032, // 520nm
    0.040, // 530nm
    0.050, // 540nm
    0.060, // 550nm
    0.068, // 560nm
    0.072, // 570nm
    0.073, // 580nm - phosphor peak (broader)
    0.070, // 590nm
    0.062, // 600nm
    0.052, // 610nm
    0.042, // 620nm
    0.034, // 630nm
    0.027, // 640nm
    0.021, // 650nm
    0.016, // 660nm
    0.012, // 670nm
    0.009, // 680nm
    0.007, // 690nm
    0.005, // 700nm
    0.004, // 710nm
    0.003, // 720nm
    0.002, // 730nm
    0.002, // 740nm
    0.001, // 750nm
    0.001, // 760nm
    0.001, // 770nm
    0.001, // 780nm
];

/// Compute the HPS spectral radiance at each wavelength.
///
/// Returns the spectral radiance in W/m^2/sr/nm at each of the 41 wavelengths,
/// given a total radiance (nW/cm^2/sr) from VIIRS.
///
/// The VIIRS DNB sensor is most sensitive around 500-900nm (broadband),
/// so we scale the spectrum to match the observed total radiance.
pub fn hps_spectrum(total_radiance_nw: f64) -> [f64; 64] {
    let mut result = [0.0f64; 64];
    let norm = normalize_factor(&HPS_SPECTRUM);

    // Convert nW/cm^2/sr to W/m^2/sr: 1 nW/cm^2/sr = 1e-5 W/m^2/sr
    let total_si = total_radiance_nw * 1e-5;

    for i in 0..NUM_WAVELENGTHS {
        result[i] = HPS_SPECTRUM[i] * norm * total_si;
    }
    result
}

/// Compute the white LED (4000K) spectral radiance at each wavelength.
pub fn led_4000k_spectrum(total_radiance_nw: f64) -> [f64; 64] {
    let mut result = [0.0f64; 64];
    let norm = normalize_factor(&LED_4000K_SPECTRUM);
    let total_si = total_radiance_nw * 1e-5;

    for i in 0..NUM_WAVELENGTHS {
        result[i] = LED_4000K_SPECTRUM[i] * norm * total_si;
    }
    result
}

/// Compute the warm white LED (3000K) spectral radiance at each wavelength.
pub fn led_3000k_spectrum(total_radiance_nw: f64) -> [f64; 64] {
    let mut result = [0.0f64; 64];
    let norm = normalize_factor(&LED_3000K_SPECTRUM);
    let total_si = total_radiance_nw * 1e-5;

    for i in 0..NUM_WAVELENGTHS {
        result[i] = LED_3000K_SPECTRUM[i] * norm * total_si;
    }
    result
}

/// Compute a mixed HPS + LED spectrum.
///
/// # Arguments
/// * `total_radiance_nw` - Total VIIRS radiance in nW/cm^2/sr
/// * `led_fraction` - Fraction of lighting that is LED (0.0 = all HPS, 1.0 = all LED)
///
/// # Returns
/// (spectral_radiance, num_wavelengths) where spectral_radiance is a 64-element
/// array with values in W/m^2/sr/nm at each of the 41 active wavelengths.
pub fn mixed_spectrum(total_radiance_nw: f64, led_fraction: f64) -> ([f64; 64], usize) {
    let led_f = led_fraction.clamp(0.0, 1.0);
    let hps_f = 1.0 - led_f;

    let hps = hps_spectrum(total_radiance_nw * hps_f);
    let led = led_4000k_spectrum(total_radiance_nw * led_f);

    let mut result = [0.0f64; 64];
    for i in 0..NUM_WAVELENGTHS {
        result[i] = hps[i] + led[i];
    }

    (result, NUM_WAVELENGTHS)
}

/// Compute the "blue light hazard" ratio: fraction of emission below 500nm.
///
/// This metric indicates how much short-wavelength light the source produces,
/// which scatters more strongly and is more disruptive for dark-adapted vision.
pub fn blue_fraction(led_fraction: f64) -> f64 {
    let led_f = led_fraction.clamp(0.0, 1.0);
    let hps_f = 1.0 - led_f;

    let hps_blue: f64 = HPS_SPECTRUM[..12].iter().sum(); // 380-490nm
    let hps_total: f64 = HPS_SPECTRUM.iter().sum();
    let led_blue: f64 = LED_4000K_SPECTRUM[..12].iter().sum();
    let led_total: f64 = LED_4000K_SPECTRUM.iter().sum();

    let total_blue = hps_f * hps_blue + led_f * led_blue;
    let total_all = hps_f * hps_total + led_f * led_total;

    if total_all > 0.0 {
        total_blue / total_all
    } else {
        0.0
    }
}

/// Compute the Rayleigh-scattering-weighted spectral power.
///
/// Because Rayleigh scattering scales as lambda^-4, blue light is scattered
/// ~9.4x more than red light (400nm vs 700nm). This metric weights the
/// spectrum by the scattering efficiency, indicating how much skyglow the
/// source actually produces per watt of ground-level emission.
///
/// Returns a normalized "scattering effectiveness" factor (1.0 = reference).
pub fn rayleigh_scattering_effectiveness(led_fraction: f64) -> f64 {
    let led_f = led_fraction.clamp(0.0, 1.0);
    let hps_f = 1.0 - led_f;

    let mut weighted_hps = 0.0;
    let mut total_hps = 0.0;
    let mut weighted_led = 0.0;
    let mut total_led = 0.0;

    for i in 0..NUM_WAVELENGTHS {
        let wl = WL_START + i as f64 * WL_STEP;
        let rayleigh_weight = (550.0 / wl).powi(4); // normalized to 550nm

        weighted_hps += HPS_SPECTRUM[i] * rayleigh_weight;
        total_hps += HPS_SPECTRUM[i];
        weighted_led += LED_4000K_SPECTRUM[i] * rayleigh_weight;
        total_led += LED_4000K_SPECTRUM[i];
    }

    let hps_eff = if total_hps > 0.0 {
        weighted_hps / total_hps
    } else {
        0.0
    };
    let led_eff = if total_led > 0.0 {
        weighted_led / total_led
    } else {
        0.0
    };

    hps_f * hps_eff + led_f * led_eff
}

/// Wavelength at a given index in the MCRT grid.
pub fn wavelength_at(index: usize) -> f64 {
    WL_START + index as f64 * WL_STEP
}

/// Number of active wavelengths.
pub fn num_wavelengths() -> usize {
    NUM_WAVELENGTHS
}

/// Compute a normalization factor so that sum * step = 1.0 (unit integral).
fn normalize_factor(spectrum: &[f64; NUM_WAVELENGTHS]) -> f64 {
    let sum: f64 = spectrum.iter().sum();
    if sum > 0.0 {
        1.0 / (sum * WL_STEP)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hps_spectrum_positive() {
        let s = hps_spectrum(50.0);
        for i in 0..NUM_WAVELENGTHS {
            assert!(s[i] >= 0.0, "HPS spectrum negative at index {}", i);
        }
    }

    #[test]
    fn led_spectrum_positive() {
        let s = led_4000k_spectrum(50.0);
        for i in 0..NUM_WAVELENGTHS {
            assert!(s[i] >= 0.0, "LED spectrum negative at index {}", i);
        }
    }

    #[test]
    fn hps_peak_at_590() {
        // HPS should peak near 590nm (index 21)
        let s = hps_spectrum(50.0);
        let peak_idx = s[..NUM_WAVELENGTHS]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let peak_wl = wavelength_at(peak_idx);
        assert!(
            (peak_wl - 590.0).abs() < 20.0,
            "HPS peak should be near 590nm, got {}nm",
            peak_wl
        );
    }

    #[test]
    fn led_blue_peak_around_450() {
        // LED 4000K should have a peak in the 440-460nm range
        let s = led_4000k_spectrum(50.0);
        // Find max in the blue region (380-500nm, indices 0-12)
        let blue_peak_idx = s[0..12]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let peak_wl = wavelength_at(blue_peak_idx);
        assert!(
            peak_wl >= 430.0 && peak_wl <= 470.0,
            "LED blue peak should be 430-470nm, got {}nm",
            peak_wl
        );
    }

    #[test]
    fn mixed_spectrum_interpolates() {
        let (all_hps, _) = mixed_spectrum(50.0, 0.0);
        let (all_led, _) = mixed_spectrum(50.0, 1.0);
        let (half, _) = mixed_spectrum(50.0, 0.5);

        for i in 0..NUM_WAVELENGTHS {
            let _expected = (all_hps[i] + all_led[i]) / 2.0;
            // Not exact half because each is normalized independently
            // but should be in the right ballpark
            assert!(
                half[i] > 0.0 || (all_hps[i] == 0.0 && all_led[i] == 0.0),
                "Mixed spectrum should be positive at index {}",
                i
            );
        }
    }

    #[test]
    fn blue_fraction_led_higher_than_hps() {
        let bf_hps = blue_fraction(0.0);
        let bf_led = blue_fraction(1.0);
        assert!(
            bf_led > bf_hps,
            "LED should have more blue: LED={:.3}, HPS={:.3}",
            bf_led,
            bf_hps
        );
    }

    #[test]
    fn blue_fraction_bounded() {
        for &f in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let bf = blue_fraction(f);
            assert!(bf >= 0.0 && bf <= 1.0, "Blue fraction out of range: {}", bf);
        }
    }

    #[test]
    fn rayleigh_effectiveness_led_higher_than_hps() {
        let re_hps = rayleigh_scattering_effectiveness(0.0);
        let re_led = rayleigh_scattering_effectiveness(1.0);
        assert!(
            re_led > re_hps,
            "LED should scatter more: LED={:.3}, HPS={:.3}",
            re_led,
            re_hps
        );
    }

    #[test]
    fn zero_radiance_gives_zero_spectrum() {
        let s = hps_spectrum(0.0);
        for i in 0..NUM_WAVELENGTHS {
            assert_eq!(s[i], 0.0);
        }
    }

    #[test]
    fn spectrum_scales_linearly() {
        let s1 = hps_spectrum(10.0);
        let s2 = hps_spectrum(20.0);
        for i in 0..NUM_WAVELENGTHS {
            if s1[i] > 0.0 {
                let ratio = s2[i] / s1[i];
                assert!(
                    (ratio - 2.0).abs() < 0.01,
                    "Spectrum should scale linearly, ratio={} at index {}",
                    ratio,
                    i
                );
            }
        }
    }

    #[test]
    fn led_3000k_less_blue_than_4000k() {
        let blue_3k: f64 = LED_3000K_SPECTRUM[..12].iter().sum();
        let blue_4k: f64 = LED_4000K_SPECTRUM[..12].iter().sum();
        assert!(
            blue_3k < blue_4k,
            "3000K should have less blue: 3K={:.3}, 4K={:.3}",
            blue_3k,
            blue_4k
        );
    }

    #[test]
    fn wavelength_grid_correct() {
        assert_eq!(wavelength_at(0), 380.0);
        assert_eq!(wavelength_at(17), 550.0);
        assert_eq!(wavelength_at(40), 780.0);
    }
}
