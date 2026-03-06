//! Luminance computation from spectral radiance using CIE vision functions.
//!
//! Converts spectral radiance I(λ) [W/m²/sr/nm] to photometric luminance [cd/m²]
//! using CIE photopic V(λ), scotopic V'(λ), and mesopic blending per CIE 191:2010.

use crate::vision::{PHOTOPIC_V, SCOTOPIC_V_PRIME};

/// Maximum luminous efficacy for photopic vision [lm/W].
/// K_m = 683.002 lm/W at λ = 555nm.
const KM_PHOTOPIC: f64 = 683.002;

/// Maximum luminous efficacy for scotopic vision [lm/W].
/// K_m' = 1700.06 lm/W at λ = 507nm.
const KM_SCOTOPIC: f64 = 1700.06;

/// Interpolate V(λ) or V'(λ) at a given wavelength from the 5nm-spaced table.
///
/// The table covers 380-780nm at 5nm intervals (81 values).
fn interpolate_vision_function(wavelength_nm: f64, values: &[f64; 81]) -> f64 {
    if !(380.0..=780.0).contains(&wavelength_nm) {
        return 0.0;
    }
    let idx_f = (wavelength_nm - 380.0) / 5.0;
    let idx = idx_f as usize;
    if idx >= 80 {
        return values[80];
    }
    let frac = idx_f - idx as f64;
    values[idx] + frac * (values[idx + 1] - values[idx])
}

/// Compute photopic luminance from spectral radiance.
///
/// L_v = K_m × ∫ L_e(λ) × V(λ) dλ
///
/// where L_e(λ) is spectral radiance in W/m²/sr/nm.
///
/// Returns luminance in cd/m² (= lm/m²/sr).
pub fn photopic_luminance(wavelengths_nm: &[f64], radiance: &[f64]) -> f64 {
    if wavelengths_nm.len() < 2 || wavelengths_nm.len() != radiance.len() {
        return 0.0;
    }

    // Trapezoidal integration of L_e(λ) × V(λ)
    let mut integral = 0.0;
    for i in 0..(wavelengths_nm.len() - 1) {
        let dw = wavelengths_nm[i + 1] - wavelengths_nm[i];
        let v0 = interpolate_vision_function(wavelengths_nm[i], &PHOTOPIC_V);
        let v1 = interpolate_vision_function(wavelengths_nm[i + 1], &PHOTOPIC_V);
        let f0 = radiance[i] * v0;
        let f1 = radiance[i + 1] * v1;
        integral += 0.5 * (f0 + f1) * dw;
    }

    KM_PHOTOPIC * integral
}

/// Compute scotopic luminance from spectral radiance.
///
/// L_v' = K_m' × ∫ L_e(λ) × V'(λ) dλ
///
/// Returns luminance in scotopic cd/m².
pub fn scotopic_luminance(wavelengths_nm: &[f64], radiance: &[f64]) -> f64 {
    if wavelengths_nm.len() < 2 || wavelengths_nm.len() != radiance.len() {
        return 0.0;
    }

    let mut integral = 0.0;
    for i in 0..(wavelengths_nm.len() - 1) {
        let dw = wavelengths_nm[i + 1] - wavelengths_nm[i];
        let v0 = interpolate_vision_function(wavelengths_nm[i], &SCOTOPIC_V_PRIME);
        let v1 = interpolate_vision_function(wavelengths_nm[i + 1], &SCOTOPIC_V_PRIME);
        let f0 = radiance[i] * v0;
        let f1 = radiance[i + 1] * v1;
        integral += 0.5 * (f0 + f1) * dw;
    }

    KM_SCOTOPIC * integral
}

/// Compute mesopic luminance using CIE 191:2010 adapted model.
///
/// The mesopic luminance blends between photopic and scotopic based on
/// adaptation level. During twilight, vision transitions from photopic
/// (cone-dominated) to scotopic (rod-dominated).
///
/// The adaptation coefficient m ranges from 0 (scotopic) to 1 (photopic).
pub fn mesopic_luminance(wavelengths_nm: &[f64], radiance: &[f64]) -> f64 {
    let l_p = photopic_luminance(wavelengths_nm, radiance);
    let l_s = scotopic_luminance(wavelengths_nm, radiance);

    // Estimate adaptation coefficient m from photopic luminance
    let m = mesopic_coefficient(l_p);

    // Mesopic luminance: L_mes = m × L_p + (1 - m) × L_s × (K_m / K_m')
    // The K_m/K_m' factor normalizes scotopic to photopic scale
    m * l_p + (1.0 - m) * l_s * (KM_PHOTOPIC / KM_SCOTOPIC)
}

/// Compute the mesopic adaptation coefficient m from photopic luminance.
///
/// Based on CIE 191:2010:
/// - L >= 5.0 cd/m²: m = 1.0 (fully photopic)
/// - L <= 0.005 cd/m²: m = 0.0 (fully scotopic)
/// - Between: logarithmic interpolation
fn mesopic_coefficient(l_photopic: f64) -> f64 {
    const L_UPPER: f64 = 5.0; // cd/m²
    const L_LOWER: f64 = 0.005; // cd/m²

    if l_photopic >= L_UPPER {
        1.0
    } else if l_photopic <= L_LOWER || l_photopic <= 0.0 {
        0.0
    } else {
        let log_l = libm::log10(l_photopic);
        let log_upper = libm::log10(L_UPPER);
        let log_lower = libm::log10(L_LOWER);
        (log_l - log_lower) / (log_upper - log_lower)
    }
}

/// Compute spectral centroid wavelength.
///
/// λ_c = ∫ λ × L_e(λ) × V(λ) dλ / ∫ L_e(λ) × V(λ) dλ
///
/// This indicates the perceived "color" of the sky:
/// - λ_c < 500nm: blue-dominated
/// - λ_c ≈ 550-580nm: white/neutral
/// - λ_c > 600nm: red-dominated
///
/// Used to distinguish shafaq al-ahmar (red) from shafaq al-abyad (white).
pub fn spectral_centroid(wavelengths_nm: &[f64], radiance: &[f64]) -> f64 {
    if wavelengths_nm.len() < 2 || wavelengths_nm.len() != radiance.len() {
        return 0.0;
    }

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for i in 0..(wavelengths_nm.len() - 1) {
        let dw = wavelengths_nm[i + 1] - wavelengths_nm[i];
        let v0 = interpolate_vision_function(wavelengths_nm[i], &PHOTOPIC_V);
        let v1 = interpolate_vision_function(wavelengths_nm[i + 1], &PHOTOPIC_V);

        let f0 = radiance[i] * v0;
        let f1 = radiance[i + 1] * v1;
        let wf0 = wavelengths_nm[i] * f0;
        let wf1 = wavelengths_nm[i + 1] * f1;

        numerator += 0.5 * (wf0 + wf1) * dw;
        denominator += 0.5 * (f0 + f1) * dw;
    }

    if denominator > 1e-30 {
        numerator / denominator
    } else {
        0.0
    }
}

/// Compute luminance in the red band only (λ > 600nm).
///
/// Used for shafaq al-ahmar detection — tracks the red glow specifically.
pub fn red_band_luminance(wavelengths_nm: &[f64], radiance: &[f64]) -> f64 {
    if wavelengths_nm.len() < 2 || wavelengths_nm.len() != radiance.len() {
        return 0.0;
    }

    let mut integral = 0.0;
    for i in 0..(wavelengths_nm.len() - 1) {
        if wavelengths_nm[i + 1] <= 600.0 {
            continue;
        }
        let dw = wavelengths_nm[i + 1] - wavelengths_nm[i];
        let v0 = interpolate_vision_function(wavelengths_nm[i], &PHOTOPIC_V);
        let v1 = interpolate_vision_function(wavelengths_nm[i + 1], &PHOTOPIC_V);
        let f0 = radiance[i] * v0;
        let f1 = radiance[i + 1] * v1;
        integral += 0.5 * (f0 + f1) * dw;
    }

    KM_PHOTOPIC * integral
}

/// Compute luminance in the blue band only (λ < 500nm).
///
/// Tracks when blue scattered light disappears during twilight.
pub fn blue_band_luminance(wavelengths_nm: &[f64], radiance: &[f64]) -> f64 {
    if wavelengths_nm.len() < 2 || wavelengths_nm.len() != radiance.len() {
        return 0.0;
    }

    let mut integral = 0.0;
    for i in 0..(wavelengths_nm.len() - 1) {
        if wavelengths_nm[i] >= 500.0 {
            continue;
        }
        let dw = wavelengths_nm[i + 1] - wavelengths_nm[i];
        let v0 = interpolate_vision_function(wavelengths_nm[i], &PHOTOPIC_V);
        let v1 = interpolate_vision_function(wavelengths_nm[i + 1], &PHOTOPIC_V);
        let f0 = radiance[i] * v0;
        let f1 = radiance[i + 1] * v1;
        integral += 0.5 * (f0 + f1) * dw;
    }

    KM_PHOTOPIC * integral
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create a flat spectrum (constant radiance across all wavelengths)
    fn flat_spectrum(value: f64) -> (Vec<f64>, Vec<f64>) {
        let wls: Vec<f64> = (0..41).map(|i| 380.0 + i as f64 * 10.0).collect();
        let rad: Vec<f64> = vec![value; 41];
        (wls, rad)
    }

    // Helper: create a monochromatic-like spectrum (peaked at one wavelength)
    fn peaked_spectrum(peak_nm: f64, width_nm: f64, peak_value: f64) -> (Vec<f64>, Vec<f64>) {
        let wls: Vec<f64> = (0..41).map(|i| 380.0 + i as f64 * 10.0).collect();
        let rad: Vec<f64> = wls
            .iter()
            .map(|&wl| {
                let dist = (wl - peak_nm).abs();
                if dist < width_nm {
                    peak_value * (1.0 - dist / width_nm)
                } else {
                    0.0
                }
            })
            .collect();
        (wls, rad)
    }

    // ── photopic_luminance ──

    #[test]
    fn photopic_luminance_zero_for_zero_radiance() {
        let (wls, rad) = flat_spectrum(0.0);
        let l = photopic_luminance(&wls, &rad);
        assert!(l.abs() < 1e-20, "Zero radiance → zero luminance, got {}", l);
    }

    #[test]
    fn photopic_luminance_positive_for_positive_radiance() {
        let (wls, rad) = flat_spectrum(1.0);
        let l = photopic_luminance(&wls, &rad);
        assert!(l > 0.0, "Positive radiance → positive luminance, got {}", l);
    }

    #[test]
    fn photopic_luminance_scales_linearly() {
        let (wls, rad1) = flat_spectrum(1.0);
        let (_, rad2) = flat_spectrum(2.0);
        let l1 = photopic_luminance(&wls, &rad1);
        let l2 = photopic_luminance(&wls, &rad2);
        assert!(
            ((l2 / l1) - 2.0).abs() < 1e-10,
            "Luminance should scale linearly: L2/L1 = {}, expected 2.0",
            l2 / l1
        );
    }

    #[test]
    fn photopic_luminance_uses_km_683() {
        // For a delta-like spectrum at 555nm (where V=1.0):
        // L_v ≈ K_m × L_e × V(555) × Δλ = 683.002 × L_e × 1.0 × Δλ
        // With a narrow peak: L_e(555nm)=1.0 over a ~10nm band
        let wls = vec![550.0, 555.0, 560.0];
        let rad = vec![0.0, 1.0, 0.0];
        let l = photopic_luminance(&wls, &rad);
        // Trapezoidal: integral = 0.5*(0*V(550) + 1*V(555))*5 + 0.5*(1*V(555) + 0*V(560))*5
        // = 0.5*1.0*5 + 0.5*1.0*5 = 5.0
        // L = 683.002 * 5.0 = 3415.01
        assert!(
            (l - 683.002 * 5.0).abs() < 10.0,
            "L at 555nm peak = {}, expected ~{}",
            l,
            683.002 * 5.0
        );
    }

    #[test]
    fn photopic_luminance_empty_input() {
        assert_eq!(photopic_luminance(&[], &[]), 0.0);
        assert_eq!(photopic_luminance(&[550.0], &[1.0]), 0.0); // only 1 point
    }

    #[test]
    fn photopic_luminance_mismatched_lengths() {
        let wls = vec![500.0, 600.0];
        let rad = vec![1.0, 2.0, 3.0]; // different length
        assert_eq!(photopic_luminance(&wls, &rad), 0.0);
    }

    // ── scotopic_luminance ──

    #[test]
    fn scotopic_luminance_zero_for_zero_radiance() {
        let (wls, rad) = flat_spectrum(0.0);
        let l = scotopic_luminance(&wls, &rad);
        assert!(l.abs() < 1e-20, "Zero radiance → zero scotopic luminance");
    }

    #[test]
    fn scotopic_luminance_positive_for_positive_radiance() {
        let (wls, rad) = flat_spectrum(1.0);
        let l = scotopic_luminance(&wls, &rad);
        assert!(l > 0.0, "Positive radiance → positive scotopic luminance");
    }

    #[test]
    fn scotopic_luminance_uses_km_1700() {
        // Scotopic K_m' = 1700.06 lm/W
        let wls = vec![505.0, 510.0];
        let rad = vec![1.0, 1.0]; // flat over 5nm band near scotopic peak
        let l = scotopic_luminance(&wls, &rad);
        // At ~507nm, V'≈0.982. Integral ≈ 0.982 × 5 = 4.91
        // L = 1700.06 × 4.91 ≈ 8347
        assert!(
            l > 5000.0 && l < 15000.0,
            "Scotopic L near 507nm = {}, expected ~8347",
            l
        );
    }

    #[test]
    fn scotopic_higher_than_photopic_for_blue_light() {
        // For blue-dominated light, scotopic luminance should exceed photopic
        let (wls, rad) = peaked_spectrum(480.0, 30.0, 1.0);
        let l_p = photopic_luminance(&wls, &rad);
        let l_s = scotopic_luminance(&wls, &rad);
        assert!(
            l_s > l_p,
            "Scotopic ({}) should exceed photopic ({}) for blue light",
            l_s,
            l_p
        );
    }

    #[test]
    fn scotopic_lower_than_photopic_for_red_light() {
        // For red light (>600nm), photopic should exceed scotopic
        let (wls, rad) = peaked_spectrum(650.0, 30.0, 1.0);
        let l_p = photopic_luminance(&wls, &rad);
        let l_s = scotopic_luminance(&wls, &rad);
        assert!(
            l_p > l_s,
            "Photopic ({}) should exceed scotopic ({}) for red light",
            l_p,
            l_s
        );
    }

    // ── mesopic_luminance ──

    #[test]
    fn mesopic_luminance_zero_for_zero_radiance() {
        let (wls, rad) = flat_spectrum(0.0);
        let l = mesopic_luminance(&wls, &rad);
        assert!(l.abs() < 1e-20, "Zero radiance → zero mesopic luminance");
    }

    #[test]
    fn mesopic_luminance_between_photopic_and_scotopic() {
        // Mesopic should be between photopic and scotopic (or close)
        let (wls, rad) = flat_spectrum(0.001); // low light level
        let l_p = photopic_luminance(&wls, &rad);
        let l_s = scotopic_luminance(&wls, &rad);
        let l_m = mesopic_luminance(&wls, &rad);

        let l_min = l_p.min(l_s * (KM_PHOTOPIC / KM_SCOTOPIC));
        let l_max = l_p.max(l_s * (KM_PHOTOPIC / KM_SCOTOPIC));
        // Mesopic should be somewhere in between (allow some margin)
        assert!(
            l_m >= l_min * 0.5 && l_m <= l_max * 2.0,
            "Mesopic ({}) should be near range [{}, {}]",
            l_m,
            l_min,
            l_max
        );
    }

    #[test]
    fn mesopic_equals_photopic_at_high_luminance() {
        // At high luminance (>5 cd/m²), mesopic coefficient = 1 → purely photopic
        // Use very high radiance to get L_p > 5 cd/m²
        let (wls, rad) = flat_spectrum(1.0); // will give L_p >> 5
        let l_p = photopic_luminance(&wls, &rad);
        let l_m = mesopic_luminance(&wls, &rad);

        if l_p >= 5.0 {
            assert!(
                ((l_m / l_p) - 1.0).abs() < 0.01,
                "At high luminance, mesopic ({}) should equal photopic ({})",
                l_m,
                l_p
            );
        }
    }

    // ── mesopic_coefficient (tested indirectly) ──

    #[test]
    fn mesopic_coefficient_boundaries() {
        // Test the mesopic coefficient at boundary values
        // L >= 5.0 → m = 1.0
        assert!((mesopic_coefficient(5.0) - 1.0).abs() < 1e-10);
        assert!((mesopic_coefficient(10.0) - 1.0).abs() < 1e-10);
        assert!((mesopic_coefficient(100.0) - 1.0).abs() < 1e-10);

        // L <= 0.005 → m = 0.0
        assert!((mesopic_coefficient(0.005) - 0.0).abs() < 1e-10);
        assert!((mesopic_coefficient(0.001) - 0.0).abs() < 1e-10);
        assert!((mesopic_coefficient(0.0) - 0.0).abs() < 1e-10);
        assert!((mesopic_coefficient(-1.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn mesopic_coefficient_midrange() {
        // At geometric mean of boundaries: sqrt(5.0 * 0.005) ≈ 0.158
        // log10(0.158) ≈ -0.801
        // m = (-0.801 - (-2.301)) / (0.699 - (-2.301)) = 1.5 / 3.0 = 0.5
        let l_mid = libm::sqrt(5.0 * 0.005);
        let m = mesopic_coefficient(l_mid);
        assert!(
            (m - 0.5).abs() < 0.01,
            "Mesopic coefficient at geometric mean = {}, expected ~0.5",
            m
        );
    }

    #[test]
    fn mesopic_coefficient_monotonically_increases() {
        let mut prev = mesopic_coefficient(0.001);
        let test_values = [0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
        for &l in &test_values {
            let m = mesopic_coefficient(l);
            assert!(
                m >= prev - 1e-10,
                "Mesopic coefficient should increase: m({})={} < m(prev)={}",
                l,
                m,
                prev
            );
            prev = m;
        }
    }

    // ── spectral_centroid ──

    #[test]
    fn spectral_centroid_monochromatic() {
        // A spectrum peaked at 600nm should have centroid near 600nm
        let (wls, rad) = peaked_spectrum(600.0, 15.0, 1.0);
        let centroid = spectral_centroid(&wls, &rad);
        assert!(
            (centroid - 600.0).abs() < 20.0,
            "Centroid of 600nm peak = {}, expected ~600nm",
            centroid
        );
    }

    #[test]
    fn spectral_centroid_blue_peak() {
        // Blue-peaked spectrum should have centroid < 500nm
        let (wls, rad) = peaked_spectrum(450.0, 30.0, 1.0);
        let centroid = spectral_centroid(&wls, &rad);
        assert!(
            centroid < 520.0,
            "Blue peak centroid = {}, expected < 520nm",
            centroid
        );
    }

    #[test]
    fn spectral_centroid_red_peak() {
        // Red-peaked spectrum should have centroid > 600nm
        let (wls, rad) = peaked_spectrum(680.0, 30.0, 1.0);
        let centroid = spectral_centroid(&wls, &rad);
        assert!(
            centroid > 600.0,
            "Red peak centroid = {}, expected > 600nm",
            centroid
        );
    }

    #[test]
    fn spectral_centroid_flat_spectrum() {
        // A flat spectrum weighted by V(λ) should have centroid near the
        // photopic peak (~555nm), maybe slightly shifted due to V(λ) asymmetry
        let (wls, rad) = flat_spectrum(1.0);
        let centroid = spectral_centroid(&wls, &rad);
        assert!(
            centroid > 530.0 && centroid < 580.0,
            "Flat spectrum centroid = {}, expected ~555nm",
            centroid
        );
    }

    #[test]
    fn spectral_centroid_zero_for_zero_radiance() {
        let (wls, rad) = flat_spectrum(0.0);
        let centroid = spectral_centroid(&wls, &rad);
        assert!(
            centroid.abs() < 1e-10,
            "Zero radiance → zero centroid, got {}",
            centroid
        );
    }

    #[test]
    fn spectral_centroid_empty_input() {
        assert_eq!(spectral_centroid(&[], &[]), 0.0);
    }

    // ── red_band_luminance ──

    #[test]
    fn red_band_only_includes_above_600nm() {
        // A spectrum with all energy below 600nm should give zero red band
        let (wls, rad) = peaked_spectrum(500.0, 50.0, 1.0);
        let l_red = red_band_luminance(&wls, &rad);
        assert!(
            l_red < 1e-10,
            "Spectrum below 600nm should have zero red band, got {}",
            l_red
        );
    }

    #[test]
    fn red_band_positive_for_red_spectrum() {
        let (wls, rad) = peaked_spectrum(650.0, 30.0, 1.0);
        let l_red = red_band_luminance(&wls, &rad);
        assert!(
            l_red > 0.0,
            "Red spectrum should have positive red band luminance"
        );
    }

    #[test]
    fn red_band_less_than_total() {
        let (wls, rad) = flat_spectrum(1.0);
        let l_total = photopic_luminance(&wls, &rad);
        let l_red = red_band_luminance(&wls, &rad);
        assert!(
            l_red < l_total,
            "Red band ({}) should be less than total ({})",
            l_red,
            l_total
        );
    }

    // ── blue_band_luminance ──

    #[test]
    fn blue_band_only_includes_below_500nm() {
        // A spectrum with all energy above 500nm should give zero blue band
        let (wls, rad) = peaked_spectrum(600.0, 50.0, 1.0);
        let l_blue = blue_band_luminance(&wls, &rad);
        assert!(
            l_blue < 1e-10,
            "Spectrum above 500nm should have zero blue band, got {}",
            l_blue
        );
    }

    #[test]
    fn blue_band_positive_for_blue_spectrum() {
        let (wls, rad) = peaked_spectrum(450.0, 30.0, 1.0);
        let l_blue = blue_band_luminance(&wls, &rad);
        assert!(
            l_blue > 0.0,
            "Blue spectrum should have positive blue band luminance"
        );
    }

    #[test]
    fn blue_band_less_than_total() {
        let (wls, rad) = flat_spectrum(1.0);
        let l_total = photopic_luminance(&wls, &rad);
        let l_blue = blue_band_luminance(&wls, &rad);
        assert!(
            l_blue < l_total,
            "Blue band ({}) should be less than total ({})",
            l_blue,
            l_total
        );
    }

    #[test]
    fn red_plus_blue_less_than_total() {
        // Red + blue bands should sum to less than total (missing 500-600nm)
        let (wls, rad) = flat_spectrum(1.0);
        let l_total = photopic_luminance(&wls, &rad);
        let l_red = red_band_luminance(&wls, &rad);
        let l_blue = blue_band_luminance(&wls, &rad);
        assert!(
            l_red + l_blue < l_total * 1.01, // small tolerance
            "Red ({}) + Blue ({}) = {} should be < total ({})",
            l_red,
            l_blue,
            l_red + l_blue,
            l_total
        );
    }

    // ── Constants ──

    #[test]
    fn km_photopic_is_683() {
        assert!(
            (KM_PHOTOPIC - 683.002).abs() < 0.01,
            "K_m photopic = {}, expected 683.002",
            KM_PHOTOPIC
        );
    }

    #[test]
    fn km_scotopic_is_1700() {
        assert!(
            (KM_SCOTOPIC - 1700.06).abs() < 0.1,
            "K_m' scotopic = {}, expected 1700.06",
            KM_SCOTOPIC
        );
    }
}
