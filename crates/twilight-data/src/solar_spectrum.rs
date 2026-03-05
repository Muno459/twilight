//! Solar spectral irradiance based on TSIS-1 HSRS (2021).
//!
//! Provides top-of-atmosphere solar spectral irradiance at 10nm intervals
//! from 380-780nm. These values are used to weight the MCRT results
//! to compute physical luminance.
//!
//! Units: W/m²/nm (spectral irradiance)
//!
//! Source: Coddington, O., et al. (2021). The TSIS-1 Hybrid Solar Reference
//! Spectrum. Geophysical Research Letters, 48.

/// Wavelengths in nm (10nm intervals, 380-780nm).
pub const SOLAR_WAVELENGTHS_NM: [f64; 41] = [
    380.0, 390.0, 400.0, 410.0, 420.0, 430.0, 440.0, 450.0, 460.0, 470.0, 480.0, 490.0, 500.0,
    510.0, 520.0, 530.0, 540.0, 550.0, 560.0, 570.0, 580.0, 590.0, 600.0, 610.0, 620.0, 630.0,
    640.0, 650.0, 660.0, 670.0, 680.0, 690.0, 700.0, 710.0, 720.0, 730.0, 740.0, 750.0, 760.0,
    770.0, 780.0,
];

/// Solar spectral irradiance at TOA [W/m²/nm].
/// Representative values from TSIS-1 HSRS at 10nm resolution.
pub const SOLAR_IRRADIANCE: [f64; 41] = [
    1.119, 1.068, 1.527, 1.714, 1.744, 1.638, 1.810, 2.087, 2.024, 1.948, 2.005, 1.946, 1.940,
    1.889, 1.863, 1.843, 1.824, 1.848, 1.833, 1.803, 1.780, 1.694, 1.704, 1.693, 1.639, 1.636,
    1.594, 1.580, 1.544, 1.515, 1.486, 1.438, 1.413, 1.389, 1.360, 1.323, 1.296, 1.265, 1.194,
    1.244, 1.216,
];

/// Number of wavelength entries.
pub const SOLAR_NUM_ENTRIES: usize = 41;

/// Get solar irradiance at a given wavelength by linear interpolation.
///
/// Returns irradiance in W/m²/nm.
pub fn solar_irradiance_at(wavelength_nm: f64) -> f64 {
    if wavelength_nm < SOLAR_WAVELENGTHS_NM[0]
        || wavelength_nm > SOLAR_WAVELENGTHS_NM[SOLAR_NUM_ENTRIES - 1]
    {
        return 0.0;
    }

    for i in 0..(SOLAR_NUM_ENTRIES - 1) {
        if wavelength_nm >= SOLAR_WAVELENGTHS_NM[i] && wavelength_nm <= SOLAR_WAVELENGTHS_NM[i + 1]
        {
            let frac = (wavelength_nm - SOLAR_WAVELENGTHS_NM[i])
                / (SOLAR_WAVELENGTHS_NM[i + 1] - SOLAR_WAVELENGTHS_NM[i]);
            return SOLAR_IRRADIANCE[i] + frac * (SOLAR_IRRADIANCE[i + 1] - SOLAR_IRRADIANCE[i]);
        }
    }

    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solar_irradiance_at_grid_points() {
        // At exact grid wavelengths, should return exact table values
        for i in 0..SOLAR_NUM_ENTRIES {
            let irr = solar_irradiance_at(SOLAR_WAVELENGTHS_NM[i]);
            assert!(
                (irr - SOLAR_IRRADIANCE[i]).abs() < 1e-10,
                "E_sun({}) = {}, expected {}",
                SOLAR_WAVELENGTHS_NM[i],
                irr,
                SOLAR_IRRADIANCE[i]
            );
        }
    }

    #[test]
    fn solar_irradiance_peak_near_450nm() {
        // TSIS-1: peak irradiance ≈ 2.087 W/m²/nm near 450nm
        let irr = solar_irradiance_at(450.0);
        assert!(
            (irr - 2.087).abs() < 0.01,
            "Peak at 450nm = {}, expected 2.087",
            irr
        );
    }

    #[test]
    fn solar_irradiance_all_positive() {
        for i in 0..SOLAR_NUM_ENTRIES {
            assert!(
                SOLAR_IRRADIANCE[i] > 0.0,
                "Solar irradiance[{}] at {}nm should be positive",
                i,
                SOLAR_WAVELENGTHS_NM[i]
            );
        }
    }

    #[test]
    fn solar_irradiance_reasonable_range() {
        // All values should be between 0.5 and 3.0 W/m²/nm in the visible
        for i in 0..SOLAR_NUM_ENTRIES {
            assert!(
                SOLAR_IRRADIANCE[i] > 0.5 && SOLAR_IRRADIANCE[i] < 3.0,
                "Solar irradiance[{}] at {}nm = {} outside expected range [0.5, 3.0]",
                i,
                SOLAR_WAVELENGTHS_NM[i],
                SOLAR_IRRADIANCE[i]
            );
        }
    }

    #[test]
    fn solar_irradiance_zero_outside_range() {
        assert_eq!(solar_irradiance_at(300.0), 0.0, "Below range should be 0");
        assert_eq!(solar_irradiance_at(379.0), 0.0, "Below range should be 0");
        assert_eq!(solar_irradiance_at(781.0), 0.0, "Above range should be 0");
        assert_eq!(solar_irradiance_at(1000.0), 0.0, "Above range should be 0");
    }

    #[test]
    fn solar_irradiance_interpolation_midpoint() {
        let wl_mid = (SOLAR_WAVELENGTHS_NM[5] + SOLAR_WAVELENGTHS_NM[6]) / 2.0;
        let irr = solar_irradiance_at(wl_mid);
        let expected = (SOLAR_IRRADIANCE[5] + SOLAR_IRRADIANCE[6]) / 2.0;
        assert!(
            (irr - expected).abs() < 0.001,
            "Midpoint interpolation: got {}, expected {}",
            irr,
            expected
        );
    }

    #[test]
    fn solar_total_irradiance_integration() {
        // Integrate solar irradiance over 380-780nm using trapezoidal rule.
        // Total Solar Irradiance in visible band should be ~600-700 W/m²
        // (total TSI is ~1361 W/m², visible is roughly half)
        let mut total = 0.0;
        for i in 0..(SOLAR_NUM_ENTRIES - 1) {
            let dw = SOLAR_WAVELENGTHS_NM[i + 1] - SOLAR_WAVELENGTHS_NM[i];
            total += 0.5 * (SOLAR_IRRADIANCE[i] + SOLAR_IRRADIANCE[i + 1]) * dw;
        }
        assert!(
            total > 500.0 && total < 800.0,
            "Visible band irradiance integral = {} W/m², expected ~600-700",
            total
        );
    }

    #[test]
    fn solar_wavelength_grid_is_uniform_10nm() {
        for i in 0..(SOLAR_NUM_ENTRIES - 1) {
            let spacing = SOLAR_WAVELENGTHS_NM[i + 1] - SOLAR_WAVELENGTHS_NM[i];
            assert!(
                (spacing - 10.0).abs() < 0.01,
                "Grid spacing at index {}: {} nm, expected 10 nm",
                i,
                spacing
            );
        }
    }

    #[test]
    fn solar_wavelength_grid_starts_at_380() {
        assert!((SOLAR_WAVELENGTHS_NM[0] - 380.0).abs() < 0.01);
    }

    #[test]
    fn solar_wavelength_grid_ends_at_780() {
        assert!((SOLAR_WAVELENGTHS_NM[SOLAR_NUM_ENTRIES - 1] - 780.0).abs() < 0.01);
    }
}
