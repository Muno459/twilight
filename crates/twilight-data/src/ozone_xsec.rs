//! Ozone absorption cross-sections from Serdyuchenko-Gorshelev (2014).
//!
//! These are representative values at 293 K for wavelengths relevant to
//! twilight (380-780 nm). The Chappuis band (440-850 nm) is the dominant
//! ozone absorption feature in the visible spectrum and is critical for
//! twilight color.
//!
//! Units: cm² per molecule
//!
//! Source: Serdyuchenko, A., et al. (2014). High spectral resolution ozone
//! absorption cross-sections. Atmos. Meas. Tech., 7, 625-636.

/// Wavelengths in nm for the ozone cross-section table.
pub const O3_WAVELENGTHS_NM: [f64; 41] = [
    380.0, 390.0, 400.0, 410.0, 420.0, 430.0, 440.0, 450.0, 460.0, 470.0, 480.0, 490.0, 500.0,
    510.0, 520.0, 530.0, 540.0, 550.0, 560.0, 570.0, 580.0, 590.0, 600.0, 610.0, 620.0, 630.0,
    640.0, 650.0, 660.0, 670.0, 680.0, 690.0, 700.0, 710.0, 720.0, 730.0, 740.0, 750.0, 760.0,
    770.0, 780.0,
];

/// Ozone absorption cross-sections at 293 K [cm²/molecule].
///
/// The Chappuis band peaks near 600 nm with σ ≈ 5.0e-21 cm².
/// Values below ~440 nm are in the Huggins band tail.
/// Values above ~700 nm are in the Chappuis band tail (weak).
pub const O3_CROSS_SECTIONS_293K: [f64; 41] = [
    // 380-430 nm: Huggins band tail (rapidly decreasing)
    2.48e-22, 7.60e-23, 2.80e-23, 1.20e-23, 6.10e-24, 3.80e-24,
    // 440-490 nm: Chappuis band onset (increasing)
    3.50e-24, 4.50e-24, 7.20e-24, 1.10e-23, 1.60e-23, 2.20e-23,
    // 500-550 nm: Chappuis band (building up)
    2.85e-23, 3.45e-23, 3.90e-23, 4.25e-23, 4.55e-23, 4.78e-23,
    // 560-610 nm: Chappuis band peak region
    4.95e-23, 5.05e-23, 5.10e-23, 5.08e-23, 5.01e-23, 4.90e-23, 4.72e-23,
    // 630-680 nm: Chappuis band (declining)
    4.48e-23, 4.15e-23, 3.72e-23, 3.20e-23, 2.60e-23, 1.95e-23, 1.38e-23,
    // 700-780 nm: Chappuis band tail (weak)
    9.20e-24, 5.80e-24, 3.50e-24, 2.00e-24, 1.10e-24, 5.80e-25, 3.00e-25, 1.50e-25, 7.00e-26,
];

/// Number of entries in the ozone cross-section table.
pub const O3_NUM_ENTRIES: usize = 41;

/// Get ozone absorption cross-section at a given wavelength by linear interpolation.
///
/// Returns cross-section in cm²/molecule.
/// Returns 0 for wavelengths outside the table range.
pub fn o3_cross_section_at(wavelength_nm: f64) -> f64 {
    if wavelength_nm < O3_WAVELENGTHS_NM[0] || wavelength_nm > O3_WAVELENGTHS_NM[O3_NUM_ENTRIES - 1]
    {
        return 0.0;
    }

    for i in 0..(O3_NUM_ENTRIES - 1) {
        if wavelength_nm >= O3_WAVELENGTHS_NM[i] && wavelength_nm <= O3_WAVELENGTHS_NM[i + 1] {
            let frac = (wavelength_nm - O3_WAVELENGTHS_NM[i])
                / (O3_WAVELENGTHS_NM[i + 1] - O3_WAVELENGTHS_NM[i]);
            return O3_CROSS_SECTIONS_293K[i]
                + frac * (O3_CROSS_SECTIONS_293K[i + 1] - O3_CROSS_SECTIONS_293K[i]);
        }
    }

    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn o3_xsec_at_grid_points() {
        // At exact grid wavelengths, should return exact table values
        for i in 0..O3_NUM_ENTRIES {
            let sigma = o3_cross_section_at(O3_WAVELENGTHS_NM[i]);
            assert!(
                (sigma - O3_CROSS_SECTIONS_293K[i]).abs() < 1e-30,
                "σ_O3({}) = {:.4e}, expected {:.4e}",
                O3_WAVELENGTHS_NM[i],
                sigma,
                O3_CROSS_SECTIONS_293K[i]
            );
        }
    }

    #[test]
    fn o3_xsec_chappuis_peak() {
        // Chappuis band peaks near 580-600 nm, σ ≈ 5e-23 cm²
        let sigma_580 = o3_cross_section_at(580.0);
        assert!(
            sigma_580 > 4e-23 && sigma_580 < 6e-23,
            "Chappuis peak σ(580nm) = {:.4e}, expected ~5e-23",
            sigma_580
        );
    }

    #[test]
    fn o3_xsec_huggins_band_larger_at_short_wavelengths() {
        // Huggins band (< 400nm) has larger cross-sections than Chappuis onset
        let sigma_380 = o3_cross_section_at(380.0);
        let sigma_440 = o3_cross_section_at(440.0);
        assert!(
            sigma_380 > sigma_440,
            "Huggins band: σ(380)={:.4e} should exceed σ(440)={:.4e}",
            sigma_380,
            sigma_440
        );
    }

    #[test]
    fn o3_xsec_decreases_toward_red() {
        // Cross-section should decrease from Chappuis peak toward 780nm
        let sigma_600 = o3_cross_section_at(600.0);
        let sigma_700 = o3_cross_section_at(700.0);
        let sigma_780 = o3_cross_section_at(780.0);
        assert!(sigma_600 > sigma_700, "σ should decrease: 600nm > 700nm");
        assert!(sigma_700 > sigma_780, "σ should decrease: 700nm > 780nm");
    }

    #[test]
    fn o3_xsec_zero_outside_range() {
        assert!(o3_cross_section_at(300.0) == 0.0, "Below range should be 0");
        assert!(o3_cross_section_at(379.0) == 0.0, "Below range should be 0");
        assert!(o3_cross_section_at(781.0) == 0.0, "Above range should be 0");
        assert!(o3_cross_section_at(900.0) == 0.0, "Above range should be 0");
    }

    #[test]
    fn o3_xsec_all_values_positive() {
        for i in 0..O3_NUM_ENTRIES {
            assert!(
                O3_CROSS_SECTIONS_293K[i] > 0.0,
                "O3 cross-section[{}] at {}nm should be positive",
                i,
                O3_WAVELENGTHS_NM[i]
            );
        }
    }

    #[test]
    fn o3_xsec_interpolation_midpoint() {
        // At midpoint between two grid values, should be average
        let wl_mid = (O3_WAVELENGTHS_NM[10] + O3_WAVELENGTHS_NM[11]) / 2.0;
        let sigma = o3_cross_section_at(wl_mid);
        let expected = (O3_CROSS_SECTIONS_293K[10] + O3_CROSS_SECTIONS_293K[11]) / 2.0;
        assert!(
            (sigma - expected).abs() / expected < 0.001,
            "Midpoint interpolation: got {:.4e}, expected {:.4e}",
            sigma,
            expected
        );
    }

    #[test]
    fn o3_wavelength_grid_is_uniform_10nm() {
        for i in 0..(O3_NUM_ENTRIES - 1) {
            let spacing = O3_WAVELENGTHS_NM[i + 1] - O3_WAVELENGTHS_NM[i];
            assert!(
                (spacing - 10.0).abs() < 0.01,
                "Grid spacing at index {}: {} nm, expected 10 nm",
                i,
                spacing
            );
        }
    }
}
