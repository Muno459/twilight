//! CIE photopic V(λ) and scotopic V'(λ) luminous efficiency functions.

/// CIE 1924 photopic luminous efficiency V(λ) at 5nm intervals, 380-780nm.
/// 81 values covering daylight-adapted human vision.
pub const PHOTOPIC_WAVELENGTHS_NM: [f64; 81] = [
    380.0, 385.0, 390.0, 395.0, 400.0, 405.0, 410.0, 415.0, 420.0, 425.0, 430.0, 435.0, 440.0,
    445.0, 450.0, 455.0, 460.0, 465.0, 470.0, 475.0, 480.0, 485.0, 490.0, 495.0, 500.0, 505.0,
    510.0, 515.0, 520.0, 525.0, 530.0, 535.0, 540.0, 545.0, 550.0, 555.0, 560.0, 565.0, 570.0,
    575.0, 580.0, 585.0, 590.0, 595.0, 600.0, 605.0, 610.0, 615.0, 620.0, 625.0, 630.0, 635.0,
    640.0, 645.0, 650.0, 655.0, 660.0, 665.0, 670.0, 675.0, 680.0, 685.0, 690.0, 695.0, 700.0,
    705.0, 710.0, 715.0, 720.0, 725.0, 730.0, 735.0, 740.0, 745.0, 750.0, 755.0, 760.0, 765.0,
    770.0, 775.0, 780.0,
];

/// CIE 1924 photopic V(λ) values at 5nm intervals, 380-780nm.
pub const PHOTOPIC_V: [f64; 81] = [
    0.0000390, 0.0000640, 0.0001200, 0.0002170, 0.0003960, 0.0006400, 0.0012100, 0.0021800,
    0.0040000, 0.0073000, 0.0116000, 0.0168000, 0.0230000, 0.0298000, 0.0380000, 0.0480000,
    0.0600000, 0.0739000, 0.0910000, 0.1126000, 0.1390000, 0.1693000, 0.2080000, 0.2586000,
    0.3230000, 0.4073000, 0.5030000, 0.6082000, 0.7100000, 0.7932000, 0.8620000, 0.9149000,
    0.9540000, 0.9803000, 0.9950000, 1.0000000, 0.9950000, 0.9786000, 0.9520000, 0.9154000,
    0.8700000, 0.8163000, 0.7570000, 0.6949000, 0.6310000, 0.5668000, 0.5030000, 0.4412000,
    0.3810000, 0.3210000, 0.2650000, 0.2170000, 0.1750000, 0.1382000, 0.1070000, 0.0816000,
    0.0610000, 0.0446000, 0.0320000, 0.0232000, 0.0170000, 0.0119000, 0.0082100, 0.0057200,
    0.0041000, 0.0029300, 0.0020900, 0.0014800, 0.0010500, 0.0007400, 0.0005200, 0.0003610,
    0.0002490, 0.0001720, 0.0001200, 0.0000848, 0.0000600, 0.0000424, 0.0000300, 0.0000212,
    0.0000150,
];

/// CIE 1951 scotopic V'(λ) values at 5nm intervals, 380-780nm.
/// Night-adapted vision, peak at 507nm.
pub const SCOTOPIC_V_PRIME: [f64; 81] = [
    0.000589, 0.001108, 0.002209, 0.004530, 0.009290, 0.018530, 0.034840, 0.060400, 0.096600,
    0.143600, 0.199800, 0.263500, 0.328100, 0.407300, 0.503000, 0.608200, 0.710000, 0.793200,
    0.862000, 0.914850, 0.951000, 0.974000, 0.982000, 0.982000, 0.971000, 0.935000, 0.904000,
    0.854450, 0.793000, 0.721000, 0.650000, 0.581100, 0.517000, 0.456300, 0.400000, 0.349100,
    0.303000, 0.260800, 0.223000, 0.189800, 0.156700, 0.128600, 0.103500, 0.081100, 0.065000,
    0.051700, 0.041000, 0.032300, 0.025500, 0.020000, 0.015600, 0.012200, 0.009600, 0.007600,
    0.006000, 0.004800, 0.003800, 0.003000, 0.002400, 0.001900, 0.001500, 0.001200, 0.000950,
    0.000760, 0.000600, 0.000480, 0.000380, 0.000300, 0.000240, 0.000190, 0.000150, 0.000120,
    0.000095, 0.000076, 0.000060, 0.000048, 0.000038, 0.000030, 0.000024, 0.000019, 0.000015,
];

#[cfg(test)]
mod tests {
    use super::*;

    // ── Photopic V(λ) ──

    #[test]
    fn photopic_peak_at_555nm() {
        // CIE 1924: V(555nm) = 1.0
        // 555nm is at index 35: (555-380)/5 = 35
        assert!(
            (PHOTOPIC_V[35] - 1.0).abs() < 1e-10,
            "V(555nm) = {}, expected 1.0",
            PHOTOPIC_V[35]
        );
    }

    #[test]
    fn photopic_wavelength_at_index_35_is_555nm() {
        assert!(
            (PHOTOPIC_WAVELENGTHS_NM[35] - 555.0).abs() < 0.01,
            "Wavelength at index 35 = {}, expected 555.0",
            PHOTOPIC_WAVELENGTHS_NM[35]
        );
    }

    #[test]
    fn photopic_values_in_valid_range() {
        // All values should be in [0, 1]
        for (i, &v) in PHOTOPIC_V.iter().enumerate() {
            assert!(
                v >= 0.0 && v <= 1.0,
                "V[{}] at {}nm = {} outside [0, 1]",
                i,
                PHOTOPIC_WAVELENGTHS_NM[i],
                v
            );
        }
    }

    #[test]
    fn photopic_peak_is_maximum() {
        // V(555nm) = 1.0 should be the maximum
        let max_v = PHOTOPIC_V.iter().cloned().fold(0.0f64, f64::max);
        assert!(
            (max_v - 1.0).abs() < 1e-10,
            "Max V(λ) = {}, expected 1.0",
            max_v
        );
    }

    #[test]
    fn photopic_symmetric_around_peak() {
        // V(λ) should be roughly symmetric: V(550nm) ≈ V(560nm) (both near 1.0)
        // Index 34 = 550nm, Index 36 = 560nm
        assert!(
            (PHOTOPIC_V[34] - PHOTOPIC_V[36]).abs() < 0.01,
            "V(550nm)={} vs V(560nm)={} should be approximately equal",
            PHOTOPIC_V[34],
            PHOTOPIC_V[36]
        );
    }

    #[test]
    fn photopic_low_at_edges() {
        // V(380nm) and V(780nm) should be very small
        assert!(
            PHOTOPIC_V[0] < 0.001,
            "V(380nm) = {}, expected < 0.001",
            PHOTOPIC_V[0]
        );
        assert!(
            PHOTOPIC_V[80] < 0.001,
            "V(780nm) = {}, expected < 0.001",
            PHOTOPIC_V[80]
        );
    }

    #[test]
    fn photopic_array_length() {
        assert_eq!(PHOTOPIC_V.len(), 81);
        assert_eq!(PHOTOPIC_WAVELENGTHS_NM.len(), 81);
    }

    #[test]
    fn photopic_wavelength_grid_uniform_5nm() {
        for i in 0..80 {
            let spacing = PHOTOPIC_WAVELENGTHS_NM[i + 1] - PHOTOPIC_WAVELENGTHS_NM[i];
            assert!(
                (spacing - 5.0).abs() < 0.01,
                "Grid spacing at index {}: {} nm, expected 5 nm",
                i,
                spacing
            );
        }
    }

    #[test]
    fn photopic_wavelength_grid_starts_380_ends_780() {
        assert!((PHOTOPIC_WAVELENGTHS_NM[0] - 380.0).abs() < 0.01);
        assert!((PHOTOPIC_WAVELENGTHS_NM[80] - 780.0).abs() < 0.01);
    }

    // ── Scotopic V'(λ) ──

    #[test]
    fn scotopic_peak_near_507nm() {
        // CIE 1951: V'(λ) peaks at ~507nm with value ~0.982
        // 505nm = index 25: (505-380)/5 = 25
        // 510nm = index 26
        // Peak should be around indices 22-25
        let peak_val = SCOTOPIC_V_PRIME.iter().cloned().fold(0.0f64, f64::max);
        assert!(
            (peak_val - 0.982).abs() < 0.01,
            "Scotopic peak = {}, expected ~0.982",
            peak_val
        );
    }

    #[test]
    fn scotopic_peak_index_near_507nm() {
        // Find the index of the maximum
        let mut max_idx = 0;
        let mut max_val = 0.0;
        for (i, &v) in SCOTOPIC_V_PRIME.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        let peak_wl = PHOTOPIC_WAVELENGTHS_NM[max_idx];
        // Peak should be near 505-510nm (indices 25-26)
        assert!(
            peak_wl >= 490.0 && peak_wl <= 515.0,
            "Scotopic peak at {}nm (index {}), expected near 507nm",
            peak_wl,
            max_idx
        );
    }

    #[test]
    fn scotopic_values_in_valid_range() {
        for (i, &v) in SCOTOPIC_V_PRIME.iter().enumerate() {
            assert!(
                v >= 0.0 && v <= 1.0,
                "V'[{}] at {}nm = {} outside [0, 1]",
                i,
                PHOTOPIC_WAVELENGTHS_NM[i],
                v
            );
        }
    }

    #[test]
    fn scotopic_shifted_blue_from_photopic() {
        // Scotopic peak (507nm) should be bluer than photopic peak (555nm)
        // At 500nm: scotopic should be higher than photopic
        let idx_500 = 24; // (500-380)/5 = 24
        assert!(
            SCOTOPIC_V_PRIME[idx_500] > PHOTOPIC_V[idx_500],
            "At 500nm: scotopic ({}) should exceed photopic ({})",
            SCOTOPIC_V_PRIME[idx_500],
            PHOTOPIC_V[idx_500]
        );
    }

    #[test]
    fn scotopic_lower_in_red() {
        // At 700nm: scotopic should be much lower than photopic
        let idx_700 = 64; // (700-380)/5 = 64
        assert!(
            SCOTOPIC_V_PRIME[idx_700] < PHOTOPIC_V[idx_700],
            "At 700nm: scotopic ({}) should be less than photopic ({})",
            SCOTOPIC_V_PRIME[idx_700],
            PHOTOPIC_V[idx_700]
        );
    }

    #[test]
    fn scotopic_array_length() {
        assert_eq!(SCOTOPIC_V_PRIME.len(), 81);
    }

    #[test]
    fn scotopic_low_at_edges() {
        assert!(
            SCOTOPIC_V_PRIME[0] < 0.01,
            "V'(380nm) = {}, expected < 0.01",
            SCOTOPIC_V_PRIME[0]
        );
        assert!(
            SCOTOPIC_V_PRIME[80] < 0.001,
            "V'(780nm) = {}, expected < 0.001",
            SCOTOPIC_V_PRIME[80]
        );
    }
}
