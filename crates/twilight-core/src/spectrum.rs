//! Wavelength grid and spectral cross-section utilities.

/// Rayleigh scattering cross-section using Bodhaine et al. (1999).
///
/// Includes wavelength-dependent King correction factor.
///
/// - `wavelength_nm`: wavelength in nanometers
///
/// Returns scattering cross-section in cm^2 per molecule.
pub fn rayleigh_cross_section(wavelength_nm: f64) -> f64 {
    use libm::pow;

    let lambda_um = wavelength_nm / 1000.0; // nm to μm
    let lambda_cm = wavelength_nm * 1e-7; // nm to cm

    // Refractive index of air (Peck & Reeder 1972)
    // (n-1) × 10^8 for standard air at 288.15 K, 1013.25 hPa
    let sigma2 = 1.0 / (lambda_um * lambda_um);
    let n_minus_1 = (5791817.0 / (238.0185 - sigma2) + 167909.0 / (57.362 - sigma2)) * 1e-8;

    // Number density at standard conditions (Loschmidt number)
    let n_s: f64 = 2.546899e19; // molecules/cm^3 at 288.15 K, 1013.25 hPa

    // King factor F(lambda) - depolarization correction
    // From Bates (1984), varies with wavelength
    // F = (6 + 3*rho) / (6 - 7*rho) where rho is depolarization ratio
    // Approximate: F ≈ 1.049 at 400nm, 1.048 at 550nm, 1.046 at 700nm
    let f_king = 1.0480 + 0.00013 * (550.0 - wavelength_nm) / 150.0;

    // Rayleigh cross-section formula (Bodhaine et al. 1999, Eq. 1)
    // σ = (24π³ / N_s²λ⁴) × ((n²-1)/(n²+2))² × F(λ)
    //
    // n²-1 = (n-1)(n+1) = 2(n-1) + (n-1)² (exact expansion)
    let n2_minus_1 = 2.0 * n_minus_1 + n_minus_1 * n_minus_1;
    let n2_plus_2 = n2_minus_1 + 3.0; // n² + 2
    let lorentz_lorenz = n2_minus_1 / n2_plus_2; // (n²-1)/(n²+2)

    (24.0 * core::f64::consts::PI * core::f64::consts::PI * core::f64::consts::PI)
        / (n_s * n_s * pow(lambda_cm, 4.0))
        * (lorentz_lorenz * lorentz_lorenz)
        * f_king
}

/// Compute Rayleigh scattering coefficient (extinction per meter) at given
/// wavelength and number density.
///
/// - `wavelength_nm`: wavelength in nanometers
/// - `number_density`: air number density in molecules/m^3
///
/// Returns scattering coefficient in 1/m.
pub fn rayleigh_scattering_coeff(wavelength_nm: f64, number_density: f64) -> f64 {
    let sigma_cm2 = rayleigh_cross_section(wavelength_nm);
    let sigma_m2 = sigma_cm2 * 1e-4; // cm² to m²
    sigma_m2 * number_density
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values validated against Bodhaine et al. (1999) and
    // standard atmosphere optical depths.
    //
    // Our implementation uses the exact Lorentz-Lorenz factor
    // ((n²-1)/(n²+2))² with Peck & Reeder refractive index and
    // Bates (1984) King factor. The resulting optical depths match
    // published values to within ~1%:
    //   τ(550nm) = 0.097 (ref: 0.098)
    //   τ(400nm) = 0.359 (ref: 0.36)

    #[test]
    fn rayleigh_xsec_400nm_value() {
        let sigma = rayleigh_cross_section(400.0);
        // Bodhaine-style formula gives ~1.669e-26 at 400nm
        assert!(
            (sigma - 1.669e-26).abs() < 0.05e-26,
            "σ(400nm) = {:.4e}, expected ~1.669e-26",
            sigma
        );
    }

    #[test]
    fn rayleigh_xsec_550nm_value() {
        let sigma = rayleigh_cross_section(550.0);
        // ~4.507e-27 at 550nm
        assert!(
            (sigma - 4.507e-27).abs() < 0.05e-27,
            "σ(550nm) = {:.4e}, expected ~4.507e-27",
            sigma
        );
    }

    #[test]
    fn rayleigh_xsec_700nm_value() {
        let sigma = rayleigh_cross_section(700.0);
        // ~1.692e-27 at 700nm
        assert!(
            (sigma - 1.692e-27).abs() < 0.02e-27,
            "σ(700nm) = {:.4e}, expected ~1.692e-27",
            sigma
        );
    }

    #[test]
    fn rayleigh_optical_depth_550nm() {
        // Full-atmosphere column density ≈ 2.15e25 mol/cm²
        // τ_ray(550nm) should be ≈ 0.098 (Bucholtz 1995, Bodhaine 1999)
        let sigma = rayleigh_cross_section(550.0);
        let n_col = 2.15e25; // molecules/cm²
        let tau = sigma * n_col;
        assert!(
            (tau - 0.098).abs() < 0.005,
            "τ_ray(550nm) = {:.4}, expected ~0.098",
            tau
        );
    }

    #[test]
    fn rayleigh_optical_depth_400nm() {
        // τ_ray(400nm) should be ≈ 0.36
        let sigma = rayleigh_cross_section(400.0);
        let n_col = 2.15e25;
        let tau = sigma * n_col;
        assert!(
            (tau - 0.36).abs() < 0.01,
            "τ_ray(400nm) = {:.4}, expected ~0.36",
            tau
        );
    }

    #[test]
    fn rayleigh_xsec_lambda_minus_four_scaling() {
        // Rayleigh scattering scales as λ⁻⁴.
        // σ(400nm)/σ(780nm) should be ≈ (780/400)⁴ ≈ 14.46
        // Allow 10% tolerance because King factor varies slightly with wavelength
        let s400 = rayleigh_cross_section(400.0);
        let s780 = rayleigh_cross_section(780.0);
        let ratio = s400 / s780;
        let expected = libm::pow(780.0 / 400.0, 4.0);
        assert!(
            (ratio / expected - 1.0).abs() < 0.10,
            "λ⁻⁴ scaling: ratio={:.2}, expected~{:.2}",
            ratio,
            expected
        );
    }

    #[test]
    fn rayleigh_xsec_is_positive() {
        for wl in (380..=780).step_by(10) {
            let sigma = rayleigh_cross_section(wl as f64);
            assert!(sigma > 0.0, "σ({}) should be positive, got {}", wl, sigma);
        }
    }

    #[test]
    fn rayleigh_xsec_decreases_with_wavelength() {
        // Cross-section should monotonically decrease with increasing wavelength
        let mut prev = rayleigh_cross_section(380.0);
        for wl in (390..=780).step_by(10) {
            let sigma = rayleigh_cross_section(wl as f64);
            assert!(
                sigma < prev,
                "σ should decrease: σ({})={:.4e} >= σ({})={:.4e}",
                wl,
                sigma,
                wl - 10,
                prev
            );
            prev = sigma;
        }
    }

    #[test]
    fn rayleigh_scattering_coeff_at_sea_level() {
        // At sea level: n ≈ 2.547e25 molecules/m³ (2.547e19 cm⁻³ × 1e6)
        // At 550nm, β_ray should be ≈ 0.013 km⁻¹ = 1.3e-5 m⁻¹
        let n_density_m3 = 2.547e19 * 1e6; // molecules/m³
        let beta = rayleigh_scattering_coeff(550.0, n_density_m3);
        assert!(
            (1.0e-5..5.0e-5).contains(&beta),
            "β_ray(550nm, sea level) = {:.4e}, expected ~1.3e-5 m⁻¹",
            beta
        );
    }

    #[test]
    fn rayleigh_scattering_coeff_zero_density() {
        let beta = rayleigh_scattering_coeff(550.0, 0.0);
        assert!(
            beta.abs() < 1e-30,
            "Zero density should give zero coefficient"
        );
    }

    #[test]
    fn rayleigh_scattering_coeff_proportional_to_density() {
        let n1 = 1e25;
        let n2 = 2e25;
        let beta1 = rayleigh_scattering_coeff(550.0, n1);
        let beta2 = rayleigh_scattering_coeff(550.0, n2);
        assert!(
            ((beta2 / beta1) - 2.0).abs() < 1e-10,
            "β should be proportional to density"
        );
    }
}
