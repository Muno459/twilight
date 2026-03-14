//! Tropospheric aerosol climatology based on OPAC aerosol models.
//!
//! Provides spectral optical properties for standard aerosol types:
//! extinction, single scattering albedo (SSA), and asymmetry parameter
//! as functions of wavelength, altitude, and relative humidity.
//!
//! Aerosol types follow Hess et al. (1998) "Optical Properties of Aerosols
//! and Clouds: The software package OPAC" (BAMS 79, 831-844).
//!
//! The wavelength dependence of extinction uses the Angstrom power law:
//!   β_ext(λ) = β_ext(550nm) × (550/λ)^α
//!
//! The vertical profile is exponential with a characteristic scale height:
//!   N(z) = N₀ × exp(-z/H)
//!
//! where z is altitude above sea level and H is the aerosol scale height.

use libm::{exp, pow};

/// Standard aerosol types from OPAC climatology.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AerosolType {
    /// Rural/background continental aerosol. Low loading, mostly sulfate
    /// and organic particles. Typical of remote continental areas.
    ContinentalClean,
    /// Moderate continental aerosol with some anthropogenic influence.
    /// Representative of average conditions over populated continents.
    ContinentalAverage,
    /// Urban/industrial aerosol. High soot content, strong absorption.
    /// Representative of polluted cities and industrial regions.
    Urban,
    /// Clean maritime aerosol. Dominated by sea salt particles.
    /// Representative of open ocean far from land.
    MaritimeClean,
    /// Polluted maritime aerosol. Sea salt plus continental outflow.
    /// Representative of coastal areas and shipping lanes.
    MaritimePolluted,
    /// Desert/mineral dust aerosol. Large coarse-mode particles.
    /// Representative of arid regions and dust events.
    Desert,
}

/// Aerosol optical properties at the 550nm reference wavelength.
///
/// These are the fundamental parameters from which spectral properties
/// are derived using the Angstrom law and empirical spectral slopes.
#[derive(Debug, Clone, Copy)]
pub struct AerosolProperties {
    /// Aerosol optical depth at 550nm (column-integrated extinction)
    pub aod_550: f64,
    /// Single scattering albedo at 550nm
    pub ssa_550: f64,
    /// Asymmetry parameter at 550nm
    pub asymmetry_550: f64,
    /// Angstrom exponent for extinction wavelength dependence
    /// AOD(λ) = AOD(550) × (550/λ)^α
    pub angstrom_exponent: f64,
    /// Aerosol scale height in meters (1/e folding altitude)
    pub scale_height_m: f64,
    /// SSA spectral slope: dSSA/dλ per nm (linear correction from 550nm)
    /// SSA(λ) = SSA(550) + ssa_slope × (λ - 550)
    pub ssa_slope: f64,
    /// Asymmetry spectral slope: dg/dλ per nm (linear correction from 550nm)
    /// g(λ) = g(550) + g_slope × (λ - 550)
    pub g_slope: f64,
}

/// Look up the default aerosol properties for a given type.
///
/// Values are representative of moderate relative humidity (~50-70%)
/// conditions, consistent with typical boundary layer environments.
///
/// # Sources
/// - Hess et al. (1998), OPAC Tables
/// - d'Almeida et al. (1991), Atmospheric Aerosols
/// - Dubovik et al. (2002), AERONET retrievals
pub fn default_properties(atype: AerosolType) -> AerosolProperties {
    match atype {
        AerosolType::ContinentalClean => AerosolProperties {
            aod_550: 0.05,
            ssa_550: 0.97,
            asymmetry_550: 0.68,
            angstrom_exponent: 1.3,
            scale_height_m: 2000.0,
            // SSA increases slightly toward red (less absorption)
            ssa_slope: 2.0e-5,
            // Asymmetry decreases with wavelength (smaller effective size parameter)
            g_slope: -1.5e-4,
        },
        AerosolType::ContinentalAverage => AerosolProperties {
            aod_550: 0.12,
            ssa_550: 0.93,
            asymmetry_550: 0.70,
            angstrom_exponent: 1.3,
            scale_height_m: 2000.0,
            ssa_slope: 1.5e-5,
            g_slope: -1.5e-4,
        },
        AerosolType::Urban => AerosolProperties {
            aod_550: 0.30,
            ssa_550: 0.88,
            asymmetry_550: 0.68,
            angstrom_exponent: 1.5,
            scale_height_m: 1500.0,
            // Urban SSA has stronger spectral dependence due to BC absorption
            // peaking in UV/blue. SSA *increases* toward longer wavelengths.
            ssa_slope: 5.0e-5,
            g_slope: -2.0e-4,
        },
        AerosolType::MaritimeClean => AerosolProperties {
            aod_550: 0.06,
            ssa_550: 0.99,
            asymmetry_550: 0.72,
            angstrom_exponent: 0.5,
            scale_height_m: 1000.0,
            // Sea salt is nearly non-absorbing; SSA barely changes with λ
            ssa_slope: 1.0e-6,
            g_slope: -1.0e-4,
        },
        AerosolType::MaritimePolluted => AerosolProperties {
            aod_550: 0.15,
            ssa_550: 0.96,
            asymmetry_550: 0.70,
            angstrom_exponent: 0.8,
            scale_height_m: 1500.0,
            ssa_slope: 1.5e-5,
            g_slope: -1.3e-4,
        },
        AerosolType::Desert => AerosolProperties {
            aod_550: 0.50,
            ssa_550: 0.92,
            asymmetry_550: 0.75,
            angstrom_exponent: 0.3,
            scale_height_m: 3000.0,
            // Dust SSA increases strongly toward red (iron oxide absorption
            // is concentrated below ~600nm)
            ssa_slope: 8.0e-5,
            // Large particles: g changes less with wavelength
            g_slope: -5.0e-5,
        },
    }
}

/// Compute the aerosol extinction coefficient [1/m] at a given wavelength
/// and altitude.
///
/// Uses the Angstrom power law for wavelength dependence and an
/// exponential vertical profile.
///
/// # Arguments
/// * `props` - Aerosol optical properties at 550nm reference
/// * `wavelength_nm` - Wavelength in nanometers
/// * `altitude_m` - Altitude above sea level in meters
///
/// # Returns
/// Aerosol extinction coefficient in 1/m
pub fn aerosol_extinction(props: &AerosolProperties, wavelength_nm: f64, altitude_m: f64) -> f64 {
    if altitude_m < 0.0 || props.aod_550 <= 0.0 || props.scale_height_m <= 0.0 {
        return 0.0;
    }

    // Column extinction at this wavelength: AOD(λ) = AOD(550) × (550/λ)^α
    let aod_lambda = props.aod_550 * pow(550.0 / wavelength_nm, props.angstrom_exponent);

    // Surface-level extinction coefficient:
    // AOD = ∫₀^∞ β₀ exp(-z/H) dz = β₀ × H
    // Therefore β₀ = AOD / H
    let beta_surface = aod_lambda / props.scale_height_m;

    // Exponential vertical profile
    beta_surface * exp(-altitude_m / props.scale_height_m)
}

/// Compute the aerosol single scattering albedo at a given wavelength.
///
/// Uses a linear spectral correction from the 550nm reference value.
/// Clamped to [0, 1].
///
/// # Arguments
/// * `props` - Aerosol optical properties at 550nm reference
/// * `wavelength_nm` - Wavelength in nanometers
pub fn aerosol_ssa(props: &AerosolProperties, wavelength_nm: f64) -> f64 {
    let ssa = props.ssa_550 + props.ssa_slope * (wavelength_nm - 550.0);
    // Clamp to physical bounds
    ssa.clamp(0.0, 1.0)
}

/// Compute the aerosol scattering asymmetry parameter at a given wavelength.
///
/// Uses a linear spectral correction from the 550nm reference value.
/// Clamped to [-1, 1].
///
/// # Arguments
/// * `props` - Aerosol optical properties at 550nm reference
/// * `wavelength_nm` - Wavelength in nanometers
pub fn aerosol_asymmetry(props: &AerosolProperties, wavelength_nm: f64) -> f64 {
    let g = props.asymmetry_550 + props.g_slope * (wavelength_nm - 550.0);
    g.clamp(-1.0, 1.0)
}

/// Compute the total column AOD at a given wavelength.
///
/// AOD(λ) = AOD(550nm) × (550/λ)^α
pub fn aerosol_aod(props: &AerosolProperties, wavelength_nm: f64) -> f64 {
    props.aod_550 * pow(550.0 / wavelength_nm, props.angstrom_exponent)
}

/// All six standard aerosol types for iteration.
pub const ALL_AEROSOL_TYPES: [AerosolType; 6] = [
    AerosolType::ContinentalClean,
    AerosolType::ContinentalAverage,
    AerosolType::Urban,
    AerosolType::MaritimeClean,
    AerosolType::MaritimePolluted,
    AerosolType::Desert,
];

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    // ── AerosolType / default_properties ──

    #[test]
    fn all_types_have_positive_aod() {
        for atype in &ALL_AEROSOL_TYPES {
            let p = default_properties(*atype);
            assert!(
                p.aod_550 > 0.0,
                "{:?} has non-positive AOD: {}",
                atype,
                p.aod_550
            );
        }
    }

    #[test]
    fn all_types_have_valid_ssa() {
        for atype in &ALL_AEROSOL_TYPES {
            let p = default_properties(*atype);
            assert!(
                p.ssa_550 > 0.0 && p.ssa_550 <= 1.0,
                "{:?} has invalid SSA: {}",
                atype,
                p.ssa_550
            );
        }
    }

    #[test]
    fn all_types_have_valid_asymmetry() {
        for atype in &ALL_AEROSOL_TYPES {
            let p = default_properties(*atype);
            assert!(
                p.asymmetry_550 >= 0.0 && p.asymmetry_550 <= 1.0,
                "{:?} has invalid asymmetry: {}",
                atype,
                p.asymmetry_550
            );
        }
    }

    #[test]
    fn all_types_have_positive_angstrom() {
        for atype in &ALL_AEROSOL_TYPES {
            let p = default_properties(*atype);
            assert!(
                p.angstrom_exponent >= 0.0,
                "{:?} has negative Angstrom exponent: {}",
                atype,
                p.angstrom_exponent
            );
        }
    }

    #[test]
    fn all_types_have_positive_scale_height() {
        for atype in &ALL_AEROSOL_TYPES {
            let p = default_properties(*atype);
            assert!(
                p.scale_height_m > 0.0,
                "{:?} has non-positive scale height: {}",
                atype,
                p.scale_height_m
            );
        }
    }

    #[test]
    fn urban_most_absorbing() {
        // Urban aerosol should have the lowest SSA (most absorbing)
        let urban = default_properties(AerosolType::Urban);
        for atype in &ALL_AEROSOL_TYPES {
            let p = default_properties(*atype);
            assert!(
                urban.ssa_550 <= p.ssa_550 + EPSILON,
                "Urban SSA {} should be <= {:?} SSA {}",
                urban.ssa_550,
                atype,
                p.ssa_550
            );
        }
    }

    #[test]
    fn maritime_clean_least_absorbing() {
        // Maritime clean should have the highest SSA
        let mc = default_properties(AerosolType::MaritimeClean);
        for atype in &ALL_AEROSOL_TYPES {
            let p = default_properties(*atype);
            assert!(
                mc.ssa_550 >= p.ssa_550 - EPSILON,
                "Maritime clean SSA {} should be >= {:?} SSA {}",
                mc.ssa_550,
                atype,
                p.ssa_550
            );
        }
    }

    #[test]
    fn desert_highest_aod() {
        // Desert should have the highest default AOD
        let desert = default_properties(AerosolType::Desert);
        for atype in &ALL_AEROSOL_TYPES {
            let p = default_properties(*atype);
            assert!(
                desert.aod_550 >= p.aod_550 - EPSILON,
                "Desert AOD {} should be >= {:?} AOD {}",
                desert.aod_550,
                atype,
                p.aod_550
            );
        }
    }

    #[test]
    fn desert_lowest_angstrom() {
        // Desert (coarse particles) should have the lowest Angstrom exponent
        let desert = default_properties(AerosolType::Desert);
        for atype in &ALL_AEROSOL_TYPES {
            let p = default_properties(*atype);
            assert!(
                desert.angstrom_exponent <= p.angstrom_exponent + EPSILON,
                "Desert Angstrom {} should be <= {:?} Angstrom {}",
                desert.angstrom_exponent,
                atype,
                p.angstrom_exponent
            );
        }
    }

    #[test]
    fn urban_highest_angstrom() {
        // Urban (fine particles) should have the highest Angstrom exponent
        let urban = default_properties(AerosolType::Urban);
        for atype in &ALL_AEROSOL_TYPES {
            let p = default_properties(*atype);
            assert!(
                urban.angstrom_exponent >= p.angstrom_exponent - EPSILON,
                "Urban Angstrom {} should be >= {:?} Angstrom {}",
                urban.angstrom_exponent,
                atype,
                p.angstrom_exponent
            );
        }
    }

    // ── aerosol_extinction ──

    #[test]
    fn extinction_positive_at_surface() {
        for atype in &ALL_AEROSOL_TYPES {
            let p = default_properties(*atype);
            let ext = aerosol_extinction(&p, 550.0, 0.0);
            assert!(
                ext > 0.0,
                "{:?} extinction at surface should be positive: {}",
                atype,
                ext
            );
        }
    }

    #[test]
    fn extinction_decreases_with_altitude() {
        for atype in &ALL_AEROSOL_TYPES {
            let p = default_properties(*atype);
            let ext_0 = aerosol_extinction(&p, 550.0, 0.0);
            let ext_5k = aerosol_extinction(&p, 550.0, 5000.0);
            let ext_10k = aerosol_extinction(&p, 550.0, 10000.0);
            assert!(
                ext_0 > ext_5k,
                "{:?}: ext(0m)={} should be > ext(5km)={}",
                atype,
                ext_0,
                ext_5k
            );
            assert!(
                ext_5k > ext_10k,
                "{:?}: ext(5km)={} should be > ext(10km)={}",
                atype,
                ext_5k,
                ext_10k
            );
        }
    }

    #[test]
    fn extinction_integrates_to_aod() {
        // Verify that ∫₀^∞ β(z) dz = AOD for each type at 550nm.
        // Since β(z) = (AOD/H) × exp(-z/H), the integral is exactly AOD.
        // We'll integrate numerically to confirm.
        for atype in &ALL_AEROSOL_TYPES {
            let p = default_properties(*atype);
            let dz = 10.0; // 10m steps
            let n_steps = 100_000; // up to 1000 km (well beyond scale height)
            let mut integral = 0.0;
            for i in 0..n_steps {
                let z = (i as f64 + 0.5) * dz;
                integral += aerosol_extinction(&p, 550.0, z) * dz;
            }
            let rel_err = (integral - p.aod_550).abs() / p.aod_550;
            assert!(
                rel_err < 0.01,
                "{:?}: ∫β dz = {:.6}, expected AOD = {:.6}, err = {:.4}",
                atype,
                integral,
                p.aod_550,
                rel_err
            );
        }
    }

    #[test]
    fn extinction_blue_greater_than_red_for_fine_particles() {
        // For fine-mode aerosols (Angstrom > 1), blue extinction should exceed red
        let p = default_properties(AerosolType::Urban); // α = 1.5
        let ext_400 = aerosol_extinction(&p, 400.0, 0.0);
        let ext_700 = aerosol_extinction(&p, 700.0, 0.0);
        assert!(
            ext_400 > ext_700,
            "Urban: ext(400nm)={:.4e} should be > ext(700nm)={:.4e}",
            ext_400,
            ext_700
        );
    }

    #[test]
    fn extinction_wavelength_ratio_matches_angstrom() {
        // β(400) / β(700) = (550/400)^α / (550/700)^α = (700/400)^α
        let p = default_properties(AerosolType::ContinentalAverage);
        let ext_400 = aerosol_extinction(&p, 400.0, 0.0);
        let ext_700 = aerosol_extinction(&p, 700.0, 0.0);
        let ratio = ext_400 / ext_700;
        let expected = pow(700.0 / 400.0, p.angstrom_exponent);
        let rel_err = (ratio - expected).abs() / expected;
        assert!(
            rel_err < 1e-10,
            "Wavelength ratio {:.4} should match (700/400)^α = {:.4}",
            ratio,
            expected
        );
    }

    #[test]
    fn extinction_zero_for_zero_aod() {
        let mut p = default_properties(AerosolType::Urban);
        p.aod_550 = 0.0;
        assert!(aerosol_extinction(&p, 550.0, 0.0).abs() < 1e-30);
    }

    #[test]
    fn extinction_zero_for_negative_altitude() {
        let p = default_properties(AerosolType::Urban);
        assert!(aerosol_extinction(&p, 550.0, -100.0).abs() < 1e-30);
    }

    #[test]
    fn extinction_desert_weak_wavelength_dependence() {
        // Desert (α=0.3) should have much weaker wavelength dependence than urban (α=1.5)
        let desert = default_properties(AerosolType::Desert);
        let urban = default_properties(AerosolType::Urban);

        let desert_ratio =
            aerosol_extinction(&desert, 400.0, 0.0) / aerosol_extinction(&desert, 700.0, 0.0);
        let urban_ratio =
            aerosol_extinction(&urban, 400.0, 0.0) / aerosol_extinction(&urban, 700.0, 0.0);

        assert!(
            urban_ratio > desert_ratio,
            "Urban ratio {:.3} should be > desert ratio {:.3}",
            urban_ratio,
            desert_ratio
        );
    }

    // ── aerosol_ssa ──

    #[test]
    fn ssa_at_550_matches_reference() {
        for atype in &ALL_AEROSOL_TYPES {
            let p = default_properties(*atype);
            let ssa = aerosol_ssa(&p, 550.0);
            assert!(
                (ssa - p.ssa_550).abs() < 1e-15,
                "{:?}: SSA(550nm) = {}, expected {}",
                atype,
                ssa,
                p.ssa_550
            );
        }
    }

    #[test]
    fn ssa_bounded_0_to_1() {
        for atype in &ALL_AEROSOL_TYPES {
            let p = default_properties(*atype);
            for wl in (380..=780).step_by(10) {
                let ssa = aerosol_ssa(&p, wl as f64);
                assert!(
                    (0.0..=1.0).contains(&ssa),
                    "{:?}: SSA({}) = {} out of bounds",
                    atype,
                    wl,
                    ssa
                );
            }
        }
    }

    #[test]
    fn ssa_urban_increases_toward_red() {
        // Urban aerosol: BC absorption peaks in blue, so SSA increases toward red
        let p = default_properties(AerosolType::Urban);
        let ssa_400 = aerosol_ssa(&p, 400.0);
        let ssa_700 = aerosol_ssa(&p, 700.0);
        assert!(
            ssa_700 > ssa_400,
            "Urban SSA should increase toward red: SSA(400)={:.4} SSA(700)={:.4}",
            ssa_400,
            ssa_700
        );
    }

    // ── aerosol_asymmetry ──

    #[test]
    fn asymmetry_at_550_matches_reference() {
        for atype in &ALL_AEROSOL_TYPES {
            let p = default_properties(*atype);
            let g = aerosol_asymmetry(&p, 550.0);
            assert!(
                (g - p.asymmetry_550).abs() < 1e-15,
                "{:?}: g(550nm) = {}, expected {}",
                atype,
                g,
                p.asymmetry_550
            );
        }
    }

    #[test]
    fn asymmetry_bounded() {
        for atype in &ALL_AEROSOL_TYPES {
            let p = default_properties(*atype);
            for wl in (380..=780).step_by(10) {
                let g = aerosol_asymmetry(&p, wl as f64);
                assert!(
                    (-1.0..=1.0).contains(&g),
                    "{:?}: g({}) = {} out of bounds",
                    atype,
                    wl,
                    g
                );
            }
        }
    }

    #[test]
    fn asymmetry_decreases_toward_red() {
        // For all types, g should decrease with wavelength
        // (smaller effective size parameter at longer wavelengths)
        for atype in &ALL_AEROSOL_TYPES {
            let p = default_properties(*atype);
            let g_400 = aerosol_asymmetry(&p, 400.0);
            let g_700 = aerosol_asymmetry(&p, 700.0);
            assert!(
                g_400 >= g_700 - EPSILON,
                "{:?}: g(400nm)={:.4} should be >= g(700nm)={:.4}",
                atype,
                g_400,
                g_700
            );
        }
    }

    // ── aerosol_aod ──

    #[test]
    fn aod_at_550_matches_reference() {
        for atype in &ALL_AEROSOL_TYPES {
            let p = default_properties(*atype);
            let aod = aerosol_aod(&p, 550.0);
            assert!(
                (aod - p.aod_550).abs() < 1e-15,
                "{:?}: AOD(550nm) = {}, expected {}",
                atype,
                aod,
                p.aod_550
            );
        }
    }

    #[test]
    fn aod_angstrom_law() {
        // Verify AOD(λ₁)/AOD(λ₂) = (λ₂/λ₁)^α
        let p = default_properties(AerosolType::ContinentalClean);
        let aod_400 = aerosol_aod(&p, 400.0);
        let aod_700 = aerosol_aod(&p, 700.0);
        let ratio = aod_400 / aod_700;
        let expected = pow(700.0 / 400.0, p.angstrom_exponent);
        assert!(
            (ratio - expected).abs() < 1e-10,
            "AOD ratio {:.4} should match Angstrom law {:.4}",
            ratio,
            expected
        );
    }

    #[test]
    fn aod_urban_largest_in_blue() {
        // Fine-mode urban aerosol: AOD should be largest in blue
        let p = default_properties(AerosolType::Urban);
        let aod_380 = aerosol_aod(&p, 380.0);
        let aod_550 = aerosol_aod(&p, 550.0);
        let aod_780 = aerosol_aod(&p, 780.0);
        assert!(aod_380 > aod_550);
        assert!(aod_550 > aod_780);
    }

    // ── Custom properties ──

    #[test]
    fn custom_properties_work() {
        let p = AerosolProperties {
            aod_550: 1.0,
            ssa_550: 0.80,
            asymmetry_550: 0.60,
            angstrom_exponent: 2.0,
            scale_height_m: 1000.0,
            ssa_slope: 0.0,
            g_slope: 0.0,
        };

        let ext = aerosol_extinction(&p, 550.0, 0.0);
        // β₀ = AOD/H = 1.0/1000 = 0.001 m⁻¹
        assert!(
            (ext - 0.001).abs() < 1e-10,
            "Custom ext = {}, expected 0.001",
            ext
        );

        let ssa = aerosol_ssa(&p, 550.0);
        assert!((ssa - 0.80).abs() < 1e-15);

        let g = aerosol_asymmetry(&p, 550.0);
        assert!((g - 0.60).abs() < 1e-15);
    }

    #[test]
    fn continental_clean_aod_matches_literature() {
        // Continental clean: AOD(550) ~ 0.05 (OPAC, d'Almeida 1991)
        let p = default_properties(AerosolType::ContinentalClean);
        assert!(
            (p.aod_550 - 0.05).abs() < 0.02,
            "Continental clean AOD = {}",
            p.aod_550
        );
    }

    #[test]
    fn urban_aod_matches_literature() {
        // Urban: AOD(550) ~ 0.2-0.4 (AERONET observations)
        let p = default_properties(AerosolType::Urban);
        assert!(
            p.aod_550 > 0.15 && p.aod_550 < 0.5,
            "Urban AOD = {}",
            p.aod_550
        );
    }

    #[test]
    fn desert_aod_matches_literature() {
        // Desert: AOD(550) ~ 0.3-0.8 (AERONET observations)
        let p = default_properties(AerosolType::Desert);
        assert!(
            p.aod_550 > 0.2 && p.aod_550 < 1.0,
            "Desert AOD = {}",
            p.aod_550
        );
    }

    #[test]
    fn urban_ssa_matches_literature() {
        // Urban SSA at 550nm: ~0.85-0.93 (AERONET, Dubovik 2002)
        let p = default_properties(AerosolType::Urban);
        assert!(
            p.ssa_550 > 0.82 && p.ssa_550 < 0.95,
            "Urban SSA = {}",
            p.ssa_550
        );
    }

    #[test]
    fn desert_ssa_matches_literature() {
        // Desert SSA at 550nm: ~0.88-0.96 (AERONET)
        let p = default_properties(AerosolType::Desert);
        assert!(
            p.ssa_550 > 0.85 && p.ssa_550 < 0.98,
            "Desert SSA = {}",
            p.ssa_550
        );
    }

    #[test]
    fn scale_heights_in_reasonable_range() {
        for atype in &ALL_AEROSOL_TYPES {
            let p = default_properties(*atype);
            // Aerosol scale heights should be 0.5-5 km
            assert!(
                p.scale_height_m > 500.0 && p.scale_height_m < 5000.0,
                "{:?}: scale height = {} m",
                atype,
                p.scale_height_m
            );
        }
    }

    #[test]
    fn extinction_at_one_scale_height() {
        // At z = H, extinction should be β₀/e
        let p = default_properties(AerosolType::Urban);
        let ext_0 = aerosol_extinction(&p, 550.0, 0.0);
        let ext_h = aerosol_extinction(&p, 550.0, p.scale_height_m);
        let ratio = ext_h / ext_0;
        let expected = 1.0 / core::f64::consts::E;
        assert!(
            (ratio - expected).abs() < 1e-10,
            "ext(H)/ext(0) = {:.6}, expected 1/e = {:.6}",
            ratio,
            expected
        );
    }

    #[test]
    fn six_aerosol_types_exist() {
        assert_eq!(ALL_AEROSOL_TYPES.len(), 6);
    }

    #[test]
    fn aerosol_types_are_distinct() {
        for i in 0..ALL_AEROSOL_TYPES.len() {
            for j in (i + 1)..ALL_AEROSOL_TYPES.len() {
                assert_ne!(
                    ALL_AEROSOL_TYPES[i], ALL_AEROSOL_TYPES[j],
                    "Types {:?} and {:?} should be distinct",
                    ALL_AEROSOL_TYPES[i], ALL_AEROSOL_TYPES[j]
                );
            }
        }
    }
}
