//! Standard cloud layer types with physically realistic defaults.
//!
//! Provides named cloud presets for use with the atmosphere builder.
//! Each type specifies base altitude, top altitude, optical depth,
//! single scattering albedo, and asymmetry parameter.
//!
//! Cloud extinction is approximately wavelength-independent in the
//! visible spectrum (size parameter x >> 1 for cloud droplets), so
//! optical depth is constant across 380-780 nm.
//!
//! Liquid water clouds (stratus, stratocumulus, cumulus, altostratus)
//! have SSA very close to 1.0 (nearly non-absorbing) and asymmetry
//! parameter ~0.85 (strong forward scattering from large droplets).
//!
//! Ice clouds (cirrus) have slightly lower SSA and smaller asymmetry
//! (~0.77) due to non-spherical crystal shapes.

/// Standard cloud types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CloudType {
    /// Thin cirrus (ice). High altitude, low optical depth.
    /// Common in fair weather. Barely visible, mostly transparent.
    ThinCirrus,
    /// Thick cirrus (ice). High altitude, moderate optical depth.
    /// Sun disk obscured. Produces halos.
    ThickCirrus,
    /// Altostratus (water/mixed). Mid-level, moderate optical depth.
    /// Diffuse light, no shadows.
    Altostratus,
    /// Stratus (water). Low overcast blanket.
    /// Uniform grey sky, steady drizzle possible.
    Stratus,
    /// Stratocumulus (water). Low, lumpy cloud sheet.
    /// Most common cloud type globally.
    Stratocumulus,
    /// Cumulus (water). Fair-weather puffy clouds.
    /// Moderate to thick.
    Cumulus,
}

/// Cloud layer properties.
#[derive(Debug, Clone, Copy)]
pub struct CloudProperties {
    /// Cloud base altitude in km above sea level
    pub base_km: f64,
    /// Cloud top altitude in km above sea level
    pub top_km: f64,
    /// Cloud optical depth (wavelength-independent in the visible)
    pub optical_depth: f64,
    /// Single scattering albedo (fraction of extinction that is scattering)
    pub ssa: f64,
    /// Scattering asymmetry parameter (Henyey-Greenstein g)
    pub asymmetry: f64,
}

/// Look up default properties for a cloud type.
///
/// Values represent typical mid-latitude conditions.
///
/// # Sources
/// - Heymsfield & Platt (1984), cirrus microphysics
/// - Stephens (1978), cloud radiative properties
/// - ISCCP cloud climatology
pub fn default_properties(ctype: CloudType) -> CloudProperties {
    match ctype {
        CloudType::ThinCirrus => CloudProperties {
            base_km: 8.0,
            top_km: 10.0,
            optical_depth: 0.3,
            ssa: 0.97,
            asymmetry: 0.77,
        },
        CloudType::ThickCirrus => CloudProperties {
            base_km: 8.0,
            top_km: 11.0,
            optical_depth: 2.0,
            ssa: 0.96,
            asymmetry: 0.77,
        },
        CloudType::Altostratus => CloudProperties {
            base_km: 3.0,
            top_km: 5.0,
            optical_depth: 5.0,
            ssa: 0.999,
            asymmetry: 0.85,
        },
        CloudType::Stratus => CloudProperties {
            base_km: 0.5,
            top_km: 1.5,
            optical_depth: 10.0,
            ssa: 0.999,
            asymmetry: 0.85,
        },
        CloudType::Stratocumulus => CloudProperties {
            base_km: 1.0,
            top_km: 2.0,
            optical_depth: 8.0,
            ssa: 0.999,
            asymmetry: 0.85,
        },
        CloudType::Cumulus => CloudProperties {
            base_km: 1.0,
            top_km: 3.0,
            optical_depth: 15.0,
            ssa: 0.999,
            asymmetry: 0.85,
        },
    }
}

/// All six standard cloud types for iteration.
pub const ALL_CLOUD_TYPES: [CloudType; 6] = [
    CloudType::ThinCirrus,
    CloudType::ThickCirrus,
    CloudType::Altostratus,
    CloudType::Stratus,
    CloudType::Stratocumulus,
    CloudType::Cumulus,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_types_have_positive_od() {
        for ctype in &ALL_CLOUD_TYPES {
            let p = default_properties(*ctype);
            assert!(p.optical_depth > 0.0, "{:?} has non-positive OD", ctype);
        }
    }

    #[test]
    fn all_types_have_valid_ssa() {
        for ctype in &ALL_CLOUD_TYPES {
            let p = default_properties(*ctype);
            assert!(
                p.ssa > 0.0 && p.ssa <= 1.0,
                "{:?} has invalid SSA: {}",
                ctype,
                p.ssa
            );
        }
    }

    #[test]
    fn all_types_have_valid_asymmetry() {
        for ctype in &ALL_CLOUD_TYPES {
            let p = default_properties(*ctype);
            assert!(
                p.asymmetry > 0.0 && p.asymmetry <= 1.0,
                "{:?} has invalid asymmetry: {}",
                ctype,
                p.asymmetry
            );
        }
    }

    #[test]
    fn all_types_have_valid_altitude() {
        for ctype in &ALL_CLOUD_TYPES {
            let p = default_properties(*ctype);
            assert!(p.base_km >= 0.0, "{:?} has negative base", ctype);
            assert!(
                p.top_km > p.base_km,
                "{:?} has top <= base: {} <= {}",
                ctype,
                p.top_km,
                p.base_km
            );
            assert!(p.top_km < 20.0, "{:?} has top > 20km: {}", ctype, p.top_km);
        }
    }

    #[test]
    fn cirrus_higher_than_stratus() {
        let cirrus = default_properties(CloudType::ThinCirrus);
        let stratus = default_properties(CloudType::Stratus);
        assert!(
            cirrus.base_km > stratus.top_km,
            "Cirrus base ({}) should be above stratus top ({})",
            cirrus.base_km,
            stratus.top_km
        );
    }

    #[test]
    fn cirrus_lower_ssa_than_water_clouds() {
        let cirrus = default_properties(CloudType::ThinCirrus);
        let stratus = default_properties(CloudType::Stratus);
        assert!(
            cirrus.ssa < stratus.ssa,
            "Cirrus SSA ({}) should be < stratus SSA ({})",
            cirrus.ssa,
            stratus.ssa
        );
    }

    #[test]
    fn cirrus_lower_asymmetry_than_water_clouds() {
        let cirrus = default_properties(CloudType::ThinCirrus);
        let stratus = default_properties(CloudType::Stratus);
        assert!(
            cirrus.asymmetry < stratus.asymmetry,
            "Cirrus g ({}) should be < stratus g ({})",
            cirrus.asymmetry,
            stratus.asymmetry
        );
    }

    #[test]
    fn water_clouds_nearly_non_absorbing() {
        // All liquid water clouds should have SSA > 0.99
        let water_types = [
            CloudType::Altostratus,
            CloudType::Stratus,
            CloudType::Stratocumulus,
            CloudType::Cumulus,
        ];
        for ctype in &water_types {
            let p = default_properties(*ctype);
            assert!(
                p.ssa > 0.99,
                "{:?} SSA should be > 0.99, got {}",
                ctype,
                p.ssa
            );
        }
    }

    #[test]
    fn water_clouds_strong_forward_scatter() {
        // All liquid water clouds should have g ~ 0.85
        let water_types = [
            CloudType::Altostratus,
            CloudType::Stratus,
            CloudType::Stratocumulus,
            CloudType::Cumulus,
        ];
        for ctype in &water_types {
            let p = default_properties(*ctype);
            assert!(
                (p.asymmetry - 0.85).abs() < 0.05,
                "{:?} g should be ~0.85, got {}",
                ctype,
                p.asymmetry
            );
        }
    }

    #[test]
    fn thin_cirrus_thinner_than_thick() {
        let thin = default_properties(CloudType::ThinCirrus);
        let thick = default_properties(CloudType::ThickCirrus);
        assert!(
            thin.optical_depth < thick.optical_depth,
            "Thin cirrus OD ({}) should be < thick cirrus OD ({})",
            thin.optical_depth,
            thick.optical_depth
        );
    }

    #[test]
    fn cumulus_thickest() {
        let cumulus = default_properties(CloudType::Cumulus);
        for ctype in &ALL_CLOUD_TYPES {
            let p = default_properties(*ctype);
            assert!(
                cumulus.optical_depth >= p.optical_depth,
                "Cumulus OD ({}) should be >= {:?} OD ({})",
                cumulus.optical_depth,
                ctype,
                p.optical_depth
            );
        }
    }

    #[test]
    fn six_cloud_types_exist() {
        assert_eq!(ALL_CLOUD_TYPES.len(), 6);
    }

    #[test]
    fn cloud_types_are_distinct() {
        for i in 0..ALL_CLOUD_TYPES.len() {
            for j in (i + 1)..ALL_CLOUD_TYPES.len() {
                assert_ne!(
                    ALL_CLOUD_TYPES[i], ALL_CLOUD_TYPES[j],
                    "Types {:?} and {:?} should be distinct",
                    ALL_CLOUD_TYPES[i], ALL_CLOUD_TYPES[j]
                );
            }
        }
    }
}
