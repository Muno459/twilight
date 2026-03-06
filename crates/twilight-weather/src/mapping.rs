//! Map weather observations to MCRT aerosol, cloud, and gas parameters.
//!
//! The mapping is physically motivated but necessarily approximate.
//! We use the measured AOD at 550nm directly (not the type default),
//! and select the aerosol "type" (which determines SSA, asymmetry,
//! Angstrom exponent, and scale height) based on dust concentration
//! and AOD magnitude.
//!
//! Cloud mapping uses the low/mid/high cloud cover breakdown to select
//! the dominant cloud type, and scales optical depth by coverage fraction.
//!
//! Gas composition mapping converts surface O3 and NO2 concentrations
//! from the CAMS-based air quality API into total column estimates for
//! the MCRT gas absorption model.

use twilight_data::aerosol::{self, AerosolProperties, AerosolType};
use twilight_data::cloud::{self, CloudProperties, CloudType};

use crate::{GasComposition, WeatherConditions};

/// Minimum AOD to consider aerosols worth modeling.
/// Below this, the atmosphere is essentially pristine.
const AOD_THRESHOLD: f64 = 0.04;

/// Dust concentration threshold (ug/m3) above which we classify as desert-type.
const DUST_THRESHOLD: f64 = 20.0;

/// Minimum cloud cover (%) to include a cloud layer.
const CLOUD_COVER_THRESHOLD: f64 = 10.0;

/// Map weather observations to aerosol properties.
///
/// The key insight is that we use the *measured* AOD from the air quality
/// API, not the default AOD for the type. The type selection only determines
/// the other optical properties (SSA, asymmetry, Angstrom exponent, scale
/// height) which depend on aerosol composition.
///
/// Type selection logic:
/// - High dust (>20 ug/m3): Desert (coarse mineral particles, low Angstrom)
/// - AOD > 0.20 with low dust: Urban (fine anthropogenic particles, high Angstrom)
/// - AOD 0.08-0.20: ContinentalAverage
/// - AOD 0.04-0.08: ContinentalClean
/// - AOD < 0.04: no aerosol (pristine)
pub fn map_aerosol(conditions: &WeatherConditions) -> Option<AerosolProperties> {
    let aod = conditions.aod_550;

    if aod < AOD_THRESHOLD {
        return None;
    }

    // Select type based on composition indicators
    let base_type = if conditions.dust_ug_m3 > DUST_THRESHOLD {
        AerosolType::Desert
    } else if aod > 0.20 {
        AerosolType::Urban
    } else if aod > 0.08 {
        AerosolType::ContinentalAverage
    } else {
        AerosolType::ContinentalClean
    };

    // Get the type's default properties for SSA, asymmetry, etc.
    let defaults = aerosol::default_properties(base_type);

    // Override AOD with the measured value
    Some(AerosolProperties {
        aod_550: aod,
        ..defaults
    })
}

/// Map weather observations to cloud layer properties.
///
/// Uses the low/mid/high cloud cover breakdown from the weather API to
/// select the dominant cloud layer type and scale its optical depth by
/// the coverage fraction.
///
/// Priority: low clouds > mid clouds > high clouds (low clouds have the
/// largest radiative impact on twilight).
///
/// The optical depth is scaled by (cover_fraction)^0.7 to account for
/// the fact that partial cloud cover doesn't linearly reduce the mean
/// optical depth (cloud edges contribute, and horizontal transport of
/// light through gaps matters).
pub fn map_cloud(conditions: &WeatherConditions) -> Option<CloudProperties> {
    let low = conditions.cloud_cover_low.clamp(0.0, 100.0);
    let mid = conditions.cloud_cover_mid.clamp(0.0, 100.0);
    let high = conditions.cloud_cover_high.clamp(0.0, 100.0);

    // Check for fog (WMO codes 45, 48)
    let is_fog = conditions.weather_code == 45 || conditions.weather_code == 48;

    // Determine dominant cloud layer
    if is_fog || low >= CLOUD_COVER_THRESHOLD {
        // Low cloud or fog
        let cover = if is_fog {
            100.0_f64.min(low.max(80.0))
        } else {
            low
        };
        let cover_frac = cover / 100.0;

        let base_type = if is_fog || conditions.visibility_m < 1000.0 {
            // Fog or very low visibility: thick stratus
            CloudType::Stratus
        } else if low > 60.0 {
            CloudType::Stratocumulus
        } else {
            // Scattered low clouds
            CloudType::Cumulus
        };

        let defaults = cloud::default_properties(base_type);
        let scaled_od = defaults.optical_depth * cover_frac.powf(0.7);

        if scaled_od < 0.01 {
            return None;
        }

        return Some(CloudProperties {
            optical_depth: scaled_od,
            ..defaults
        });
    }

    if mid >= CLOUD_COVER_THRESHOLD {
        let cover_frac = (mid / 100.0).min(1.0);
        let defaults = cloud::default_properties(CloudType::Altostratus);
        let scaled_od = defaults.optical_depth * cover_frac.powf(0.7);

        if scaled_od < 0.01 {
            return None;
        }

        return Some(CloudProperties {
            optical_depth: scaled_od,
            ..defaults
        });
    }

    if high >= CLOUD_COVER_THRESHOLD {
        let cover_frac = (high / 100.0).min(1.0);

        let base_type = if high > 60.0 {
            CloudType::ThickCirrus
        } else {
            CloudType::ThinCirrus
        };

        let defaults = cloud::default_properties(base_type);
        let scaled_od = defaults.optical_depth * cover_frac.powf(0.7);

        if scaled_od < 0.01 {
            return None;
        }

        return Some(CloudProperties {
            optical_depth: scaled_od,
            ..defaults
        });
    }

    None
}

// ── Gas composition mapping ─────────────────────────────────────────────

/// NO2 molar mass (g/mol).
const NO2_MOLAR_MASS: f64 = 46.0;

/// Avogadro's number (molecules/mol).
const AVOGADRO: f64 = 6.022e23;

/// Empirical relationship between surface O3 concentration and total column.
///
/// The total column O3 is dominated by the stratospheric layer (peak at
/// 20-25 km), not surface O3. However, surface O3 correlates loosely with
/// the total column through large-scale atmospheric dynamics (tropopause
/// folding, stratosphere-troposphere exchange).
///
/// Typical surface O3: 20-80 ug/m3 (rural), 40-200 ug/m3 (urban episodes).
/// Typical total column: 220-450 DU (global range), ~300 DU (mid-latitude).
///
/// We use a conservative linear mapping with a floor of 250 DU and a
/// ceiling of 450 DU. The baseline is 300 DU at 60 ug/m3 surface O3.
fn estimate_o3_column_du(surface_o3_ug_m3: f64) -> f64 {
    // Baseline: 300 DU corresponds to ~60 ug/m3 surface O3
    // Sensitivity: ~0.5 DU per ug/m3 deviation (weak correlation)
    let baseline_du = 300.0;
    let baseline_surface = 60.0;
    let sensitivity = 0.5; // DU per ug/m3

    let du = baseline_du + (surface_o3_ug_m3 - baseline_surface) * sensitivity;

    // Clamp to physically reasonable range
    du.clamp(220.0, 450.0)
}

/// Convert surface NO2 in ug/m3 to number density in molecules/m3.
///
/// n [molecules/m3] = (concentration [ug/m3] * 1e-6 [g/ug]) / M [g/mol] * N_A [molecules/mol] * 1e6 [cm3/m3... wait]
///
/// Actually: n [molecules/m3] = (C [ug/m3] * 1e-6 [g/ug] * N_A [molecules/mol]) / (M [g/mol] * 1e-3 [kg/g] * 1e3 [L/m3] * 22.4 [L/mol at STP])
///
/// Convert NO2 surface concentration from ug/m3 to molecules/m3.
///
/// The concentration C [ug/m3] is already a mass per unit volume, so the
/// conversion is straightforward:
///   n [molecules/m3] = C [ug/m3] * 1e-6 [g/ug] / M [g/mol] * N_A [molecules/mol]
///   n = C * 1e-6 * 6.022e23 / 46.0
///   n = C * 1.309e16
///
/// At 40 ug/m3 (moderate urban): n ~ 5.2e17 molecules/m3
fn no2_ug_m3_to_molecules_m3(no2_ug_m3: f64) -> f64 {
    no2_ug_m3 * 1e-6 * AVOGADRO / NO2_MOLAR_MASS
}

/// Map weather observations to gas composition overrides.
///
/// Converts surface O3 and NO2 concentrations from the CAMS-based air
/// quality API into values usable by the MCRT gas absorption model:
///
/// - **O3**: Surface concentration is used to estimate total column O3
///   in Dobson Units via an empirical relationship. This adjusts the
///   standard ~347 DU column to actual conditions (ozone holes, seasonal
///   variation, latitude dependence).
///
/// - **NO2**: Surface concentration is converted to number density
///   (molecules/m^3) to scale the tropospheric NO2 profile. This matters
///   for Huggins/Chappuis band absorption, especially in polluted urban areas.
///
/// Returns `None` if both O3 and NO2 are zero or missing (no data from API).
pub fn map_gas_composition(conditions: &WeatherConditions) -> Option<GasComposition> {
    let has_o3 = conditions.ozone_ug_m3 > 0.0;
    let has_no2 = conditions.nitrogen_dioxide_ug_m3 > 0.0;

    if !has_o3 && !has_no2 {
        return None;
    }

    let o3_column_du = if has_o3 {
        Some(estimate_o3_column_du(conditions.ozone_ug_m3))
    } else {
        None
    };

    let no2_surface_density = if has_no2 {
        Some(no2_ug_m3_to_molecules_m3(conditions.nitrogen_dioxide_ug_m3))
    } else {
        None
    };

    Some(GasComposition {
        o3_column_du,
        no2_surface_density,
    })
}

/// Generate a human-readable description of the atmospheric conditions.
pub fn describe(
    conditions: &WeatherConditions,
    aerosol: &Option<AerosolProperties>,
    cloud: &Option<CloudProperties>,
    gas: &Option<GasComposition>,
) -> String {
    let mut parts = Vec::new();

    // Aerosol description
    match aerosol {
        None => parts.push("Clear sky (pristine, AOD < 0.04)".to_string()),
        Some(props) => {
            let type_name = if conditions.dust_ug_m3 > DUST_THRESHOLD {
                "desert dust"
            } else if props.aod_550 > 0.20 {
                "urban haze"
            } else if props.aod_550 > 0.08 {
                "continental haze"
            } else {
                "light haze"
            };
            parts.push(format!("AOD {:.2} ({})", props.aod_550, type_name));
        }
    }

    // Cloud description
    match cloud {
        None => {} // don't mention absence of clouds if already clear
        Some(props) => {
            let cloud_name = if props.base_km >= 7.0 {
                if props.optical_depth > 1.0 {
                    "thick cirrus"
                } else {
                    "thin cirrus"
                }
            } else if props.base_km >= 2.5 {
                "altostratus"
            } else if props.base_km < 0.6 {
                "stratus/fog"
            } else if props.optical_depth > 10.0 {
                "stratocumulus"
            } else {
                "low cloud"
            };
            parts.push(format!(
                "Cloud: {} (OD {:.1}, {:.0}-{:.0}km)",
                cloud_name, props.optical_depth, props.base_km, props.top_km
            ));
        }
    }

    // Gas composition
    if let Some(gc) = gas {
        let mut gas_parts = Vec::new();
        if let Some(du) = gc.o3_column_du {
            gas_parts.push(format!("O3 {:.0} DU", du));
        }
        if gc.no2_surface_density.is_some() {
            gas_parts.push(format!(
                "NO2 {:.0} ug/m3",
                conditions.nitrogen_dioxide_ug_m3
            ));
        }
        if !gas_parts.is_empty() {
            parts.push(format!("Gas: {}", gas_parts.join(", ")));
        }
    }

    // Visibility
    if conditions.visibility_m < 10000.0 {
        parts.push(format!(
            "Visibility {:.1}km",
            conditions.visibility_m / 1000.0
        ));
    }

    parts.join(". ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WeatherConditions;

    fn base_conditions() -> WeatherConditions {
        WeatherConditions {
            aod_550: 0.10,
            dust_ug_m3: 0.0,
            pm2_5_ug_m3: 10.0,
            pm10_ug_m3: 15.0,
            ozone_ug_m3: 50.0,
            nitrogen_dioxide_ug_m3: 10.0,
            cloud_cover_total: 0.0,
            cloud_cover_low: 0.0,
            cloud_cover_mid: 0.0,
            cloud_cover_high: 0.0,
            visibility_m: 20000.0,
            relative_humidity: 50.0,
            weather_code: 0,
            timestamp: String::new(),
            api_latitude: 50.0,
            api_longitude: 10.0,
        }
    }

    // ── Aerosol type selection ──

    #[test]
    fn aod_below_threshold_no_aerosol() {
        let mut c = base_conditions();
        c.aod_550 = 0.02;
        assert!(map_aerosol(&c).is_none());
    }

    #[test]
    fn aod_at_threshold_produces_aerosol() {
        let mut c = base_conditions();
        c.aod_550 = 0.05;
        assert!(map_aerosol(&c).is_some());
    }

    #[test]
    fn continental_clean_range() {
        let mut c = base_conditions();
        c.aod_550 = 0.06;
        c.dust_ug_m3 = 0.0;
        let props = map_aerosol(&c).unwrap();
        // Continental clean has angstrom ~1.3
        assert!((props.angstrom_exponent - 1.3).abs() < 0.01);
        assert!((props.aod_550 - 0.06).abs() < 0.001);
    }

    #[test]
    fn continental_average_range() {
        let mut c = base_conditions();
        c.aod_550 = 0.15;
        c.dust_ug_m3 = 0.0;
        let props = map_aerosol(&c).unwrap();
        assert!((props.angstrom_exponent - 1.3).abs() < 0.01);
        assert!((props.aod_550 - 0.15).abs() < 0.001);
    }

    #[test]
    fn urban_range() {
        let mut c = base_conditions();
        c.aod_550 = 0.30;
        c.dust_ug_m3 = 0.0;
        let props = map_aerosol(&c).unwrap();
        // Urban has angstrom ~1.5
        assert!((props.angstrom_exponent - 1.5).abs() < 0.01);
    }

    #[test]
    fn desert_when_dusty() {
        let mut c = base_conditions();
        c.aod_550 = 0.40;
        c.dust_ug_m3 = 50.0;
        let props = map_aerosol(&c).unwrap();
        // Desert has angstrom ~0.3
        assert!((props.angstrom_exponent - 0.3).abs() < 0.01);
    }

    #[test]
    fn measured_aod_always_used() {
        let mut c = base_conditions();
        c.aod_550 = 0.42;
        c.dust_ug_m3 = 0.0;
        let props = map_aerosol(&c).unwrap();
        assert!((props.aod_550 - 0.42).abs() < 0.001);
    }

    // ── Cloud type selection ──

    #[test]
    fn no_cloud_below_threshold() {
        let mut c = base_conditions();
        c.cloud_cover_total = 5.0;
        c.cloud_cover_low = 5.0;
        assert!(map_cloud(&c).is_none());
    }

    #[test]
    fn high_cloud_thin_cirrus() {
        let mut c = base_conditions();
        c.cloud_cover_high = 30.0;
        let cloud = map_cloud(&c).unwrap();
        assert!(
            cloud.base_km >= 7.0,
            "Should be cirrus, base={}km",
            cloud.base_km
        );
        assert!(cloud.asymmetry < 0.80, "Cirrus should have lower asymmetry");
    }

    #[test]
    fn high_cloud_thick_cirrus() {
        let mut c = base_conditions();
        c.cloud_cover_high = 80.0;
        let cloud = map_cloud(&c).unwrap();
        assert!(cloud.base_km >= 7.0);
        // Thick cirrus has higher OD than thin
        assert!(
            cloud.optical_depth > 0.5,
            "Thick cirrus should have OD > 0.5"
        );
    }

    #[test]
    fn mid_cloud_altostratus() {
        let mut c = base_conditions();
        c.cloud_cover_mid = 60.0;
        let cloud = map_cloud(&c).unwrap();
        assert!(
            cloud.base_km >= 2.5 && cloud.base_km <= 6.0,
            "Altostratus base should be 3-5km, got {}km",
            cloud.base_km
        );
    }

    #[test]
    fn low_cloud_scattered() {
        let mut c = base_conditions();
        c.cloud_cover_low = 30.0;
        let cloud = map_cloud(&c).unwrap();
        assert!(cloud.base_km < 3.0, "Low cloud base should be <3km");
    }

    #[test]
    fn low_cloud_overcast_stratocumulus() {
        let mut c = base_conditions();
        c.cloud_cover_low = 80.0;
        let cloud = map_cloud(&c).unwrap();
        assert!(cloud.base_km < 3.0);
        assert!(
            cloud.optical_depth > 2.0,
            "Overcast should have substantial OD"
        );
    }

    #[test]
    fn fog_gives_stratus() {
        let mut c = base_conditions();
        c.weather_code = 45;
        c.cloud_cover_low = 100.0;
        let cloud = map_cloud(&c).unwrap();
        assert!(
            cloud.base_km < 1.0,
            "Fog/stratus base should be <1km, got {}km",
            cloud.base_km
        );
    }

    #[test]
    fn low_cloud_priority_over_high() {
        // When both low and high clouds are present, low should dominate
        let mut c = base_conditions();
        c.cloud_cover_low = 70.0;
        c.cloud_cover_high = 50.0;
        let cloud = map_cloud(&c).unwrap();
        assert!(cloud.base_km < 3.0, "Low cloud should take priority");
    }

    #[test]
    fn od_scales_with_coverage_fraction() {
        let mut c1 = base_conditions();
        c1.cloud_cover_high = 20.0;
        let mut c2 = base_conditions();
        c2.cloud_cover_high = 90.0;

        let cloud1 = map_cloud(&c1).unwrap();
        let cloud2 = map_cloud(&c2).unwrap();

        assert!(
            cloud2.optical_depth > cloud1.optical_depth,
            "90% cover should give more OD than 20%: {:.2} vs {:.2}",
            cloud2.optical_depth,
            cloud1.optical_depth
        );
    }

    // ── Description ──

    #[test]
    fn describe_pristine() {
        let mut c = base_conditions();
        c.aod_550 = 0.02;
        let a = map_aerosol(&c);
        let cl = map_cloud(&c);
        let g = map_gas_composition(&c);
        let desc = describe(&c, &a, &cl, &g);
        assert!(desc.contains("pristine") || desc.contains("Clear"));
    }

    #[test]
    fn describe_hazy_with_cloud() {
        let mut c = base_conditions();
        c.aod_550 = 0.25;
        c.cloud_cover_high = 40.0;
        let a = map_aerosol(&c);
        let cl = map_cloud(&c);
        let g = map_gas_composition(&c);
        let desc = describe(&c, &a, &cl, &g);
        assert!(desc.contains("0.25"));
        assert!(desc.contains("cirrus") || desc.contains("Cloud"));
    }

    #[test]
    fn describe_low_visibility() {
        let mut c = base_conditions();
        c.visibility_m = 5000.0;
        let a = map_aerosol(&c);
        let cl = map_cloud(&c);
        let g = map_gas_composition(&c);
        let desc = describe(&c, &a, &cl, &g);
        assert!(desc.contains("5.0km") || desc.contains("Visibility"));
    }

    // ── Gas composition mapping ──

    #[test]
    fn gas_composition_from_typical_conditions() {
        let c = base_conditions(); // O3=50, NO2=10
        let gc = map_gas_composition(&c).expect("Should produce gas composition");
        assert!(gc.o3_column_du.is_some());
        let du = gc.o3_column_du.unwrap();
        assert!(
            du >= 220.0 && du <= 450.0,
            "O3 column should be in range: {} DU",
            du
        );
    }

    #[test]
    fn gas_composition_none_when_zero() {
        let mut c = base_conditions();
        c.ozone_ug_m3 = 0.0;
        c.nitrogen_dioxide_ug_m3 = 0.0;
        assert!(map_gas_composition(&c).is_none());
    }

    #[test]
    fn gas_composition_o3_column_scales_with_surface() {
        let mut c1 = base_conditions();
        c1.ozone_ug_m3 = 30.0;
        let mut c2 = base_conditions();
        c2.ozone_ug_m3 = 100.0;
        let gc1 = map_gas_composition(&c1).unwrap();
        let gc2 = map_gas_composition(&c2).unwrap();
        assert!(
            gc2.o3_column_du.unwrap() > gc1.o3_column_du.unwrap(),
            "Higher surface O3 should give higher column estimate"
        );
    }

    #[test]
    fn gas_composition_o3_column_clamped() {
        let mut c = base_conditions();
        c.ozone_ug_m3 = 500.0; // extremely high
        let gc = map_gas_composition(&c).unwrap();
        assert!(
            gc.o3_column_du.unwrap() <= 450.0,
            "Should be clamped to 450 DU"
        );

        c.ozone_ug_m3 = 1.0; // extremely low
        let gc = map_gas_composition(&c).unwrap();
        assert!(
            gc.o3_column_du.unwrap() >= 220.0,
            "Should be clamped to 220 DU"
        );
    }

    #[test]
    fn gas_composition_no2_conversion_reasonable() {
        let mut c = base_conditions();
        c.nitrogen_dioxide_ug_m3 = 40.0; // moderate urban
        let gc = map_gas_composition(&c).unwrap();
        let n = gc.no2_surface_density.unwrap();
        // At 40 ug/m3 NO2: n = 40e-6 * 6.022e23 / 46.0 ~ 5.2e17 molecules/m3
        assert!(
            n > 1e17 && n < 1e19,
            "NO2 density should be ~5e17, got {:.2e}",
            n
        );
    }

    #[test]
    fn gas_composition_o3_only() {
        let mut c = base_conditions();
        c.ozone_ug_m3 = 60.0;
        c.nitrogen_dioxide_ug_m3 = 0.0;
        let gc = map_gas_composition(&c).unwrap();
        assert!(gc.o3_column_du.is_some());
        assert!(gc.no2_surface_density.is_none());
    }

    #[test]
    fn gas_composition_no2_only() {
        let mut c = base_conditions();
        c.ozone_ug_m3 = 0.0;
        c.nitrogen_dioxide_ug_m3 = 20.0;
        let gc = map_gas_composition(&c).unwrap();
        assert!(gc.o3_column_du.is_none());
        assert!(gc.no2_surface_density.is_some());
    }
}
