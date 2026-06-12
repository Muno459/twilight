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
//! Gas composition mapping converts surface NO2 concentration from the
//! CAMS-based air quality API into a boundary-layer number-density override.
//! Surface O3 produces no override (it does not determine the column).

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

    // Select type based on composition indicators.
    //
    // Maritime detection heuristic (no land/sea mask in the API): clean
    // marine air has low dust, low fine-particle load relative to AOD
    // (sea salt is coarse: PM2.5/PM10 ratio low), and high humidity. The
    // sea-salt Angstrom exponent (~0.3) vs continental (~1.3) materially
    // changes the blue/red extinction split at twilight, so reaching the
    // maritime types matters; previously they were unreachable from
    // weather data.
    let pm_ratio = if conditions.pm10_ug_m3 > 1.0 {
        conditions.pm2_5_ug_m3 / conditions.pm10_ug_m3
    } else {
        1.0
    };
    let maritime_like = conditions.dust_ug_m3 < 5.0
        && conditions.relative_humidity > 70.0
        && pm_ratio < 0.55;
    let base_type = if conditions.dust_ug_m3 > DUST_THRESHOLD {
        AerosolType::Desert
    } else if maritime_like && aod <= 0.15 {
        AerosolType::MaritimeClean
    } else if maritime_like {
        AerosolType::MaritimePolluted
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
/// Uses ALL THREE altitude bands (low/mid/high) from the API, combined via
/// the independent-column approximation: each band transmits
///
///   T_band = (1 - f) + f * T_diff(OD_band)
///
/// where f is the band's cover fraction and T_diff is the Eddington diffuse
/// transmission of the band's type OD (matching the engine's cloud
/// transport). The bands' transmissions multiply (independent overlap), and
/// the combined transmission is inverted back into one EFFECTIVE optical
/// depth. This replaces two earlier defects: a priority early-return where
/// 10% scattered low cloud completely masked 90% high overcast, and an
/// uncited OD*f^0.7 partial-cover fudge.
///
/// The effective layer takes the geometry (base/top) and droplet asymmetry
/// of the band with the largest transmission DEFICIT (the radiatively
/// dominant band).
pub fn map_cloud(conditions: &WeatherConditions) -> Option<CloudProperties> {
    let is_fog = conditions.weather_code == 45 || conditions.weather_code == 48;

    // (cover %, type) per band — low band type refined by cover/visibility.
    let low_cover = if is_fog {
        100.0_f64.min(conditions.cloud_cover_low.max(80.0))
    } else {
        conditions.cloud_cover_low.clamp(0.0, 100.0)
    };
    let low_type = if is_fog || conditions.visibility_m < 1000.0 {
        CloudType::Stratus
    } else if low_cover > 60.0 {
        CloudType::Stratocumulus
    } else {
        CloudType::Cumulus
    };
    let mid_cover = conditions.cloud_cover_mid.clamp(0.0, 100.0);
    let high_cover = conditions.cloud_cover_high.clamp(0.0, 100.0);
    let high_type = if high_cover > 60.0 {
        CloudType::ThickCirrus
    } else {
        CloudType::ThinCirrus
    };

    // Eddington diffuse transmission of a delta-scaled type OD.
    let t_diff = |ctype: CloudType| -> (f64, CloudProperties) {
        let p = cloud::default_properties(ctype);
        let f_peak = p.asymmetry * p.asymmetry;
        let de_scale = 1.0 - p.ssa * f_peak;
        let g_scaled = p.asymmetry / (1.0 + p.asymmetry);
        let tau_star = p.optical_depth * de_scale;
        (1.0 / (1.0 + 0.75 * tau_star * (1.0 - g_scaled)), p)
    };

    let mut t_total = 1.0_f64;
    let mut dominant: Option<(f64, CloudProperties)> = None; // (deficit, props)
    for (cover, ctype) in [
        (low_cover, low_type),
        (mid_cover, CloudType::Altostratus),
        (high_cover, high_type),
    ] {
        if cover < CLOUD_COVER_THRESHOLD {
            continue;
        }
        let f = (cover / 100.0).clamp(0.0, 1.0);
        let (t_band_cloudy, props) = t_diff(ctype);
        let t_band = (1.0 - f) + f * t_band_cloudy;
        t_total *= t_band;
        let deficit = 1.0 - t_band;
        if dominant.as_ref().map(|(d, _)| deficit > *d).unwrap_or(true) {
            dominant = Some((deficit, props));
        }
    }

    let (_, dom) = dominant?;
    if t_total >= 0.995 {
        return None; // optically negligible
    }

    // Invert the combined diffuse transmission back to one effective
    // UNSCALED optical depth for the dominant band's droplet properties
    // (the builder re-applies delta scaling).
    let f_peak = dom.asymmetry * dom.asymmetry;
    let de_scale = 1.0 - dom.ssa * f_peak;
    let g_scaled = dom.asymmetry / (1.0 + dom.asymmetry);
    let tau_star_eff = (1.0 / t_total - 1.0) / (0.75 * (1.0 - g_scaled));
    let od_eff = tau_star_eff / de_scale;

    Some(CloudProperties {
        optical_depth: od_eff,
        ..dom
    })
}


/// Build cloud properties from a SATELLITE sample (GIBS MODIS COT + CTH),
/// blended with the sunward-path samples ("2.5D"): at twilight the shadow
/// path crosses the cloud field 50-300 km toward the sun, so the
/// radiatively relevant optical depth is a blend of the overhead and
/// sunward columns.
///
/// Layer placement uses the measured cloud-top height; geometric thickness
/// is estimated from COT (thicker optically -> deeper layer, clamped to
/// 200-3000 m); droplet properties are chosen by the measured top height
/// (ice cirrus above ~6 km, mixed alto 2.5-6 km, water stratiform below).
///
/// Returns None when the satellite saw clear sky everywhere (callers may
/// still fall back to the model/forecast cloud).
pub fn map_cloud_satellite(
    sat: &crate::satellite::SatelliteCloudPath,
) -> Option<CloudProperties> {
    let obs_cot = sat.observer.map(|s| s.cot).unwrap_or(0.0);
    // Sunward weighting: the path fraction scales how much of the sunward
    // mean enters; with no sunward cloud the observer column stands alone.
    let eff_cot = if sat.n_path_samples > 0 {
        let w_path = 0.5 * sat.path_cloud_fraction;
        obs_cot * (1.0 - w_path) + sat.path_mean_cot * w_path
    } else {
        obs_cot
    };
    if eff_cot < 0.3 {
        return None; // optically negligible
    }

    let top_m = sat
        .observer
        .and_then(|s| s.cloud_top_m)
        .unwrap_or(2500.0);
    let thickness_m = (eff_cot * 80.0).clamp(200.0, 3000.0);
    let top_km = (top_m / 1000.0).clamp(0.4, 14.0);
    let base_km = (top_km - thickness_m / 1000.0).max(0.15);

    // Droplet properties by measured top height (phase proxy).
    let (ssa, g) = if top_km > 6.0 {
        (0.9995, 0.77) // ice cirrus-like
    } else if top_km > 2.5 {
        (0.999, 0.82) // mixed/alto
    } else {
        (0.999, 0.85) // warm stratiform
    };

    Some(CloudProperties {
        base_km,
        top_km,
        optical_depth: eff_cot,
        ssa,
        asymmetry: g,
    })
}

// ── Gas composition mapping ─────────────────────────────────────────────

/// NO2 molar mass (g/mol).
const NO2_MOLAR_MASS: f64 = 46.0;

/// Avogadro's number (molecules/mol).
const AVOGADRO: f64 = 6.022e23;

/// Surface O3 (a boundary-layer photochemical quantity) does NOT determine
/// the total O3 column, which is dominated by the stratospheric reservoir
/// and governed by latitude/season/dynamics. The previous code invented a
/// linear surface-to-column proxy; it has been removed. Open-Meteo's air
/// quality API provides surface O3 only, so no column override is produced
/// — the engine keeps its standard-atmosphere column (345 DU) unless a
/// real measured column is supplied by the caller.

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
/// - **O3**: surface concentration is reported for display but produces
///   NO column override — a surface reading does not determine the column
///   (see note above). The engine keeps its standard 345 DU column.
///
/// - **NO2**: Surface concentration is converted to number density
///   (molecules/m^3) to scale the tropospheric NO2 profile. This matters
///   for NO2's visible absorption band (~400-500 nm), especially in
///   polluted urban areas.
///
/// Returns `None` if both O3 and NO2 are zero or missing (no data from API).
pub fn map_gas_composition(conditions: &WeatherConditions) -> Option<GasComposition> {
    let has_o3 = conditions.ozone_ug_m3 > 0.0;
    let has_no2 = conditions.nitrogen_dioxide_ug_m3 > 0.0;

    let _ = has_o3; // surface O3 is reported but produces no override
    if !has_no2 {
        return None;
    }

    // Surface O3 cannot be converted to a column (see note above):
    // no override. Surface NO2 IS a usable boundary-layer override.
    let o3_column_du: Option<f64> = None;

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

    // ── Satellite cloud mapping ──

    #[test]
    fn satellite_cloud_places_layer_at_measured_height() {
        use crate::satellite::{SatelliteCloud, SatelliteCloudPath};
        let sat = SatelliteCloudPath {
            observer: Some(SatelliteCloud {
                cot: 12.0,
                cloud_top_m: Some(1800.0),
                age_days: 0,
            }),
            path_mean_cot: 10.0,
            path_cloud_fraction: 1.0,
            n_path_samples: 4,
        };
        let c = map_cloud_satellite(&sat).unwrap();
        assert!((c.top_km - 1.8).abs() < 1e-9, "top at satellite CTH");
        assert!(c.base_km < c.top_km && c.base_km > 0.0);
        // blended COT between observer and path
        assert!(c.optical_depth > 10.0 && c.optical_depth < 12.5);
        assert!((c.asymmetry - 0.85).abs() < 1e-9, "warm stratiform g");
    }

    #[test]
    fn satellite_high_cloud_gets_ice_properties() {
        use crate::satellite::{SatelliteCloud, SatelliteCloudPath};
        let sat = SatelliteCloudPath {
            observer: Some(SatelliteCloud {
                cot: 2.0,
                cloud_top_m: Some(9500.0),
                age_days: 0,
            }),
            path_mean_cot: 0.0,
            path_cloud_fraction: 0.0,
            n_path_samples: 4,
        };
        let c = map_cloud_satellite(&sat).unwrap();
        assert!((c.asymmetry - 0.77).abs() < 1e-9, "cirrus-like g for high top");
        assert!(c.top_km > 9.0);
    }

    #[test]
    fn satellite_clear_sky_yields_none() {
        use crate::satellite::SatelliteCloudPath;
        let sat = SatelliteCloudPath {
            observer: None,
            path_mean_cot: 0.0,
            path_cloud_fraction: 0.0,
            n_path_samples: 4,
        };
        assert!(map_cloud_satellite(&sat).is_none());
    }

    #[test]
    fn maritime_types_reachable_from_marine_conditions() {
        // Coastal/marine signature: humid, coarse-dominated, low dust.
        let mut c = base_conditions();
        c.dust_ug_m3 = 1.0;
        c.relative_humidity = 85.0;
        c.pm2_5_ug_m3 = 4.0;
        c.pm10_ug_m3 = 18.0; // coarse sea salt
        c.aod_550 = 0.10;
        let a = map_aerosol(&c).expect("aerosol expected");
        assert!(
            a.angstrom_exponent < 0.8,
            "marine air should select a maritime (low-Angstrom) type, got alpha={}",
            a.angstrom_exponent
        );
    }
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
        // Surface O3 must NOT be converted into a column override
        // (a surface reading does not determine the stratospheric column).
        assert!(gc.o3_column_du.is_none());
        assert!(gc.no2_surface_density.is_some());
    }

    #[test]
    fn gas_composition_none_when_zero() {
        let mut c = base_conditions();
        c.ozone_ug_m3 = 0.0;
        c.nitrogen_dioxide_ug_m3 = 0.0;
        assert!(map_gas_composition(&c).is_none());
    }

    #[test]
    fn gas_composition_never_invents_o3_column() {
        // Any surface O3 value — tiny or extreme — must produce no column
        // override. The old code mapped these through an invented linear
        // proxy clamped to [220, 450] DU.
        for o3 in [1.0, 30.0, 100.0, 500.0] {
            let mut c = base_conditions();
            c.ozone_ug_m3 = o3;
            let gc = map_gas_composition(&c).unwrap();
            assert!(
                gc.o3_column_du.is_none(),
                "surface O3 {} ug/m3 must not become a column",
                o3
            );
        }
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
        // O3-only input: no overrides at all -> None (nothing to apply).
        let mut c = base_conditions();
        c.ozone_ug_m3 = 60.0;
        c.nitrogen_dioxide_ug_m3 = 0.0;
        assert!(map_gas_composition(&c).is_none());
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
