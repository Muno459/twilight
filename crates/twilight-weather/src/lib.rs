//! Real-time weather data for twilight MCRT simulations.
//!
//! Fetches current atmospheric conditions from the Open-Meteo API (no key
//! required) and maps them to the aerosol and cloud models used by the
//! twilight radiative transfer engine.
//!
//! Two API endpoints are used:
//! - Weather Forecast API: cloud cover (low/mid/high), visibility, humidity
//! - Air Quality API: aerosol optical depth at 550nm, dust concentration
//!
//! The mapping from weather observations to MCRT parameters is physically
//! motivated but approximate. Real aerosol composition and cloud micro-
//! physics are more complex than what a weather API can provide.

pub mod api;
pub mod mapping;

use twilight_data::aerosol::AerosolProperties;
use twilight_data::cloud::CloudProperties;

/// Current weather conditions relevant to twilight radiative transfer.
#[derive(Debug, Clone)]
pub struct WeatherConditions {
    /// Aerosol optical depth at 550nm (dimensionless)
    pub aod_550: f64,
    /// Dust concentration at surface (ug/m3)
    pub dust_ug_m3: f64,
    /// PM2.5 concentration at surface (ug/m3)
    pub pm2_5_ug_m3: f64,
    /// PM10 concentration at surface (ug/m3)
    pub pm10_ug_m3: f64,
    /// Total cloud cover (%)
    pub cloud_cover_total: f64,
    /// Low cloud cover, below 3km (%)
    pub cloud_cover_low: f64,
    /// Mid cloud cover, 3-8km (%)
    pub cloud_cover_mid: f64,
    /// High cloud cover, above 8km (%)
    pub cloud_cover_high: f64,
    /// Visibility in meters
    pub visibility_m: f64,
    /// Relative humidity at 2m (%)
    pub relative_humidity: f64,
    /// WMO weather code
    pub weather_code: i32,
    /// Timestamp of the observation (ISO8601)
    pub timestamp: String,
    /// Latitude actually used by the API (grid cell center)
    pub api_latitude: f64,
    /// Longitude actually used by the API (grid cell center)
    pub api_longitude: f64,
}

/// Atmospheric parameters derived from weather observations, ready to
/// pass to `twilight_data::builder::build_full()`.
#[derive(Debug, Clone)]
pub struct AtmosphericParams {
    /// Aerosol properties (None = clear sky, no aerosol)
    pub aerosol: Option<AerosolProperties>,
    /// Cloud properties (None = clear sky, no cloud)
    pub cloud: Option<CloudProperties>,
    /// Human-readable summary of what was detected
    pub description: String,
    /// The raw weather conditions used to derive these params
    pub conditions: WeatherConditions,
}

/// Fetch current weather conditions and derive MCRT parameters.
///
/// This is the main entry point. It makes two HTTP requests to Open-Meteo
/// (weather + air quality), maps the observations to aerosol and cloud
/// properties, and returns everything ready for the MCRT pipeline.
///
/// # Errors
/// Returns an error string if the API requests fail.
pub fn fetch_atmospheric_params(lat: f64, lon: f64) -> Result<AtmosphericParams, String> {
    let conditions = api::fetch_weather(lat, lon)?;
    let aerosol = mapping::map_aerosol(&conditions);
    let cloud = mapping::map_cloud(&conditions);
    let description = mapping::describe(&conditions, &aerosol, &cloud);

    Ok(AtmosphericParams {
        aerosol,
        cloud,
        description,
        conditions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_clear_conditions() -> WeatherConditions {
        WeatherConditions {
            aod_550: 0.03,
            dust_ug_m3: 0.0,
            pm2_5_ug_m3: 3.0,
            pm10_ug_m3: 5.0,
            cloud_cover_total: 0.0,
            cloud_cover_low: 0.0,
            cloud_cover_mid: 0.0,
            cloud_cover_high: 0.0,
            visibility_m: 30000.0,
            relative_humidity: 40.0,
            weather_code: 0,
            timestamp: "2026-03-05T12:00".to_string(),
            api_latitude: 54.82,
            api_longitude: 9.36,
        }
    }

    fn make_hazy_conditions() -> WeatherConditions {
        WeatherConditions {
            aod_550: 0.25,
            dust_ug_m3: 2.0,
            pm2_5_ug_m3: 25.0,
            pm10_ug_m3: 40.0,
            cloud_cover_total: 20.0,
            cloud_cover_low: 0.0,
            cloud_cover_mid: 0.0,
            cloud_cover_high: 20.0,
            visibility_m: 8000.0,
            relative_humidity: 70.0,
            weather_code: 2,
            timestamp: "2026-03-05T12:00".to_string(),
            api_latitude: 54.82,
            api_longitude: 9.36,
        }
    }

    fn make_overcast_conditions() -> WeatherConditions {
        WeatherConditions {
            aod_550: 0.10,
            dust_ug_m3: 0.0,
            pm2_5_ug_m3: 8.0,
            pm10_ug_m3: 12.0,
            cloud_cover_total: 100.0,
            cloud_cover_low: 90.0,
            cloud_cover_mid: 30.0,
            cloud_cover_high: 10.0,
            visibility_m: 5000.0,
            relative_humidity: 95.0,
            weather_code: 3,
            timestamp: "2026-03-05T12:00".to_string(),
            api_latitude: 54.82,
            api_longitude: 9.36,
        }
    }

    fn make_dusty_conditions() -> WeatherConditions {
        WeatherConditions {
            aod_550: 0.60,
            dust_ug_m3: 80.0,
            pm2_5_ug_m3: 15.0,
            pm10_ug_m3: 120.0,
            cloud_cover_total: 0.0,
            cloud_cover_low: 0.0,
            cloud_cover_mid: 0.0,
            cloud_cover_high: 0.0,
            visibility_m: 4000.0,
            relative_humidity: 25.0,
            weather_code: 0,
            timestamp: "2026-03-05T12:00".to_string(),
            api_latitude: 21.42,
            api_longitude: 39.83,
        }
    }

    // ── Aerosol mapping tests ──

    #[test]
    fn clear_sky_no_aerosol() {
        let c = make_clear_conditions();
        let aerosol = mapping::map_aerosol(&c);
        assert!(aerosol.is_none(), "AOD 0.03 should produce no aerosol");
    }

    #[test]
    fn moderate_aod_produces_aerosol() {
        let c = make_hazy_conditions();
        let aerosol = mapping::map_aerosol(&c);
        assert!(aerosol.is_some(), "AOD 0.25 should produce aerosol");
        let props = aerosol.unwrap();
        assert!(
            (props.aod_550 - 0.25).abs() < 0.01,
            "Should use measured AOD, got {}",
            props.aod_550
        );
    }

    #[test]
    fn high_dust_gives_desert_type() {
        let c = make_dusty_conditions();
        let aerosol = mapping::map_aerosol(&c);
        assert!(aerosol.is_some());
        let props = aerosol.unwrap();
        // Desert type has low angstrom (large particles)
        assert!(
            props.angstrom_exponent < 0.5,
            "High dust should give desert-type aerosol (low angstrom), got {}",
            props.angstrom_exponent
        );
        assert!(
            (props.aod_550 - 0.60).abs() < 0.01,
            "Should use measured AOD"
        );
    }

    #[test]
    fn urban_haze_high_angstrom() {
        let mut c = make_hazy_conditions();
        c.dust_ug_m3 = 0.0; // no dust, pure urban haze
        c.aod_550 = 0.35;
        let aerosol = mapping::map_aerosol(&c);
        assert!(aerosol.is_some());
        let props = aerosol.unwrap();
        // Urban type has high angstrom (fine particles)
        assert!(
            props.angstrom_exponent > 1.0,
            "Urban haze should have high angstrom, got {}",
            props.angstrom_exponent
        );
    }

    #[test]
    fn aerosol_aod_uses_measured_value() {
        // Whatever type is selected, the AOD should always be the measured value
        for aod in &[0.06, 0.15, 0.30, 0.50] {
            let mut c = make_hazy_conditions();
            c.aod_550 = *aod;
            c.dust_ug_m3 = 0.0;
            let aerosol = mapping::map_aerosol(&c);
            if let Some(props) = aerosol {
                assert!(
                    (props.aod_550 - aod).abs() < 0.01,
                    "AOD should be {}, got {}",
                    aod,
                    props.aod_550
                );
            }
        }
    }

    // ── Cloud mapping tests ──

    #[test]
    fn no_cloud_when_clear() {
        let c = make_clear_conditions();
        let cloud = mapping::map_cloud(&c);
        assert!(cloud.is_none(), "0% cloud cover should produce no cloud");
    }

    #[test]
    fn high_cloud_gives_cirrus() {
        let mut c = make_clear_conditions();
        c.cloud_cover_high = 50.0;
        c.cloud_cover_total = 50.0;
        let cloud = mapping::map_cloud(&c);
        assert!(cloud.is_some(), "50% high cloud should produce cirrus");
        let props = cloud.unwrap();
        assert!(
            props.base_km >= 7.0,
            "Cirrus should be high altitude, got base {}km",
            props.base_km
        );
    }

    #[test]
    fn low_cloud_gives_low_type() {
        let c = make_overcast_conditions();
        let cloud = mapping::map_cloud(&c);
        assert!(cloud.is_some(), "90% low cloud should produce cloud");
        let props = cloud.unwrap();
        assert!(
            props.base_km < 3.0,
            "Low overcast should give low cloud, got base {}km",
            props.base_km
        );
    }

    #[test]
    fn cloud_od_scales_with_coverage() {
        // Higher cloud cover should give higher optical depth
        let mut c1 = make_clear_conditions();
        c1.cloud_cover_high = 30.0;
        c1.cloud_cover_total = 30.0;
        let mut c2 = make_clear_conditions();
        c2.cloud_cover_high = 80.0;
        c2.cloud_cover_total = 80.0;

        let cloud1 = mapping::map_cloud(&c1);
        let cloud2 = mapping::map_cloud(&c2);

        if let (Some(p1), Some(p2)) = (cloud1, cloud2) {
            assert!(
                p2.optical_depth >= p1.optical_depth,
                "More cloud cover should give more OD: {:.1} vs {:.1}",
                p2.optical_depth,
                p1.optical_depth
            );
        }
    }

    #[test]
    fn fog_gives_stratus() {
        let mut c = make_clear_conditions();
        c.weather_code = 45; // fog
        c.cloud_cover_low = 100.0;
        c.cloud_cover_total = 100.0;
        c.visibility_m = 500.0;
        let cloud = mapping::map_cloud(&c);
        assert!(cloud.is_some());
        let props = cloud.unwrap();
        assert!(
            props.base_km < 1.0,
            "Fog should give very low cloud, got base {}km",
            props.base_km
        );
    }

    // ── Description tests ──

    #[test]
    fn description_clear_sky() {
        let c = make_clear_conditions();
        let aerosol = mapping::map_aerosol(&c);
        let cloud = mapping::map_cloud(&c);
        let desc = mapping::describe(&c, &aerosol, &cloud);
        assert!(
            desc.contains("clear") || desc.contains("Clean") || desc.contains("pristine"),
            "Clear sky description should mention clear: {}",
            desc
        );
    }

    #[test]
    fn description_includes_aod() {
        let c = make_hazy_conditions();
        let aerosol = mapping::map_aerosol(&c);
        let cloud = mapping::map_cloud(&c);
        let desc = mapping::describe(&c, &aerosol, &cloud);
        assert!(
            desc.contains("0.25") || desc.contains("AOD"),
            "Hazy description should mention AOD: {}",
            desc
        );
    }

    #[test]
    fn description_includes_cloud() {
        let c = make_overcast_conditions();
        let aerosol = mapping::map_aerosol(&c);
        let cloud = mapping::map_cloud(&c);
        let desc = mapping::describe(&c, &aerosol, &cloud);
        assert!(
            desc.contains("cloud") || desc.contains("overcast") || desc.contains("Cloud"),
            "Overcast description should mention cloud: {}",
            desc
        );
    }

    // ── Edge cases ──

    #[test]
    fn zero_visibility_handled() {
        let mut c = make_clear_conditions();
        c.visibility_m = 0.0;
        // Should not panic
        let _ = mapping::map_aerosol(&c);
        let _ = mapping::map_cloud(&c);
    }

    #[test]
    fn negative_aod_treated_as_zero() {
        let mut c = make_clear_conditions();
        c.aod_550 = -0.01; // bad data
        let aerosol = mapping::map_aerosol(&c);
        assert!(aerosol.is_none(), "Negative AOD should be treated as clear");
    }

    #[test]
    fn cloud_cover_over_100_clamped() {
        let mut c = make_clear_conditions();
        c.cloud_cover_total = 120.0;
        c.cloud_cover_low = 120.0;
        // Should not panic, treat as 100%
        let cloud = mapping::map_cloud(&c);
        assert!(cloud.is_some());
    }
}
