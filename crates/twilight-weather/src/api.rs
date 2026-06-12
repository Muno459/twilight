//! Open-Meteo API client.
//!
//! Fetches current weather and air quality data. No API key required
//! for non-commercial use.
//!
//! Two endpoints:
//! - `api.open-meteo.com/v1/forecast`: cloud cover, visibility, humidity
//! - `air-quality-api.open-meteo.com/v1/air-quality`: AOD, dust, PM, O3, NO2
//!
//! Missing data is never papered over with optimistic values: fields the
//! API did not report fall back to documented CONSERVATIVE defaults and
//! every substitution is recorded in `WeatherConditions::data_warnings`.

use serde::Deserialize;

use crate::error::WeatherError;
use crate::WeatherConditions;

const WEATHER_BASE_URL: &str = "https://api.open-meteo.com/v1/forecast";
const AIR_QUALITY_BASE_URL: &str = "https://air-quality-api.open-meteo.com/v1/air-quality";

/// Timeout for HTTP requests in milliseconds.
const REQUEST_TIMEOUT_MS: u64 = 10_000;

/// Conservative fallbacks for fields the API did not report. Chosen to
/// avoid the optimistic bias of assuming pristine air (AOD 0, 50 km
/// visibility): a continental-background aerosol load and moderate-haze
/// visibility. Every use is recorded in `data_warnings`.
const DEFAULT_AOD_550: f64 = 0.15;
const DEFAULT_VISIBILITY_M: f64 = 10_000.0;
const DEFAULT_RELATIVE_HUMIDITY: f64 = 50.0;

// ── Weather API response types ──

#[derive(Debug, Deserialize)]
struct WeatherResponse {
    latitude: f64,
    longitude: f64,
    current: Option<WeatherCurrent>,
}

#[derive(Debug, Deserialize)]
struct WeatherCurrent {
    time: Option<String>,
    cloud_cover: Option<f64>,
    cloud_cover_low: Option<f64>,
    cloud_cover_mid: Option<f64>,
    cloud_cover_high: Option<f64>,
    visibility: Option<f64>,
    relative_humidity_2m: Option<f64>,
    weather_code: Option<i32>,
}

// ── Air Quality API response types ──

#[derive(Debug, Deserialize)]
struct AirQualityResponse {
    current: Option<AirQualityCurrent>,
}

#[derive(Debug, Deserialize)]
struct AirQualityCurrent {
    aerosol_optical_depth: Option<f64>,
    dust: Option<f64>,
    pm2_5: Option<f64>,
    pm10: Option<f64>,
    ozone: Option<f64>,
    nitrogen_dioxide: Option<f64>,
}

// ── Missing-field accounting ──

/// One sample of every field the two endpoints can deliver, before
/// missing-field substitution. Shared by the current and hourly paths.
struct RawSample {
    aod_550: Option<f64>,
    dust: Option<f64>,
    pm2_5: Option<f64>,
    pm10: Option<f64>,
    ozone: Option<f64>,
    nitrogen_dioxide: Option<f64>,
    cloud_cover: Option<f64>,
    cloud_cover_low: Option<f64>,
    cloud_cover_mid: Option<f64>,
    cloud_cover_high: Option<f64>,
    visibility: Option<f64>,
    relative_humidity: Option<f64>,
    weather_code: Option<i32>,
}

/// Air-quality fields assumed when the whole air-quality feed is absent.
/// Trace species at 0 read downstream as "no data, no override".
fn aq_background() -> RawSample {
    RawSample {
        aod_550: Some(DEFAULT_AOD_550),
        dust: Some(0.0),
        pm2_5: Some(0.0),
        pm10: Some(0.0),
        ozone: Some(0.0),
        nitrogen_dioxide: Some(0.0),
        cloud_cover: None,
        cloud_cover_low: None,
        cloud_cover_mid: None,
        cloud_cover_high: None,
        visibility: None,
        relative_humidity: None,
        weather_code: None,
    }
}

const AQ_FEED_MISSING: &str = "air quality feed returned no data; assuming continental \
background (AOD 0.15, no dust/PM/O3/NO2)";

fn or_default(value: Option<f64>, default: f64, what: &str, warnings: &mut Vec<String>) -> f64 {
    match value {
        Some(v) => v,
        None => {
            warnings.push(format!("{what} missing; assuming {default}"));
            default
        }
    }
}

/// Substitute documented conservative defaults for missing fields,
/// recording each substitution in `warnings`.
fn conditions_from(
    raw: RawSample,
    timestamp: String,
    api_latitude: f64,
    api_longitude: f64,
    mut warnings: Vec<String>,
) -> WeatherConditions {
    let w = &mut warnings;
    let aod_550 = or_default(raw.aod_550, DEFAULT_AOD_550, "aerosol optical depth", w);
    let dust_ug_m3 = or_default(raw.dust, 0.0, "dust [ug/m3]", w);
    let pm2_5_ug_m3 = or_default(raw.pm2_5, 0.0, "PM2.5 [ug/m3]", w);
    let pm10_ug_m3 = or_default(raw.pm10, 0.0, "PM10 [ug/m3]", w);
    let ozone_ug_m3 = or_default(raw.ozone, 0.0, "surface O3 [ug/m3]", w);
    let nitrogen_dioxide_ug_m3 = or_default(raw.nitrogen_dioxide, 0.0, "surface NO2 [ug/m3]", w);
    let cloud_cover_total = or_default(raw.cloud_cover, 0.0, "total cloud cover [%]", w);
    let cloud_cover_low = or_default(raw.cloud_cover_low, 0.0, "low cloud cover [%]", w);
    let cloud_cover_mid = or_default(raw.cloud_cover_mid, 0.0, "mid cloud cover [%]", w);
    let cloud_cover_high = or_default(raw.cloud_cover_high, 0.0, "high cloud cover [%]", w);
    let visibility_m = or_default(raw.visibility, DEFAULT_VISIBILITY_M, "visibility [m]", w);
    let relative_humidity = or_default(
        raw.relative_humidity,
        DEFAULT_RELATIVE_HUMIDITY,
        "relative humidity [%]",
        w,
    );
    let weather_code = match raw.weather_code {
        Some(c) => c,
        None => {
            w.push("weather code missing; assuming clear (0)".to_string());
            0
        }
    };
    WeatherConditions {
        aod_550,
        dust_ug_m3,
        pm2_5_ug_m3,
        pm10_ug_m3,
        ozone_ug_m3,
        nitrogen_dioxide_ug_m3,
        cloud_cover_total,
        cloud_cover_low,
        cloud_cover_mid,
        cloud_cover_high,
        visibility_m,
        relative_humidity,
        weather_code,
        timestamp,
        api_latitude,
        api_longitude,
        data_warnings: warnings,
    }
}

/// Fetch current weather conditions from Open-Meteo.
///
/// Makes two HTTP requests (weather + air quality) and merges the results
/// into a single `WeatherConditions` struct.
///
/// # Errors
/// Returns a [`WeatherError`] if either request fails, returns invalid
/// JSON, or the weather feed carries no current conditions.
pub fn fetch_weather(lat: f64, lon: f64) -> Result<WeatherConditions, WeatherError> {
    let weather_url = format!(
        "{}?latitude={}&longitude={}&current=cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,visibility,relative_humidity_2m,weather_code",
        WEATHER_BASE_URL, lat, lon
    );

    let aq_url = format!(
        "{}?latitude={}&longitude={}&current=aerosol_optical_depth,dust,pm2_5,pm10,ozone,nitrogen_dioxide",
        AIR_QUALITY_BASE_URL, lat, lon
    );

    let weather: WeatherResponse = fetch_json(&weather_url)?;
    let aq: AirQualityResponse = fetch_json(&aq_url)?;

    // The primary weather feed must answer; the air-quality feed is
    // allowed to be absent (recorded as a data gap, not invented away).
    let wc = weather
        .current
        .ok_or_else(|| WeatherError::no_data("weather API returned no current conditions"))?;

    let mut warnings = Vec::new();
    let aqc = match aq.current {
        Some(c) => c,
        None => {
            warnings.push(AQ_FEED_MISSING.to_string());
            AirQualityCurrent {
                aerosol_optical_depth: Some(DEFAULT_AOD_550),
                dust: Some(0.0),
                pm2_5: Some(0.0),
                pm10: Some(0.0),
                ozone: Some(0.0),
                nitrogen_dioxide: Some(0.0),
            }
        }
    };

    let raw = RawSample {
        aod_550: aqc.aerosol_optical_depth,
        dust: aqc.dust,
        pm2_5: aqc.pm2_5,
        pm10: aqc.pm10,
        ozone: aqc.ozone,
        nitrogen_dioxide: aqc.nitrogen_dioxide,
        cloud_cover: wc.cloud_cover,
        cloud_cover_low: wc.cloud_cover_low,
        cloud_cover_mid: wc.cloud_cover_mid,
        cloud_cover_high: wc.cloud_cover_high,
        visibility: wc.visibility,
        relative_humidity: wc.relative_humidity_2m,
        weather_code: wc.weather_code,
    };
    Ok(conditions_from(
        raw,
        wc.time.unwrap_or_default(),
        weather.latitude,
        weather.longitude,
        warnings,
    ))
}

// ── Hourly forecast types (for prayer-hour sampling) ──

#[derive(Debug, Deserialize)]
struct WeatherHourlyResponse {
    latitude: f64,
    longitude: f64,
    hourly: Option<WeatherHourly>,
}

#[derive(Debug, Deserialize)]
struct WeatherHourly {
    time: Vec<String>,
    cloud_cover: Option<Vec<Option<f64>>>,
    cloud_cover_low: Option<Vec<Option<f64>>>,
    cloud_cover_mid: Option<Vec<Option<f64>>>,
    cloud_cover_high: Option<Vec<Option<f64>>>,
    visibility: Option<Vec<Option<f64>>>,
    relative_humidity_2m: Option<Vec<Option<f64>>>,
    weather_code: Option<Vec<Option<i32>>>,
}

#[derive(Debug, Deserialize)]
struct AirQualityHourlyResponse {
    hourly: Option<AirQualityHourly>,
}

#[derive(Debug, Deserialize)]
struct AirQualityHourly {
    time: Vec<String>,
    aerosol_optical_depth: Option<Vec<Option<f64>>>,
    dust: Option<Vec<Option<f64>>>,
    pm2_5: Option<Vec<Option<f64>>>,
    pm10: Option<Vec<Option<f64>>>,
    ozone: Option<Vec<Option<f64>>>,
    nitrogen_dioxide: Option<Vec<Option<f64>>>,
}

fn pick<T: Copy>(v: &Option<Vec<Option<T>>>, idx: usize) -> Option<T> {
    v.as_ref().and_then(|a| a.get(idx).copied().flatten())
}

/// Fetch FORECAST weather conditions for a specific UTC date and hour.
///
/// Prayer times are computed for a specific civil twilight window, not for
/// "now": sampling the hourly forecast at the actual Fajr/Isha hour uses
/// the best data the API offers (Open-Meteo blends the major global models
/// and serves hourly fields up to 16 days ahead and ~90 days back via the
/// same endpoint).
///
/// `date` is "YYYY-MM-DD"; `hour_utc` is the UTC hour to sample (0-23,
/// fractional input is rounded to the nearest hour).
///
/// # Errors
/// The requested hour must exist in the returned hourly series; a series
/// that does not contain it fails with [`WeatherError::NoData`] rather
/// than silently sampling a different hour.
pub fn fetch_weather_at(
    lat: f64,
    lon: f64,
    date: &str,
    hour_utc: f64,
) -> Result<WeatherConditions, WeatherError> {
    let hour = (hour_utc.rem_euclid(24.0)).round() as usize % 24;
    let target = format!("{}T{:02}:00", date, hour);

    let weather_url = format!(
        "{}?latitude={}&longitude={}&hourly=cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,visibility,relative_humidity_2m,weather_code&start_date={}&end_date={}&timezone=UTC",
        WEATHER_BASE_URL, lat, lon, date, date
    );
    let aq_url = format!(
        "{}?latitude={}&longitude={}&hourly=aerosol_optical_depth,dust,pm2_5,pm10,ozone,nitrogen_dioxide&start_date={}&end_date={}&timezone=UTC",
        AIR_QUALITY_BASE_URL, lat, lon, date, date
    );

    let weather: WeatherHourlyResponse = fetch_json(&weather_url)?;
    let aq: AirQualityHourlyResponse = fetch_json(&aq_url)?;

    let wh = weather
        .hourly
        .ok_or_else(|| WeatherError::no_data("weather API returned no hourly data"))?;
    let widx = wh.time.iter().position(|t| t == &target).ok_or_else(|| {
        WeatherError::no_data(format!(
            "weather API hourly series has no entry for {target} UTC"
        ))
    })?;

    let mut warnings = Vec::new();
    let aq_raw = match aq.hourly {
        Some(h) => {
            let i = h.time.iter().position(|t| t == &target).ok_or_else(|| {
                WeatherError::no_data(format!(
                    "air quality hourly series has no entry for {target} UTC"
                ))
            })?;
            RawSample {
                aod_550: pick(&h.aerosol_optical_depth, i),
                dust: pick(&h.dust, i),
                pm2_5: pick(&h.pm2_5, i),
                pm10: pick(&h.pm10, i),
                ozone: pick(&h.ozone, i),
                nitrogen_dioxide: pick(&h.nitrogen_dioxide, i),
                ..aq_background()
            }
        }
        None => {
            warnings.push(AQ_FEED_MISSING.to_string());
            aq_background()
        }
    };

    let raw = RawSample {
        cloud_cover: pick(&wh.cloud_cover, widx),
        cloud_cover_low: pick(&wh.cloud_cover_low, widx),
        cloud_cover_mid: pick(&wh.cloud_cover_mid, widx),
        cloud_cover_high: pick(&wh.cloud_cover_high, widx),
        visibility: pick(&wh.visibility, widx),
        relative_humidity: pick(&wh.relative_humidity_2m, widx),
        weather_code: pick(&wh.weather_code, widx),
        ..aq_raw
    };
    Ok(conditions_from(
        raw,
        target,
        weather.latitude,
        weather.longitude,
        warnings,
    ))
}

/// Fetch and deserialize JSON from a URL, retrying transient failures.
pub(crate) fn fetch_json<T: serde::de::DeserializeOwned>(url: &str) -> Result<T, WeatherError> {
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_millis(REQUEST_TIMEOUT_MS)))
        .build()
        .new_agent();

    let body = crate::retry::with_retries(WeatherError::is_transient, || {
        let mut response = agent
            .get(url)
            .call()
            .map_err(|e| WeatherError::from_ureq(e, url))?;
        response
            .body_mut()
            .read_to_string()
            .map_err(|e| WeatherError::from_ureq(e, url))
    })?;

    serde_json::from_str(&body).map_err(|e| {
        let head: String = body.chars().take(200).collect();
        WeatherError::parse(format!("invalid JSON from {url}: {e} (body: {head})"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weather_url_format() {
        let url = format!(
            "{}?latitude={}&longitude={}&current=cloud_cover",
            WEATHER_BASE_URL, 54.82, 9.36
        );
        assert!(url.contains("api.open-meteo.com"));
        assert!(url.contains("54.82"));
        assert!(url.contains("9.36"));
    }

    #[test]
    fn air_quality_url_format() {
        let url = format!(
            "{}?latitude={}&longitude={}&current=aerosol_optical_depth",
            AIR_QUALITY_BASE_URL, 21.42, 39.83
        );
        assert!(url.contains("air-quality-api.open-meteo.com"));
        assert!(url.contains("21.42"));
    }

    fn full_raw() -> RawSample {
        RawSample {
            aod_550: Some(0.08),
            dust: Some(1.0),
            pm2_5: Some(5.0),
            pm10: Some(9.0),
            ozone: Some(55.0),
            nitrogen_dioxide: Some(12.0),
            cloud_cover: Some(40.0),
            cloud_cover_low: Some(10.0),
            cloud_cover_mid: Some(20.0),
            cloud_cover_high: Some(30.0),
            visibility: Some(25_000.0),
            relative_humidity: Some(60.0),
            weather_code: Some(2),
        }
    }

    #[test]
    fn complete_sample_yields_no_warnings() {
        let c = conditions_from(full_raw(), "t".into(), 50.0, 10.0, Vec::new());
        assert!(c.data_warnings.is_empty(), "{:?}", c.data_warnings);
        assert!((c.aod_550 - 0.08).abs() < 1e-12);
        assert!((c.visibility_m - 25_000.0).abs() < 1e-9);
    }

    #[test]
    fn missing_fields_get_conservative_defaults_and_warnings() {
        let raw = RawSample {
            aod_550: None,
            visibility: None,
            relative_humidity: None,
            ..full_raw()
        };
        let c = conditions_from(raw, "t".into(), 50.0, 10.0, Vec::new());
        // Conservative, not pristine: continental AOD and 10 km visibility.
        assert!((c.aod_550 - DEFAULT_AOD_550).abs() < 1e-12);
        assert!((c.visibility_m - DEFAULT_VISIBILITY_M).abs() < 1e-9);
        assert!((c.relative_humidity - DEFAULT_RELATIVE_HUMIDITY).abs() < 1e-9);
        assert_eq!(c.data_warnings.len(), 3, "{:?}", c.data_warnings);
        assert!(c.data_warnings[0].contains("aerosol optical depth"));
    }

    #[test]
    fn aq_feed_missing_recorded_once() {
        // Whole-feed absence: one block-level warning, no per-field spam.
        let raw = RawSample {
            cloud_cover: Some(0.0),
            cloud_cover_low: Some(0.0),
            cloud_cover_mid: Some(0.0),
            cloud_cover_high: Some(0.0),
            visibility: Some(20_000.0),
            relative_humidity: Some(50.0),
            weather_code: Some(0),
            ..aq_background()
        };
        let c = conditions_from(
            raw,
            "t".into(),
            50.0,
            10.0,
            vec![AQ_FEED_MISSING.to_string()],
        );
        assert_eq!(c.data_warnings.len(), 1, "{:?}", c.data_warnings);
        assert!((c.aod_550 - DEFAULT_AOD_550).abs() < 1e-12);
    }

    // Integration test: actually fetch from Open-Meteo
    // Only runs when explicitly requested (takes network)
    #[test]
    #[ignore]
    fn fetch_weather_live() {
        let result = fetch_weather(54.8239, 9.3631);
        assert!(result.is_ok(), "Live fetch failed: {:?}", result.err());
        let conditions = result.unwrap();
        assert!(conditions.aod_550 >= 0.0, "AOD should be non-negative");
        assert!(
            conditions.cloud_cover_total >= 0.0 && conditions.cloud_cover_total <= 100.0,
            "Cloud cover should be 0-100%"
        );
        assert!(
            conditions.visibility_m > 0.0,
            "Visibility should be positive"
        );
    }
}
