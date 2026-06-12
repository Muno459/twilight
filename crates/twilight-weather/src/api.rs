//! Open-Meteo API client.
//!
//! Fetches current weather and air quality data. No API key required
//! for non-commercial use.
//!
//! Two endpoints:
//! - `api.open-meteo.com/v1/forecast`: cloud cover, visibility, humidity
//! - `air-quality-api.open-meteo.com/v1/air-quality`: AOD, dust, PM, O3, NO2

use serde::Deserialize;

use crate::WeatherConditions;

const WEATHER_BASE_URL: &str = "https://api.open-meteo.com/v1/forecast";
const AIR_QUALITY_BASE_URL: &str = "https://air-quality-api.open-meteo.com/v1/air-quality";

/// Timeout for HTTP requests in milliseconds.
const REQUEST_TIMEOUT_MS: u64 = 10_000;

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

/// Fetch current weather conditions from Open-Meteo.
///
/// Makes two HTTP requests (weather + air quality) and merges the results
/// into a single `WeatherConditions` struct.
///
/// # Errors
/// Returns an error string if either request fails or returns invalid JSON.
pub fn fetch_weather(lat: f64, lon: f64) -> Result<WeatherConditions, String> {
    let weather_url = format!(
        "{}?latitude={}&longitude={}&current=cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,visibility,relative_humidity_2m,weather_code",
        WEATHER_BASE_URL, lat, lon
    );

    let aq_url = format!(
        "{}?latitude={}&longitude={}&current=aerosol_optical_depth,dust,pm2_5,pm10,ozone,nitrogen_dioxide",
        AIR_QUALITY_BASE_URL, lat, lon
    );

    // Fetch weather data
    let weather: WeatherResponse =
        fetch_json(&weather_url).map_err(|e| format!("Weather API error: {}", e))?;

    // Fetch air quality data
    let aq: AirQualityResponse =
        fetch_json(&aq_url).map_err(|e| format!("Air Quality API error: {}", e))?;

    let wc = weather.current.unwrap_or(WeatherCurrent {
        time: None,
        cloud_cover: None,
        cloud_cover_low: None,
        cloud_cover_mid: None,
        cloud_cover_high: None,
        visibility: None,
        relative_humidity_2m: None,
        weather_code: None,
    });

    let aqc = aq.current.unwrap_or(AirQualityCurrent {
        aerosol_optical_depth: None,
        dust: None,
        pm2_5: None,
        pm10: None,
        ozone: None,
        nitrogen_dioxide: None,
    });

    Ok(WeatherConditions {
        aod_550: aqc.aerosol_optical_depth.unwrap_or(0.0),
        dust_ug_m3: aqc.dust.unwrap_or(0.0),
        pm2_5_ug_m3: aqc.pm2_5.unwrap_or(0.0),
        pm10_ug_m3: aqc.pm10.unwrap_or(0.0),
        ozone_ug_m3: aqc.ozone.unwrap_or(0.0),
        nitrogen_dioxide_ug_m3: aqc.nitrogen_dioxide.unwrap_or(0.0),
        cloud_cover_total: wc.cloud_cover.unwrap_or(0.0),
        cloud_cover_low: wc.cloud_cover_low.unwrap_or(0.0),
        cloud_cover_mid: wc.cloud_cover_mid.unwrap_or(0.0),
        cloud_cover_high: wc.cloud_cover_high.unwrap_or(0.0),
        visibility_m: wc.visibility.unwrap_or(50000.0),
        relative_humidity: wc.relative_humidity_2m.unwrap_or(50.0),
        weather_code: wc.weather_code.unwrap_or(0),
        timestamp: wc.time.unwrap_or_default(),
        api_latitude: weather.latitude,
        api_longitude: weather.longitude,
    })
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
pub fn fetch_weather_at(
    lat: f64,
    lon: f64,
    date: &str,
    hour_utc: f64,
) -> Result<WeatherConditions, String> {
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

    let weather: WeatherHourlyResponse =
        fetch_json(&weather_url).map_err(|e| format!("Weather API error: {}", e))?;
    let aq: AirQualityHourlyResponse =
        fetch_json(&aq_url).map_err(|e| format!("Air Quality API error: {}", e))?;

    let wh = weather
        .hourly
        .ok_or_else(|| "Weather API returned no hourly data".to_string())?;
    let widx = wh
        .time
        .iter()
        .position(|t| t == &target)
        .unwrap_or_else(|| hour.min(wh.time.len().saturating_sub(1)));

    let (aqidx, aqh) = match aq.hourly {
        Some(h) => {
            let i = h
                .time
                .iter()
                .position(|t| t == &target)
                .unwrap_or_else(|| hour.min(h.time.len().saturating_sub(1)));
            (i, Some(h))
        }
        None => (0, None),
    };

    Ok(WeatherConditions {
        aod_550: aqh.as_ref().and_then(|h| pick(&h.aerosol_optical_depth, aqidx)).unwrap_or(0.0),
        dust_ug_m3: aqh.as_ref().and_then(|h| pick(&h.dust, aqidx)).unwrap_or(0.0),
        pm2_5_ug_m3: aqh.as_ref().and_then(|h| pick(&h.pm2_5, aqidx)).unwrap_or(0.0),
        pm10_ug_m3: aqh.as_ref().and_then(|h| pick(&h.pm10, aqidx)).unwrap_or(0.0),
        ozone_ug_m3: aqh.as_ref().and_then(|h| pick(&h.ozone, aqidx)).unwrap_or(0.0),
        nitrogen_dioxide_ug_m3: aqh
            .as_ref()
            .and_then(|h| pick(&h.nitrogen_dioxide, aqidx))
            .unwrap_or(0.0),
        cloud_cover_total: pick(&wh.cloud_cover, widx).unwrap_or(0.0),
        cloud_cover_low: pick(&wh.cloud_cover_low, widx).unwrap_or(0.0),
        cloud_cover_mid: pick(&wh.cloud_cover_mid, widx).unwrap_or(0.0),
        cloud_cover_high: pick(&wh.cloud_cover_high, widx).unwrap_or(0.0),
        visibility_m: pick(&wh.visibility, widx).unwrap_or(50000.0),
        relative_humidity: pick(&wh.relative_humidity_2m, widx).unwrap_or(50.0),
        weather_code: pick(&wh.weather_code, widx).unwrap_or(0),
        timestamp: target,
        api_latitude: weather.latitude,
        api_longitude: weather.longitude,
    })
}

/// Fetch and deserialize JSON from a URL.
fn fetch_json<T: serde::de::DeserializeOwned>(url: &str) -> Result<T, String> {
    let agent = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_millis(REQUEST_TIMEOUT_MS)))
        .build()
        .new_agent();

    let mut response = agent
        .get(url)
        .call()
        .map_err(|e| format!("HTTP request failed: {}", e))?;

    let body = response
        .body_mut()
        .read_to_string()
        .map_err(|e| format!("Failed to read response body: {}", e))?;

    serde_json::from_str(&body).map_err(|e| {
        format!(
            "Failed to parse JSON: {} (body: {})",
            e,
            &body[..body.len().min(200)]
        )
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
