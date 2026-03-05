//! Open-Meteo API client.
//!
//! Fetches current weather and air quality data. No API key required
//! for non-commercial use.
//!
//! Two endpoints:
//! - `api.open-meteo.com/v1/forecast`: cloud cover, visibility, humidity
//! - `air-quality-api.open-meteo.com/v1/air-quality`: AOD, dust, PM

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
        "{}?latitude={}&longitude={}&current=aerosol_optical_depth,dust,pm2_5,pm10",
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
    });

    Ok(WeatherConditions {
        aod_550: aqc.aerosol_optical_depth.unwrap_or(0.0),
        dust_ug_m3: aqc.dust.unwrap_or(0.0),
        pm2_5_ug_m3: aqc.pm2_5.unwrap_or(0.0),
        pm10_ug_m3: aqc.pm10.unwrap_or(0.0),
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

/// Fetch and deserialize JSON from a URL.
fn fetch_json<T: serde::de::DeserializeOwned>(url: &str) -> Result<T, String> {
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_millis(REQUEST_TIMEOUT_MS))
        .build();

    let response = agent
        .get(url)
        .call()
        .map_err(|e| format!("HTTP request failed: {}", e))?;

    let body = response
        .into_string()
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
