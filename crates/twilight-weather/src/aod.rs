//! Measured aerosol optical depth (AOD550) from the CAMS-based
//! Open-Meteo Air Quality API, expressed as EXCESS over the engine's
//! desert-calibration baseline.
//!
//! WHY EXCESS, NOT ABSOLUTE: the khayt edge factor was calibrated so
//! that a NO-aerosol run lands on the desert campaign cluster
//! (KACST/Hail/Aswan/Mecca). That calibration air was not a vacuum: it
//! already contained clean desert aerosol. Feeding the engine an
//! ABSOLUTE measured AOD therefore double-counts the baseline load -
//! the measured warning case is Hail with the desert type constant
//! (AOD 0.5): 14.46 -> 9.01 deg (validation/RESULTS_CRITERION_SITES.md
//! section 9.4). Every aerosol input to the engine expresses excess
//! over that calibration air (section 9.7 recipe, point 3).
//!
//! Historical coverage: the Open-Meteo air-quality archive serves
//! hourly AOD550 from [`AOD_ARCHIVE_START`] onward (measured 2026-07:
//! earlier dates return rows of nulls, not errors). Requests before
//! that date fail loudly with [`WeatherError::NoData`]; nothing is
//! fabricated.

use std::path::Path;

use serde::Deserialize;

use crate::api;
use crate::cache;
use crate::error::WeatherError;

/// Open-Meteo Air Quality endpoint (CAMS global forecast/analysis;
/// free, no key).
const AIR_QUALITY_BASE_URL: &str = "https://air-quality-api.open-meteo.com/v1/air-quality";

/// First date with actual (non-null) AOD values in the Open-Meteo
/// air-quality archive, measured empirically 2026-07-03 by probing the
/// API: 2022-06-15 and everything earlier returns all-null AOD rows;
/// 2022-08-15 onward returns values. Open-Meteo documents the CAMS
/// archive as beginning 2022-07-29.
pub const AOD_ARCHIVE_START: &str = "2022-07-29";

/// The engine's aerosol calibration baseline [AOD550, CAMS scale].
///
/// This is the AOD the CAMS product assigns to the AIR THE CRITERION
/// WAS CALIBRATED IN, so `excess = measured - baseline` is a difference
/// of two values in the SAME retrieval scale and the product's absolute
/// bias largely cancels. Provenance for 0.10:
///
/// 1. Measured survey (tools/aod_baseline_survey.py, run 2026-07-03):
///    fajr-hour CAMS AOD550 at the four desert calibration sites
///    (Mecca, Hail, Riyadh desert, Aswan), winter mornings 2023-2025,
///    n = 1080: p10 = 0.09, p25 = 0.14, median = 0.21. The calibration
///    campaigns selected CLEAR mornings (observers only score cleanly
///    visible dawns), so the calibrated air sits near the clean decile:
///    ~0.10.
/// 2. Independent cross-check: section 9.6 measured the Assiut
///    campaign's implied excess at ~0.08; CAMS at Assiut in the
///    campaign season reads ~0.15-0.20 at the fajr hour (representative
///    recent dates; the 2012-2014 campaign dates predate the archive),
///    implying a baseline of 0.07-0.12.
/// 3. The alternative 0.05 (the ContinentalClean OPAC constant, the
///    "clean bracket" endpoint of section 9) was considered and
///    rejected: it is a MODEL constant, not a CAMS-scale measurement.
///    On the measured 2025-12-21 Mecca morning (CAMS 0.11) it would
///    charge the calibration site itself with 0.06 of excess (about
///    -1 deg of fajr depression), breaking the desert control that
///    DEFINES the calibration.
pub const AOD_BASELINE_550: f64 = 0.10;

/// Excess of a measured absolute AOD550 over the calibration baseline,
/// floored at zero: the engine cannot model air CLEANER than the air it
/// was calibrated in (there is no negative aerosol input). Callers
/// should surface the clamp (see [`MeasuredAod::below_baseline`]).
pub fn excess_over_baseline(aod_550: f64) -> f64 {
    (aod_550 - AOD_BASELINE_550).max(0.0)
}

/// 1-sigma uncertainty envelope [AOD550] for a CAMS value.
///
/// The API publishes NO per-value uncertainty, so this is an envelope,
/// not a retrieval sigma, and it is labeled as such downstream. Scale:
/// ECMWF's quarterly CAMS validation reports (o-suite vs AERONET) put
/// global AOD550 MAE at roughly 0.06-0.12 depending on region and
/// season, with relative bias up to ~30 percent in dust regions -
/// comparable to the classical satellite envelope 0.03 + 0.2 x AOD
/// (the MODIS dark-target form), which is what we use. Evaluated on the
/// ABSOLUTE value (the retrieval error lives there, not in the excess).
pub fn aod_sigma_envelope(aod_550_abs: f64) -> f64 {
    0.03 + 0.20 * aod_550_abs.max(0.0)
}

/// One measured AOD sample at a prayer-relevant hour.
#[derive(Debug, Clone)]
pub struct MeasuredAod {
    /// Absolute AOD550 as CAMS reports it.
    pub aod_550: f64,
    /// Excess over [`AOD_BASELINE_550`], floored at 0: the value the
    /// engine's aerosol input should carry.
    pub excess_550: f64,
    /// 1-sigma envelope uncertainty of the absolute value
    /// ([`aod_sigma_envelope`]).
    pub sigma_550: f64,
    /// True when the measured value sits BELOW the calibration baseline
    /// (the excess was clamped to 0): the site's air is cleaner than
    /// the engine can express; the residual is at most
    /// `baseline - measured` worth of AOD, on the shallow side.
    pub below_baseline: bool,
    /// The exact hourly sample used, "YYYY-MM-DDTHH:00" UTC.
    pub timestamp: String,
}

#[derive(Debug, Deserialize)]
struct AodHourlyResponse {
    hourly: Option<AodHourly>,
}

#[derive(Debug, Deserialize)]
struct AodHourly {
    time: Vec<String>,
    aerosol_optical_depth: Option<Vec<Option<f64>>>,
}

/// Extract the AOD sample at `target` from a raw API response body.
/// Typed failures, no fabricated defaults: a missing hourly block, a
/// missing hour, or a null value (the archive's pre-2022-07-29
/// behavior) each produce a distinct [`WeatherError::NoData`].
fn aod_from_body(body: &str, url: &str, target: &str) -> Result<MeasuredAod, WeatherError> {
    let resp: AodHourlyResponse = api::parse_json(body, url)?;
    let hourly = resp
        .hourly
        .ok_or_else(|| WeatherError::no_data("air quality API returned no hourly block"))?;
    let idx = hourly
        .time
        .iter()
        .position(|t| t == target)
        .ok_or_else(|| {
            WeatherError::no_data(format!(
                "air quality hourly series has no entry for {target} UTC"
            ))
        })?;
    let value = hourly
        .aerosol_optical_depth
        .as_ref()
        .and_then(|v| v.get(idx).copied().flatten())
        .ok_or_else(|| {
            WeatherError::no_data(format!(
                "AOD550 is null at {target} UTC (the CAMS archive begins \
                 {AOD_ARCHIVE_START}; earlier dates return null rows)"
            ))
        })?;
    if !(0.0..10.0).contains(&value) {
        return Err(WeatherError::parse(format!(
            "AOD550 {value} at {target} UTC is outside the physical range [0, 10)"
        )));
    }
    Ok(MeasuredAod {
        aod_550: value,
        excess_550: excess_over_baseline(value),
        sigma_550: aod_sigma_envelope(value),
        below_baseline: value < AOD_BASELINE_550,
        timestamp: target.to_string(),
    })
}

/// Fetch the measured AOD550 for a specific UTC date and hour - the
/// prayer-relevant hour, same rounding as the weather feed
/// (`api::hourly_target`).
///
/// Discipline matches the existing feeds: bounded retries on transient
/// failures (`retry::with_retries`, inside `api::fetch_text`), and an
/// optional on-disk cache (`cache_dir`) of the raw response with atomic
/// writes and the shared 14-day prune (`cache.rs`), so a validation
/// sweep re-running the same site/date does not re-hit the API.
///
/// # Errors
/// [`WeatherError::NoData`] when the requested hour has no value (dates
/// before [`AOD_ARCHIVE_START`], or beyond the forecast horizon);
/// transport/HTTP/parse errors otherwise. Never a made-up value.
pub fn fetch_aod_550_at(
    lat: f64,
    lon: f64,
    date: &str,
    hour_utc: f64,
    cache_dir: Option<&Path>,
) -> Result<MeasuredAod, WeatherError> {
    let target = api::hourly_target(date, hour_utc);
    let url = format!(
        "{}?latitude={}&longitude={}&hourly=aerosol_optical_depth&start_date={}&end_date={}&timezone=UTC",
        AIR_QUALITY_BASE_URL, lat, lon, date, date
    );

    // Cache key: site to 3 decimals (~100 m) + date. One file holds the
    // whole day's hourly series, so Fajr and Isha share a fetch.
    let cache_path = cache_dir.map(|d| {
        d.join(format!(
            "aod_{:.3}_{:.3}_{}.json",
            lat,
            lon,
            date.replace('/', "-")
        ))
    });

    if let Some(path) = &cache_path {
        if let Ok(body) = std::fs::read_to_string(path) {
            match aod_from_body(&body, &url, &target) {
                Ok(m) => return Ok(m),
                // NoData in a cached body is real (null hour); only a
                // corrupt/unparseable cache falls through to a refetch.
                Err(WeatherError::NoData(msg)) => return Err(WeatherError::NoData(msg)),
                Err(_) => {}
            }
        }
    }

    let body = api::fetch_text(&url)?;
    let measured = aod_from_body(&body, &url, &target)?;
    if let Some(path) = &cache_path {
        // Best-effort cache write + prune; a cache failure must not
        // fail a fetch that already succeeded.
        let _ = cache::write_atomic(path, body.as_bytes());
        if let Some(dir) = path.parent() {
            cache::prune_stale(dir, "aod_");
        }
    }
    Ok(measured)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Canned Open-Meteo air-quality response (shape verified against
    /// the live API 2026-07-03; values abbreviated to four hours).
    const FIXTURE: &str = r#"{
        "latitude": 52.4, "longitude": -2.0, "utc_offset_seconds": 0,
        "hourly_units": {"time": "iso8601", "aerosol_optical_depth": ""},
        "hourly": {
            "time": ["2025-12-10T03:00", "2025-12-10T04:00",
                     "2025-12-10T05:00", "2025-12-10T06:00"],
            "aerosol_optical_depth": [0.07, 0.07, null, 0.08]
        }
    }"#;

    #[test]
    fn fixture_parses_and_samples_the_requested_hour() {
        let m = aod_from_body(FIXTURE, "test://fixture", "2025-12-10T06:00").unwrap();
        assert!((m.aod_550 - 0.08).abs() < 1e-12);
        assert_eq!(m.timestamp, "2025-12-10T06:00");
        // Excess semantics: 0.08 sits below the 0.10 baseline.
        assert_eq!(m.excess_550, 0.0);
        assert!(m.below_baseline);
        // Envelope: 0.03 + 0.2 * 0.08.
        assert!((m.sigma_550 - 0.046).abs() < 1e-12);
    }

    #[test]
    fn null_hour_fails_loudly_with_archive_note() {
        let err = aod_from_body(FIXTURE, "test://fixture", "2025-12-10T05:00").unwrap_err();
        match err {
            WeatherError::NoData(msg) => {
                assert!(msg.contains("null"), "{msg}");
                assert!(msg.contains(AOD_ARCHIVE_START), "{msg}");
            }
            other => panic!("expected NoData, got {other:?}"),
        }
    }

    #[test]
    fn missing_hour_fails_loudly() {
        let err = aod_from_body(FIXTURE, "test://fixture", "2025-12-10T12:00").unwrap_err();
        assert!(matches!(err, WeatherError::NoData(_)));
    }

    #[test]
    fn missing_hourly_block_fails_loudly() {
        let err =
            aod_from_body(r#"{"latitude": 1.0}"#, "test://fixture", "2025-12-10T06:00")
                .unwrap_err();
        assert!(matches!(err, WeatherError::NoData(_)));
    }

    #[test]
    fn unphysical_value_rejected() {
        let body = r#"{"hourly": {"time": ["2025-12-10T06:00"],
                        "aerosol_optical_depth": [42.0]}}"#;
        let err = aod_from_body(body, "test://fixture", "2025-12-10T06:00").unwrap_err();
        assert!(matches!(err, WeatherError::Parse(_)));
    }

    #[test]
    fn excess_arithmetic() {
        // Above baseline: plain difference.
        assert!((excess_over_baseline(0.19) - 0.09).abs() < 1e-12);
        // At and below baseline: clamped to zero, never negative.
        assert_eq!(excess_over_baseline(AOD_BASELINE_550), 0.0);
        assert_eq!(excess_over_baseline(0.03), 0.0);
    }

    #[test]
    fn sigma_envelope_arithmetic() {
        assert!((aod_sigma_envelope(0.0) - 0.03).abs() < 1e-12);
        assert!((aod_sigma_envelope(0.5) - 0.13).abs() < 1e-12);
        // Negative input (bad data) does not shrink the floor.
        assert!((aod_sigma_envelope(-0.5) - 0.03).abs() < 1e-12);
    }

    #[test]
    fn cache_roundtrip_and_prune_discipline() {
        let dir = std::env::temp_dir().join("twilight_weather_aod_cache_test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("aod_52.440_-1.950_2025-12-10.json");
        cache::write_atomic(&path, FIXTURE.as_bytes()).unwrap();
        // A cached body is served without any network access.
        let m = fetch_aod_550_at(52.44, -1.95, "2025-12-10", 6.0, Some(&dir)).unwrap();
        assert!((m.aod_550 - 0.08).abs() < 1e-12);
        // A cached NULL hour is a real NoData, not a refetch trigger.
        let err = fetch_aod_550_at(52.44, -1.95, "2025-12-10", 5.0, Some(&dir)).unwrap_err();
        assert!(matches!(err, WeatherError::NoData(_)));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Live smoke test (network): a recent date inside the archive must
    /// return a physical value; run with --ignored.
    #[test]
    #[ignore]
    fn fetch_aod_live() {
        let m = fetch_aod_550_at(21.4225, 39.8262, "2025-12-21", 2.0, None)
            .expect("live AOD fetch failed");
        assert!(m.aod_550 >= 0.0 && m.aod_550 < 5.0, "AOD {}", m.aod_550);
        assert_eq!(m.timestamp, "2025-12-21T02:00");
        // Pre-archive date must fail loudly, never fabricate.
        let err = fetch_aod_550_at(21.4225, 39.8262, "2015-12-21", 2.0, None).unwrap_err();
        assert!(matches!(err, WeatherError::NoData(_)), "{err}");
    }
}
