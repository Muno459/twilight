//! Live solar 10.7 cm radio flux (F10.7) from NOAA SWPC.
//!
//! F10.7 is a real, daily-measured quantity (Penticton radio observatory,
//! distributed by NOAA's Space Weather Prediction Center) and is the
//! standard proxy for the solar EUV activity that drives airglow
//! brightness: the night-sky airglow roughly doubles between solar
//! minimum (~70 sfu) and maximum (~200 sfu). Fetching the measured value
//! replaces the mid-cycle assumption (130 sfu) in the night-sky
//! background model.
//!
//! Endpoint (no key required):
//! <https://services.swpc.noaa.gov/json/f107_cm_flux.json>
//! Records are newest-first; `ninety_day_mean` is populated on the noon
//! reporting entries only.

use serde::Deserialize;

use crate::error::WeatherError;

const F107_URL: &str = "https://services.swpc.noaa.gov/json/f107_cm_flux.json";

/// Physical sanity bounds for F10.7 (sfu). The historical record spans
/// ~64 (deep minimum) to ~380 (extreme flare-day readings).
const F107_MIN: f64 = 50.0;
const F107_MAX: f64 = 450.0;

#[derive(Debug, Deserialize)]
struct F107Record {
    time_tag: String,
    flux: f64,
    ninety_day_mean: Option<f64>,
}

/// Measured solar radio flux state.
#[derive(Debug, Clone)]
pub struct SolarFlux {
    /// Most recent reported observed flux (sfu).
    pub latest_sfu: f64,
    /// 90-day running mean (sfu), when reported.
    pub ninety_day_mean_sfu: Option<f64>,
    /// Timestamp of the latest observation (ISO8601).
    pub time_tag: String,
}

impl SolarFlux {
    /// Activity proxy for the airglow scaling: the 557.7/630.0 nm airglow
    /// responds to EUV integrated over days-to-weeks, so the mean of the
    /// daily value and the 90-day mean captures both the cycle level and
    /// the current rotation's state (matches the practice in night-sky
    /// brightness modelling, e.g. Walker 1988 used monthly means).
    pub fn effective_sfu(&self) -> f64 {
        match self.ninety_day_mean_sfu {
            Some(m) => 0.5 * (self.latest_sfu + m),
            None => self.latest_sfu,
        }
    }
}

/// Fetch the latest measured F10.7 from NOAA SWPC.
pub fn fetch_f107() -> Result<SolarFlux, WeatherError> {
    let records: Vec<F107Record> = crate::api::fetch_json(F107_URL)?;
    parse_records(records)
}

fn parse_records(records: Vec<F107Record>) -> Result<SolarFlux, WeatherError> {
    let latest = records
        .iter()
        .find(|r| (F107_MIN..=F107_MAX).contains(&r.flux))
        .ok_or_else(|| WeatherError::no_data("F10.7 feed: no physically valid records"))?;
    let mean = records
        .iter()
        .filter_map(|r| r.ninety_day_mean)
        .find(|m| (F107_MIN..=F107_MAX).contains(m));
    Ok(SolarFlux {
        latest_sfu: latest.flux,
        ninety_day_mean_sfu: mean,
        time_tag: latest.time_tag.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_swpc_format() {
        // Real response shape (2026-06-11): newest first, scientific
        // notation floats, ninety_day_mean only on noon entries.
        let body = r#"[
            {"time_tag":"2026-06-11T22:00:00","frequency":2800,"flux":1.29e+002,"reporting_schedule":"Afternoon","avg_begin_date":null,"ninety_day_mean":null,"rec_count":null},
            {"time_tag":"2026-06-11T20:00:00","frequency":2800,"flux":1.27e+002,"reporting_schedule":"Noon","avg_begin_date":"2026-03-14T20:00:00","ninety_day_mean":1.26e+002,"rec_count":90}
        ]"#;
        let records: Vec<F107Record> = serde_json::from_str(body).unwrap();
        let flux = parse_records(records).unwrap();
        assert!((flux.latest_sfu - 129.0).abs() < 1e-9);
        assert!((flux.ninety_day_mean_sfu.unwrap() - 126.0).abs() < 1e-9);
        assert!((flux.effective_sfu() - 127.5).abs() < 1e-9);
    }

    #[test]
    fn rejects_unphysical_records() {
        let body = r#"[
            {"time_tag":"2026-06-11T22:00:00","frequency":2800,"flux":0.0,"ninety_day_mean":null},
            {"time_tag":"2026-06-11T20:00:00","frequency":2800,"flux":131.0,"ninety_day_mean":9999.0}
        ]"#;
        let records: Vec<F107Record> = serde_json::from_str(body).unwrap();
        let flux = parse_records(records).unwrap();
        assert!((flux.latest_sfu - 131.0).abs() < 1e-9);
        assert!(flux.ninety_day_mean_sfu.is_none(), "9999 must be rejected");
    }

    // Live network test - run with `cargo test -- --ignored`.
    #[test]
    #[ignore]
    fn live_fetch() {
        let flux = fetch_f107().expect("SWPC fetch failed");
        eprintln!("F10.7: {:?} -> effective {}", flux, flux.effective_sfu());
        assert!((F107_MIN..=F107_MAX).contains(&flux.latest_sfu));
    }
}
