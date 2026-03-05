//! End-to-end prayer time computation pipeline.
//!
//! Given a location and date, computes physically-based Fajr and Isha times
//! by running the MCRT engine across twilight solar zenith angles and applying
//! the spectral threshold model.
//!
//! Pipeline:
//! 1. Solar engine: compute solar position -> get declination, azimuth at sunset
//! 2. Solar engine: find sunset/sunrise times via zenith crossing
//! 3. Solar engine: check maximum SZA to detect persistent twilight
//! 4. MCRT Pass 1: coarse scan (2° steps) to locate threshold regions
//! 5. MCRT Pass 2: fine scan (0.1° steps) around each crossing
//! 6. Threshold: compute luminance, classify twilight, find crossings
//! 7. Solar engine: convert threshold SZA -> clock time via binary search
//!
//! The solar engine uses JPL DE440 ephemeris when a BSP file path is provided,
//! falling back to NREL SPA otherwise. DE440 provides ~1000x more precise
//! solar positions but requires the ~114 MB data file.

use twilight_data::aerosol::{self, AerosolProperties, AerosolType};
use twilight_data::atmosphere_profiles::AtmosphereType;
use twilight_data::builder;
use twilight_data::cloud::{self, CloudProperties, CloudType};
use twilight_solar::de440::De440;
use twilight_solar::spa::{self, SpaInput};
use twilight_threshold::threshold::{self, ThresholdConfig, TwilightAnalysis};

use crate::simulation::{self, ScatteringMode, SimulationConfig, SpectralResult};

/// Input for the prayer time pipeline.
#[derive(Debug, Clone)]
pub struct PrayerTimeInput {
    /// Observer latitude (degrees, north positive)
    pub latitude: f64,
    /// Observer longitude (degrees, east positive)
    pub longitude: f64,
    /// Observer elevation above sea level (meters)
    pub elevation: f64,
    /// Year
    pub year: i32,
    /// Month (1-12)
    pub month: i32,
    /// Day (1-31)
    pub day: i32,
    /// Timezone offset from UTC (hours)
    pub timezone: f64,
    /// Delta T (TT - UT1) in seconds
    pub delta_t: f64,
    /// Surface albedo (0-1)
    pub surface_albedo: f64,
    /// SZA scan resolution (degrees) for coarse pass.
    /// Default: 0.5
    pub sza_step: f64,
    /// Aerosol type. None for clear sky.
    pub aerosol_type: Option<AerosolType>,
    /// Cloud type. None for clear sky.
    pub cloud_type: Option<CloudType>,
    /// Custom aerosol properties (overrides aerosol_type when set).
    /// Used by the weather API integration to pass measured AOD values.
    pub custom_aerosol: Option<AerosolProperties>,
    /// Custom cloud properties (overrides cloud_type when set).
    /// Used by the weather API integration to pass derived cloud params.
    pub custom_cloud: Option<CloudProperties>,
    /// Threshold configuration
    pub threshold_config: ThresholdConfig,
    /// Path to DE440 BSP file. When provided, the pipeline uses JPL DE440
    /// as the primary solar position engine instead of SPA.
    pub de440_path: Option<String>,
    /// Scattering mode: single (deterministic) or multiple (Monte Carlo).
    pub scattering_mode: ScatteringMode,
    /// Number of photons per wavelength for MC mode. Ignored in single mode.
    pub photons_per_wavelength: usize,
}

impl Default for PrayerTimeInput {
    fn default() -> Self {
        Self {
            latitude: 21.4225, // Mecca
            longitude: 39.8262,
            elevation: 0.0,
            year: 2024,
            month: 1,
            day: 1,
            timezone: 3.0, // AST
            delta_t: 69.184,
            surface_albedo: 0.15,
            sza_step: 0.5,
            aerosol_type: None,
            cloud_type: None,
            custom_aerosol: None,
            custom_cloud: None,
            threshold_config: ThresholdConfig::default(),
            de440_path: None,
            scattering_mode: ScatteringMode::Single,
            photons_per_wavelength: 10_000,
        }
    }
}

/// Which solar position engine was used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EphemerisUsed {
    /// NREL Solar Position Algorithm (analytical, always available)
    Spa,
    /// JPL DE440 planetary ephemeris (requires BSP file)
    De440,
}

/// Output of the prayer time pipeline.
#[derive(Debug, Clone)]
pub struct PrayerTimeOutput {
    /// Fajr time (fractional hour, local time). None if not determinable.
    pub fajr_time: Option<f64>,
    /// Isha time per shafaq al-abyad (fractional hour, local time)
    pub isha_abyad_time: Option<f64>,
    /// Isha time per shafaq al-ahmar (fractional hour, local time)
    pub isha_ahmar_time: Option<f64>,
    /// Sunrise time (fractional hour, local time)
    pub sunrise_time: Option<f64>,
    /// Sunset time (fractional hour, local time)
    pub sunset_time: Option<f64>,
    /// Solar zenith angle at Fajr threshold
    pub fajr_sza_deg: Option<f64>,
    /// Solar zenith angle at Isha al-abyad threshold
    pub isha_abyad_sza_deg: Option<f64>,
    /// Solar zenith angle at Isha al-ahmar threshold
    pub isha_ahmar_sza_deg: Option<f64>,
    /// Equivalent solar depression angle for Fajr
    pub fajr_depression_deg: Option<f64>,
    /// Equivalent solar depression angle for Isha al-abyad
    pub isha_abyad_depression_deg: Option<f64>,
    /// Equivalent solar depression angle for Isha al-ahmar
    pub isha_ahmar_depression_deg: Option<f64>,
    /// Whether persistent twilight was detected (sun never drops below threshold)
    pub persistent_twilight: bool,
    /// Maximum solar zenith angle reached on this date (for persistent twilight)
    pub max_sza_deg: Option<f64>,
    /// Full twilight analysis data (for diagnostics)
    pub twilight_analyses: Vec<TwilightAnalysis>,
    /// MCRT spectral results (for diagnostics)
    pub spectral_results: Vec<SpectralResult>,
    /// Computation time in milliseconds
    pub computation_time_ms: u64,
    /// Which solar position engine was used
    pub ephemeris: EphemerisUsed,
}

// ── Solar position engine abstraction ──────────────────────────────

/// Internal solar engine that dispatches between DE440 and SPA.
///
/// DE440 is used when available (primary). SPA is the fallback.
/// Both provide the same interface: zenith at a given hour, and
/// bisection search for zenith crossings.
struct SolarEngine {
    de440: Option<De440>,
    spa_input: SpaInput,
}

impl SolarEngine {
    fn new(input: &PrayerTimeInput) -> (Self, EphemerisUsed) {
        let spa_input = SpaInput {
            year: input.year,
            month: input.month,
            day: input.day,
            hour: 0,
            minute: 0,
            second: 0,
            timezone: input.timezone,
            latitude: input.latitude,
            longitude: input.longitude,
            elevation: input.elevation,
            pressure: 1013.25,
            temperature: 15.0,
            delta_t: input.delta_t,
            slope: 0.0,
            azm_rotation: 0.0,
            atmos_refract: 0.5667,
        };

        // Try to open DE440 if path is provided
        let (de440, ephemeris) = match &input.de440_path {
            Some(path) => match De440::open(path) {
                Ok(de) => (Some(de), EphemerisUsed::De440),
                Err(_) => (None, EphemerisUsed::Spa),
            },
            None => (None, EphemerisUsed::Spa),
        };

        (SolarEngine { de440, spa_input }, ephemeris)
    }

    /// Get solar zenith angle at a fractional hour (local time).
    fn zenith_at_hour(&mut self, fractional_hour: f64) -> Option<f64> {
        if let Some(ref mut de) = self.de440 {
            // Convert local fractional hour to UTC
            let utc_hour = fractional_hour - self.spa_input.timezone;
            de.zenith_at_hour(
                self.spa_input.year,
                self.spa_input.month,
                self.spa_input.day,
                utc_hour,
                self.spa_input.delta_t,
                self.spa_input.latitude,
                self.spa_input.longitude,
                self.spa_input.elevation,
            )
            .ok()
        } else {
            let mut input = self.spa_input.clone();
            set_time_from_fractional_hour(&mut input, fractional_hour);
            spa::solar_position(&input).ok().map(|o| o.zenith)
        }
    }

    /// Get solar azimuth angle at a fractional hour (local time).
    fn azimuth_at_hour(&mut self, fractional_hour: f64) -> Option<f64> {
        if let Some(ref mut de) = self.de440 {
            let utc_hour = fractional_hour - self.spa_input.timezone;
            de.solar_position(
                self.spa_input.year,
                self.spa_input.month,
                self.spa_input.day,
                // Convert fractional UTC hour to h/m/s
                (utc_hour as i32).max(0),
                (((utc_hour - (utc_hour as i32) as f64) * 60.0) as i32).max(0),
                0,
                self.spa_input.delta_t,
                self.spa_input.latitude,
                self.spa_input.longitude,
                self.spa_input.elevation,
            )
            .ok()
            .map(|t| t.azimuth)
        } else {
            let mut input = self.spa_input.clone();
            set_time_from_fractional_hour(&mut input, fractional_hour);
            spa::solar_position(&input).ok().map(|o| o.azimuth)
        }
    }

    /// Find the fractional hour when zenith angle crosses `target_zenith`.
    /// Searches within `[start_hour, end_hour]` (local time).
    fn find_zenith_crossing(
        &mut self,
        target_zenith: f64,
        start_hour: f64,
        end_hour: f64,
        tolerance: f64,
    ) -> Option<f64> {
        if let Some(ref mut de) = self.de440 {
            // Convert local hours to UTC for DE440
            let utc_start = start_hour - self.spa_input.timezone;
            let utc_end = end_hour - self.spa_input.timezone;

            match de.find_zenith_crossing(
                self.spa_input.year,
                self.spa_input.month,
                self.spa_input.day,
                target_zenith,
                utc_start,
                utc_end,
                tolerance,
                self.spa_input.delta_t,
                self.spa_input.latitude,
                self.spa_input.longitude,
                self.spa_input.elevation,
            ) {
                Ok(Some(utc_hour)) => {
                    // Convert UTC result back to local
                    Some(utc_hour + self.spa_input.timezone)
                }
                _ => None,
            }
        } else {
            spa::find_zenith_crossing(
                &self.spa_input,
                target_zenith,
                start_hour,
                end_hour,
                tolerance,
            )
        }
    }

    /// Compute the maximum solar zenith angle on this date.
    fn compute_max_sza(&mut self) -> Option<f64> {
        let mut max_sza = 0.0f64;
        let mut hour = 0.0f64;
        while hour < 24.0 {
            if let Some(z) = self.zenith_at_hour(hour) {
                if z > max_sza {
                    max_sza = z;
                }
            }
            hour += 0.5;
        }
        if max_sza > 0.0 {
            Some(max_sza)
        } else {
            None
        }
    }
}

// ── Main pipeline ──────────────────────────────────────────────────

/// Run the full prayer time computation pipeline.
///
/// Uses a two-pass adaptive scan:
/// 1. Coarse scan at `sza_step` resolution to locate threshold regions
/// 2. Fine scan at 0.1° around each crossing for sub-minute precision
///
/// Also detects persistent twilight at high latitudes in summer.
///
/// When `de440_path` is set in the input, the pipeline uses JPL DE440
/// for all solar position computations. Otherwise falls back to SPA.
pub fn compute_prayer_times(input: &PrayerTimeInput) -> PrayerTimeOutput {
    let start = std::time::Instant::now();

    // Step 1: Build atmosphere model
    // Custom properties (from weather API) take priority over type-based defaults.
    let aerosol_props = input
        .custom_aerosol
        .clone()
        .or_else(|| input.aerosol_type.map(|at| aerosol::default_properties(at)));
    let cloud_props = input
        .custom_cloud
        .clone()
        .or_else(|| input.cloud_type.map(|ct| cloud::default_properties(ct)));
    let atm = builder::build_full(
        AtmosphereType::UsStandard,
        input.surface_albedo,
        aerosol_props.as_ref(),
        cloud_props.as_ref(),
    );

    // Step 2: Initialize solar engine (DE440 primary, SPA fallback)
    let (mut engine, ephemeris) = SolarEngine::new(input);

    // Step 3: Find sunrise/sunset times
    let sunrise_time = engine.find_zenith_crossing(90.8333, 0.0, 12.0, 0.0001);
    let sunset_time = engine.find_zenith_crossing(90.8333, 12.0, 24.0, 0.0001);

    // Step 4: Check maximum SZA to detect persistent twilight
    let max_sza_deg = engine.compute_max_sza();
    let persistent_twilight = max_sza_deg.map(|sza| sza < 106.0).unwrap_or(false);

    // Step 5: Determine solar azimuth at sunset for view direction
    let solar_azimuth_evening = if let Some(sunset_h) = sunset_time {
        engine.azimuth_at_hour(sunset_h).unwrap_or(270.0)
    } else {
        270.0
    };

    // Step 6: Determine the upper bound of the scan based on max SZA
    let sza_upper = max_sza_deg.map(|s| s.min(108.0)).unwrap_or(108.0);

    // Step 7: MCRT Pass 1 -- Coarse scan to locate threshold regions
    let config = SimulationConfig {
        latitude: input.latitude,
        longitude: input.longitude,
        elevation: input.elevation,
        solar_azimuth: solar_azimuth_evening,
        view_zenith: 85.0,
        apply_solar_irradiance: true,
        scattering_mode: input.scattering_mode,
        photons_per_wavelength: input.photons_per_wavelength,
    };

    let coarse_results =
        simulation::simulate_twilight_scan(&atm, &config, 90.0, sza_upper, input.sza_step);

    let coarse_analyses: Vec<TwilightAnalysis> = coarse_results
        .iter()
        .map(|sr| {
            threshold::analyze_twilight(
                sr.sza_deg,
                &sr.wavelengths_nm,
                &sr.radiance,
                &input.threshold_config,
            )
        })
        .collect();

    // Find approximate crossing regions from coarse scan
    let coarse_prayer =
        threshold::determine_prayer_times(coarse_analyses.clone(), &input.threshold_config);

    // Step 8: MCRT Pass 2 -- Fine scan around each crossing
    let mut refine_regions: Vec<(f64, f64)> = Vec::new();
    let margin = input.sza_step + 0.1;

    if let Some(sza) = coarse_prayer.fajr_sza_deg {
        add_refine_region(&mut refine_regions, sza - margin, sza + margin, sza_upper);
    }
    if let Some(sza) = coarse_prayer.isha_abyad_sza_deg {
        add_refine_region(&mut refine_regions, sza - margin, sza + margin, sza_upper);
    }
    if let Some(sza) = coarse_prayer.isha_ahmar_sza_deg {
        add_refine_region(&mut refine_regions, sza - margin, sza + margin, sza_upper);
    }

    let fine_step = 0.1;
    let mut fine_results: Vec<SpectralResult> = Vec::new();

    for (lo, hi) in &refine_regions {
        let region = simulation::simulate_twilight_scan(&atm, &config, *lo, *hi, fine_step);
        fine_results.extend(region);
    }

    // Combine coarse + fine results, sort by SZA, deduplicate
    let mut all_results: Vec<SpectralResult> = coarse_results;
    all_results.extend(fine_results);
    all_results.sort_by(|a, b| a.sza_deg.partial_cmp(&b.sza_deg).unwrap());

    let mut deduped_results: Vec<SpectralResult> = Vec::new();
    for r in all_results {
        if let Some(last) = deduped_results.last() {
            if (r.sza_deg - last.sza_deg).abs() < 0.05 {
                deduped_results.pop();
            }
        }
        deduped_results.push(r);
    }

    // Step 9: Re-analyze with combined high-resolution data
    let all_analyses: Vec<TwilightAnalysis> = deduped_results
        .iter()
        .map(|sr| {
            threshold::analyze_twilight(
                sr.sza_deg,
                &sr.wavelengths_nm,
                &sr.radiance,
                &input.threshold_config,
            )
        })
        .collect();

    let prayer_result = threshold::determine_prayer_times(all_analyses, &input.threshold_config);

    // Step 10: Convert threshold SZAs to clock times
    let fajr_time = prayer_result
        .fajr_sza_deg
        .and_then(|sza| engine.find_zenith_crossing(sza, 0.0, 12.0, 0.0001));

    let isha_abyad_time = prayer_result
        .isha_abyad_sza_deg
        .and_then(|sza| engine.find_zenith_crossing(sza, 12.0, 24.0, 0.0001));

    let isha_ahmar_time = prayer_result
        .isha_ahmar_sza_deg
        .and_then(|sza| engine.find_zenith_crossing(sza, 12.0, 24.0, 0.0001));

    let elapsed = start.elapsed();

    PrayerTimeOutput {
        fajr_time,
        isha_abyad_time,
        isha_ahmar_time,
        sunrise_time,
        sunset_time,
        fajr_sza_deg: prayer_result.fajr_sza_deg,
        isha_abyad_sza_deg: prayer_result.isha_abyad_sza_deg,
        isha_ahmar_sza_deg: prayer_result.isha_ahmar_sza_deg,
        fajr_depression_deg: prayer_result.fajr_sza_deg.map(|s| s - 90.0),
        isha_abyad_depression_deg: prayer_result.isha_abyad_sza_deg.map(|s| s - 90.0),
        isha_ahmar_depression_deg: prayer_result.isha_ahmar_sza_deg.map(|s| s - 90.0),
        persistent_twilight,
        max_sza_deg,
        twilight_analyses: prayer_result.analyses,
        spectral_results: deduped_results,
        computation_time_ms: elapsed.as_millis() as u64,
        ephemeris,
    }
}

/// Compute the maximum solar zenith angle on a given date (SPA-only helper for tests).
#[allow(dead_code)]
fn compute_max_sza(spa_input: &SpaInput) -> Option<f64> {
    let mut max_sza = 0.0f64;
    let mut hour = 0.0f64;
    while hour < 24.0 {
        let mut input = spa_input.clone();
        set_time_from_fractional_hour(&mut input, hour);
        if let Ok(result) = spa::solar_position(&input) {
            if result.zenith > max_sza {
                max_sza = result.zenith;
            }
        }
        hour += 0.5;
    }
    if max_sza > 0.0 {
        Some(max_sza)
    } else {
        None
    }
}

/// Add a refinement region, clamping to valid bounds and merging overlaps.
fn add_refine_region(regions: &mut Vec<(f64, f64)>, lo: f64, hi: f64, max_sza: f64) {
    let lo = lo.max(90.0);
    let hi = hi.min(max_sza);
    if hi <= lo {
        return;
    }

    // Check if this overlaps with an existing region
    for region in regions.iter_mut() {
        if lo <= region.1 + 0.5 && hi >= region.0 - 0.5 {
            // Merge
            region.0 = region.0.min(lo);
            region.1 = region.1.max(hi);
            return;
        }
    }

    regions.push((lo, hi));
}

/// Set hour/minute/second fields from a fractional hour value.
///
/// Converts to total integer seconds (with rounding) first, then decomposes
/// with integer arithmetic to avoid floating-point truncation errors.
fn set_time_from_fractional_hour(input: &mut SpaInput, fractional_hour: f64) {
    let total_seconds = (fractional_hour * 3600.0).round() as i32;
    input.hour = total_seconds / 3600;
    input.minute = (total_seconds % 3600) / 60;
    input.second = total_seconds % 60;
}

/// Format fractional hour as HH:MM:SS string.
///
/// Converts to total integer seconds (with rounding) first, then decomposes
/// with integer arithmetic to avoid floating-point truncation errors.
pub fn format_time(h: f64) -> String {
    if h < 0.0 || h > 24.0 {
        return "N/A".to_string();
    }
    let total_seconds = (h * 3600.0).round() as u32;
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;
    format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── PrayerTimeInput defaults ──

    #[test]
    fn default_input_mecca() {
        let input = PrayerTimeInput::default();
        assert!((input.latitude - 21.4225).abs() < 0.01);
        assert!((input.longitude - 39.8262).abs() < 0.01);
        assert!((input.elevation - 0.0).abs() < 0.01);
        assert_eq!(input.year, 2024);
        assert_eq!(input.month, 1);
        assert_eq!(input.day, 1);
        assert!((input.timezone - 3.0).abs() < 0.01);
        assert!((input.delta_t - 69.184).abs() < 0.01);
        assert!((input.surface_albedo - 0.15).abs() < 0.01);
        assert!((input.sza_step - 0.5).abs() < 0.01);
    }

    // ── format_time ──

    #[test]
    fn format_time_midnight() {
        assert_eq!(format_time(0.0), "00:00:00");
    }

    #[test]
    fn format_time_noon() {
        assert_eq!(format_time(12.0), "12:00:00");
    }

    #[test]
    fn format_time_end_of_day() {
        assert_eq!(format_time(24.0), "24:00:00");
    }

    #[test]
    fn format_time_fractional() {
        // 6.5 hours = 06:30:00
        assert_eq!(format_time(6.5), "06:30:00");
    }

    #[test]
    fn format_time_with_seconds() {
        // 12.5083... hours = 12:30:30
        let h = 12.0 + 30.0 / 60.0 + 30.0 / 3600.0;
        let formatted = format_time(h);
        assert_eq!(formatted, "12:30:30");
    }

    #[test]
    fn format_time_negative() {
        assert_eq!(format_time(-1.0), "N/A");
    }

    #[test]
    fn format_time_over_24() {
        assert_eq!(format_time(25.0), "N/A");
    }

    #[test]
    fn format_time_fajr_typical() {
        // Typical Fajr: ~5:30 = 5.5 hours
        let formatted = format_time(5.5);
        assert_eq!(formatted, "05:30:00");
    }

    // ── compute_prayer_times (end-to-end integration test) ──

    #[test]
    fn compute_prayer_times_mecca_produces_results() {
        // Mecca equinox: should produce valid Fajr and Isha times
        let input = PrayerTimeInput {
            latitude: 21.4225,
            longitude: 39.8262,
            year: 2024,
            month: 3,
            day: 15,
            timezone: 3.0,
            sza_step: 1.0, // coarser for speed
            ..PrayerTimeInput::default()
        };

        let output = compute_prayer_times(&input);

        // Should find sunrise and sunset
        assert!(output.sunrise_time.is_some(), "Should find sunrise");
        assert!(output.sunset_time.is_some(), "Should find sunset");

        // Sunrise should be in morning hours (4-7 local)
        if let Some(sunrise) = output.sunrise_time {
            assert!(
                sunrise > 4.0 && sunrise < 8.0,
                "Sunrise at {}, expected 4-8 local",
                sunrise
            );
        }

        // Sunset should be in evening hours (17-20 local)
        if let Some(sunset) = output.sunset_time {
            assert!(
                sunset > 16.0 && sunset < 20.0,
                "Sunset at {}, expected 16-20 local",
                sunset
            );
        }

        // Not persistent twilight at 21°N
        assert!(
            !output.persistent_twilight,
            "Mecca should not have persistent twilight"
        );
    }

    #[test]
    fn compute_prayer_times_mecca_depression_near_15_deg() {
        // The single-scatter engine produces ~15° depression angle
        let input = PrayerTimeInput {
            latitude: 21.4225,
            longitude: 39.8262,
            year: 2024,
            month: 3,
            day: 15,
            timezone: 3.0,
            sza_step: 0.5,
            ..PrayerTimeInput::default()
        };

        let output = compute_prayer_times(&input);

        // Check depression angles are reasonable (around 14-16°)
        if let Some(dep) = output.fajr_depression_deg {
            assert!(
                dep > 12.0 && dep < 18.0,
                "Fajr depression = {}°, expected 12-18°",
                dep
            );
        }
    }

    #[test]
    fn compute_prayer_times_fajr_before_sunrise() {
        let input = PrayerTimeInput {
            latitude: 21.4225,
            longitude: 39.8262,
            year: 2024,
            month: 3,
            day: 15,
            timezone: 3.0,
            sza_step: 1.0,
            ..PrayerTimeInput::default()
        };

        let output = compute_prayer_times(&input);

        if let (Some(fajr), Some(sunrise)) = (output.fajr_time, output.sunrise_time) {
            assert!(
                fajr < sunrise,
                "Fajr ({}) should be before sunrise ({})",
                fajr,
                sunrise
            );
        }
    }

    #[test]
    fn compute_prayer_times_isha_after_sunset() {
        let input = PrayerTimeInput {
            latitude: 21.4225,
            longitude: 39.8262,
            year: 2024,
            month: 3,
            day: 15,
            timezone: 3.0,
            sza_step: 1.0,
            ..PrayerTimeInput::default()
        };

        let output = compute_prayer_times(&input);

        if let (Some(isha), Some(sunset)) = (output.isha_abyad_time, output.sunset_time) {
            assert!(
                isha > sunset,
                "Isha ({}) should be after sunset ({})",
                isha,
                sunset
            );
        }
    }

    #[test]
    fn compute_prayer_times_has_spectral_data() {
        let input = PrayerTimeInput {
            latitude: 21.4225,
            longitude: 39.8262,
            year: 2024,
            month: 3,
            day: 15,
            timezone: 3.0,
            sza_step: 2.0, // coarse for speed
            ..PrayerTimeInput::default()
        };

        let output = compute_prayer_times(&input);
        assert!(
            !output.spectral_results.is_empty(),
            "Should have spectral results"
        );
        assert!(
            !output.twilight_analyses.is_empty(),
            "Should have twilight analyses"
        );
    }

    #[test]
    fn compute_prayer_times_timing() {
        let input = PrayerTimeInput {
            latitude: 21.4225,
            longitude: 39.8262,
            year: 2024,
            month: 3,
            day: 15,
            timezone: 3.0,
            sza_step: 1.0,
            ..PrayerTimeInput::default()
        };

        let output = compute_prayer_times(&input);
        // Should complete in reasonable time (< 10 seconds even on slow hardware)
        assert!(
            output.computation_time_ms < 10000,
            "Computation took {}ms, expected < 10000ms",
            output.computation_time_ms
        );
    }

    #[test]
    fn compute_prayer_times_london_winter() {
        // London, winter solstice: should have normal twilight (sun gets deep enough)
        let input = PrayerTimeInput {
            latitude: 51.5,
            longitude: -0.1,
            year: 2024,
            month: 12,
            day: 21,
            timezone: 0.0,
            sza_step: 1.0,
            ..PrayerTimeInput::default()
        };

        let output = compute_prayer_times(&input);
        assert!(
            !output.persistent_twilight,
            "London winter should not have persistent twilight"
        );
        assert!(output.sunrise_time.is_some(), "Should find sunrise");
        assert!(output.sunset_time.is_some(), "Should find sunset");
    }

    #[test]
    fn compute_prayer_times_max_sza_populated() {
        let input = PrayerTimeInput {
            latitude: 21.4225,
            longitude: 39.8262,
            year: 2024,
            month: 6,
            day: 21,
            timezone: 3.0,
            sza_step: 2.0,
            ..PrayerTimeInput::default()
        };

        let output = compute_prayer_times(&input);
        assert!(output.max_sza_deg.is_some(), "Should compute max SZA");
        let max_sza = output.max_sza_deg.unwrap();
        // At 21°N in June, max SZA should be > 90° (sun goes below horizon)
        assert!(
            max_sza > 90.0,
            "Max SZA at 21°N in June = {}, expected > 90°",
            max_sza
        );
    }

    // ── add_refine_region ──

    #[test]
    fn add_refine_region_basic() {
        let mut regions: Vec<(f64, f64)> = Vec::new();
        add_refine_region(&mut regions, 95.0, 100.0, 108.0);
        assert_eq!(regions.len(), 1);
        assert!((regions[0].0 - 95.0).abs() < 0.01);
        assert!((regions[0].1 - 100.0).abs() < 0.01);
    }

    #[test]
    fn add_refine_region_clamps_to_90() {
        let mut regions: Vec<(f64, f64)> = Vec::new();
        add_refine_region(&mut regions, 85.0, 95.0, 108.0);
        assert!((regions[0].0 - 90.0).abs() < 0.01, "Should clamp lo to 90");
    }

    #[test]
    fn add_refine_region_clamps_to_max_sza() {
        let mut regions: Vec<(f64, f64)> = Vec::new();
        add_refine_region(&mut regions, 95.0, 115.0, 108.0);
        assert!(
            (regions[0].1 - 108.0).abs() < 0.01,
            "Should clamp hi to max_sza"
        );
    }

    #[test]
    fn add_refine_region_merges_overlapping() {
        let mut regions: Vec<(f64, f64)> = Vec::new();
        add_refine_region(&mut regions, 95.0, 100.0, 108.0);
        add_refine_region(&mut regions, 99.0, 105.0, 108.0);
        // Should merge into one region [95, 105]
        assert_eq!(regions.len(), 1, "Overlapping regions should merge");
        assert!((regions[0].0 - 95.0).abs() < 0.01);
        assert!((regions[0].1 - 105.0).abs() < 0.01);
    }

    #[test]
    fn add_refine_region_rejects_inverted() {
        let mut regions: Vec<(f64, f64)> = Vec::new();
        add_refine_region(&mut regions, 100.0, 95.0, 108.0); // lo > hi after clamping
                                                             // lo=100, hi=95 → hi <= lo → should not add
        assert_eq!(regions.len(), 0, "Inverted region should not be added");
    }

    // ── set_time_from_fractional_hour ──

    #[test]
    fn set_time_noon() {
        let mut input = SpaInput::default();
        set_time_from_fractional_hour(&mut input, 12.0);
        assert_eq!(input.hour, 12);
        assert_eq!(input.minute, 0);
        assert_eq!(input.second, 0);
    }

    #[test]
    fn set_time_with_minutes() {
        let mut input = SpaInput::default();
        set_time_from_fractional_hour(&mut input, 12.5); // 12:30:00
        assert_eq!(input.hour, 12);
        assert_eq!(input.minute, 30);
        assert_eq!(input.second, 0);
    }

    #[test]
    fn set_time_with_seconds() {
        let mut input = SpaInput::default();
        let h = 12.0 + 30.0 / 60.0 + 30.0 / 3600.0; // 12:30:30
        set_time_from_fractional_hour(&mut input, h);
        assert_eq!(input.hour, 12);
        assert_eq!(input.minute, 30);
        assert_eq!(input.second, 30);
    }
}
