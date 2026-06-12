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

use twilight_skyglow::SkyglowResult;
use twilight_solar::de440::De440;
use twilight_solar::spa::{self, SpaInput};
use twilight_terrain::horizon;
use twilight_terrain::HorizonProfile;
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
    /// Full vertical cloud profile as independent layers (overrides both
    /// `custom_cloud` and `cloud_type` when set). Produced by the cloud3d
    /// satellite reconstruction (80-level IWC profile collapsed into
    /// contiguous layers) - real measured 3D cloud structure.
    pub cloud_layers: Option<Vec<CloudProperties>>,
    /// Threshold configuration
    pub threshold_config: ThresholdConfig,
    /// Path to DE440 BSP file. When provided, the pipeline uses JPL DE440
    /// as the primary solar position engine instead of SPA.
    pub de440_path: Option<String>,
    /// Scattering mode: single (deterministic) or multiple (Monte Carlo).
    pub scattering_mode: ScatteringMode,
    /// Number of photons per wavelength for MC mode. Ignored in single mode.
    pub photons_per_wavelength: usize,
    /// Horizon profile from terrain masking. When present, sunrise/sunset SZA
    /// is adjusted based on terrain obstruction at the sun's azimuth.
    pub horizon_profile: Option<HorizonProfile>,
    /// Light pollution skyglow result. When present, the artificial spectral
    /// radiance is added to the MCRT-computed natural twilight radiance before
    /// threshold analysis, shifting Fajr/Isha times.
    pub skyglow: Option<SkyglowResult>,
    /// Override total column O3 (Dobson Units). When set, the standard
    /// atmosphere O3 profile is scaled to match this total column.
    /// Typical range: 220-450 DU.
    pub o3_column_du: Option<f64>,
    /// Override surface NO2 density (molecules/m^3). When set, the standard
    /// atmosphere NO2 profile is scaled so the surface value matches.
    pub no2_surface_density: Option<f64>,
    /// Enable full Stokes [I,Q,U,V] polarization tracking (default: true).
    /// When false (`--fast` mode), uses scalar phase function only.
    pub polarized: bool,
    /// Solar 10.7 cm radio flux (sfu) for the airglow background. The real
    /// measured value from NOAA SWPC (fetched by the CLI) - airglow scales
    /// ~2.4x from solar minimum (~70 sfu) to maximum (~200 sfu).
    /// None = mid-cycle default (130).
    pub solar_f107: Option<f64>,
    /// Print scan progress and diagnostics to stderr. Default false: the
    /// pipeline is a library and must stay silent; everything a caller
    /// needs to display travels in [`PrayerTimeOutput`] fields.
    pub verbose: bool,
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
            cloud_layers: None,
            threshold_config: ThresholdConfig::default(),
            de440_path: None,
            scattering_mode: ScatteringMode::Single,
            photons_per_wavelength: 10_000,
            horizon_profile: None,
            skyglow: None,
            o3_column_du: None,
            no2_surface_density: None,
            polarized: true,
            solar_f107: None,
            verbose: false,
        }
    }
}

// ── Scan constants ─────────────────────────────────────────────────
//
// The coarse/fine scan, the crossing fits, and the background model are
// coupled through these values; each comment states the constraint that
// keeps them consistent.

/// Coarse scan floor [deg SZA]: the sun's upper limb at the horizon.
/// No twilight threshold sits above it.
const SCAN_FLOOR_SZA: f64 = 90.0;

/// Coarse scan ceiling [deg SZA] (18 deg depression, astronomical dark).
/// Deliberately 2 deg DEEPER than [`PERSISTENT_TWILIGHT_SZA`]: the deepest
/// absolute threshold crosses near 105-106 deg, and scanning past it
/// (a) samples the dark night floor that the relative mode and the
/// celestial refloat float their thresholds on, and (b) keeps the
/// [`FIT_WINDOW_DEG`] fit window of a deep crossing inside the scan.
const SCAN_CEILING_SZA: f64 = 108.0;

/// Persistent-twilight cutoff [deg SZA]: when the nightly maximum SZA
/// stays below this, the sky never reaches full darkness and the
/// absolute thresholds may never cross. Distinct from
/// [`SCAN_CEILING_SZA`] (see there); a night peaking between the two is
/// NOT flagged persistent yet is still scanned in full.
const PERSISTENT_TWILIGHT_SZA: f64 = 106.0;

/// View zenith [deg] of the MCRT patch: 5 deg above the horizon toward
/// the sun, inside the twilight arch. The celestial-background refloat
/// must evaluate the night sky for this SAME patch - both feed the same
/// threshold comparison.
const VIEW_ZENITH_DEG: f64 = 85.0;

/// Independent seed-salted MC estimates per SZA (forced to 1 for
/// deterministic Single mode). Production noise control: the threshold
/// search runs on the MEAN curve and the per-SZA standard error feeds a
/// confidence interval on the prayer minute.
const K_SEEDS: usize = 4;

/// Fine scan step [deg] around each coarse crossing.
const FINE_STEP_DEG: f64 = 0.1;

/// Duplicate tolerance [deg] when merging coarse and fine results: half
/// a fine step, so re-scanned points replace their coarse versions while
/// genuine fine-grid neighbors survive.
const DEDUP_TOL_DEG: f64 = FINE_STEP_DEG / 2.0;

/// Half-width [deg] of the standard-error lookup window around a fitted
/// crossing. Must exceed [`FINE_STEP_DEG`] so at least one fine-scan
/// sample always falls inside, and stay below [`FIT_WINDOW_DEG`] so the
/// SE describes the same neighborhood the fit used.
const SE_WINDOW_DEG: f64 = 0.35;

/// Half-width [deg] of the local log-linear crossing-fit window. Must
/// not exceed the refine margin (`sza_step + FINE_STEP_DEG`, = 0.6 at
/// the default 0.5 deg coarse step) or the window would reach past the
/// fine-scanned region into coarse-only samples.
const FIT_WINDOW_DEG: f64 = 0.6;

/// Night-sky V-band extinction [mag/airmass]: gas-only component
/// (Rayleigh + ozone) of the zenith extinction coefficient.
const EXTINCTION_K_GAS: f64 = 0.16;

/// Aerosol V-band extinction per unit AOD at 550 nm [mag/airmass]; the
/// measured AOD from the weather feed converts through this slope.
const EXTINCTION_K_PER_AOD: f64 = 1.2;

/// Assumed aerosol extinction term [mag/airmass] when no measured AOD
/// is available (clear continental background).
const EXTINCTION_K_DEFAULT_AEROSOL: f64 = 0.05;

/// Celestial-background refloat trigger: re-float the detection
/// thresholds when the physical background at a crossing differs from
/// the dark-sky constant by more than this fraction (moonlit nights,
/// strong airglow, Milky Way pointing).
const REFLOAT_TRIGGER_FRACTION: f64 = 0.25;

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
    /// 1-sigma statistical uncertainty on the Fajr minute from MC noise
    /// (None for deterministic runs or when no fit was possible).
    pub fajr_uncertainty_min: Option<f64>,
    /// 1-sigma uncertainty on the Isha al-abyad minute.
    pub isha_abyad_uncertainty_min: Option<f64>,
    /// 1-sigma uncertainty on the Isha al-ahmar minute.
    pub isha_ahmar_uncertainty_min: Option<f64>,
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
    /// True when prayer times were determined in HIGH-LATITUDE RELATIVE
    /// mode: the sky never reached the dark-night background, so the
    /// detection thresholds were floated on tonight's actual sky-brightness
    /// minimum (threshold = (L_min + background) x detection_factor). Fajr
    /// is then the physically detectable onset of dawn brightening - the
    /// engine reports the physics; substitute fiqh rules remain the
    /// user's choice.
    pub high_latitude_relative_thresholds: bool,
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
    /// Horizon elevation angle at sunrise azimuth (degrees). None if no terrain.
    pub sunrise_horizon_deg: Option<f64>,
    /// Horizon elevation angle at sunset azimuth (degrees). None if no terrain.
    pub sunset_horizon_deg: Option<f64>,
    /// Effective sunrise SZA after terrain adjustment (degrees). None if no terrain.
    pub sunrise_sza_effective: Option<f64>,
    /// Effective sunset SZA after terrain adjustment (degrees). None if no terrain.
    pub sunset_sza_effective: Option<f64>,
    /// Terrain source name (e.g., "Copernicus DEM GLO-30 (30m)")
    pub terrain_source: Option<String>,
    /// Artificial sky brightness at zenith (mcd/m^2). None if no skyglow model.
    pub skyglow_zenith_mcd: Option<f64>,
    /// Effective Bortle class (1-9). None if no skyglow model.
    pub skyglow_bortle: Option<u8>,
    /// Estimated prayer time shift due to light pollution (minutes).
    pub skyglow_shift_minutes: Option<f64>,
    /// Note describing the celestial-background threshold refloat when it
    /// triggered (moonlit nights, strong airglow). None when the physical
    /// background matched the dark-sky constant within the trigger margin.
    pub celestial_refloat: Option<String>,
    /// The khayt al-abyad (Quran 2:187) contrast-criterion times - the
    /// PRIMARY determination: Fajr = white thread distinct WITH lateral
    /// spread; Isha = shafaq distinctness disappears (ahmar primary).
    /// The absolute-threshold times above remain as the legacy
    /// comparison method.
    pub khayt: KhaytTimes,
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

    /// Civil-date day increment (handles month/year rollover, leap years).
    ///
    /// One of three hand-rolled date implementations in the workspace
    /// (the others: `twilight-weather/satellite.rs date_minus` and the
    /// SPA calendar in `twilight-solar/spa.rs julian_day`); a future
    /// shared calendar helper should absorb all three.
    fn next_civil_day(year: i32, month: i32, day: i32) -> (i32, i32, i32) {
        debug_assert!((1..=12).contains(&month), "month out of range: {month}");
        let month = month.clamp(1, 12);
        let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
        let month_len = match month {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
            4 | 6 | 9 | 11 => 30,
            2 => {
                if leap {
                    29
                } else {
                    28
                }
            }
            // Clamped to 1..=12 above.
            _ => unreachable!(),
        };
        if day < month_len {
            (year, month, day + 1)
        } else if month < 12 {
            (year, month + 1, 1)
        } else {
            (year + 1, 1, 1)
        }
    }

    /// SPA input for a local fractional hour, rolling hours >= 24 into the
    /// NEXT civil day - required because at high latitudes the night's
    /// threshold crossings (Isha under persistent twilight) can fall
    /// after local midnight.
    fn spa_input_at(&self, fractional_hour: f64) -> SpaInput {
        let mut input = self.spa_input.clone();
        let mut h = fractional_hour;
        while h >= 24.0 {
            let (y, m, d) = Self::next_civil_day(input.year, input.month, input.day);
            input.year = y;
            input.month = m;
            input.day = d;
            h -= 24.0;
        }
        set_time_from_fractional_hour(&mut input, h);
        input
    }

    /// Get solar zenith angle at a fractional hour (local time).
    fn zenith_at_hour(&mut self, fractional_hour: f64) -> Option<f64> {
        if let Some(ref mut de) = self.de440 {
            // Local -> UTC can go negative (east timezones) or past 24;
            // the DE440 path resolves both through pure Julian-day
            // arithmetic, no calendar rollover needed.
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
            let input = self.spa_input_at(fractional_hour);
            spa::solar_position(&input).ok().map(|o| o.zenith)
        }
    }

    /// Get solar azimuth angle at a fractional hour (local time).
    ///
    /// Same civil-day handling as [`Self::zenith_at_hour`]: the DE440
    /// path keeps the fractional hour (seconds included) and lets the
    /// Julian-day conversion absorb negative / past-24 values; the SPA
    /// path rolls past-24 hours into the next civil day.
    fn azimuth_at_hour(&mut self, fractional_hour: f64) -> Option<f64> {
        if let Some(ref mut de) = self.de440 {
            let utc_hour = fractional_hour - self.spa_input.timezone;
            de.solar_position_at_hour(
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
            .map(|t| t.azimuth)
        } else {
            let input = self.spa_input_at(fractional_hour);
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

    /// Robust zenith-crossing finder for windows that may CONTAIN the
    /// nightly zenith maximum (solar midnight).
    ///
    /// Plain bisection needs the endpoints to bracket the target; near
    /// high-latitude solar midnight the target SZA is reached only INSIDE
    /// the window (the zenith rises through it and falls back), so the
    /// endpoint test fails and the crossing is lost - exactly the case for
    /// brightness-based Fajr under persistent twilight. This scans the
    /// window at 6-minute resolution, picks the bracketing segment whose
    /// slope matches the requested phase (descending zenith = morning/
    /// Fajr side, ascending = evening/Isha side), then bisects inside
    /// that locally monotone segment.
    fn find_zenith_crossing_robust(
        &mut self,
        target_zenith: f64,
        start_hour: f64,
        end_hour: f64,
        tolerance: f64,
        descending: bool,
    ) -> Option<f64> {
        let step = 0.1;
        let mut prev_h = start_hour;
        let mut prev_z = self.zenith_at_hour(prev_h)?;
        let mut h = start_hour + step;
        while h <= end_hour + 1e-9 {
            let z = self.zenith_at_hour(h)?;
            let crossed = (prev_z - target_zenith) * (z - target_zenith) <= 0.0;
            let slope_ok = if descending { z < prev_z } else { z > prev_z };
            if crossed && slope_ok {
                // In-engine bisection over zenith_at_hour (which handles
                // past-midnight hours), locally monotone on this segment.
                let (mut lo, mut hi) = (prev_h, h);
                let mut z_lo = prev_z;
                for _ in 0..64 {
                    let mid = 0.5 * (lo + hi);
                    let z_mid = self.zenith_at_hour(mid)?;
                    if (z_mid - target_zenith).abs() < tolerance {
                        return Some(mid);
                    }
                    if (z_lo - target_zenith) * (z_mid - target_zenith) < 0.0 {
                        hi = mid;
                    } else {
                        lo = mid;
                        z_lo = z_mid;
                    }
                }
                return Some(0.5 * (lo + hi));
            }
            prev_h = h;
            prev_z = z;
            h += step;
        }
        // Fall back to the plain bracketing search over the whole window.
        self.find_zenith_crossing(target_zenith, start_hour, end_hour, tolerance)
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

// ── Build helpers ──────────────────────────────────────────────────

/// Build the atmosphere model from pipeline input.
///
/// Custom properties (from weather API) take priority over type-based defaults.
/// Gas composition overrides (O3 column, NO2 density) are applied when present.
fn build_atmosphere(input: &PrayerTimeInput) -> twilight_core::atmosphere::AtmosphereModel {
    let aerosol_props = input
        .custom_aerosol
        .or_else(|| input.aerosol_type.map(aerosol::default_properties));

    // Full vertical profile (cloud3d satellite reconstruction) wins over
    // any single-layer description.
    if let Some(layers) = &input.cloud_layers {
        return builder::build_full_with_gas_layers(
            AtmosphereType::UsStandard,
            input.surface_albedo,
            aerosol_props.as_ref(),
            layers,
            input.o3_column_du,
            input.no2_surface_density,
        );
    }

    let cloud_props = input
        .custom_cloud
        .or_else(|| input.cloud_type.map(cloud::default_properties));

    if input.o3_column_du.is_some() || input.no2_surface_density.is_some() {
        builder::build_full_with_gas(
            AtmosphereType::UsStandard,
            input.surface_albedo,
            aerosol_props.as_ref(),
            cloud_props.as_ref(),
            input.o3_column_du,
            input.no2_surface_density,
        )
    } else {
        builder::build_full(
            AtmosphereType::UsStandard,
            input.surface_albedo,
            aerosol_props.as_ref(),
            cloud_props.as_ref(),
        )
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
    let atm = build_atmosphere(input);
    let scan =
        |atm: &twilight_core::atmosphere::AtmosphereModel,
         config: &SimulationConfig,
         start: f64,
         end: f64,
         step: f64| { simulation::simulate_twilight_scan(atm, config, start, end, step) };
    compute_prayer_times_inner(input, &atm, &scan, None)
}

/// Run the prayer time pipeline with GPU-accelerated MCRT simulation.
///
/// Uploads the atmosphere model to the GPU, then uses the GPU backend for
/// both coarse and fine scan passes. If any GPU operation fails, falls back
/// to the CPU pipeline automatically.
///
/// The caller must have already initialized the GPU backend via
/// `twilight_gpu::try_init()`. The atmosphere is uploaded here.
#[cfg(feature = "gpu")]
pub fn compute_prayer_times_gpu(
    input: &PrayerTimeInput,
    gpu: &mut dyn twilight_gpu::GpuBackend,
) -> PrayerTimeOutput {
    use crate::gpu_dispatch;

    let atm = build_atmosphere(input);

    // Cloud fields (cloud_extinction / cloud_g_scaled) ship in the v3
    // packed buffers and the Metal kernels apply the Eddington diffuse
    // transmission - GPU and CPU share the single-representation cloud
    // transport.

    // Upload atmosphere to GPU. On failure, fall back to CPU entirely.
    if let Err(e) = gpu.upload_atmosphere(&atm) {
        if input.verbose {
            eprintln!(
                "Warning: GPU atmosphere upload failed ({}), falling back to CPU",
                e
            );
        }
        let scan = |atm: &twilight_core::atmosphere::AtmosphereModel,
                    config: &SimulationConfig,
                    start: f64,
                    end: f64,
                    step: f64| {
            simulation::simulate_twilight_scan(atm, config, start, end, step)
        };
        return compute_prayer_times_inner(input, &atm, &scan, None);
    }

    // Reborrow as shared for dispatch (upload is done, no more &mut needed).
    let gpu_ref: &dyn twilight_gpu::GpuBackend = &*gpu;
    let scan = |atm: &twilight_core::atmosphere::AtmosphereModel,
                config: &SimulationConfig,
                start: f64,
                end: f64,
                step: f64| {
        gpu_dispatch::simulate_twilight_scan_gpu(gpu_ref, atm, config, start, end, step)
            .unwrap_or_else(|e| {
                if input.verbose {
                    eprintln!(
                        "Warning: GPU dispatch error ({}), CPU fallback for this scan",
                        e
                    );
                }
                simulation::simulate_twilight_scan(atm, config, start, end, step)
            })
    };
    let scan_list = |atm: &twilight_core::atmosphere::AtmosphereModel,
                     config: &SimulationConfig,
                     sza_values: &[f64]| {
        gpu_dispatch::simulate_twilight_szalist_gpu(gpu_ref, atm, config, sza_values)
            .unwrap_or_else(|e| {
                if input.verbose {
                    eprintln!(
                        "Warning: GPU batch dispatch error ({}), CPU fallback for fine pass",
                        e
                    );
                }
                let mut out = Vec::with_capacity(sza_values.len());
                for &sza in sza_values {
                    out.push(simulation::simulate_at_sza(atm, config, sza));
                }
                out
            })
    };
    compute_prayer_times_inner(input, &atm, &scan, Some(&scan_list))
}


/// Scan closure: simulate a SZA range (start, end, step) for one config.
type ScanFn<'a> = dyn Fn(
        &twilight_core::atmosphere::AtmosphereModel,
        &SimulationConfig,
        f64,
        f64,
        f64,
    ) -> Vec<SpectralResult>
    + 'a;

/// Batched scan closure: simulate an explicit SZA list for one config.
type ScanListFn<'a> = dyn Fn(
        &twilight_core::atmosphere::AtmosphereModel,
        &SimulationConfig,
        &[f64],
    ) -> Vec<SpectralResult>
    + 'a;

/// Per-SZA statistics from K independent seed-salted MC runs.
struct SzaStats {
    /// Mean spectra (one per requested SZA, in order).
    results: Vec<SpectralResult>,
    /// Relative standard error of the mean mesopic luminance per SZA
    /// (0.0 for deterministic runs or K = 1).
    rel_se: Vec<f64>,
}

/// Run the scan over `szas` K times with independent seed salts and return
/// per-SZA mean spectra plus the relative standard error of the mesopic
/// luminance. For `ScatteringMode::Single` (deterministic) K is forced
/// to 1. This is the production noise control: a single MC estimate per
/// SZA previously fed the threshold search directly, so one firefly seed
/// could shift the prayer minute.
#[allow(clippy::type_complexity)]
fn scan_szas_with_stats(
    atm: &twilight_core::atmosphere::AtmosphereModel,
    config: &SimulationConfig,
    szas: &[f64],
    k: usize,
    scan: &ScanFn,
    scan_list: Option<&ScanListFn>,
) -> SzaStats {
    let k = if matches!(config.scattering_mode, ScatteringMode::Single) {
        1
    } else {
        k.max(1)
    };

    let run_once = |salt: u64| -> Vec<SpectralResult> {
        let mut cfg = config.clone();
        cfg.seed_salt = salt;
        if let Some(f) = scan_list {
            f(atm, &cfg, szas)
        } else {
            // The range-based scan closure evaluated point-wise.
            szas.iter()
                .flat_map(|&sza| scan(atm, &cfg, sza, sza, 1.0))
                .collect()
        }
    };

    let runs: Vec<Vec<SpectralResult>> = (0..k as u64).map(run_once).collect();
    let n_sza = runs[0].len();

    let mut results = Vec::with_capacity(n_sza);
    let mut rel_se = Vec::with_capacity(n_sza);
    for i in 0..n_sza {
        let mut mean = runs[0][i].clone();
        let num_wl = mean.radiance.len();
        for run in runs.iter().skip(1) {
            for w in 0..num_wl {
                mean.radiance[w] += run[i].radiance[w];
            }
        }
        for w in 0..num_wl {
            mean.radiance[w] /= k as f64;
        }

        // Mesopic luminance per run -> SE of the mean.
        let se = if k > 1 {
            let lums: Vec<f64> = runs
                .iter()
                .map(|run| {
                    twilight_threshold::luminance::mesopic_luminance(
                        &run[i].wavelengths_nm,
                        &run[i].radiance,
                    )
                })
                .collect();
            let m: f64 = lums.iter().sum::<f64>() / k as f64;
            if m > 1e-300 {
                let var: f64 =
                    lums.iter().map(|l| (l - m) * (l - m)).sum::<f64>() / (k as f64 - 1.0);
                (var.sqrt() / (k as f64).sqrt()) / m
            } else {
                0.0
            }
        } else {
            0.0
        };
        results.push(mean);
        rel_se.push(se);
    }
    SzaStats { results, rel_se }
}

/// Physical night-sky background [cd/m^2] toward a view direction at a
/// local fractional hour. `view_azimuth` None = toward the sun's azimuth
/// at that instant. Moon from the JPL ephemeris when loaded.
fn night_sky_total(
    input: &PrayerTimeInput,
    engine: &mut SolarEngine,
    t_local: f64,
    view_zenith_deg: f64,
    view_azimuth_deg: Option<f64>,
) -> f64 {
    let az = view_azimuth_deg
        .or_else(|| engine.azimuth_at_hour(t_local))
        .unwrap_or(270.0);
    // Raw UTC hour, deliberately NOT wrapped into [0,24): both the Meeus
    // and DE440 paths convert through pure Julian-day arithmetic, so
    // hour 25.5 lands on the correct NEXT civil day. (Wrapping would put
    // a past-midnight Isha moon a full day early - a 13 deg lunar
    // position error.)
    let hour_utc = t_local - input.timezone;
    let inp = twilight_threshold::night_sky::NightSkyInput {
        latitude: input.latitude,
        longitude: input.longitude,
        year: input.year,
        month: input.month,
        day: input.day,
        hour_utc,
        view_zenith_deg,
        view_azimuth_deg: az,
        solar_f107: input.solar_f107.unwrap_or(130.0),
        extinction_k: EXTINCTION_K_GAS
            + input
                .custom_aerosol
                .map(|a| EXTINCTION_K_PER_AOD * a.aod_550)
                .unwrap_or(EXTINCTION_K_DEFAULT_AEROSOL),
    };
    // Real JPL ephemeris for the Moon when the BSP is loaded; truncated
    // Meeus series otherwise.
    let lum = match engine.de440.as_mut() {
        Some(de) => match de.moon_state(
            input.year,
            input.month,
            input.day,
            hour_utc,
            input.delta_t,
            input.latitude,
            input.longitude,
            input.elevation,
        ) {
            Ok(moon) => {
                twilight_threshold::night_sky::night_sky_luminance_with_moon(&inp, &moon)
            }
            Err(_) => twilight_threshold::night_sky::night_sky_luminance(&inp),
        },
        None => twilight_threshold::night_sky::night_sky_luminance(&inp),
    };
    lum.total
}

/// Times and diagnostics from the khayt al-abyad pass (see
/// [`crate::khayt`]). All times are local fractional hours.
#[derive(Debug, Clone, Default)]
pub struct KhaytTimes {
    /// Fajr sadiq: white-thread distinctness WITH lateral spread.
    pub fajr_time: Option<f64>,
    pub fajr_sza_deg: Option<f64>,
    /// Al-fajr al-kadhib: central distinctness without spread (the
    /// zodiacal wedge), when it precedes sadiq.
    pub kadhib_time: Option<f64>,
    /// Isha per shafaq al-ahmar (red-band distinctness disappears) -
    /// Shafi'i/Maliki/Hanbali.
    pub isha_ahmar_time: Option<f64>,
    pub isha_ahmar_sza_deg: Option<f64>,
    /// Isha per shafaq al-abyad (white distinctness disappears) - Hanafi.
    pub isha_abyad_time: Option<f64>,
    pub isha_abyad_sza_deg: Option<f64>,
    /// Contrast margin per band azimuth at the Fajr crossing (diagnostic).
    pub fajr_margins: Vec<f64>,
}

/// The khayt al-abyad pass: simulate the fan of sky patches, compose
/// per-patch totals (MCRT + celestial background + skyglow), and detect
/// the Quranic contrast events on both sides of the night.
///
/// The MCRT radiance depends only on (SZA, relative azimuth), so ONE fan
/// scan serves both Fajr and Isha; only the celestial background (moon,
/// zodiacal geometry) is evaluated per side.
#[allow(clippy::too_many_arguments)]
fn khayt_pass(
    input: &PrayerTimeInput,
    atm: &twilight_core::atmosphere::AtmosphereModel,
    base_config: &SimulationConfig,
    scan: &ScanFn,
    scan_list: Option<&ScanListFn>,
    engine: &mut SolarEngine,
    max_sza: f64,
) -> KhaytTimes {
    use crate::khayt::{detect, KhaytParams, KhaytScan, PatchLum};

    let params = KhaytParams::default();
    let lo = 93.0_f64;
    let hi = (max_sza - 0.25).min(111.0);
    if hi - lo < 3.0 {
        return KhaytTimes::default(); // never dark enough for the fan
    }
    let n_steps = ((hi - lo) / 1.0).floor() as usize + 1;
    let szas: Vec<f64> = (0..n_steps).map(|i| lo + i as f64).collect();

    // ── MCRT fan: per-direction luminance curves (side-independent) ──
    let thr_cfg = threshold::ThresholdConfig::default();
    let sim_dir = |offset: f64| -> Vec<(f64, f64)> {
        let mut cfg = base_config.clone();
        cfg.view_zenith = params.band_zenith_deg;
        cfg.view_azimuth = Some(base_config.solar_azimuth + offset);
        let stats = scan_szas_with_stats(atm, &cfg, &szas, 2, scan, scan_list);
        stats
            .results
            .iter()
            .map(|r| {
                let a = threshold::analyze_twilight(
                    r.sza_deg,
                    &r.wavelengths_nm,
                    &r.radiance,
                    &thr_cfg,
                );
                (a.luminance_mesopic, a.luminance_red)
            })
            .collect()
    };
    let band_mcrt: Vec<Vec<(f64, f64)>> =
        params.band_offsets_deg.iter().map(|&o| sim_dir(o)).collect();
    let ref_mcrt: Vec<Vec<(f64, f64)>> =
        params.ref_offsets_deg.iter().map(|&o| sim_dir(o)).collect();

    // ── Skyglow: identical additive veil on every patch of the low ring
    // (azimuthal structure of city glow is unknown from the atlas; an
    // equal veil still raises adaptation and suppresses contrast, which
    // is its physically dominant effect on detection).
    let (sg_mes, sg_red) = match &input.skyglow {
        Some(sg) => {
            let elev = 90.0 - params.band_zenith_deg;
            let lift = twilight_skyglow::angular::enhancement_factor(elev);
            let mes = sg.zenith_luminance * lift;
            // Red share from the skyglow's own spectrum (HPS vs LED mix).
            let n = sg.num_wavelengths.min(sg.spectral_radiance.len());
            let wl: Vec<f64> = (0..n).map(|i| 380.0 + 10.0 * i as f64).collect();
            let mes_s = twilight_threshold::luminance::mesopic_luminance(
                &wl,
                &sg.spectral_radiance[..n],
            );
            let red_s = twilight_threshold::luminance::red_band_luminance(
                &wl,
                &sg.spectral_radiance[..n],
            );
            let red_frac = if mes_s > 1e-30 { red_s / mes_s } else { 0.3 };
            (mes, mes * red_frac)
        }
        None => (0.0, 0.0),
    };

    // ── Celestial background per patch per side: anchors in SZA,
    // piecewise-linear in between. The absolute azimuth of each patch is
    // (real sun azimuth at that instant) + offset.
    let anchor_szas: Vec<f64> = [97.0, 101.0, 105.0, 109.0]
        .iter()
        .copied()
        .filter(|&s| s > lo && s < hi)
        .collect();
    let mut celestial_side = |morning: bool| -> Vec<(f64, Vec<f64>, Vec<f64>)> {
        // (anchor_sza, band_bg[j], ref_bg[j])
        let (w0, w1) = if morning { (0.0, 12.0) } else { (12.0, 28.0) };
        anchor_szas
            .iter()
            .filter_map(|&sza| {
                let t = engine.find_zenith_crossing_robust(sza, w0, w1, 0.001, morning)?;
                let sun_az = engine.azimuth_at_hour(t).unwrap_or(base_config.solar_azimuth);
                let band: Vec<f64> = params
                    .band_offsets_deg
                    .iter()
                    .map(|&o| {
                        night_sky_total(
                            input,
                            engine,
                            t,
                            params.band_zenith_deg,
                            Some(sun_az + o),
                        )
                    })
                    .collect();
                let refs: Vec<f64> = params
                    .ref_offsets_deg
                    .iter()
                    .map(|&o| {
                        night_sky_total(
                            input,
                            engine,
                            t,
                            params.band_zenith_deg,
                            Some(sun_az + o),
                        )
                    })
                    .collect();
                Some((sza, band, refs))
            })
            .collect()
    };
    let interp = |anchors: &[(f64, Vec<f64>, Vec<f64>)], sza: f64, j: usize, is_ref: bool| -> f64 {
        let pick = |a: &(f64, Vec<f64>, Vec<f64>)| if is_ref { a.2[j] } else { a.1[j] };
        if anchors.is_empty() {
            return threshold::NIGHT_SKY_LUMINANCE;
        }
        if sza <= anchors[0].0 {
            return pick(&anchors[0]);
        }
        for w in anchors.windows(2) {
            if sza <= w[1].0 {
                let f = (sza - w[0].0) / (w[1].0 - w[0].0);
                return pick(&w[0]) * (1.0 - f) + pick(&w[1]) * f;
            }
        }
        pick(anchors.last().unwrap())
    };

    let assemble = |anchors: &[(f64, Vec<f64>, Vec<f64>)]| -> KhaytScan {
        let band = szas
            .iter()
            .enumerate()
            .map(|(i, &sza)| {
                (0..params.band_offsets_deg.len())
                    .map(|j| {
                        let bg = interp(anchors, sza, j, false);
                        PatchLum {
                            mesopic: band_mcrt[j][i].0 + bg + sg_mes,
                            red: band_mcrt[j][i].1
                                + bg * params.celestial_red_fraction
                                + sg_red,
                        }
                    })
                    .collect()
            })
            .collect();
        let refs = szas
            .iter()
            .enumerate()
            .map(|(i, &sza)| {
                (0..params.ref_offsets_deg.len())
                    .map(|j| {
                        let bg = interp(anchors, sza, j, true);
                        PatchLum {
                            mesopic: ref_mcrt[j][i].0 + bg + sg_mes,
                            red: ref_mcrt[j][i].1
                                + bg * params.celestial_red_fraction
                                + sg_red,
                        }
                    })
                    .collect()
            })
            .collect();
        KhaytScan {
            szas: szas.clone(),
            band,
            refs,
        }
    };

    let morning_anchors = celestial_side(true);
    let evening_anchors = celestial_side(false);
    let morning = detect(&assemble(&morning_anchors), &params.for_side(true));
    let evening = detect(&assemble(&evening_anchors), &params.for_side(false));

    let mut out = KhaytTimes::default();
    if let Some(c) = morning.sadiq {
        out.fajr_sza_deg = Some(c.sza_deg);
        out.fajr_time = engine.find_zenith_crossing_robust(c.sza_deg, 0.0, 12.0, 0.0001, true);
        out.fajr_margins = morning.margins_at_sadiq.clone();
    }
    if let Some(c) = morning.kadhib {
        out.kadhib_time = engine.find_zenith_crossing_robust(c.sza_deg, 0.0, 12.0, 0.0001, true);
    }
    if let Some(c) = evening.ahmar {
        out.isha_ahmar_sza_deg = Some(c.sza_deg);
        out.isha_ahmar_time =
            engine.find_zenith_crossing_robust(c.sza_deg, 12.0, 28.0, 0.0001, false);
    }
    if let Some(c) = evening.sadiq {
        out.isha_abyad_sza_deg = Some(c.sza_deg);
        out.isha_abyad_time =
            engine.find_zenith_crossing_robust(c.sza_deg, 12.0, 28.0, 0.0001, false);
    }
    out
}

// ── Pipeline stages ────────────────────────────────────────────────
//
// `compute_prayer_times_inner` runs these in order; the data handed
// between stages travels in the small structs below.

/// Sunrise/sunset clock times with terrain diagnostics.
struct SunEvents {
    sunrise_time: Option<f64>,
    sunset_time: Option<f64>,
    /// Horizon elevation at the sunrise/sunset azimuth [deg]; None
    /// without a terrain profile.
    sunrise_horizon_deg: Option<f64>,
    sunset_horizon_deg: Option<f64>,
    /// Effective SZA after terrain adjustment [deg]; None when terrain
    /// does not obstruct.
    sunrise_sza_effective: Option<f64>,
    sunset_sza_effective: Option<f64>,
}

/// Maximum solar depth of the night and the scan bound derived from it.
struct TwilightExtent {
    /// Maximum SZA reached on this date; None when the probe never saw
    /// the sun.
    max_sza_deg: Option<f64>,
    /// Sun never reaches full darkness (max SZA < [`PERSISTENT_TWILIGHT_SZA`]).
    persistent_twilight: bool,
    /// Upper bound of the SZA scan.
    sza_upper: f64,
}

/// Combined coarse + fine MCRT scan output.
struct ScanData {
    /// Spectral results sorted by SZA, near-duplicates removed
    /// (fine-scan points replace their coarse versions).
    results: Vec<SpectralResult>,
    /// (SZA, relative standard error of the mesopic luminance) for every
    /// scanned point, coarse and fine.
    se_by_sza: Vec<(f64, f64)>,
}

/// Threshold-crossing SZAs [deg] for the three absolute-threshold events.
#[derive(Clone, Copy, Default)]
struct CrossingSzas {
    fajr: Option<f64>,
    isha_abyad: Option<f64>,
    isha_ahmar: Option<f64>,
}

/// 1-sigma uncertainties [deg] on the crossing SZAs from MC noise.
#[derive(Clone, Copy, Default)]
struct CrossingSigmas {
    fajr: Option<f64>,
    isha_abyad: Option<f64>,
    isha_ahmar: Option<f64>,
}

/// Clock times [local fractional hours] for the three events.
#[derive(Clone, Copy, Default)]
struct PrayerClockTimes {
    fajr: Option<f64>,
    isha_abyad: Option<f64>,
    isha_ahmar: Option<f64>,
}

/// Outcome of the celestial-background refloat: possibly updated
/// crossings plus the human-readable note for the caller.
struct CelestialRefloat {
    szas: CrossingSzas,
    times: PrayerClockTimes,
    note: String,
}

/// 1-sigma uncertainties on the prayer minutes [min].
struct UncertaintyMinutes {
    fajr: Option<f64>,
    isha_abyad: Option<f64>,
    isha_ahmar: Option<f64>,
}

/// Find sunrise/sunset, adjusted for terrain when a horizon profile is
/// present.
///
/// Standard SZA for sunrise/sunset = 90.8333 degrees (refraction +
/// semi-diameter). When terrain blocks the horizon, the sun must be
/// higher to clear it, so the effective SZA is smaller (sun clears
/// terrain later at sunrise, earlier at sunset). The standard-SZA
/// crossing fixes the sun's azimuth, the terrain profile gives the
/// horizon angle at that azimuth, and the crossing is re-found at the
/// adjusted SZA.
fn sun_events_with_terrain(input: &PrayerTimeInput, engine: &mut SolarEngine) -> SunEvents {
    let standard_sunrise = engine.find_zenith_crossing(90.8333, 0.0, 12.0, 0.0001);
    let standard_sunset = engine.find_zenith_crossing(90.8333, 12.0, 24.0, 0.0001);

    let Some(profile) = &input.horizon_profile else {
        return SunEvents {
            sunrise_time: standard_sunrise,
            sunset_time: standard_sunset,
            sunrise_horizon_deg: None,
            sunset_horizon_deg: None,
            sunrise_sza_effective: None,
            sunset_sza_effective: None,
        };
    };

    // One side: standard crossing -> sun azimuth -> terrain angle ->
    // crossing re-found at the terrain-adjusted SZA.
    let mut adjust = |standard: Option<f64>,
                      lo: f64,
                      hi: f64,
                      default_az: f64|
     -> (Option<f64>, Option<f64>, Option<f64>) {
        let Some(h) = standard else {
            return (standard, None, None);
        };
        let az = engine.azimuth_at_hour(h).unwrap_or(default_az);
        let hz = profile.angle_at(az);
        if hz > 0.01 {
            let eff_sza = horizon::effective_sunrise_sza(hz);
            let adjusted = engine.find_zenith_crossing(eff_sza, lo, hi, 0.0001);
            (adjusted.or(standard), Some(hz), Some(eff_sza))
        } else {
            (standard, Some(hz), None)
        }
    };

    let (sunrise_time, sunrise_horizon_deg, sunrise_sza_effective) =
        adjust(standard_sunrise, 0.0, 12.0, 90.0);
    let (sunset_time, sunset_horizon_deg, sunset_sza_effective) =
        adjust(standard_sunset, 12.0, 24.0, 270.0);

    SunEvents {
        sunrise_time,
        sunset_time,
        sunrise_horizon_deg,
        sunset_horizon_deg,
        sunrise_sza_effective,
        sunset_sza_effective,
    }
}

/// Probe the night's maximum solar depth: persistent-twilight flag and
/// the scan ceiling.
fn twilight_extent(engine: &mut SolarEngine) -> TwilightExtent {
    let max_sza_deg = engine.compute_max_sza();
    TwilightExtent {
        max_sza_deg,
        persistent_twilight: max_sza_deg
            .map(|sza| sza < PERSISTENT_TWILIGHT_SZA)
            .unwrap_or(false),
        sza_upper: max_sza_deg
            .map(|s| s.min(SCAN_CEILING_SZA))
            .unwrap_or(SCAN_CEILING_SZA),
    }
}

/// MCRT Pass 1 + 2: coarse scan at `sza_step` over the full twilight
/// range, threshold search on the coarse curve to select refine regions,
/// fine scan at [`FINE_STEP_DEG`] inside them, then merge and dedup.
#[allow(clippy::type_complexity)]
fn run_adaptive_scan(
    input: &PrayerTimeInput,
    atm: &twilight_core::atmosphere::AtmosphereModel,
    config: &SimulationConfig,
    scan: &ScanFn,
    scan_list: Option<&ScanListFn>,
    sza_upper: f64,
) -> ScanData {
    let seeds_label = if matches!(config.scattering_mode, ScatteringMode::Single) {
        1
    } else {
        K_SEEDS
    };

    // Pass 1: coarse scan to locate threshold regions.
    let coarse_t0 = std::time::Instant::now();
    let mut coarse_szas = Vec::new();
    {
        let mut sza = SCAN_FLOOR_SZA;
        while sza <= sza_upper + 1e-6 {
            coarse_szas.push(sza);
            sza += input.sza_step;
        }
    }
    if input.verbose {
        eprint!(
            "Coarse scan: {} points x {} seeds ... ",
            coarse_szas.len(),
            seeds_label
        );
        let _ = std::io::Write::flush(&mut std::io::stderr());
    }
    let coarse_stats = scan_szas_with_stats(atm, config, &coarse_szas, K_SEEDS, scan, scan_list);
    if input.verbose {
        eprintln!("{:.1?}", coarse_t0.elapsed());
    }
    let mut se_by_sza: Vec<(f64, f64)> = coarse_szas
        .iter()
        .zip(coarse_stats.rel_se.iter())
        .map(|(&s, &e)| (s, e))
        .collect();
    let coarse_results = coarse_stats.results;

    let coarse_analyses: Vec<TwilightAnalysis> = coarse_results
        .iter()
        .map(|sr| {
            let analysis = threshold::analyze_twilight(
                sr.sza_deg,
                &sr.wavelengths_nm,
                &sr.radiance,
                &input.threshold_config,
            );
            if input.verbose {
                eprintln!(
                    "  SZA {:.1}: mesopic={:.4e} total_rad={:.4e}",
                    sr.sza_deg,
                    analysis.luminance_mesopic,
                    sr.radiance.iter().sum::<f64>()
                );
            }
            analysis
        })
        .collect();

    // Approximate crossing regions from the coarse curve.
    let coarse_prayer = threshold::determine_prayer_times(coarse_analyses, &input.threshold_config);

    // Pass 2: fine scan around each coarse crossing. The margin spans
    // one coarse step plus one fine step so the fit window around the
    // refined crossing stays inside fine-scanned territory.
    let mut refine_regions: Vec<(f64, f64)> = Vec::new();
    let margin = input.sza_step + FINE_STEP_DEG;
    for sza in [
        coarse_prayer.fajr_sza_deg,
        coarse_prayer.isha_abyad_sza_deg,
        coarse_prayer.isha_ahmar_sza_deg,
    ]
    .into_iter()
    .flatten()
    {
        add_refine_region(&mut refine_regions, sza - margin, sza + margin, sza_upper);
    }

    let mut fine_results: Vec<SpectralResult> = Vec::new();
    if !refine_regions.is_empty() {
        let mut fine_szas = Vec::new();
        for (lo, hi) in &refine_regions {
            let mut sza = *lo;
            while sza <= *hi + 1e-6 {
                fine_szas.push(sza);
                sza += FINE_STEP_DEG;
            }
        }

        let fine_t0 = std::time::Instant::now();
        if input.verbose {
            eprint!(
                "Fine scan: {} points x {} seeds ... ",
                fine_szas.len(),
                seeds_label
            );
            let _ = std::io::Write::flush(&mut std::io::stderr());
        }
        let fine_stats = scan_szas_with_stats(atm, config, &fine_szas, K_SEEDS, scan, scan_list);
        if input.verbose {
            eprintln!("{:.1?}", fine_t0.elapsed());
        }
        for (i, r) in fine_stats.results.into_iter().enumerate() {
            se_by_sza.push((r.sza_deg, fine_stats.rel_se[i]));
            fine_results.push(r);
        }
    }

    // Merge coarse + fine, sort by SZA, drop near-duplicates. The sort
    // is stable, so at equal SZA the fine point follows its coarse twin
    // and the pop-then-push keeps the fine one.
    let mut all_results = coarse_results;
    all_results.extend(fine_results);
    all_results.sort_by(|a, b| a.sza_deg.total_cmp(&b.sza_deg));

    let mut results: Vec<SpectralResult> = Vec::new();
    for r in all_results {
        if let Some(last) = results.last() {
            if (r.sza_deg - last.sza_deg).abs() < DEDUP_TOL_DEG {
                results.pop();
            }
        }
        results.push(r);
    }

    ScanData { results, se_by_sza }
}

/// Inject artificial skyglow radiance into the natural MCRT spectra,
/// shifting the threshold crossings for urban sky brightness.
fn inject_skyglow(results: &mut [SpectralResult], sg: &SkyglowResult) {
    for sr in results.iter_mut() {
        let n = sr.radiance.len().min(sg.num_wavelengths);
        for i in 0..n {
            sr.radiance[i] += sg.spectral_radiance[i];
        }
    }
}

/// HIGH-LATITUDE RELATIVE-DETECTION MODE.
///
/// The engine's observable is SKY BRIGHTNESS, not a depression-angle
/// convention. The absolute thresholds are themselves derived as
/// detection_factor x dark-night background (see ThresholdConfig docs).
/// Under persistent twilight the sky never reaches the dark background,
/// but it still has a luminance MINIMUM at solar midnight and then
/// measurably brightens - an SQM records it, an observer sees the glow
/// spread. The general law, valid at every latitude, is therefore
///
///   threshold = (L_min_tonight + L_dark_background) x detection_factor
///
/// At normal latitudes L_min ~ 0 and this reduces exactly to the
/// absolute constants; at high latitudes it floats on tonight's bright
/// floor and Fajr = the moment the dawn brightening becomes detectable
/// above it (al-fajr al-sadiq as a physical event, not a convention).
/// The fiqh question of whether to follow this or a substitute rule
/// (aqrab al-ayyam etc.) is the user's; the engine reports the physics.
///
/// Returns the relative-threshold result and `true` when the mode
/// engaged; otherwise hands `base` back unchanged.
fn apply_high_latitude_relative_mode(
    input: &PrayerTimeInput,
    base: threshold::PrayerTimeResult,
) -> (threshold::PrayerTimeResult, bool) {
    if base.fajr_sza_deg.is_some() || base.analyses.len() < 3 {
        return (base, false);
    }
    // Tonight's floor: the dimmest scanned point (solar-midnight side).
    let l_min_mesopic = base
        .analyses
        .iter()
        .map(|a| a.luminance_mesopic)
        .fold(f64::MAX, f64::min);
    let l_min_red = base
        .analyses
        .iter()
        .map(|a| a.luminance_red)
        .fold(f64::MAX, f64::min);
    if !(l_min_mesopic.is_finite() && l_min_mesopic > 0.0) {
        return (base, false);
    }
    let bg = threshold::NIGHT_SKY_LUMINANCE;
    // TVI contrast detection at the eye's actual adaptation level
    // (the spectral->mesopic chain models the eye's response; the
    // TVI models its CONTRAST sensitivity, which improves sharply
    // against brighter skies - Blackwell-anchored, dark-site limit
    // bit-compatible with the absolute constants).
    let floor_mesopic = l_min_mesopic + bg;
    let floor_red = l_min_red.max(0.0) + bg;
    let relative_config = threshold::ThresholdConfig {
        fajr_luminance: threshold::detection_threshold(
            floor_mesopic,
            input.threshold_config.fajr_luminance,
        ),
        isha_abyad_luminance: threshold::detection_threshold(
            floor_mesopic,
            input.threshold_config.isha_abyad_luminance,
        ),
        isha_ahmar_red_luminance: threshold::detection_threshold(
            floor_red,
            input.threshold_config.isha_ahmar_red_luminance,
        ),
        ..input.threshold_config.clone()
    };
    let relative_result =
        threshold::determine_prayer_times(base.analyses.clone(), &relative_config);
    if relative_result.fajr_sza_deg.is_none() {
        return (base, false);
    }
    if input.verbose {
        eprintln!(
            "High-latitude mode: TVI contrast detection on tonight's sky \
             floor (L_min mesopic = {:.3e} cd/m^2 -> Fajr threshold \
             {:.3e} cd/m^2, a {:.0}% rise).",
            l_min_mesopic,
            relative_config.fajr_luminance,
            (relative_config.fajr_luminance / floor_mesopic - 1.0) * 100.0
        );
    }
    (relative_result, true)
}

/// Crossing-on-fit + confidence interval.
///
/// Refine each pairwise crossing with a local log-linear least-squares
/// fit (robust to single noisy MC samples), and propagate the per-SZA
/// standard error into a sigma on the crossing SZA:
///   sigma_sza = rel_SE(luminance) / |d(lnL)/dSZA|.
fn fit_crossings(
    input: &PrayerTimeInput,
    result: &threshold::PrayerTimeResult,
    se_by_sza: &[(f64, f64)],
    used_relative_thresholds: bool,
) -> (CrossingSzas, CrossingSigmas) {
    let lookup_rel_se = |sza: f64| -> f64 {
        se_by_sza
            .iter()
            .filter(|(s, _)| (s - sza).abs() <= SE_WINDOW_DEG)
            .map(|(_, e)| *e)
            .fold(0.0f64, f64::max)
    };
    let fit_window = |pick: &dyn Fn(&TwilightAnalysis) -> f64, around: f64| -> Vec<(f64, f64)> {
        result
            .analyses
            .iter()
            .filter(|a| (a.sza_deg - around).abs() <= FIT_WINDOW_DEG)
            .map(|a| (a.sza_deg, pick(a)))
            .collect()
    };
    let refine = |sza_opt: Option<f64>,
                  pick: &dyn Fn(&TwilightAnalysis) -> f64,
                  thresh: f64|
     -> (Option<f64>, Option<f64>) {
        let Some(sza0) = sza_opt else {
            return (None, None);
        };
        let window = fit_window(pick, sza0);
        match threshold::fit_crossing_loglinear(&window, thresh) {
            Some((fitted, slope)) => {
                let sigma_sza = lookup_rel_se(fitted) / slope.abs().max(1e-6);
                (Some(fitted), Some(sigma_sza))
            }
            None => (Some(sza0), None),
        }
    };
    // Under high-latitude mode the fit must target the floated thresholds.
    let floor_of = |pick: &dyn Fn(&TwilightAnalysis) -> f64| -> f64 {
        result
            .analyses
            .iter()
            .map(pick)
            .fold(f64::MAX, f64::min)
            .max(0.0)
            + threshold::NIGHT_SKY_LUMINANCE
    };
    let effective = |dark_anchor: f64, pick: &dyn Fn(&TwilightAnalysis) -> f64| -> f64 {
        if used_relative_thresholds {
            threshold::detection_threshold(floor_of(pick), dark_anchor)
        } else {
            dark_anchor
        }
    };
    let cfg = &input.threshold_config;
    let (fajr, fajr_sigma) = refine(
        result.fajr_sza_deg,
        &|a| a.luminance_mesopic,
        effective(cfg.fajr_luminance, &|a| a.luminance_mesopic),
    );
    let (isha_abyad, abyad_sigma) = refine(
        result.isha_abyad_sza_deg,
        &|a| a.luminance_mesopic,
        effective(cfg.isha_abyad_luminance, &|a| a.luminance_mesopic),
    );
    let (isha_ahmar, ahmar_sigma) = refine(
        result.isha_ahmar_sza_deg,
        &|a| a.luminance_red,
        effective(cfg.isha_ahmar_red_luminance, &|a| a.luminance_red),
    );
    (
        CrossingSzas {
            fajr,
            isha_abyad,
            isha_ahmar,
        },
        CrossingSigmas {
            fajr: fajr_sigma,
            isha_abyad: abyad_sigma,
            isha_ahmar: ahmar_sigma,
        },
    )
}

/// Convert threshold SZAs to clock times (slope-aware: the morning
/// crossing is on the DESCENDING zenith branch, the evening on the
/// ASCENDING - required when the crossing lies near solar midnight,
/// where plain endpoint-bracketing bisection fails).
fn crossings_to_times(engine: &mut SolarEngine, szas: CrossingSzas) -> PrayerClockTimes {
    PrayerClockTimes {
        fajr: szas
            .fajr
            .and_then(|sza| engine.find_zenith_crossing_robust(sza, 0.0, 12.0, 0.0001, true)),
        isha_abyad: szas
            .isha_abyad
            .and_then(|sza| engine.find_zenith_crossing_robust(sza, 12.0, 28.0, 0.0001, false)),
        isha_ahmar: szas
            .isha_ahmar
            .and_then(|sza| engine.find_zenith_crossing_robust(sza, 12.0, 28.0, 0.0001, false)),
    }
}

/// CELESTIAL-BACKGROUND REFINEMENT (hyperaccuracy pass).
///
/// The TVI floor so far used the constant dark-sky background. The real
/// background at the crossing instant varies with airglow (solar
/// activity), zodiacal light, integrated starlight, and - dominantly,
/// when the moon is up - scattered moonlight (Krisciunas & Schaefer
/// 1991): a bright moon near dawn raises the detection floor and
/// physically delays the perceptible fajr al-sadiq. One refinement
/// pass: evaluate the physical background at each crossing time with
/// the actual view geometry, re-float the thresholds, re-determine the
/// crossings, and re-convert the times.
///
/// Returns None when the physical background matches the dark-sky
/// constant within [`REFLOAT_TRIGGER_FRACTION`] (no refloat needed).
fn refloat_on_celestial_background(
    input: &PrayerTimeInput,
    engine: &mut SolarEngine,
    analyses: &[TwilightAnalysis],
    szas: CrossingSzas,
    times: PrayerClockTimes,
) -> Option<CelestialRefloat> {
    let mut night_sky_at = |t_local: Option<f64>| -> Option<f64> {
        let t = t_local?;
        Some(night_sky_total(input, engine, t, VIEW_ZENITH_DEG, None))
    };
    let bgf = night_sky_at(times.fajr)?;
    let bgi = night_sky_at(times.isha_abyad)?;

    // Only re-float when the physical background differs materially from
    // the constant the first pass assumed: moonlit nights, strong
    // airglow, Milky Way pointing, etc.
    let bg0 = threshold::NIGHT_SKY_LUMINANCE;
    if (bgf - bg0).abs() / bg0 <= REFLOAT_TRIGGER_FRACTION
        && (bgi - bg0).abs() / bg0 <= REFLOAT_TRIGGER_FRACTION
    {
        return None;
    }

    let note = format!(
        "Celestial background: fajr-side {:.3e} cd/m^2, isha-side {:.3e} \
         (dark-sky const {:.3e}) - re-floating thresholds.",
        bgf, bgi, bg0
    );
    if input.verbose {
        eprintln!("{}", note);
    }

    let scan_floor = analyses
        .iter()
        .map(|a| a.luminance_mesopic)
        .fold(f64::MAX, f64::min)
        .max(0.0);
    let scan_floor_red = analyses
        .iter()
        .map(|a| a.luminance_red)
        .fold(f64::MAX, f64::min)
        .max(0.0);

    let refined_config = threshold::ThresholdConfig {
        fajr_luminance: threshold::detection_threshold(
            scan_floor + bgf,
            input.threshold_config.fajr_luminance,
        ),
        isha_abyad_luminance: threshold::detection_threshold(
            scan_floor + bgi,
            input.threshold_config.isha_abyad_luminance,
        ),
        isha_ahmar_red_luminance: threshold::detection_threshold(
            scan_floor_red + bgi,
            input.threshold_config.isha_ahmar_red_luminance,
        ),
        ..input.threshold_config.clone()
    };
    let refined = threshold::determine_prayer_times(analyses.to_vec(), &refined_config);
    if refined.fajr_sza_deg.is_none() && refined.isha_abyad_sza_deg.is_none() {
        // Trigger fired but the refloated thresholds found no crossings:
        // keep the first-pass values, still report the note.
        return Some(CelestialRefloat { szas, times, note });
    }
    let new_szas = CrossingSzas {
        fajr: refined.fajr_sza_deg.or(szas.fajr),
        isha_abyad: refined.isha_abyad_sza_deg.or(szas.isha_abyad),
        isha_ahmar: refined.isha_ahmar_sza_deg.or(szas.isha_ahmar),
    };
    let new_times = crossings_to_times(engine, new_szas);
    Some(CelestialRefloat {
        szas: new_szas,
        times: new_times,
        note,
    })
}

/// Convert crossing-SZA sigmas to minutes using the local dt/dSZA slope
/// (about 3-5 min/deg depending on latitude/season).
fn propagate_uncertainty(
    engine: &mut SolarEngine,
    szas: CrossingSzas,
    times: PrayerClockTimes,
    sigmas: CrossingSigmas,
) -> UncertaintyMinutes {
    let mut sigma_minutes = |sza: Option<f64>,
                             t: Option<f64>,
                             sigma: Option<f64>,
                             lo: f64,
                             hi: f64,
                             descending: bool|
     -> Option<f64> {
        let (Some(s0), Some(t0), Some(sig)) = (sza, t, sigma) else {
            return None;
        };
        let t1 = engine.find_zenith_crossing_robust(s0 + 0.25, lo, hi, 0.0001, descending)?;
        let dtdsza = ((t1 - t0) / 0.25).abs() * 60.0; // minutes per degree
        Some(sig * dtdsza)
    };
    UncertaintyMinutes {
        fajr: sigma_minutes(szas.fajr, times.fajr, sigmas.fajr, 0.0, 12.0, true),
        isha_abyad: sigma_minutes(
            szas.isha_abyad,
            times.isha_abyad,
            sigmas.isha_abyad,
            12.0,
            24.0,
            false,
        ),
        isha_ahmar: sigma_minutes(
            szas.isha_ahmar,
            times.isha_ahmar,
            sigmas.isha_ahmar,
            12.0,
            24.0,
            false,
        ),
    }
}

/// Inner pipeline implementation parameterized by scan function.
///
/// Both `compute_prayer_times` (CPU) and `compute_prayer_times_gpu`
/// delegate to this function. The only difference is the `scan` closure
/// that produces `Vec<SpectralResult>` for a given SZA range.
fn compute_prayer_times_inner(
    input: &PrayerTimeInput,
    atm: &twilight_core::atmosphere::AtmosphereModel,
    scan: &ScanFn,
    scan_list: Option<&ScanListFn>,
) -> PrayerTimeOutput {
    let start = std::time::Instant::now();

    // Solar engine (DE440 primary, SPA fallback).
    let (mut engine, ephemeris) = SolarEngine::new(input);

    let sun = sun_events_with_terrain(input, &mut engine);
    let extent = twilight_extent(&mut engine);

    // View direction: toward the sun's azimuth at sunset.
    let solar_azimuth_evening = sun
        .sunset_time
        .and_then(|h| engine.azimuth_at_hour(h))
        .unwrap_or(270.0);

    let config = SimulationConfig {
        latitude: input.latitude,
        longitude: input.longitude,
        elevation: input.elevation,
        solar_azimuth: solar_azimuth_evening,
        view_zenith: VIEW_ZENITH_DEG,
        view_azimuth: None,
        apply_solar_irradiance: true,
        scattering_mode: input.scattering_mode,
        photons_per_wavelength: input.photons_per_wavelength,
        polarized: input.polarized,
        seed_salt: 0,
    };

    // MCRT passes 1 + 2.
    let ScanData {
        results: mut spectral_results,
        se_by_sza,
    } = run_adaptive_scan(input, atm, &config, scan, scan_list, extent.sza_upper);

    if let Some(ref sg) = input.skyglow {
        inject_skyglow(&mut spectral_results, sg);
    }

    // Threshold analysis on the combined high-resolution data (now
    // including skyglow if set).
    let all_analyses: Vec<TwilightAnalysis> = spectral_results
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
    let base_result = threshold::determine_prayer_times(all_analyses, &input.threshold_config);

    let (prayer_result, used_relative_thresholds) =
        apply_high_latitude_relative_mode(input, base_result);

    let (fitted_szas, sigmas) =
        fit_crossings(input, &prayer_result, &se_by_sza, used_relative_thresholds);
    let fitted_times = crossings_to_times(&mut engine, fitted_szas);

    let (final_szas, final_times, celestial_refloat) = match refloat_on_celestial_background(
        input,
        &mut engine,
        &prayer_result.analyses,
        fitted_szas,
        fitted_times,
    ) {
        Some(r) => (r.szas, r.times, Some(r.note)),
        None => (fitted_szas, fitted_times, None),
    };

    let uncertainty = propagate_uncertainty(&mut engine, final_szas, final_times, sigmas);

    // ── THE KHAYT AL-ABYAD PASS (primary criterion) ────────────────
    // Simulate the horizon fan and detect the Quranic contrast events:
    // the white thread distinct from the black with lateral spread
    // (Fajr sadiq), the narrow wedge alone (fajr kadhib), and the
    // mirrored disappearances for Isha (ahmar primary).
    let khayt = match extent.max_sza_deg {
        Some(ms) => khayt_pass(input, atm, &config, scan, scan_list, &mut engine, ms),
        None => KhaytTimes::default(),
    };

    let elapsed = start.elapsed();

    PrayerTimeOutput {
        khayt,
        fajr_time: final_times.fajr,
        isha_abyad_time: final_times.isha_abyad,
        isha_ahmar_time: final_times.isha_ahmar,
        sunrise_time: sun.sunrise_time,
        sunset_time: sun.sunset_time,
        fajr_sza_deg: final_szas.fajr,
        isha_abyad_sza_deg: final_szas.isha_abyad,
        isha_ahmar_sza_deg: final_szas.isha_ahmar,
        fajr_uncertainty_min: uncertainty.fajr,
        isha_abyad_uncertainty_min: uncertainty.isha_abyad,
        isha_ahmar_uncertainty_min: uncertainty.isha_ahmar,
        fajr_depression_deg: final_szas.fajr.map(|s| s - 90.0),
        isha_abyad_depression_deg: final_szas.isha_abyad.map(|s| s - 90.0),
        isha_ahmar_depression_deg: final_szas.isha_ahmar.map(|s| s - 90.0),
        persistent_twilight: extent.persistent_twilight,
        high_latitude_relative_thresholds: used_relative_thresholds,
        max_sza_deg: extent.max_sza_deg,
        twilight_analyses: prayer_result.analyses,
        spectral_results,
        computation_time_ms: elapsed.as_millis() as u64,
        ephemeris,
        sunrise_horizon_deg: sun.sunrise_horizon_deg,
        sunset_horizon_deg: sun.sunset_horizon_deg,
        sunrise_sza_effective: sun.sunrise_sza_effective,
        sunset_sza_effective: sun.sunset_sza_effective,
        terrain_source: input
            .horizon_profile
            .as_ref()
            .map(|p| p.source_name.clone()),
        skyglow_zenith_mcd: input.skyglow.as_ref().map(|sg| {
            twilight_skyglow::bortle::radiance_to_zenith_luminance(sg.integrated_radiance)
        }),
        skyglow_bortle: input.skyglow.as_ref().map(|sg| sg.bortle_class),
        skyglow_shift_minutes: input.skyglow.as_ref().map(|sg| {
            let lum =
                twilight_skyglow::bortle::radiance_to_zenith_luminance(sg.integrated_radiance);
            twilight_skyglow::bortle::estimated_prayer_shift_minutes(lum)
        }),
        celestial_refloat,
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
    let lo = lo.max(SCAN_FLOOR_SZA);
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
    if !(0.0..=24.0).contains(&h) {
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

    // ── SolarEngine date/hour handling ──

    #[test]
    fn next_civil_day_rollovers() {
        assert_eq!(SolarEngine::next_civil_day(2024, 6, 15), (2024, 6, 16));
        assert_eq!(SolarEngine::next_civil_day(2024, 4, 30), (2024, 5, 1));
        assert_eq!(SolarEngine::next_civil_day(2024, 12, 31), (2025, 1, 1));
        assert_eq!(SolarEngine::next_civil_day(2024, 2, 28), (2024, 2, 29)); // leap
        assert_eq!(SolarEngine::next_civil_day(2023, 2, 28), (2023, 3, 1));
        assert_eq!(SolarEngine::next_civil_day(2100, 2, 28), (2100, 3, 1)); // century non-leap
    }

    /// Past-midnight hours must address the NEXT civil day: azimuth at
    /// 25.5h on day N equals azimuth at 1.5h on day N+1 (SPA path; the
    /// DE440 path is covered in twilight-solar).
    #[test]
    fn azimuth_at_hour_rolls_past_midnight() {
        let input = PrayerTimeInput {
            latitude: 59.91,
            longitude: 10.75,
            year: 2024,
            month: 6,
            day: 1,
            timezone: 2.0,
            ..PrayerTimeInput::default()
        };
        let (mut engine, _) = SolarEngine::new(&input);
        let rolled = engine.azimuth_at_hour(25.5).expect("azimuth");

        let next_day = PrayerTimeInput {
            day: 2,
            ..input.clone()
        };
        let (mut engine_next, _) = SolarEngine::new(&next_day);
        let direct = engine_next.azimuth_at_hour(1.5).expect("azimuth");

        assert!(
            (rolled - direct).abs() < 1e-9,
            "rolled {rolled} vs direct {direct}"
        );
    }

    // ── Polar day (midnight sun) - regression for the empty-scan panic ──

    /// Tromsø (69.6°N) at the June solstice: the sun never sets, the coarse
    /// scan never starts, and the pipeline previously panicked with
    /// `index out of bounds: the len is 0` in determine_prayer_times.
    /// It must instead return Fajr/Isha = None with persistent_twilight set.
    #[test]
    fn polar_day_returns_none_instead_of_panicking() {
        let input = PrayerTimeInput {
            latitude: 69.6492,
            longitude: 18.9553,
            year: 2024,
            month: 6,
            day: 21,
            timezone: 2.0,
            scattering_mode: ScatteringMode::Single,
            ..PrayerTimeInput::default()
        };
        let out = compute_prayer_times(&input);
        assert!(out.fajr_time.is_none(), "no Fajr under the midnight sun");
        assert!(
            out.isha_abyad_time.is_none(),
            "no Isha under the midnight sun"
        );
        assert!(
            out.persistent_twilight || out.max_sza_deg.map(|s| s < 90.0).unwrap_or(true),
            "polar day should be flagged: persistent_twilight={} max_sza={:?}",
            out.persistent_twilight,
            out.max_sza_deg
        );
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
