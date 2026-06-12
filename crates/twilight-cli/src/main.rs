use clap::{Parser, Subcommand, ValueEnum};
use twilight_cpu::pipeline::{self, PrayerTimeInput};
use twilight_cpu::simulation::{self, ScatteringMode, SimulationConfig, SpectralResult};
use twilight_data::aerosol::AerosolType;
use twilight_data::atmosphere_profiles::AtmosphereType;
use twilight_data::builder;
use twilight_data::cloud::CloudType;
use twilight_solar::de440::De440;
use twilight_solar::spa::{self, SpaInput};
use twilight_terrain::horizon;
use twilight_threshold::threshold::TwilightColor;

/// Twilight — Monte Carlo Radiative Transfer engine for Fajr/Isha prayer times.
#[derive(Parser)]
#[command(name = "twilight")]
#[command(about = "Compute solar position and twilight times for any location and date")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Render a sky map at a specific solar zenith angle
    Render {
        /// Solar Zenith Angle in degrees
        #[arg(long, default_value = "96.0")]
        sza: f64,
        /// Image width
        #[arg(long, default_value = "800")]
        width: u32,
        /// Image height
        #[arg(long, default_value = "400")]
        height: u32,
        /// Number of rays per pixel per wavelength
        #[arg(short, long, default_value = "50")]
        rays: usize,
        /// Output filename
        #[arg(long, default_value = "sky_96.png")]
        out: String,
    },
    /// Show solar position and conventional twilight times
    Solar {
        /// Latitude in degrees (north positive)
        #[arg(short, long)]
        lat: f64,
        /// Longitude in degrees (east positive)
        #[arg(short = 'n', long)]
        lon: f64,
        /// Date in YYYY-MM-DD format
        #[arg(short, long)]
        date: String,
        /// Timezone offset from UTC (hours)
        #[arg(short, long, default_value = "0")]
        tz: f64,
        /// Elevation above sea level in meters
        #[arg(short, long, default_value = "0")]
        elevation: f64,
        /// Delta T (TT - UT1) in seconds
        #[arg(long, default_value = "69.184")]
        delta_t: f64,
        /// Path to DE440 BSP file for JPL ephemeris comparison
        #[arg(long)]
        de440: Option<String>,
    },
    /// Run MCRT simulation across twilight solar zenith angles
    Mcrt {
        /// Latitude in degrees (north positive)
        #[arg(short, long)]
        lat: f64,
        /// Longitude in degrees (east positive)
        #[arg(short = 'n', long)]
        lon: f64,
        /// Start solar zenith angle (degrees)
        #[arg(long, default_value = "90")]
        sza_start: f64,
        /// End solar zenith angle (degrees)
        #[arg(long, default_value = "108")]
        sza_end: f64,
        /// SZA step size (degrees)
        #[arg(long, default_value = "2")]
        sza_step: f64,
        /// Number of secondary rays per wavelength per step (hybrid/MC modes)
        #[arg(short, long, default_value = "100")]
        photons: usize,
        /// Surface albedo (0-1)
        #[arg(long, default_value = "0.15")]
        albedo: f64,
        /// Solar azimuth angle (degrees, 0=north, 270=west for Isha)
        #[arg(long, default_value = "270")]
        solar_azimuth: f64,
        /// View zenith angle (degrees from straight up)
        #[arg(long, default_value = "75")]
        view_zenith: f64,
        /// Aerosol type (default: none = clear sky)
        #[arg(long, value_enum, default_value = "none")]
        aerosol: CliAerosol,
        /// Cloud type (default: none = clear sky)
        #[arg(long, value_enum, default_value = "none")]
        cloud: CliCloud,
        /// Scattering mode: single, mc, or hybrid (default: hybrid)
        #[arg(long, value_enum, default_value = "hybrid")]
        scattering: CliScattering,
        /// Fetch live weather data from Open-Meteo (overrides --aerosol and --cloud)
        #[arg(long)]
        weather: bool,
        /// Force CPU-only computation (skip GPU)
        #[arg(long)]
        cpu: bool,
        /// Preferred GPU backend: metal (auto-detect if omitted)
        #[arg(long, value_enum)]
        gpu_backend: Option<CliGpuBackend>,
        /// Scalar radiance mode (skip Stokes polarization tracking).
        /// Default is full Stokes [I,Q,U,V]. Use --fast for ~10% speedup
        /// at the cost of ~0.5-2% polarization correction to intensity.
        #[arg(long)]
        fast: bool,
    },
    /// Compute physically-based Fajr and Isha prayer times using MCRT
    Pray {
        /// Latitude in degrees (north positive)
        #[arg(short, long)]
        lat: f64,
        /// Longitude in degrees (east positive)
        #[arg(short = 'n', long)]
        lon: f64,
        /// Date in YYYY-MM-DD format
        #[arg(short, long)]
        date: String,
        /// Timezone offset from UTC (hours)
        #[arg(short, long, default_value = "0")]
        tz: f64,
        /// Elevation above sea level in meters
        #[arg(short, long, default_value = "0")]
        elevation: f64,
        /// Surface albedo (0-1)
        #[arg(long, default_value = "0.15")]
        albedo: f64,
        /// Delta T (TT - UT1) in seconds
        #[arg(long, default_value = "69.184")]
        delta_t: f64,
        /// SZA scan resolution in degrees (smaller = more accurate, slower)
        #[arg(long, default_value = "0.5")]
        sza_step: f64,
        /// Aerosol type (default: none = clear sky)
        #[arg(long, value_enum, default_value = "none")]
        aerosol: CliAerosol,
        /// Cloud type (default: none = clear sky)
        #[arg(long, value_enum, default_value = "none")]
        cloud: CliCloud,
        /// Path to DE440 BSP file for JPL ephemeris (primary engine)
        #[arg(long)]
        de440: Option<String>,
        /// Scattering mode: single, mc, or hybrid (default: hybrid)
        #[arg(long, value_enum, default_value = "hybrid")]
        scattering: CliScattering,
        /// Number of secondary rays per wavelength per step (hybrid/MC modes)
        #[arg(short, long, default_value = "100")]
        photons: usize,
        /// Show detailed twilight analysis
        #[arg(long)]
        verbose: bool,
        /// Fetch live weather data from Open-Meteo (overrides --aerosol and --cloud)
        #[arg(long)]
        weather: bool,
        /// Enable terrain masking using digital elevation data.
        /// Downloads Copernicus GLO-30 (30m) tiles on first use.
        #[arg(long)]
        terrain: bool,
        /// Cache directory for DEM tiles (default: data/dem)
        #[arg(long, default_value = "data/dem")]
        dem_dir: String,
        /// Horizon scan radius in km (default: 30)
        #[arg(long, default_value = "30")]
        horizon_radius: f64,
        /// Enable light pollution skyglow model.
        /// Adds artificial sky brightness to MCRT luminance, shifting prayer times.
        #[arg(long)]
        skyglow: bool,
        /// Bortle dark-sky class (1-9). Alternative to --radiance.
        /// 1=pristine, 5=suburban, 8=city, 9=inner city.
        #[arg(long, value_parser = clap::value_parser!(u8).range(1..=9))]
        bortle: Option<u8>,
        /// VIIRS nighttime radiance at the observer (nW/cm^2/sr).
        /// Use instead of --bortle for precise input.
        #[arg(long)]
        radiance: Option<f64>,
        /// LED fraction of local lighting (0.0 = all HPS sodium, 1.0 = all LED).
        /// Default 0.5 (typical mixed modern city).
        #[arg(long, default_value = "0.5")]
        led_fraction: f64,
        /// Force CPU-only computation (skip GPU)
        #[arg(long)]
        cpu: bool,
        /// Preferred GPU backend: metal (auto-detect if omitted)
        #[arg(long, value_enum)]
        gpu_backend: Option<CliGpuBackend>,
        /// Scalar radiance mode (skip Stokes polarization tracking).
        /// Default is full Stokes [I,Q,U,V]. Use --fast for ~10% speedup
        /// at the cost of ~0.5-2% polarization correction to intensity.
        #[arg(long)]
        fast: bool,
    },
    /// Emit machine-readable spectral radiance for external RT comparison
    /// (e.g. libRadtran/uvspec). CSV to stdout:
    /// sza_deg,view_zenith_deg,rel_azimuth_deg,wavelength_nm,radiance_w_m2_sr_nm
    Compare {
        /// Latitude in degrees (north positive)
        #[arg(short, long, default_value = "21.4225")]
        lat: f64,
        /// Longitude in degrees (east positive)
        #[arg(short = 'n', long, default_value = "39.8262")]
        lon: f64,
        /// Observer elevation above sea level (meters)
        #[arg(short, long, default_value = "0")]
        elevation: f64,
        /// Solar zenith angles in degrees (comma-separated)
        #[arg(long, value_delimiter = ',', default_value = "90,92,94,96,98,100,102,104,106,108")]
        sza: Vec<f64>,
        /// View zenith angles in degrees from straight up (comma-separated)
        #[arg(long, value_delimiter = ',', default_value = "0")]
        view_zenith: Vec<f64>,
        /// Relative azimuth view-minus-sun in degrees (comma-separated;
        /// 0 = principal plane toward the sun, 180 = anti-solar)
        #[arg(long, value_delimiter = ',', default_value = "0")]
        rel_azimuth: Vec<f64>,
        /// Solar azimuth angle (degrees, 0=north, clockwise)
        #[arg(long, default_value = "270")]
        solar_azimuth: f64,
        /// Surface albedo (0-1)
        #[arg(long, default_value = "0.15")]
        albedo: f64,
        /// Pure Rayleigh atmosphere (no gas absorption, no aerosol, no cloud).
        /// This is the Tier-1 geometry/phase/optics check vs DISORT.
        #[arg(long)]
        rayleigh_only: bool,
        /// Aerosol type (ignored with --rayleigh-only)
        #[arg(long, value_enum, default_value = "none")]
        aerosol: CliAerosol,
        /// Cloud type (ignored with --rayleigh-only)
        #[arg(long, value_enum, default_value = "none")]
        cloud: CliCloud,
        /// Override O3 total column in Dobson Units (matched to mol_modify O3)
        #[arg(long)]
        o3_du: Option<f64>,
        /// Scattering mode (default: single — deterministic, f64, the
        /// apples-to-apples baseline vs DISORT single-scattering output)
        #[arg(long, value_enum, default_value = "single")]
        scattering: CliScattering,
        /// Secondary rays per LOS step (hybrid/MC modes)
        #[arg(short, long, default_value = "10000")]
        photons: usize,
        /// Scalar radiance mode (skip Stokes polarization)
        #[arg(long)]
        fast: bool,
        /// Disable atmospheric refraction (for apples-to-apples comparison
        /// against RT codes that trace straight shadow rays, e.g. MYSTIC)
        #[arg(long)]
        no_refraction: bool,
    },
}

/// CLI aerosol type selector.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliAerosol {
    /// No aerosols (clear sky)
    None,
    /// Rural/background continental
    ContinentalClean,
    /// Moderate continental
    ContinentalAverage,
    /// Urban/industrial (high soot)
    Urban,
    /// Open ocean sea salt
    MaritimeClean,
    /// Coastal/shipping lane
    MaritimePolluted,
    /// Mineral dust
    Desert,
}

impl CliAerosol {
    fn to_aerosol_type(self) -> Option<AerosolType> {
        match self {
            CliAerosol::None => Option::None,
            CliAerosol::ContinentalClean => Some(AerosolType::ContinentalClean),
            CliAerosol::ContinentalAverage => Some(AerosolType::ContinentalAverage),
            CliAerosol::Urban => Some(AerosolType::Urban),
            CliAerosol::MaritimeClean => Some(AerosolType::MaritimeClean),
            CliAerosol::MaritimePolluted => Some(AerosolType::MaritimePolluted),
            CliAerosol::Desert => Some(AerosolType::Desert),
        }
    }
}

/// CLI cloud type selector.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliCloud {
    /// No cloud (clear sky)
    None,
    /// Thin high-altitude ice cloud
    ThinCirrus,
    /// Thick high-altitude ice cloud
    ThickCirrus,
    /// Mid-level overcast
    Altostratus,
    /// Low grey overcast
    Stratus,
    /// Low lumpy cloud sheet
    Stratocumulus,
    /// Fair-weather puffy clouds
    Cumulus,
}

impl CliCloud {
    fn to_cloud_type(self) -> Option<CloudType> {
        match self {
            CliCloud::None => Option::None,
            CliCloud::ThinCirrus => Some(CloudType::ThinCirrus),
            CliCloud::ThickCirrus => Some(CloudType::ThickCirrus),
            CliCloud::Altostratus => Some(CloudType::Altostratus),
            CliCloud::Stratus => Some(CloudType::Stratus),
            CliCloud::Stratocumulus => Some(CloudType::Stratocumulus),
            CliCloud::Cumulus => Some(CloudType::Cumulus),
        }
    }
}

/// CLI scattering mode selector.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliScattering {
    /// Deterministic single-scattering (fast, no noise)
    Single,
    /// Monte Carlo multiple scattering (all orders, noisy)
    Multiple,
    /// Hybrid: exact single-scatter + MC for orders 2+ (recommended)
    Hybrid,
}

impl CliScattering {
    fn to_scattering_mode(self) -> ScatteringMode {
        match self {
            CliScattering::Single => ScatteringMode::Single,
            CliScattering::Multiple => ScatteringMode::Multiple,
            CliScattering::Hybrid => ScatteringMode::Hybrid,
        }
    }
}

/// CLI GPU backend selector.
#[derive(Debug, Clone, Copy, ValueEnum)]
enum CliGpuBackend {
    /// Apple Metal (macOS / iOS) — the only implemented GPU backend
    Metal,
}

/// Try to initialize a GPU backend. Returns the backend or prints a
/// warning and returns None (caller should fall back to CPU).
#[cfg(feature = "gpu")]
fn try_init_gpu(
    preferred: Option<CliGpuBackend>,
    photons: usize,
) -> Option<Box<dyn twilight_gpu::GpuBackend>> {
    let preferred_backend = preferred.map(|b| match b {
        CliGpuBackend::Metal => twilight_gpu::BackendKind::Metal,
    });

    let config = twilight_gpu::GpuConfig {
        preferred_backend,
        photons_per_wavelength: photons as u32,
        secondary_rays_per_step: photons as u32,
        ..twilight_gpu::GpuConfig::default()
    };

    match twilight_gpu::try_init(&config) {
        Ok(backend) => {
            let info = backend.device_info();
            if info.memory_bytes > 0 {
                println!(
                    "GPU:        {} ({}, {:.0} MB)",
                    info.name,
                    info.backend,
                    info.memory_bytes as f64 / (1024.0 * 1024.0),
                );
            } else {
                println!("GPU:        {} ({})", info.name, info.backend);
            }
            Some(backend)
        }
        Err(e) => {
            eprintln!("Warning: GPU init failed ({}), falling back to CPU", e);
            None
        }
    }
}

/// Conventional solar depression angles for twilight boundaries.
struct TwilightAngle {
    name: &'static str,
    zenith: f64,
}

const TWILIGHT_ANGLES: &[TwilightAngle] = &[
    TwilightAngle {
        name: "Sunrise/Sunset",
        zenith: 90.8333,
    },
    TwilightAngle {
        name: "Civil twilight",
        zenith: 96.0,
    },
    TwilightAngle {
        name: "Nautical twilight",
        zenith: 102.0,
    },
    TwilightAngle {
        name: "Astronomical twilight",
        zenith: 108.0,
    },
    TwilightAngle {
        name: "Fajr (18° MWL/ISNA)",
        zenith: 108.0,
    },
    TwilightAngle {
        name: "Fajr (15° Egypt/UOIF)",
        zenith: 105.0,
    },
    TwilightAngle {
        name: "Fajr (19.5° Umm al-Qura)",
        zenith: 109.5,
    },
    TwilightAngle {
        name: "Isha (17° MWL)",
        zenith: 107.0,
    },
    TwilightAngle {
        name: "Isha (17.5° Egypt)",
        zenith: 107.5,
    },
    TwilightAngle {
        name: "Isha (18° ISNA)",
        zenith: 108.0,
    },
];


/// Format a 1-sigma uncertainty in minutes as " +/-N.Nmin" (empty when
/// not available, e.g. deterministic runs).
fn format_uncertainty(min: Option<f64>) -> String {
    match min {
        Some(m) if m >= 0.05 => format!(" \u{00b1}{:.1}min", m),
        Some(_) => " \u{00b1}<0.1min".to_string(),
        None => String::new(),
    }
}

fn format_fractional_hour(h: f64) -> String {
    if !(0.0..=48.0).contains(&h) {
        return "N/A".to_string();
    }
    // Hours >= 24 are past-midnight events (high-latitude Isha can fall on
    // the next civil day) — display wrapped with a +1d marker.
    let (h, next_day) = if h >= 24.0 { (h - 24.0, true) } else { (h, false) };
    let hours = h as u32;
    let minutes = ((h - hours as f64) * 60.0) as u32;
    let seconds = ((h - hours as f64 - minutes as f64 / 60.0) * 3600.0) as u32;
    if next_day {
        format!("{:02}:{:02}:{:02} (+1d)", hours, minutes, seconds)
    } else {
        format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
    }
}

fn cmd_solar(
    lat: f64,
    lon: f64,
    date: &str,
    tz: f64,
    elevation: f64,
    delta_t: f64,
    de440_path: Option<&str>,
) {
    let parts: Vec<&str> = date.split('-').collect();
    if parts.len() != 3 {
        eprintln!("Error: date must be in YYYY-MM-DD format");
        std::process::exit(1);
    }
    let year: i32 = parts[0].parse().unwrap_or(2024);
    let month: i32 = parts[1].parse().unwrap_or(1);
    let day: i32 = parts[2].parse().unwrap_or(1);

    println!("Twilight Solar Position Calculator");
    println!("==================================");
    println!("Date:      {}-{:02}-{:02}", year, month, day);
    println!("Location:  {:.4}°N, {:.4}°E", lat, lon);
    println!("Elevation: {:.0} m", elevation);
    println!("Timezone:  UTC{:+.1}", tz);

    let ephemeris_label = if de440_path.is_some() {
        "JPL DE440 + SPA"
    } else {
        "NREL SPA"
    };
    println!("Ephemeris: {}", ephemeris_label);
    println!();

    let noon_input = SpaInput {
        year,
        month,
        day,
        hour: 12,
        minute: 0,
        second: 0,
        timezone: tz,
        latitude: lat,
        longitude: lon,
        elevation,
        pressure: 1013.25,
        temperature: 15.0,
        delta_t,
        slope: 0.0,
        azm_rotation: 0.0,
        atmos_refract: 0.5667,
    };

    match spa::solar_position(&noon_input) {
        Ok(noon) => {
            println!("Solar Position at Local Noon (SPA):");
            println!("  Zenith:       {:.4}deg", noon.zenith);
            println!("  Azimuth:      {:.4}deg", noon.azimuth);
            println!("  Declination:  {:.4}deg", noon.delta);
            println!("  Earth-Sun:    {:.6} AU", noon.r);
            println!("  Eq. of Time:  {:.2} min", noon.eot);

            // DE440 comparison if available
            if let Some(path) = de440_path {
                match De440::open(path) {
                    Ok(mut de) => {
                        // Convert local noon to UTC
                        let utc_hour = (12.0 - tz) as i32;
                        let utc_minute = (((12.0 - tz) - utc_hour as f64) * 60.0) as i32;
                        match de.solar_position(
                            year, month, day, utc_hour, utc_minute, 0, delta_t, lat, lon, elevation,
                        ) {
                            Ok(topo) => {
                                println!();
                                println!("Solar Position at Local Noon (DE440):");
                                println!("  Zenith:       {:.4}deg", topo.zenith);
                                println!("  Azimuth:      {:.4}deg", topo.azimuth);
                                println!("  RA:           {:.4}deg", topo.right_ascension);
                                println!("  Dec:          {:.4}deg", topo.declination);
                                println!("  Distance:     {:.0} km", topo.distance_km);
                                println!();
                                println!("DE440 vs SPA difference:");
                                println!("  Zenith:  {:.6}deg", (topo.zenith - noon.zenith).abs());
                                println!(
                                    "  Azimuth: {:.6}deg",
                                    (topo.azimuth - noon.azimuth).abs()
                                );
                            }
                            Err(e) => {
                                eprintln!("Warning: DE440 query failed: {}", e);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Warning: failed to open DE440 file: {}", e);
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("Error: {:?}", e);
            std::process::exit(1);
        }
    }

    println!();
    println!("Twilight Times (conventional fixed-angle):");
    println!("{:-<60}", "");
    println!("{:<35} {:>10}  {:>10}", "Event", "Morning", "Evening");
    println!("{:-<60}", "");

    let base_input = SpaInput {
        year,
        month,
        day,
        hour: 0,
        minute: 0,
        second: 0,
        timezone: tz,
        latitude: lat,
        longitude: lon,
        elevation,
        pressure: 1013.25,
        temperature: 15.0,
        delta_t,
        slope: 0.0,
        azm_rotation: 0.0,
        atmos_refract: 0.5667,
    };

    for angle in TWILIGHT_ANGLES {
        let morning = spa::find_zenith_crossing(&base_input, angle.zenith, 0.0, 12.0, 0.0001);
        let evening = spa::find_zenith_crossing(&base_input, angle.zenith, 12.0, 24.0, 0.0001);

        let morning_str = morning
            .map(format_fractional_hour)
            .unwrap_or("N/A".to_string());
        let evening_str = evening
            .map(format_fractional_hour)
            .unwrap_or("N/A".to_string());

        println!(
            "{:<35} {:>10}  {:>10}",
            angle.name, morning_str, evening_str
        );
    }

    println!();
    println!("Note: These are CONVENTIONAL times using fixed solar depression angles.");
    println!("      Use 'twilight pray' to compute physically-based times.");
    if de440_path.is_none() {
        println!("      Use --de440 <path> to enable JPL DE440 ephemeris comparison.");
    }
}

#[allow(clippy::too_many_arguments)] // CLI dispatch: all params come from parsed command-line args
fn cmd_mcrt(
    lat: f64,
    lon: f64,
    sza_start: f64,
    sza_end: f64,
    sza_step: f64,
    photons: usize,
    albedo: f64,
    solar_azimuth: f64,
    view_zenith: f64,
    aerosol: CliAerosol,
    cloud: CliCloud,
    scattering: CliScattering,
    use_weather: bool,
    force_cpu: bool,
    gpu_backend: Option<CliGpuBackend>,
    fast: bool,
) {
    // Suppress unused warnings when gpu feature is disabled
    #[cfg(not(feature = "gpu"))]
    let _ = gpu_backend;

    println!("Twilight MCRT Simulation");
    println!("=======================");
    println!("Location:     {:.4}°N, {:.4}°E", lat, lon);
    println!(
        "SZA range:    {:.1}° to {:.1}° (step {:.1}°)",
        sza_start, sza_end, sza_step
    );
    println!("Photons/λ:    {}", photons);
    println!("Wavelengths:  380-780 nm (41 bands, 10nm steps)");

    // Resolve atmosphere: weather API or manual flags
    let (aerosol_props, cloud_props, gas_composition, atm_desc) =
        if use_weather {
            match twilight_weather::fetch_atmospheric_params(lat, lon) {
                Ok(params) => {
                    println!("Weather:      {}", params.description);
                    let c = &params.conditions;
                    println!(
                    "  API data:   AOD={:.2}, cloud={:.0}% (L:{:.0}/M:{:.0}/H:{:.0}), vis={:.0}m",
                    c.aod_550, c.cloud_cover_total, c.cloud_cover_low,
                    c.cloud_cover_mid, c.cloud_cover_high, c.visibility_m
                );
                    if c.ozone_ug_m3 > 0.0 || c.nitrogen_dioxide_ug_m3 > 0.0 {
                        println!(
                            "  Gas data:   O3={:.0} ug/m3, NO2={:.0} ug/m3",
                            c.ozone_ug_m3, c.nitrogen_dioxide_ug_m3
                        );
                    }
                    (
                        params.aerosol,
                        params.cloud,
                        params.gas_composition,
                        format!("Live weather: {}", params.description),
                    )
                }
                Err(e) => {
                    eprintln!("Warning: failed to fetch weather: {}", e);
                    eprintln!("Falling back to clear sky.");
                    (
                        None,
                        None,
                        None,
                        "US Standard 1976 (clear sky, weather fetch failed)".to_string(),
                    )
                }
            }
        } else {
            let aerosol_type = aerosol.to_aerosol_type();
            let cloud_type = cloud.to_cloud_type();
            let ap = aerosol_type.map(twilight_data::aerosol::default_properties);
            let cp = cloud_type.map(twilight_data::cloud::default_properties);
            let desc = format_atm_desc(aerosol_type, cloud_type);
            (ap, cp, None, desc)
        };

    println!("Atmosphere:   {}", atm_desc);
    println!("Surface:      albedo = {:.2}", albedo);
    let scattering_mode = scattering.to_scattering_mode();
    let polarized = !fast;
    let mode_str = match scattering_mode {
        ScatteringMode::Single => "Single-scatter (deterministic)".to_string(),
        ScatteringMode::Multiple => format!("Multiple-scatter MC ({} photons/wl)", photons),
        ScatteringMode::Hybrid => format!("Hybrid SS+MC ({} secondary rays/step)", photons),
    };
    println!("Scattering:   {}", mode_str);
    let pol_str = if polarized {
        "Stokes [I,Q,U,V]"
    } else {
        "Scalar (--fast)"
    };
    println!("Polarization: {}", pol_str);
    println!(
        "View:         zenith {:.0}°, azimuth {:.0}°",
        view_zenith, solar_azimuth
    );
    println!();

    // Build atmosphere
    let atm = if let Some(ref gc) = gas_composition {
        builder::build_full_with_gas(
            AtmosphereType::UsStandard,
            albedo,
            aerosol_props.as_ref(),
            cloud_props.as_ref(),
            gc.o3_column_du,
            gc.no2_surface_density,
        )
    } else {
        builder::build_full(
            AtmosphereType::UsStandard,
            albedo,
            aerosol_props.as_ref(),
            cloud_props.as_ref(),
        )
    };

    let config = SimulationConfig {
        latitude: lat,
        longitude: lon,
        elevation: 0.0,
        solar_azimuth,
        view_zenith,
        view_azimuth: None,
        apply_solar_irradiance: true,
        scattering_mode,
        photons_per_wavelength: photons,
        polarized,
        seed_salt: 0,
    };

    // GPU initialization (default unless --cpu is passed).
    // TODO: GPU dispatch currently sends one SZA at a time synchronously.
    // Batching all SZA points into a single dispatch would significantly
    // improve GPU throughput for the prayer pipeline.
    #[cfg(feature = "gpu")]
    let mut gpu_backend = if force_cpu {
        None
    } else {
        try_init_gpu(gpu_backend, photons)
    };

    let compute_label = {
        #[cfg(feature = "gpu")]
        {
            if gpu_backend.is_some() {
                // Honest label: a backend initialized, but individual
                // kernels may still fall back to CPU on dispatch failure.
                "GPU (Metal; per-kernel CPU fallback)"
            } else {
                "CPU (rayon)"
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = force_cpu;
            "CPU (rayon)"
        }
    };
    println!("Compute:      {}", compute_label);
    println!("Running MCRT ({})...", mode_str);
    println!();

    let start = std::time::Instant::now();

    // GPU path: upload atmosphere and dispatch via GPU
    #[cfg(feature = "gpu")]
    let results = if let Some(ref mut gpu) = gpu_backend {
        match gpu.upload_atmosphere(&atm) {
            Ok(()) => twilight_cpu::gpu_dispatch::simulate_twilight_scan_gpu(
                gpu.as_ref(),
                &atm,
                &config,
                sza_start,
                sza_end,
                sza_step,
            )
            .unwrap_or_else(|e| {
                eprintln!("Warning: GPU dispatch failed ({}), falling back to CPU", e);
                simulation::simulate_twilight_scan(&atm, &config, sza_start, sza_end, sza_step)
            }),
            Err(e) => {
                eprintln!("Warning: GPU upload failed ({}), falling back to CPU", e);
                simulation::simulate_twilight_scan(&atm, &config, sza_start, sza_end, sza_step)
            }
        }
    } else {
        simulation::simulate_twilight_scan(&atm, &config, sza_start, sza_end, sza_step)
    };

    #[cfg(not(feature = "gpu"))]
    let results = simulation::simulate_twilight_scan(&atm, &config, sza_start, sza_end, sza_step);

    let elapsed = start.elapsed();

    // Print spectral results table
    println!("Results (radiance in W/m²/sr/nm):");
    println!("{:-<80}", "");
    println!(
        "{:>6}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
        "SZA°", "Total", "Blue(450)", "Green(550)", "Red(650)", "DeepRed(700)"
    );
    println!("{:-<80}", "");

    for result in &results {
        let total = simulation::total_radiance(result);

        // Extract specific wavelength channels
        let blue = get_radiance_at_wl(result, 450.0);
        let green = get_radiance_at_wl(result, 550.0);
        let red = get_radiance_at_wl(result, 650.0);
        let deep_red = get_radiance_at_wl(result, 700.0);

        println!(
            "{:>6.1}  {:>12.6e}  {:>12.6e}  {:>12.6e}  {:>12.6e}  {:>12.6e}",
            result.sza_deg, total, blue, green, red, deep_red
        );
    }

    println!("{:-<80}", "");

    // Print luminance analysis
    println!();
    println!("Luminance Analysis:");
    println!("{:-<90}", "");
    println!(
        "{:>6}  {:>12}  {:>12}  {:>12}  {:>10}  {:>12}",
        "SZA°", "L_photopic", "L_scotopic", "L_mesopic", "Centroid", "Color"
    );
    println!(
        "{:>6}  {:>12}  {:>12}  {:>12}  {:>10}  {:>12}",
        "", "(cd/m²)", "(cd/m²)", "(cd/m²)", "(nm)", ""
    );
    println!("{:-<90}", "");

    let threshold_config = twilight_threshold::threshold::ThresholdConfig::default();
    for result in &results {
        let analysis = twilight_threshold::threshold::analyze_twilight(
            result.sza_deg,
            &result.wavelengths_nm,
            &result.radiance,
            &threshold_config,
        );

        let color_str = match analysis.color {
            TwilightColor::Blue => "Blue",
            TwilightColor::White => "White (abyad)",
            TwilightColor::Orange => "Orange",
            TwilightColor::Red => "Red (ahmar)",
            TwilightColor::Dark => "Dark",
        };

        println!(
            "{:>6.1}  {:>12.6e}  {:>12.6e}  {:>12.6e}  {:>10.1}  {:>12}",
            analysis.sza_deg,
            analysis.luminance_photopic,
            analysis.luminance_scotopic,
            analysis.luminance_mesopic,
            analysis.spectral_centroid_nm,
            color_str,
        );
    }
    println!("{:-<90}", "");

    println!();
    println!("Simulation completed in {:.2?}", elapsed);

    let total_photons = photons * atm.num_wavelengths * results.len();
    println!(
        "Total photons traced: {} ({:.1}M)",
        total_photons,
        total_photons as f64 / 1e6
    );
    println!(
        "Throughput: {:.1}M photons/sec",
        total_photons as f64 / elapsed.as_secs_f64() / 1e6
    );
}

/// Emit machine-readable spectral radiance for external RT comparison.
///
/// Runs the CPU engine (f64) at each (sza, view_zenith, rel_azimuth) grid
/// point and prints CSV rows in physical units [W/m^2/sr/nm]. Designed to be
/// consumed by tools/validate_libradtran.py, which generates matched uvspec
/// decks and compares the two codes per wavelength.
#[allow(clippy::too_many_arguments)] // CLI dispatch: all params come from parsed command-line args
fn cmd_compare(
    lat: f64,
    lon: f64,
    elevation: f64,
    szas: &[f64],
    view_zeniths: &[f64],
    rel_azimuths: &[f64],
    solar_azimuth: f64,
    albedo: f64,
    rayleigh_only: bool,
    aerosol: CliAerosol,
    cloud: CliCloud,
    o3_du: Option<f64>,
    scattering: CliScattering,
    photons: usize,
    fast: bool,
    no_refraction: bool,
) {
    // Build the atmosphere once.
    let mut atm = if rayleigh_only {
        builder::build_clear_sky(AtmosphereType::UsStandard, albedo)
    } else {
        let ap = aerosol
            .to_aerosol_type()
            .map(twilight_data::aerosol::default_properties);
        let cp = cloud
            .to_cloud_type()
            .map(twilight_data::cloud::default_properties);
        builder::build_full_with_gas(
            AtmosphereType::UsStandard,
            albedo,
            ap.as_ref(),
            cp.as_ref(),
            o3_du,
            None,
        )
    };
    if no_refraction {
        // Straight-ray comparison mode (e.g. vs MYSTIC without refraction).
        for n in atm.refractive_index.iter_mut() {
            *n = 1.0;
        }
    }

    // Header with enough metadata to reproduce the run.
    println!(
        "# twilight compare: lat={} lon={} elev={} albedo={} rayleigh_only={} o3_du={:?} scattering={:?} photons={} polarized={}",
        lat, lon, elevation, albedo, rayleigh_only, o3_du, scattering.to_scattering_mode(), photons, !fast
    );
    println!("sza_deg,view_zenith_deg,rel_azimuth_deg,wavelength_nm,radiance_w_m2_sr_nm");

    for &vz in view_zeniths {
        for &ra in rel_azimuths {
            let config = SimulationConfig {
                latitude: lat,
                longitude: lon,
                elevation,
                solar_azimuth,
                view_zenith: vz,
                view_azimuth: Some(solar_azimuth + ra),
                apply_solar_irradiance: true,
                scattering_mode: scattering.to_scattering_mode(),
                photons_per_wavelength: photons,
                polarized: !fast,
                seed_salt: 0,
            };
            for &sza in szas {
                let result = simulation::simulate_at_sza(&atm, &config, sza);
                for (wl, rad) in result.wavelengths_nm.iter().zip(result.radiance.iter()) {
                    println!("{},{},{},{},{:e}", sza, vz, ra, wl, rad);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)] // CLI dispatch: all params come from parsed command-line args
fn cmd_pray(
    lat: f64,
    lon: f64,
    date: &str,
    tz: f64,
    elevation: f64,
    albedo: f64,
    delta_t: f64,
    sza_step: f64,
    aerosol: CliAerosol,
    cloud: CliCloud,
    de440_path: Option<&str>,
    scattering: CliScattering,
    photons: usize,
    verbose: bool,
    use_weather: bool,
    use_terrain: bool,
    dem_dir: &str,
    horizon_radius: f64,
    use_skyglow: bool,
    bortle: Option<u8>,
    radiance_nw: Option<f64>,
    led_fraction: f64,
    force_cpu: bool,
    gpu_backend_pref: Option<CliGpuBackend>,
    fast: bool,
) {
    // Suppress unused warnings when gpu feature is disabled
    #[cfg(not(feature = "gpu"))]
    let _ = gpu_backend_pref;

    let parts: Vec<&str> = date.split('-').collect();
    if parts.len() != 3 {
        eprintln!("Error: date must be in YYYY-MM-DD format");
        std::process::exit(1);
    }
    let year: i32 = parts[0].parse().unwrap_or(2024);
    let month: i32 = parts[1].parse().unwrap_or(1);
    let day: i32 = parts[2].parse().unwrap_or(1);

    println!("Twilight MCRT Prayer Time Calculator");
    println!("====================================");
    println!("Date:       {}-{:02}-{:02}", year, month, day);
    println!("Location:   {:.4}°N, {:.4}°E", lat, lon);
    println!("Elevation:  {:.0} m", elevation);
    println!("Timezone:   UTC{:+.1}", tz);
    println!("Albedo:     {:.2}", albedo);
    println!("SZA step:   {:.2}°", sza_step);

    // Resolve atmosphere: weather API or manual flags.
    //
    // Production weather sampling: prayer times happen at specific twilight
    // hours, so the hourly FORECAST is sampled at the evening-twilight hour
    // (sunset + 45 min) for the scan — the best data the API offers — with
    // a fallback to current conditions if the forecast fetch fails.
    let twilight_hour_utc: Option<f64> = {
        // Quick SPA sunset for the forecast hour (UTC fractional hours).
        let spa_in = SpaInput {
            year,
            month,
            day,
            hour: 12,
            minute: 0,
            second: 0,
            timezone: 0.0,
            latitude: lat,
            longitude: lon,
            elevation,
            pressure: 1013.25,
            temperature: 15.0,
            delta_t,
            slope: 0.0,
            azm_rotation: 0.0,
            atmos_refract: 0.5667,
        };
        spa::solar_position(&spa_in)
            .ok()
            .map(|o| (o.sunset + 0.75).rem_euclid(24.0))
    };
    let date_iso = format!("{}-{:02}-{:02}", year, month, day);
    let (weather_aerosol, weather_cloud, weather_gas, atm_desc) = if use_weather {
        let fetched = match twilight_hour_utc {
            Some(h) => twilight_weather::fetch_atmospheric_params_at(lat, lon, &date_iso, h)
                .map(|p| (p, format!("forecast @ {} {:02}:00 UTC", date_iso, h.round() as u32 % 24)))
                .or_else(|e| {
                    eprintln!("Note: hourly forecast unavailable ({}); using current conditions.", e);
                    twilight_weather::fetch_atmospheric_params(lat, lon)
                        .map(|p| (p, "current conditions".to_string()))
                }),
            None => twilight_weather::fetch_atmospheric_params(lat, lon)
                .map(|p| (p, "current conditions".to_string())),
        };
        match fetched.map(|(p, src)| { println!("Weather src: {}", src); p }) {
            Ok(params) => {
                let c = &params.conditions;
                println!(
                    "Weather:    AOD={:.2}, cloud={:.0}% (L:{:.0}/M:{:.0}/H:{:.0}), vis={:.0}m",
                    c.aod_550,
                    c.cloud_cover_total,
                    c.cloud_cover_low,
                    c.cloud_cover_mid,
                    c.cloud_cover_high,
                    c.visibility_m
                );
                if c.ozone_ug_m3 > 0.0 || c.nitrogen_dioxide_ug_m3 > 0.0 {
                    println!(
                        "Gas:        O3={:.0} ug/m3, NO2={:.0} ug/m3",
                        c.ozone_ug_m3, c.nitrogen_dioxide_ug_m3
                    );
                    if let Some(ref gc) = params.gas_composition {
                        if let Some(du) = gc.o3_column_du {
                            println!("            O3 column estimate: {:.0} DU", du);
                        }
                    }
                }
                let desc = format!("Live weather: {}", params.description);
                (params.aerosol, params.cloud, params.gas_composition, desc)
            }
            Err(e) => {
                eprintln!("Warning: failed to fetch weather: {}", e);
                eprintln!("Falling back to clear sky.");
                (
                    None,
                    None,
                    None,
                    "US Standard 1976 (clear sky, weather fetch failed)".to_string(),
                )
            }
        }
    } else {
        let aerosol_type = aerosol.to_aerosol_type();
        let cloud_type = cloud.to_cloud_type();
        let ap = aerosol_type.map(twilight_data::aerosol::default_properties);
        let cp = cloud_type.map(twilight_data::cloud::default_properties);
        let desc = format_atm_desc(aerosol_type, cloud_type);
        (ap, cp, None, desc)
    };

    println!("Atmosphere: {}", atm_desc);
    let ephemeris_label = if de440_path.is_some() {
        "JPL DE440"
    } else {
        "NREL SPA"
    };
    let scattering_mode = scattering.to_scattering_mode();
    let polarized = !fast;
    let method_str = match scattering_mode {
        ScatteringMode::Single => "Single-scatter MCRT + CIE mesopic vision".to_string(),
        ScatteringMode::Multiple => format!(
            "Multiple-scatter MC ({} photons/wl) + CIE mesopic vision",
            photons
        ),
        ScatteringMode::Hybrid => format!(
            "Hybrid SS+MC ({} sec. rays/step) + CIE mesopic vision",
            photons
        ),
    };
    println!("Ephemeris:  {}", ephemeris_label);
    println!("Method:     {}", method_str);
    let pol_str = if polarized {
        "Stokes [I,Q,U,V]"
    } else {
        "Scalar (--fast)"
    };
    println!("Polarize:   {}", pol_str);

    // Terrain masking
    let horizon_profile = if use_terrain {
        let dem_path = std::path::Path::new(dem_dir);
        let mut source = twilight_terrain::resolve_source(lat, lon, dem_path);

        // Compute bounding box for the horizon scan radius
        let radius_deg = horizon_radius / 111.0; // approximate degrees for bbox
        match source.prepare(
            lat - radius_deg,
            lon - radius_deg,
            lat + radius_deg,
            lon + radius_deg,
        ) {
            Ok(()) => {
                println!(
                    "Terrain:    {} (radius {:.0} km)",
                    source.name(),
                    horizon_radius
                );
                let profile = horizon::compute_horizon(source.as_ref(), lat, lon, horizon_radius);
                let max_hz = profile.max_angle();
                let min_hz = profile.min_angle();
                println!(
                    "  Observer: {:.1}m elevation, horizon range {:.3}° to {:.3}°",
                    profile.observer_elev_m, min_hz, max_hz
                );
                if max_hz > 0.1 {
                    println!(
                        "  Terrain masking active: up to {:.1}° obstruction ({:.0} min shift)",
                        max_hz,
                        max_hz * 4.0
                    );
                }
                Some(profile)
            }
            Err(e) => {
                eprintln!("Warning: failed to load terrain data: {}", e);
                eprintln!("         Continuing without terrain masking.");
                None
            }
        }
    } else {
        None
    };

    // Light pollution skyglow
    let skyglow_result = if use_skyglow || bortle.is_some() || radiance_nw.is_some() {
        let radiance = if let Some(r) = radiance_nw {
            r
        } else if let Some(b) = bortle {
            twilight_skyglow::bortle::bortle_to_radiance(b)
        } else {
            // Default: suburban (Bortle 5) -- a reasonable guess for someone
            // who enabled --skyglow without specifying a value
            twilight_skyglow::bortle::bortle_to_radiance(5)
        };

        let result = twilight_skyglow::quick_estimate_at_angle(radiance, led_fraction, 10.0);
        let lum_mcd = twilight_skyglow::bortle::radiance_to_zenith_luminance(radiance);
        println!(
            "Skyglow:    Bortle {}, zenith {:.2} mcd/m^2, LED fraction {:.0}%",
            result.bortle_class,
            lum_mcd,
            led_fraction * 100.0
        );
        let shift = twilight_skyglow::bortle::estimated_prayer_shift_minutes(lum_mcd);
        if shift > 0.5 {
            println!("  Estimated prayer time shift: ~{:.0} minutes", shift);
        }
        Some(result)
    } else {
        None
    };

    println!();

    // When using weather API, pass custom properties directly.
    // When using manual flags, use the type-based approach.
    let (aerosol_type, cloud_type, custom_aerosol, custom_cloud, o3_du, no2_density) =
        if use_weather {
            let (o3, no2) = weather_gas
                .map(|gc| (gc.o3_column_du, gc.no2_surface_density))
                .unwrap_or((None, None));
            (None, None, weather_aerosol, weather_cloud, o3, no2)
        } else {
            let at = aerosol.to_aerosol_type();
            let ct = cloud.to_cloud_type();
            (at, ct, None, None, None, None)
        };

    let input = PrayerTimeInput {
        latitude: lat,
        longitude: lon,
        elevation,
        year,
        month,
        day,
        timezone: tz,
        delta_t,
        surface_albedo: albedo,
        sza_step,
        aerosol_type,
        cloud_type,
        custom_aerosol,
        custom_cloud,
        de440_path: de440_path.map(|s| s.to_string()),
        scattering_mode,
        photons_per_wavelength: photons,
        horizon_profile,
        skyglow: skyglow_result,
        o3_column_du: o3_du,
        no2_surface_density: no2_density,
        polarized,
        ..Default::default()
    };

    // GPU initialization. GPU is the default compute backend for all modes.
    // Single-scatter benefits from batched dispatch (2.5x faster than serial).
    // MC/hybrid benefits from massive parallelism across photons/wavelengths.
    // Use --cpu to opt out.

    #[cfg(feature = "gpu")]
    let mut gpu_backend = if force_cpu {
        None
    } else {
        try_init_gpu(gpu_backend_pref, photons)
    };

    let compute_label = {
        #[cfg(feature = "gpu")]
        {
            if gpu_backend.is_some() {
                // Honest label: a backend initialized, but individual
                // kernels may still fall back to CPU on dispatch failure.
                "GPU (Metal; per-kernel CPU fallback)"
            } else {
                "CPU (rayon)"
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = force_cpu;
            "CPU (rayon)"
        }
    };
    println!("Compute:    {}", compute_label);

    println!("Computing...");

    #[cfg(feature = "gpu")]
    let output = if let Some(ref mut gpu) = gpu_backend {
        pipeline::compute_prayer_times_gpu(&input, gpu.as_mut())
    } else {
        pipeline::compute_prayer_times(&input)
    };

    #[cfg(not(feature = "gpu"))]
    let output = pipeline::compute_prayer_times(&input);

    let actual_ephemeris = match output.ephemeris {
        pipeline::EphemerisUsed::De440 => "DE440",
        pipeline::EphemerisUsed::Spa => "SPA (fallback)",
    };
    println!(
        "Done in {} ms (ephemeris: {}, compute: {})",
        output.computation_time_ms, actual_ephemeris, compute_label
    );
    println!();

    // Print results
    println!("Prayer Times (MCRT-derived):");
    println!("{:-<65}", "");

    // Sunrise/Sunset
    println!(
        "  Sunrise:              {}",
        output
            .sunrise_time
            .map(format_fractional_hour)
            .unwrap_or("N/A".to_string())
    );
    println!(
        "  Sunset:               {}",
        output
            .sunset_time
            .map(format_fractional_hour)
            .unwrap_or("N/A".to_string())
    );

    // Show terrain adjustment info
    if let Some(ref source) = output.terrain_source {
        if let (Some(sr_hz), Some(ss_hz)) = (output.sunrise_horizon_deg, output.sunset_horizon_deg)
        {
            if sr_hz > 0.01 || ss_hz > 0.01 {
                println!();
                println!("  Terrain adjustment ({}):", source);
                if sr_hz > 0.01 {
                    println!(
                        "    Sunrise horizon: {:.3}° (effective SZA: {:.4}°, ~{:.1} min later)",
                        sr_hz,
                        output.sunrise_sza_effective.unwrap_or(90.8333),
                        sr_hz * 4.0
                    );
                }
                if ss_hz > 0.01 {
                    println!(
                        "    Sunset horizon:  {:.3}° (effective SZA: {:.4}°, ~{:.1} min earlier)",
                        ss_hz,
                        output.sunset_sza_effective.unwrap_or(90.8333),
                        ss_hz * 4.0
                    );
                }
            }
        }
    }
    println!();

    // Show light pollution info
    if let Some(bortle) = output.skyglow_bortle {
        if let Some(shift) = output.skyglow_shift_minutes {
            if shift > 0.5 {
                println!(
                    "  Light pollution: Bortle {} (~{:.0} min shift)",
                    bortle, shift
                );
            }
        }
    }

    // Persistent twilight warning
    if output.persistent_twilight {
        if let Some(max_sza) = output.max_sza_deg {
            println!(
                "  ** PERSISTENT TWILIGHT: Sun only reaches {:.1}° max depression ({:.1}° SZA)",
                max_sza - 90.0,
                max_sza
            );
            println!("     Twilight never fully ends on this date at this latitude.");
            println!();
        }
    }

    // Fajr
    if let (Some(time), Some(sza), Some(dep)) = (
        output.fajr_time,
        output.fajr_sza_deg,
        output.fajr_depression_deg,
    ) {
        println!(
            "  Fajr (true dawn):     {}{}  (SZA {:.2}°, depression {:.2}°)",
            format_fractional_hour(time),
            format_uncertainty(output.fajr_uncertainty_min),
            sza,
            dep
        );
        if output.high_latitude_relative_thresholds {
            println!(
                "    └ high-latitude mode: sky never reaches full darkness; this is the"
            );
            println!(
                "      TVI-detectable onset of dawn brightening above tonight's sky floor"
            );
        }
    } else if output.persistent_twilight {
        println!("  Fajr (true dawn):     N/A (no night at all — midnight sun)");
    } else {
        println!("  Fajr (true dawn):     N/A (threshold not crossed in scan range)");
    }

    // Isha al-abyad
    if let (Some(time), Some(sza), Some(dep)) = (
        output.isha_abyad_time,
        output.isha_abyad_sza_deg,
        output.isha_abyad_depression_deg,
    ) {
        println!(
            "  Isha (al-abyad):      {}{}  (SZA {:.2}°, depression {:.2}°)",
            format_fractional_hour(time),
            format_uncertainty(output.isha_abyad_uncertainty_min),
            sza,
            dep
        );
        println!("    └ Hanafi school — white twilight disappears");
    } else {
        println!("  Isha (al-abyad):      N/A (threshold not crossed in scan range)");
    }

    // Isha al-ahmar
    if let (Some(time), Some(sza), Some(dep)) = (
        output.isha_ahmar_time,
        output.isha_ahmar_sza_deg,
        output.isha_ahmar_depression_deg,
    ) {
        println!(
            "  Isha (al-ahmar):      {}{}  (SZA {:.2}°, depression {:.2}°)",
            format_fractional_hour(time),
            format_uncertainty(output.isha_ahmar_uncertainty_min),
            sza,
            dep
        );
        println!("    └ Shafi'i/Maliki/Hanbali — red glow disappears");
    } else {
        println!("  Isha (al-ahmar):      N/A (threshold not crossed in scan range)");
    }

    println!("{:-<65}", "");

    // Compare with conventional
    println!();
    println!("Comparison with conventional fixed-angle methods:");
    println!("{:-<65}", "");

    let base_input = SpaInput {
        year,
        month,
        day,
        hour: 0,
        minute: 0,
        second: 0,
        timezone: tz,
        latitude: lat,
        longitude: lon,
        elevation,
        pressure: 1013.25,
        temperature: 15.0,
        delta_t,
        slope: 0.0,
        azm_rotation: 0.0,
        atmos_refract: 0.5667,
    };

    let conventions = [
        ("Fajr 18° (MWL/ISNA)", 108.0, true),
        ("Fajr 15° (Egypt)", 105.0, true),
        ("Fajr 19.5° (Umm al-Qura)", 109.5, true),
        ("Isha 17° (MWL)", 107.0, false),
        ("Isha 17.5° (Egypt)", 107.5, false),
        ("Isha 18° (ISNA)", 108.0, false),
    ];

    for (name, zenith, is_morning) in conventions {
        let time = if is_morning {
            spa::find_zenith_crossing(&base_input, zenith, 0.0, 12.0, 0.0001)
        } else {
            spa::find_zenith_crossing(&base_input, zenith, 12.0, 24.0, 0.0001)
        };

        let mcrt_time = if is_morning {
            output.fajr_time
        } else {
            // Use al-ahmar for Shafi'i comparison, al-abyad for Hanafi
            output.isha_ahmar_time
        };

        let time_str = time
            .map(format_fractional_hour)
            .unwrap_or("N/A".to_string());

        let diff_str = match (time, mcrt_time) {
            (Some(t1), Some(t2)) => {
                let diff_min = (t2 - t1) * 60.0;
                format!("{:+.1} min", diff_min)
            }
            _ => "---".to_string(),
        };

        println!("  {:<28} {}  (diff: {})", name, time_str, diff_str);
    }
    println!("{:-<65}", "");

    // Verbose: print full twilight analysis
    if verbose {
        println!();
        println!("Detailed Twilight Analysis:");
        println!("{:-<100}", "");
        println!(
            "{:>6}  {:>12}  {:>12}  {:>12}  {:>10}  {:>10}  {:>10}  {:>12}",
            "SZA°", "L_photopic", "L_scotopic", "L_mesopic", "L_red", "L_blue", "Centroid", "Color"
        );
        println!("{:-<100}", "");

        for a in &output.twilight_analyses {
            let color_str = match a.color {
                TwilightColor::Blue => "Blue",
                TwilightColor::White => "White",
                TwilightColor::Orange => "Orange",
                TwilightColor::Red => "Red",
                TwilightColor::Dark => "Dark",
            };

            println!(
                "{:>6.1}  {:>12.4e}  {:>12.4e}  {:>12.4e}  {:>10.4e}  {:>10.4e}  {:>10.1}  {:>12}",
                a.sza_deg,
                a.luminance_photopic,
                a.luminance_scotopic,
                a.luminance_mesopic,
                a.luminance_red,
                a.luminance_blue,
                a.spectral_centroid_nm,
                color_str,
            );
        }
        println!("{:-<100}", "");
    }

    println!();
    println!("Notes:");
    println!("  - These times are computed from first-principles radiative transfer (MCRT).");
    match scattering_mode {
        ScatteringMode::Single => {
            println!("  - Current model: US Standard 1976 atmosphere, single scattering.");
        }
        ScatteringMode::Multiple => {
            println!("  - Current model: US Standard 1976 atmosphere, multiple scattering (MC).");
            println!("  - MC noise decreases with more photons. Use --photons to adjust.");
        }
        ScatteringMode::Hybrid => {
            println!(
                "  - Current model: US Standard 1976 atmosphere, hybrid single+multi scatter."
            );
            println!(
                "  - Order 1 is exact, orders 2+ are MC. Use --photons to adjust convergence."
            );
        }
    }
    let has_aerosol = aerosol_type.is_some() || custom_aerosol.is_some();
    let has_cloud = cloud_type.is_some() || custom_cloud.is_some();
    if use_weather {
        println!("  - Atmosphere configured from live Open-Meteo weather data.");
        if has_aerosol {
            println!("  - Aerosol optical properties derived from measured AOD.");
        }
        if has_cloud {
            println!("  - Cloud layer derived from observed cloud cover.");
        }
        if o3_du.is_some() || no2_density.is_some() {
            println!("  - Gas absorption scaled from live O3/NO2 measurements.");
        }
    } else if has_aerosol || has_cloud {
        if has_aerosol {
            println!("  - Tropospheric aerosols included (OPAC climatology).");
        }
        if has_cloud {
            println!("  - Cloud layer included (Henyey-Greenstein forward scattering).");
        }
    } else {
        println!("  - No aerosols or clouds. Use --aerosol/--cloud or --weather to add them.");
    }
    println!("  - The 'depression' angle is the equivalent fixed angle that gives the same time.");
    println!(
        "  - Differences from conventional times reflect atmospheric conditions vs fixed angles."
    );
}

/// Format a human-readable atmosphere description.
fn format_atm_desc(aerosol: Option<AerosolType>, cloud: Option<CloudType>) -> String {
    match (aerosol, cloud) {
        (None, None) => "US Standard 1976 (clear sky)".to_string(),
        (Some(at), None) => format!("US Standard 1976 + {:?} aerosol", at),
        (None, Some(ct)) => format!("US Standard 1976 + {:?} cloud", ct),
        (Some(at), Some(ct)) => format!("US Standard 1976 + {:?} aerosol + {:?} cloud", at, ct),
    }
}

/// Get radiance at a specific wavelength from a SpectralResult.
fn get_radiance_at_wl(result: &SpectralResult, target_nm: f64) -> f64 {
    let mut closest_idx = 0;
    let mut closest_dist = f64::MAX;
    for (i, wl) in result.wavelengths_nm.iter().enumerate() {
        let dist = (wl - target_nm).abs();
        if dist < closest_dist {
            closest_dist = dist;
            closest_idx = i;
        }
    }
    result.radiance[closest_idx]
}

fn xyz_to_rgb(x: f64, y: f64, z: f64) -> (f64, f64, f64) {
    let r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    let g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
    let b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
    (r, g, b)
}

fn wyman_xyz(lambda: f64) -> (f64, f64, f64) {
    let x = {
        let t1 = (lambda - 442.0) * (if lambda < 442.0 { 0.0624 } else { 0.0374 });
        let t2 = (lambda - 599.8) * (if lambda < 599.8 { 0.0264 } else { 0.0323 });
        let t3 = (lambda - 501.1) * (if lambda < 501.1 { 0.0490 } else { 0.0382 });
        0.362 * (-0.5 * t1 * t1).exp() + 1.056 * (-0.5 * t2 * t2).exp()
            - 0.065 * (-0.5 * t3 * t3).exp()
    };

    let y = {
        let t1 = (lambda - 568.8) * (if lambda < 568.8 { 0.0213 } else { 0.0247 });
        let t2 = (lambda - 530.9) * (if lambda < 530.9 { 0.0613 } else { 0.0322 });
        0.821 * (-0.5 * t1 * t1).exp() + 0.286 * (-0.5 * t2 * t2).exp()
    };

    let z = {
        let t1 = (lambda - 437.0) * (if lambda < 437.0 { 0.0845 } else { 0.0278 });
        let t2 = (lambda - 459.0) * (if lambda < 459.0 { 0.0385 } else { 0.0725 });
        1.217 * (-0.5 * t1 * t1).exp() + 0.681 * (-0.5 * t2 * t2).exp()
    };
    (x, y, z)
}

fn cmd_render(sza: f64, width: u32, height: u32, rays: usize, out: &str) {
    println!("Rendering Sky Map");
    println!("SZA: {:.1} deg", sza);
    println!("Resolution: {}x{}", width, height);
    println!("Rays per pixel: {}", rays);
    println!("Output: {}", out);

    let lat = 0.0;
    let lon = 0.0;
    let elevation = 0.0;
    let albedo = 0.15;

    let atm = builder::build_full(AtmosphereType::UsStandard, albedo, None, None);

    let observer_pos = twilight_core::geometry::geographic_to_ecef(lat, lon, elevation);
    let sun_dir = twilight_core::geometry::solar_direction_ecef(sza, 180.0, lat, lon);

    let start = std::time::Instant::now();
    let num_wl = atm.num_wavelengths;
    let completed = std::sync::atomic::AtomicUsize::new(0);

    let mut pixels = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            pixels.push((x, y));
        }
    }

    use rayon::prelude::*;
    let result_pixels: Vec<(u32, u32, f64, f64, f64)> = pixels
        .into_par_iter()
        .map(|(x, y)| {
            let view_azimuth = (x as f64 / width as f64) * 360.0;
            let view_zenith = (y as f64 / height as f64) * 90.0;

            let view_dir =
                twilight_core::geometry::solar_direction_ecef(view_zenith, view_azimuth, lat, lon);

            let sza_bits = sza.to_bits();
            let mut rng = sza_bits
                .wrapping_mul(6364136223846793005)
                .wrapping_add((y as u64 * width as u64 + x as u64) ^ 0x12345678);

            let radiance_array = twilight_core::photon::hybrid_scatter_radiance_alis(
                &atm,
                observer_pos,
                view_dir,
                sun_dir,
                rays,
                &mut rng,
            );

            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut sum_z = 0.0;

            #[allow(clippy::needless_range_loop)]
            for w in 0..num_wl {
                let irrad = if w < twilight_data::solar_spectrum::SOLAR_IRRADIANCE.len() {
                    twilight_data::solar_spectrum::SOLAR_IRRADIANCE[w]
                } else {
                    1.0
                };
                let r = radiance_array[w] * irrad;
                let wl_nm = atm.wavelengths_nm[w];
                let (cx, cy, cz) = wyman_xyz(wl_nm);
                let dw = if w < num_wl - 1 {
                    atm.wavelengths_nm[w + 1] - wl_nm
                } else if w > 0 {
                    wl_nm - atm.wavelengths_nm[w - 1]
                } else {
                    10.0
                };
                sum_x += r * cx * dw;
                sum_y += r * cy * dw;
                sum_z += r * cz * dw;
            }

            let (r, g, b) = xyz_to_rgb(sum_x, sum_y, sum_z);

            let count = completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if count > 0 && count.is_multiple_of(width as usize * 10) {
                print!("\rRendered {} / {} pixels", count, width * height);
                use std::io::Write;
                std::io::stdout().flush().unwrap();
            }

            (x, y, r, g, b)
        })
        .collect();

    println!("\rRendered {} / {} pixels", width * height, width * height);

    let mut max_val = 0.0f64;
    for &(_, _, r, g, b) in &result_pixels {
        max_val = max_val.max(r).max(g).max(b);
    }

    // Tonemap
    // We want the max value to be scaled to something reasonable for Reinhard.
    // If we scale by 2.0 / max_val, then max value becomes 2.0. Reinhard: 2.0 / 3.0 = 0.66.
    // That means the brightest pixel will be gray (0.66).
    // Let's scale by 5.0 / max_val so max is 5.0. Reinhard: 5.0 / 6.0 = 0.83 (bright).
    let exposure = if max_val > 0.0 { 5.0 / max_val } else { 1.0 };

    let mut img = image::ImageBuffer::new(width, height);
    for (x, y, r, g, b) in result_pixels {
        let mut r = (r * exposure).max(0.0);
        let mut g = (g * exposure).max(0.0);
        let mut b = (b * exposure).max(0.0);

        // Reinhard
        r = r / (1.0 + r);
        g = g / (1.0 + g);
        b = b / (1.0 + b);

        // Gamma
        let gamma = 1.0 / 2.2;
        r = r.powf(gamma).clamp(0.0, 1.0);
        g = g.powf(gamma).clamp(0.0, 1.0);
        b = b.powf(gamma).clamp(0.0, 1.0);

        let rgb_tuple = image::Rgb([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]);
        img.put_pixel(x, y, rgb_tuple);
    }

    img.save(out).expect("Failed to save image");
    println!("Rendered in {:.2?}", start.elapsed());
}
fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Solar {
            lat,
            lon,
            date,
            tz,
            elevation,
            delta_t,
            de440,
        } => {
            cmd_solar(lat, lon, &date, tz, elevation, delta_t, de440.as_deref());
        }
        Commands::Mcrt {
            lat,
            lon,
            sza_start,
            sza_end,
            sza_step,
            photons,
            albedo,
            solar_azimuth,
            view_zenith,
            aerosol,
            cloud,
            scattering,
            weather,
            cpu,
            gpu_backend,
            fast,
        } => {
            cmd_mcrt(
                lat,
                lon,
                sza_start,
                sza_end,
                sza_step,
                photons,
                albedo,
                solar_azimuth,
                view_zenith,
                aerosol,
                cloud,
                scattering,
                weather,
                cpu,
                gpu_backend,
                fast,
            );
        }
        Commands::Render {
            sza,
            width,
            height,
            rays,
            out,
        } => {
            cmd_render(sza, width, height, rays, &out);
        }
        Commands::Pray {
            lat,
            lon,
            date,
            tz,
            elevation,
            albedo,
            delta_t,
            sza_step,
            aerosol,
            cloud,
            de440,
            scattering,
            photons,
            verbose,
            weather,
            terrain,
            dem_dir,
            horizon_radius,
            skyglow,
            bortle,
            radiance,
            led_fraction,
            cpu,
            gpu_backend,
            fast,
        } => {
            cmd_pray(
                lat,
                lon,
                &date,
                tz,
                elevation,
                albedo,
                delta_t,
                sza_step,
                aerosol,
                cloud,
                de440.as_deref(),
                scattering,
                photons,
                verbose,
                weather,
                terrain,
                &dem_dir,
                horizon_radius,
                skyglow,
                bortle,
                radiance,
                led_fraction,
                cpu,
                gpu_backend,
                fast,
            );
        }
        Commands::Compare {
            lat,
            lon,
            elevation,
            sza,
            view_zenith,
            rel_azimuth,
            solar_azimuth,
            albedo,
            rayleigh_only,
            aerosol,
            cloud,
            o3_du,
            scattering,
            photons,
            fast,
            no_refraction,
        } => {
            cmd_compare(
                lat,
                lon,
                elevation,
                &sza,
                &view_zenith,
                &rel_azimuth,
                solar_azimuth,
                albedo,
                rayleigh_only,
                aerosol,
                cloud,
                o3_du,
                scattering,
                photons,
                fast,
                no_refraction,
            );
        }
    }
}
