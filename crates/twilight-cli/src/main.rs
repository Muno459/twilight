use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::{Path, PathBuf};
use twilight_cpu::pipeline::{self, PrayerTimeInput, PrayerTimeOutput};
use twilight_cpu::simulation::{self, ScatteringMode, SimulationConfig, SpectralResult};
use twilight_data::aerosol::{AerosolProperties, AerosolType};
use twilight_data::atmosphere_profiles::AtmosphereType;
use twilight_data::builder;
use twilight_data::cloud::{CloudProperties, CloudType};
use twilight_skyglow::SkyglowResult;
use twilight_solar::de440::De440;
use twilight_solar::spa::{self, SpaInput};
use twilight_terrain::horizon;
use twilight_terrain::HorizonProfile;
use twilight_threshold::threshold::TwilightColor;
use twilight_weather::GasComposition;

/// Twilight - Monte Carlo Radiative Transfer engine for Fajr/Isha prayer times.
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
        #[arg(allow_negative_numbers = true, short, long)]
        lat: f64,
        /// Longitude in degrees (east positive)
        #[arg(allow_negative_numbers = true, short = 'n', long)]
        lon: f64,
        /// Date in YYYY-MM-DD format
        #[arg(short, long)]
        date: String,
        /// Timezone offset from UTC (hours)
        #[arg(allow_negative_numbers = true, short, long, default_value = "0")]
        tz: f64,
        /// Elevation above sea level in meters
        #[arg(short, long, default_value = "0")]
        elevation: f64,
        /// Delta T (TT - UT1) in seconds
        #[arg(allow_negative_numbers = true, long, default_value = "69.184")]
        delta_t: f64,
        /// Path to DE440 BSP file for JPL ephemeris comparison
        #[arg(long)]
        de440: Option<String>,
    },
    /// Run MCRT simulation across twilight solar zenith angles
    Mcrt {
        /// Latitude in degrees (north positive)
        #[arg(allow_negative_numbers = true, short, long)]
        lat: f64,
        /// Longitude in degrees (east positive)
        #[arg(allow_negative_numbers = true, short = 'n', long)]
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
    Pray(PrayArgs),
    /// Emit machine-readable spectral radiance for external RT comparison
    /// (e.g. libRadtran/uvspec). CSV to stdout:
    /// sza_deg,view_zenith_deg,rel_azimuth_deg,wavelength_nm,radiance_w_m2_sr_nm
    Compare {
        /// Latitude in degrees (north positive)
        #[arg(allow_negative_numbers = true, short, long, default_value = "21.4225")]
        lat: f64,
        /// Longitude in degrees (east positive)
        #[arg(allow_negative_numbers = true, short = 'n', long, default_value = "39.8262")]
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
        /// Scattering mode (default: single - deterministic, f64, the
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
        /// Custom uniform cloud deck: UNSCALED optical depth (the builder
        /// applies delta-Eddington similarity scaling internally). Overrides
        /// --cloud. Combined with --rayleigh-only this yields Rayleigh +
        /// cloud only (no gas absorption, no aerosol): the external
        /// slab-referee configuration (gate G2).
        #[arg(long)]
        cloud_tau: Option<f64>,
        /// Custom cloud base altitude in km (with --cloud-tau)
        #[arg(long, default_value = "1.0")]
        cloud_base_km: f64,
        /// Custom cloud top altitude in km (with --cloud-tau)
        #[arg(long, default_value = "2.0")]
        cloud_top_km: f64,
        /// Custom cloud single-scattering albedo (with --cloud-tau)
        #[arg(long, default_value = "0.999")]
        cloud_ssa: f64,
        /// Custom cloud Henyey-Greenstein asymmetry g (with --cloud-tau)
        #[arg(long, default_value = "0.85")]
        cloud_g: f64,
        /// RNG seed salt for the MC scattering modes (multi-seed error bars)
        #[arg(long, default_value = "0")]
        seed_salt: u64,
        /// 3D cloud field sidecar (same loader/validation as `pray
        /// --cloud-field`). The field owns ALL cloud (transport
        /// contract), so it cannot be combined with --cloud/--cloud-tau.
        /// This is the field-radiance referee surface (gate G2b).
        #[arg(long)]
        cloud_field: Option<std::path::PathBuf>,
    },
    /// Sky Quality Meter field calibration: predict a night's zenith
    /// sky-brightness curve and compare it against measured SQM logs
    Sqm {
        #[command(subcommand)]
        action: SqmCommands,
    },
}

#[derive(Subcommand)]
enum SqmCommands {
    /// Predict the zenith sky-brightness curve for one night
    /// (local sunset to next sunrise). CSV to stdout or --out, summary
    /// to stderr (stdout when --out is set, keeping the CSV clean).
    Predict(SqmArgs),
    /// Compare a measured SQM log against the predicted curve and
    /// report the offset binned by solar depression - the
    /// threshold-calibration measurement (see docs/SQM_CAMPAIGN.md)
    Compare(SqmCompareArgs),
}

/// Shared arguments for the sqm subcommands.
#[derive(Args)]
struct SqmArgs {
    /// Latitude in degrees (north positive)
    #[arg(allow_negative_numbers = true, short, long)]
    lat: f64,
    /// Longitude in degrees (east positive)
    #[arg(allow_negative_numbers = true, short = 'n', long)]
    lon: f64,
    /// Date in YYYY-MM-DD format: the evening the night STARTS
    #[arg(short, long)]
    date: String,
    /// Timezone offset from UTC (hours). Determined automatically from
    /// the coordinates when omitted (IANA tzdb). Set only to override.
    #[arg(short, long)]
    tz: Option<f64>,
    /// Elevation above sea level in meters
    #[arg(short, long, default_value = "0")]
    elevation: f64,
    /// Surface albedo (0-1)
    #[arg(long, default_value = "0.15")]
    albedo: f64,
    /// Delta T (TT - UT1) in seconds
    #[arg(allow_negative_numbers = true, long, default_value = "69.184")]
    delta_t: f64,
    /// Fetch live weather data from Open-Meteo (aerosol/cloud/gas).
    /// Provenance lines print to stdout, so pair with --out when piping
    /// the CSV.
    #[arg(long)]
    weather: bool,
    /// Light pollution: satellite atlas auto mode (Falchi atlas at the
    /// observer; no DNB temporal rescale, unlike `pray --skyglow`)
    #[arg(long)]
    skyglow: bool,
    /// Light pollution: Bortle dark-sky class (1-9)
    #[arg(long, value_parser = clap::value_parser!(u8).range(1..=9))]
    bortle: Option<u8>,
    /// Light pollution: VIIRS nighttime radiance (nW/cm^2/sr)
    #[arg(long)]
    radiance: Option<f64>,
    /// Scattering mode (default: single - deterministic and fast, the
    /// right speed for a whole-night scan; use hybrid for full MC)
    #[arg(long, value_enum, default_value = "single")]
    scattering: CliScattering,
    /// Secondary rays per step (hybrid/MC modes)
    #[arg(short, long, default_value = "100")]
    photons: usize,
    /// Time step in minutes
    #[arg(long, default_value = "5")]
    step_min: f64,
    /// Meter beam full-width at half-maximum in degrees. 0 (default)
    /// predicts the zenith POINT, which is what the meter measures only
    /// in the limit of a narrow beam. Pass 20 for the lensed Unihedron
    /// units (SQM-L/LE/LU/LU-DL) or 84 for the original wide-angle SQM,
    /// and the prediction is integrated over the beam's angular response
    /// instead. This matters in twilight and not at night: the twilight
    /// sky carries a steep gradient toward the sun, so a wide cone reads
    /// brighter than the zenith by an amount that GROWS with solar
    /// depression (measured at Padborg, 550 nm, sunward: 2.9x at SZA 96
    /// and 12.5x at SZA 100 at 42 degrees off zenith). Costs one MCRT
    /// evaluation per beam sample per time step.
    #[arg(long, default_value = "0")]
    beam_fwhm: f64,
    /// Write the CSV to this file instead of stdout
    #[arg(long)]
    out: Option<String>,
}

/// Offsets and weights sampling a circular beam of the given FWHM.
///
/// Returns `(view_zenith_deg, view_azimuth_deg, weight)` triples whose
/// weights sum to 1. The response is taken Gaussian in the off-axis
/// angle, `R(psi) = exp(-4 ln2 (psi/FWHM)^2)`, which is the standard
/// description of the Unihedron lensed optics and is what the published
/// half-angle figures encode. Each ring is weighted by its response and
/// by `sin(psi)` for solid angle; rings are sampled at four azimuths so
/// the sunward gradient is captured rather than averaged away by
/// assuming azimuthal symmetry.
///
/// `fwhm <= 0` degenerates to the single on-axis sample, which is the
/// historical zenith-point behaviour.
fn beam_samples(fwhm_deg: f64, solar_azimuth_deg: f64) -> Vec<(f64, f64, f64)> {
    if fwhm_deg <= 0.0 {
        return vec![(0.0, solar_azimuth_deg, 1.0)];
    }
    // Integrate to 1.2x FWHM: beyond that the Gaussian response has
    // fallen below 1e-2 and contributes under a part in 1e3 of the total.
    //
    // Clamped to the visible hemisphere. A wide beam (the 84-degree
    // original SQM) would otherwise place its outer rings past 90 degrees
    // of zenith angle, i.e. pointing into the ground, and the sky model
    // evaluated below the horizon returns a grazing-path radiance that
    // swamps the whole integral. HORIZON_MARGIN keeps the outermost ring
    // off the horizon itself, where the airmass integral is stiff and the
    // meter's own housing occludes in any case.
    const HORIZON_MARGIN_DEG: f64 = 85.0;
    let outer = (1.2 * fwhm_deg).min(HORIZON_MARGIN_DEG);
    let rings = 3;
    let azimuths = 4;
    let mut out = vec![];
    // On-axis sample: solid angle element vanishes at psi = 0, so give it
    // the weight of the disc it represents.
    let d_psi = outer / rings as f64;
    let sigma_factor = 4.0 * std::f64::consts::LN_2;
    let response = |psi: f64| (-sigma_factor * (psi / fwhm_deg).powi(2)).exp();
    let centre_w = {
        let half = 0.5 * d_psi;
        // integral of sin(psi) dpsi over [0, half), small-angle exact enough
        (1.0 - half.to_radians().cos()) * response(0.0)
    };
    out.push((0.0, solar_azimuth_deg, centre_w));
    for r in 1..=rings {
        let psi = r as f64 * d_psi;
        let w_ring = response(psi) * psi.to_radians().sin() * d_psi.to_radians();
        for a in 0..azimuths {
            let az = solar_azimuth_deg + 360.0 * a as f64 / azimuths as f64;
            out.push((psi, az, w_ring / azimuths as f64));
        }
    }
    let total: f64 = out.iter().map(|(_, _, w)| w).sum();
    if total > 0.0 {
        for s in out.iter_mut() {
            s.2 /= total;
        }
    }
    out
}

#[derive(Args)]
struct SqmCompareArgs {
    #[command(flatten)]
    base: SqmArgs,
    /// SQM log file: Unihedron SQM-LE/LU format (semicolon-separated,
    /// `#` comment headers, UTC timestamp first, mag last) or a simple
    /// 2-column CSV (timestamp_iso,mag). Autodetected.
    #[arg(long)]
    log: String,
}

/// Arguments to `pray`, passed around as one unit instead of 28
/// positional parameters.
#[derive(Args)]
struct PrayArgs {
    /// Latitude in degrees (north positive)
    #[arg(allow_negative_numbers = true, short, long)]
    lat: f64,
    /// Longitude in degrees (east positive)
    #[arg(allow_negative_numbers = true, short = 'n', long)]
    lon: f64,
    /// Date in YYYY-MM-DD format
    #[arg(short, long)]
    date: String,
    /// Timezone offset from UTC (hours). Determined automatically from
    /// the coordinates when omitted (IANA tzdb: correct zone AND the
    /// DST state for the specific date). Set only to override.
    #[arg(short, long)]
    tz: Option<f64>,
    /// Elevation above sea level in meters
    #[arg(short, long, default_value = "0")]
    elevation: f64,
    /// Surface albedo (0-1)
    #[arg(long, default_value = "0.15")]
    albedo: f64,
    /// Delta T (TT - UT1) in seconds
    #[arg(allow_negative_numbers = true, long, default_value = "69.184")]
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
    /// Azimuthally-resolved skyglow veil for the khayt criterion:
    /// fetch the VIIRS Black Marble grid around the site, bin it into
    /// ground light sources, and give every khayt fan patch its own
    /// Garstang slant veil at its real azimuth (an observer south of a
    /// city sees a darker eastern dawn horizon than its bright northern
    /// sky). STRUCTURE only: the amplitude stays on the atlas/Falchi
    /// rail of --skyglow (which this flag requires). Falls back loudly
    /// to the isotropic veil when no VIIRS grid is available.
    #[arg(long, requires = "skyglow")]
    skyglow_directional: bool,
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
    /// 3D cloud vertical profile from the cloud3d satellite model
    /// (SegFormer trained on CloudSat radar profiles). Pass "auto" to
    /// run tools/cloud3d_profile.py on live GOES imagery (needs
    /// python3+torch; Americas/Pacific coverage), or a path to a
    /// sidecar-produced JSON. Overrides single-layer cloud sources.
    #[arg(long)]
    cloud3d: Option<String>,
    /// Full 3D cloud field from a sidecar --field-out export (raw f32
    /// grid + .json header). The transport engine then traces every
    /// light path through the actual voxel structure: sun rays through
    /// real cloud gaps instead of a horizontally uniform deck.
    /// Overrides --cloud3d and all other cloud sources. The observer
    /// (--lat/--lon) must lie inside the field footprint; a field
    /// exported for another location is a hard error. Fields whose
    /// data timestamp is older than 3 hours print a staleness warning
    /// (clouds advect ~10 km per 10 min). Field runs execute on the
    /// CPU reference scan (GPU field port pending re-verification).
    #[arg(long, value_name = "PATH")]
    cloud_field: Option<std::path::PathBuf>,
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
    /// Apple Metal (macOS / iOS) - the only implemented GPU backend
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

/// Format a fractional hour as HH:MM:SS via the one canonical
/// implementation, `pipeline::format_time` (integer-seconds rounding,
/// no float truncation drift).
fn format_fractional_hour(h: f64) -> String {
    if !(0.0..=48.0).contains(&h) {
        return "N/A".to_string();
    }
    // Hours >= 24 are past-midnight events (high-latitude Isha can fall on
    // the next civil day) - display wrapped with a +1d marker.
    if h >= 24.0 {
        format!("{} (+1d)", pipeline::format_time(h - 24.0))
    } else {
        pipeline::format_time(h)
    }
}

/// Parse and validate a YYYY-MM-DD date string. Malformed input is a
/// hard error: a mistyped date must fail, never silently compute times
/// for some other day.
fn resolve_date(date: &str) -> (i32, i32, i32) {
    fn bail(date: &str, reason: &str) -> ! {
        eprintln!(
            "Error: invalid date '{}': {}; expected YYYY-MM-DD (e.g. 2026-06-13)",
            date, reason
        );
        std::process::exit(1);
    }
    let parts: Vec<&str> = date.split('-').collect();
    if parts.len() != 3 {
        bail(date, "wrong number of components");
    }
    let year: i32 = parts[0]
        .parse()
        .unwrap_or_else(|_| bail(date, "year is not a number"));
    let month: i32 = parts[1]
        .parse()
        .unwrap_or_else(|_| bail(date, "month is not a number"));
    let day: i32 = parts[2]
        .parse()
        .unwrap_or_else(|_| bail(date, "day is not a number"));
    if !(1..=12).contains(&month) {
        bail(date, "month must be 1-12");
    }
    if !(1..=31).contains(&day) {
        bail(date, "day must be 1-31");
    }
    (year, month, day)
}

/// Canonical SpaInput for this CLI: local midnight on the given date,
/// with the standard-atmosphere refraction constants used everywhere
/// (1013.25 hPa, 15 C, 0.5667 deg refraction, flat horizon). Callers
/// needing another instant override the time fields or use
/// `sun_azimuth_at`.
#[allow(clippy::too_many_arguments)] // one date + one observer frame, fixed by SPA
fn spa_input_for(
    lat: f64,
    lon: f64,
    elevation: f64,
    tz: f64,
    delta_t: f64,
    year: i32,
    month: i32,
    day: i32,
) -> SpaInput {
    SpaInput {
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
    }
}

/// Solar azimuth (degrees) at a fractional hour of day on `base`'s date
/// and frame. Integer-seconds rounding, not float truncation.
fn sun_azimuth_at(base: &SpaInput, fractional_hour: f64) -> Option<f64> {
    let mut inp = base.clone();
    let total = (fractional_hour * 3600.0).round() as i32;
    inp.hour = total / 3600;
    inp.minute = (total % 3600) / 60;
    inp.second = total % 60;
    spa::solar_position(&inp).ok().map(|o| o.azimuth)
}

/// Current UTC civil date from the system clock. `mcrt --weather` has no
/// --date flag but the forecast and satellite samplers need one.
/// Days-to-civil conversion per Howard Hinnant's date algorithms.
fn current_utc_date() -> (i32, i32, i32) {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let z = secs.div_euclid(86_400) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + if month <= 2 { 1 } else { 0 };
    (year as i32, month as i32, day as i32)
}

/// Workspace root for sidecar and data paths: the nearest ancestor of
/// the running executable containing Cargo.toml (target/release lives
/// inside the workspace). Falls back to the CWD for installed binaries.
fn repo_root() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|exe| {
            exe.ancestors()
                .find(|p| p.join("Cargo.toml").is_file())
                .map(Path::to_path_buf)
        })
        .unwrap_or_else(|| PathBuf::from("."))
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
    let (year, month, day) = resolve_date(date);

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
        hour: 12,
        ..spa_input_for(lat, lon, elevation, tz, delta_t, year, month, day)
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

    let base_input = spa_input_for(lat, lon, elevation, tz, delta_t, year, month, day);

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
    println!("Twilight MCRT Simulation");
    println!("=======================");
    println!("Location:     {:.4}°N, {:.4}°E", lat, lon);
    println!(
        "SZA range:    {:.1}° to {:.1}° (step {:.1}°)",
        sza_start, sza_end, sza_step
    );
    println!("Photons/λ:    {}", photons);
    println!("Wavelengths:  380-780 nm (41 bands, 10nm steps)");

    // Resolve atmosphere: weather API or manual flags. mcrt has no
    // --date flag, so the forecast is sampled on today's UTC date at
    // this evening's twilight hour, same as `pray`.
    let (aerosol_props, cloud_props, gas_composition, atm_desc) = if use_weather {
        let (year, month, day) = current_utc_date();
        let date_iso = format!("{}-{:02}-{:02}", year, month, day);
        // Sea level and the 2024 delta-T default: minutes of slack in
        // the sampling hour cannot move the hourly forecast bucket.
        let utc_spa = spa_input_for(lat, lon, 0.0, 0.0, 69.184, year, month, day);
        let twilight_hour_utc = spa::solar_position(&SpaInput {
            hour: 12,
            ..utc_spa.clone()
        })
        .ok()
        .map(|o| (o.sunset + 0.75).rem_euclid(24.0));
        let sun_azimuth =
            sun_azimuth_at(&utc_spa, twilight_hour_utc.unwrap_or(21.0)).unwrap_or(270.0);
        let w = weather_block(
            lat,
            lon,
            &date_iso,
            twilight_hour_utc,
            sun_azimuth,
            Path::new("data/satellite"),
        );
        (w.aerosol, w.cloud, w.gas, w.description)
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
            if let Some(backend) = gpu_backend.as_ref() {
                // Honest label: a backend initialized, but individual
                // kernels may still fall back to CPU on dispatch failure.
                // Name the backend that actually initialized rather than
                // assuming Metal; on non-Apple hosts this is wgpu.
                format!(
                    "GPU ({}; per-kernel CPU fallback)",
                    backend.device_info().backend
                )
            } else {
                "CPU (rayon)".to_string()
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            // gpu feature off: the GPU flags exist but cannot take effect.
            let _ = (force_cpu, gpu_backend);
            "CPU (rayon)".to_string()
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
                simulation::simulate_twilight_scan(
                    &atm, &config, sza_start, sza_end, sza_step, None,
                )
            }),
            Err(e) => {
                eprintln!("Warning: GPU upload failed ({}), falling back to CPU", e);
                simulation::simulate_twilight_scan(
                    &atm, &config, sza_start, sza_end, sza_step, None,
                )
            }
        }
    } else {
        simulation::simulate_twilight_scan(&atm, &config, sza_start, sza_end, sza_step, None)
    };

    #[cfg(not(feature = "gpu"))]
    let results =
        simulation::simulate_twilight_scan(&atm, &config, sza_start, sza_end, sza_step, None);

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
    custom_cloud: Option<CloudProperties>,
    o3_du: Option<f64>,
    scattering: CliScattering,
    photons: usize,
    fast: bool,
    no_refraction: bool,
    seed_salt: u64,
    cloud_field: Option<twilight_data::cloud_field_builder::OwnedCloudField>,
) {
    // Transport contract: when a 3D field is present it owns ALL cloud,
    // so the shells must stay cloud-free (same rule as the pipeline's
    // build_atmosphere). Refuse conflicting cloud options outright.
    if cloud_field.is_some()
        && (custom_cloud.is_some() || cloud.to_cloud_type().is_some())
    {
        eprintln!(
            "Error: --cloud-field owns all cloud; it cannot be combined \
             with --cloud or --cloud-tau."
        );
        std::process::exit(1);
    }
    // Build the atmosphere once.
    let mut atm = if rayleigh_only {
        match &custom_cloud {
            // G2 slab referee: Rayleigh + delta-Eddington-scaled cloud,
            // no gas absorption, no aerosol (matches a libRadtran deck
            // with `no_absorption mol` + a scaled HG water cloud).
            Some(cp) => {
                builder::build_with_cloud_properties(AtmosphereType::UsStandard, albedo, cp)
            }
            None => builder::build_clear_sky(AtmosphereType::UsStandard, albedo),
        }
    } else {
        let ap = aerosol
            .to_aerosol_type()
            .map(twilight_data::aerosol::default_properties);
        let cp = custom_cloud.or_else(|| {
            cloud
                .to_cloud_type()
                .map(twilight_data::cloud::default_properties)
        });
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
    // 3D field: carry the field's asymmetry on the (cloud-free) shells
    // for the Eddington diffuse factors, exactly as the pipeline does.
    if let Some(f) = &cloud_field {
        atm.cloud_g_scaled = f.g_default;
    }
    let field_view = cloud_field.as_ref().map(|f| f.view());

    // Header with enough metadata to reproduce the run.
    println!(
        "# twilight compare: lat={} lon={} elev={} albedo={} rayleigh_only={} o3_du={:?} scattering={:?} photons={} polarized={} seed_salt={}",
        lat, lon, elevation, albedo, rayleigh_only, o3_du, scattering.to_scattering_mode(), photons, !fast, seed_salt
    );
    if let Some(cp) = &custom_cloud {
        // Delta-Eddington scaled quantities actually carried by the medium
        // (see twilight-data builder::add_cloud_layer): the external referee
        // must be configured with THESE, not the unscaled inputs.
        let f = cp.asymmetry * cp.asymmetry;
        let de_scale = 1.0 - cp.ssa * f;
        let ssa_s = ((1.0 - f) * cp.ssa / de_scale).clamp(0.0, 1.0);
        let g_s = cp.asymmetry / (1.0 + cp.asymmetry);
        println!(
            "# custom cloud: base_km={} top_km={} tau_unscaled={} ssa={} g={} | scaled: tau*={:.6} ssa*={:.6} g*={:.6} tau_scat*={:.6} tau_abs={:.6}",
            cp.base_km, cp.top_km, cp.optical_depth, cp.ssa, cp.asymmetry,
            cp.optical_depth * de_scale, ssa_s, g_s,
            cp.optical_depth * de_scale * ssa_s,
            cp.optical_depth * de_scale * (1.0 - ssa_s),
        );
    }
    if let Some(f) = &cloud_field {
        println!(
            "# cloud field: {}x{}x{} voxels, g_default={:.7}, source={} @ {}",
            f.nz, f.nlat, f.nlon, f.g_default, f.source, f.timestamp
        );
    }
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
                seed_salt,
            };
            for &sza in szas {
                let result =
                    simulation::simulate_at_sza(&atm, &config, sza, field_view.as_ref());
                for (wl, rad) in result.wavelengths_nm.iter().zip(result.radiance.iter()) {
                    println!("{},{},{},{},{:e}", sza, vz, ra, wl, rad);
                }
            }
        }
    }
}

/// Atmosphere inputs resolved from live weather, plus the description
/// for the run header.
struct WeatherBlock {
    aerosol: Option<AerosolProperties>,
    cloud: Option<CloudProperties>,
    gas: Option<GasComposition>,
    /// Input sigma of the measured AOD product (None: climatology
    /// bracket applies downstream). Flows into
    /// `PrayerTimeInput::aod_sigma_550` for the background-uncertainty
    /// propagation.
    aod_sigma_550: Option<f64>,
    description: String,
}

/// Live-weather resolution shared by `pray` and `mcrt`.
///
/// Production weather sampling: prayer times happen at specific twilight
/// hours, so the hourly FORECAST is sampled at the evening-twilight hour
/// (sunset + 45 min) for the scan - the best data the API offers - with
/// a fallback to current conditions if the forecast fetch fails. The
/// satellite cloud override then replaces the forecast cloud layer when
/// the satellite measured one.
fn weather_block(
    lat: f64,
    lon: f64,
    date_iso: &str,
    twilight_hour_utc: Option<f64>,
    sun_azimuth: f64,
    satellite_cache: &Path,
) -> WeatherBlock {
    let fetched = match twilight_hour_utc {
        Some(h) => twilight_weather::fetch_atmospheric_params_at(lat, lon, date_iso, h)
            .map(|p| {
                (
                    p,
                    format!("forecast @ {} {:02}:00 UTC", date_iso, h.round() as u32 % 24),
                )
            })
            .or_else(|e| {
                eprintln!(
                    "Note: hourly forecast unavailable ({}); using current conditions.",
                    e
                );
                twilight_weather::fetch_atmospheric_params(lat, lon)
                    .map(|p| (p, "current conditions".to_string()))
            }),
        None => twilight_weather::fetch_atmospheric_params(lat, lon)
            .map(|p| (p, "current conditions".to_string())),
    };
    let sat_cloud = resolve_satellite_cloud(satellite_cache, date_iso, lat, lon, sun_azimuth);

    match fetched {
        Ok((mut params, src)) => {
            println!("Weather src: {}", src);
            if let Some(sc) = sat_cloud {
                params.cloud = Some(sc);
                params.description = format!("{} + satellite cloud", params.description);
            }
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
            for w in &c.data_warnings {
                eprintln!("Weather data gap: {}", w);
            }
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
            WeatherBlock {
                description: format!("Live weather: {}", params.description),
                aerosol: params.aerosol,
                cloud: params.cloud,
                gas: params.gas_composition,
                aod_sigma_550: params.aod_sigma_550,
            }
        }
        Err(e) => {
            eprintln!("Warning: failed to fetch weather: {}", e);
            eprintln!("Falling back to clear sky.");
            WeatherBlock {
                aerosol: None,
                cloud: None,
                gas: None,
                aod_sigma_550: None,
                description: "US Standard 1976 (clear sky, weather fetch failed)".to_string(),
            }
        }
    }
}

/// SATELLITE CLOUD ENHANCEMENT: sample the GIBS MODIS cloud field
/// (COT + cloud-top height) at the observer and along the sun
/// azimuth ("2.5D" - the twilight shadow path crosses the cloud
/// field tens to hundreds of km sunward). When the satellite saw
/// cloud, it overrides the model forecast's cloud layer: measured
/// optical depth at measured altitude beats a model cover fraction.
fn resolve_satellite_cloud(
    cache: &Path,
    date_iso: &str,
    lat: f64,
    lon: f64,
    sun_azimuth: f64,
) -> Option<CloudProperties> {
    let sp = twilight_weather::satellite::sample_cloud_path(cache, date_iso, lat, lon, sun_azimuth);
    if sp.observer.is_none() && sp.path_cloud_fraction <= 0.0 {
        return None;
    }
    if let Some(obs) = sp.observer {
        println!(
            "Satellite:  MODIS COT {:.1} @ top {:.1} km (age {}d), sunward cloud {:.0}% (mean COT {:.1})",
            obs.cot,
            obs.cloud_top_m.unwrap_or(0.0) / 1000.0,
            obs.age_days,
            sp.path_cloud_fraction * 100.0,
            sp.path_mean_cot
        );
    } else {
        println!(
            "Satellite:  clear overhead; sunward cloud {:.0}% (mean COT {:.1})",
            sp.path_cloud_fraction * 100.0,
            sp.path_mean_cot
        );
    }
    twilight_weather::mapping::map_cloud_satellite(&sp)
}

/// Parse the cloud3d sidecar error protocol: handled failures print one
/// stdout JSON line, {"error": code, "detail": ...}.
fn sidecar_error(stdout: &str) -> Option<(String, String)> {
    stdout.lines().rev().find_map(|line| {
        let v: serde_json::Value = serde_json::from_str(line.trim()).ok()?;
        let code = v.get("error")?.as_str()?.to_string();
        let detail = v
            .get("detail")
            .and_then(|d| d.as_str())
            .unwrap_or("")
            .to_string();
        Some((code, detail))
    })
}

/// Launch tools/cloud3d_profile.py on live GOES imagery. The script and
/// its model/data live in the source tree, so paths resolve against the
/// repo root, not the CWD. Sidecar exit codes: 2 = coverage,
/// 3 = environment, 4 = network, each with the JSON error line.
fn run_cloud3d_sidecar(
    lat: f64,
    lon: f64,
    date_iso: &str,
    hour_utc: f64,
    sun_azimuth: f64,
) -> Option<PathBuf> {
    let root = repo_root();
    let script = root.join("tools/cloud3d_profile.py");
    if !script.is_file() {
        eprintln!(
            "Note: cloud3d sidecar not found at {}; continuing without 3D profile.",
            script.display()
        );
        return None;
    }
    let out_dir = root.join("data/cloud3d");
    let _ = std::fs::create_dir_all(&out_dir);
    let out = out_dir.join("profile.json");
    let model = std::env::var("CLOUD3D_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| out_dir.join("iwc.jit.pt"));
    eprintln!("Cloud3D:    running sidecar on live GOES imagery...");
    let result = std::process::Command::new("python3")
        .arg(&script)
        .args(["--lat", &lat.to_string(), "--lon", &lon.to_string()])
        .args(["--date", date_iso, "--hour", &format!("{hour_utc:.2}")])
        .args(["--azimuth", &format!("{sun_azimuth:.1}")])
        .arg("--model")
        .arg(&model)
        .arg("--out")
        .arg(&out)
        .stderr(std::process::Stdio::inherit())
        .output();
    match result {
        Ok(o) if o.status.success() => {
            // Pass the captured protocol success line through unchanged.
            print!("{}", String::from_utf8_lossy(&o.stdout));
            Some(out)
        }
        Ok(o) => {
            let stdout = String::from_utf8_lossy(&o.stdout);
            let cause = match sidecar_error(&stdout) {
                Some((code, detail)) => {
                    let what = match code.as_str() {
                        "outside_coverage" => {
                            "location outside GOES-East/West coverage (Americas/Pacific only)"
                        }
                        "no_granules" => "no GOES granules published for this date/hour",
                        "below_horizon" => "the satellite cannot see this location",
                        "missing_deps" => "missing Python dependencies",
                        "network" => "network failure fetching GOES data",
                        other => other,
                    };
                    format!("{}: {}", what, detail)
                }
                None => format!("exited with {}", o.status),
            };
            eprintln!(
                "Note: cloud3d sidecar: {}; continuing without 3D profile.",
                cause
            );
            None
        }
        Err(e) => {
            eprintln!("Note: cloud3d sidecar failed to launch ({e}); continuing.");
            None
        }
    }
}

/// 3D CLOUDS (cloud3d): an 80-level ice-water-content profile
/// reconstructed from live geostationary imagery by the cloud3d model
/// (trained on CloudSat radar). Real measured VERTICAL STRUCTURE -
/// multiple independent layers - replacing any single-slab source.
/// `spec` is "auto" (run the sidecar) or a path to sidecar JSON.
#[allow(clippy::too_many_arguments)] // observer, instant and cache, no natural grouping left
fn resolve_cloud3d(
    spec: &str,
    lat: f64,
    lon: f64,
    date_iso: &str,
    hour_utc: f64,
    sun_azimuth: f64,
    satellite_cache: &Path,
) -> Option<Vec<CloudProperties>> {
    let json_path: PathBuf = if spec == "auto" {
        run_cloud3d_sidecar(lat, lon, date_iso, hour_utc, sun_azimuth)?
    } else {
        PathBuf::from(spec)
    };
    match twilight_weather::cloud3d::load(&json_path) {
        Ok(p) => {
            // cloud3d gives the vertical STRUCTURE; the NRT MODIS
            // COT (when the satellite measured one here, cached
            // from the weather step) gives the AMPLITUDE.
            let measured_cot =
                twilight_weather::satellite::sample_cloud(satellite_cache, date_iso, lat, lon)
                    .map(|s| s.cot)
                    .filter(|&c| c > 0.5);
            let layers = p.to_cloud_layers(measured_cot);
            if let Some(cot) = measured_cot {
                if !layers.is_empty() {
                    println!(
                        "Cloud3D:    amplitude rescaled to measured MODIS COT {:.1}",
                        cot
                    );
                }
            }
            if layers.is_empty() {
                println!(
                    "Cloud3D:    {} {} - clear column (window cloud fraction {:.0}%)",
                    p.satellite,
                    p.time_utc,
                    p.cloud_fraction * 100.0
                );
                None
            } else {
                let tau: f64 = layers.iter().map(|l| l.optical_depth).sum();
                println!(
                    "Cloud3D:    {} {}: {} layer(s), total tau {:.2}, window cloud fraction {:.0}%",
                    p.satellite,
                    p.time_utc,
                    layers.len(),
                    tau,
                    p.cloud_fraction * 100.0
                );
                for l in &layers {
                    println!(
                        "            {:.1}-{:.1} km, tau {:.2} (g {:.2})",
                        l.base_km, l.top_km, l.optical_depth, l.asymmetry
                    );
                }
                Some(layers)
            }
        }
        Err(e) => {
            eprintln!("Note: cloud3d profile load failed ({e}); continuing.");
            None
        }
    }
}

/// Terrain masking: horizon profile from DEM tiles around the observer.
fn resolve_terrain(
    lat: f64,
    lon: f64,
    dem_dir: &str,
    horizon_radius: f64,
) -> Option<HorizonProfile> {
    let dem_path = Path::new(dem_dir);
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
}

/// Light pollution skyglow: explicit --radiance/--bortle when given,
/// otherwise the satellite auto mode.
fn resolve_skyglow(args: &PrayArgs, year: i32, month: i32, day: i32) -> Option<SkyglowResult> {
    if !(args.skyglow || args.bortle.is_some() || args.radiance.is_some()) {
        return None;
    }
    let radiance = if let Some(r) = args.radiance {
        r
    } else if let Some(b) = args.bortle {
        twilight_skyglow::bortle::bortle_to_radiance(b)
    } else {
        // SATELLITE AUTO MODE - two independent satellite feeds:
        //  1. Lorenz atlas: PROPAGATED artificial zenith brightness
        //     (the right observable), frozen at its 2024 epoch.
        //  2. VIIRS Black Marble (GIBS, daily): CURRENT upward
        //     nighttime-lights radiance, moonlight-removed and
        //     BRDF-corrected. Same sensor at both epochs gives a
        //     temporal ratio that brings the atlas to today; where
        //     the atlas has no data, the live DNB stands alone.
        let cache = Path::new(&args.dem_dir)
            .parent()
            .map(|p| p.join("skyglow"))
            .unwrap_or_else(|| PathBuf::from("data/skyglow"));
        let atlas = twilight_skyglow::atlas::artificial_zenith(&cache, args.lat, args.lon);
        let today = (year, month as u32, day as u32);
        match atlas {
            Some(a) => {
                let mut mcd = a.zenith_mcd;
                match twilight_skyglow::dnb::epoch_ratio(
                    &cache,
                    args.lat,
                    args.lon,
                    today,
                    a.year as i32,
                ) {
                    Some((ratio, now, epoch_nw)) => {
                        mcd *= ratio;
                        println!(
                            "Skyglow:    satellite atlas {} {:.3} mcd/m^2 x DNB trend {:.2} \
                             ({:.1} nW now vs {:.1} nW {}) -> {:.3} mcd/m^2",
                            a.year, a.zenith_mcd, ratio, now.radiance_nw, epoch_nw, a.year, mcd
                        );
                    }
                    None => {
                        // No same-sensor data at the atlas epoch (the
                        // GapFilled layers reach back ~1 year). Fall
                        // back to a ONE-SIDED live cross-check: a
                        // bright local pixel proves new lights the
                        // 2024 atlas missed (raise to the DNB-implied
                        // floor) - but a dim local pixel proves
                        // nothing, because the atlas value is
                        // PROPAGATED sky brightness that may come
                        // from a metro tens of km away (Brondby's sky
                        // is lit by central Copenhagen, not its own
                        // pixel). Never darken from a point sample.
                        let atlas_nw = twilight_skyglow::bortle::zenith_luminance_to_radiance(mcd);
                        match twilight_skyglow::dnb::measure(&cache, args.lat, args.lon, today, 1) {
                            Some(s) if s.radiance_nw > 0.05 => {
                                let r = s.radiance_nw / atlas_nw.max(1e-6);
                                if r > 3.0 {
                                    mcd = twilight_skyglow::bortle::radiance_to_zenith_luminance(
                                        s.radiance_nw,
                                    );
                                    println!(
                                        "Skyglow:    atlas {} {:.3} mcd/m^2 raised by live \
                                         DNB {:.1} nW ({:.1}x brighter than atlas-implied) \
                                         -> {:.3} mcd/m^2",
                                        a.year, a.zenith_mcd, s.radiance_nw, r, mcd
                                    );
                                } else {
                                    println!(
                                        "Skyglow:    satellite atlas {} {:.3} mcd/m^2 \
                                         (live DNB cross-check: {:.1} nW local vs {:.1} nW \
                                         implied - consistent)",
                                        a.year, a.zenith_mcd, s.radiance_nw, atlas_nw
                                    );
                                }
                            }
                            _ => {
                                println!(
                                    "Skyglow:    satellite atlas {} -> artificial zenith \
                                     {:.3} mcd/m^2 (no live DNB here)",
                                    a.year, a.zenith_mcd
                                );
                            }
                        }
                    }
                }
                twilight_skyglow::bortle::zenith_luminance_to_radiance(mcd)
            }
            None => {
                // Atlas gap: live Black Marble alone.
                match twilight_skyglow::dnb::measure(&cache, args.lat, args.lon, today, 1) {
                    Some(s) => {
                        println!(
                            "Skyglow:    live VIIRS Black Marble {:.1} nW/cm^2/sr \
                             (median of {} nights, {})",
                            s.radiance_nw,
                            s.dates_used.len(),
                            s.layer
                        );
                        s.radiance_nw
                    }
                    None => {
                        eprintln!("Note: no satellite skyglow data here; using Bortle 5 default.");
                        twilight_skyglow::bortle::bortle_to_radiance(5)
                    }
                }
            }
        }
    };

    // The legacy scan observes at VIEW_ZENITH_DEG = 85 (5 deg elevation);
    // the injected veil must be lifted for THAT elevation, not a
    // hard-coded 10 deg (which under-veiled the legacy path by the
    // 5-vs-10-deg enhancement ratio: urban legacy fajr ~1 min early,
    // review round 2).
    let result = twilight_skyglow::quick_estimate_at_angle(radiance, args.led_fraction, 5.0);
    let lum_mcd = twilight_skyglow::bortle::radiance_to_zenith_luminance(radiance);
    println!(
        "Skyglow:    Bortle {}, zenith {:.2} mcd/m^2, LED fraction {:.0}%",
        result.bortle_class,
        lum_mcd,
        args.led_fraction * 100.0
    );
    let shift = twilight_skyglow::bortle::estimated_prayer_shift_minutes(lum_mcd);
    if shift > 0.5 {
        println!("  Estimated prayer time shift: ~{:.0} minutes", shift);
    }
    Some(result)
}

/// Azimuthally-resolved skyglow (--skyglow-directional): VIIRS Black
/// Marble grid around the site -> binned ground sources for the khayt
/// Garstang slant veils. STRUCTURE only; the veil amplitude stays on
/// the --skyglow atlas/Falchi rail (`radiance_nw` must be the SAME
/// value the isotropic path resolved). Returns None LOUDLY when no
/// usable grid exists; the pipeline then keeps the isotropic veil.
///
/// The GapFilled Black Marble layers reach back only about a year, so
/// the grid is opened near TODAY even for historical run dates: a
/// city's lighting GEOMETRY is essentially time-invariant, and only
/// the geometry is consumed here.
fn resolve_directional_skyglow(
    args: &PrayArgs,
    radiance_nw: f64,
    dawn_azimuth_deg: Option<f64>,
) -> Option<twilight_skyglow::DirectionalSkyglow> {
    let cache = Path::new(&args.dem_dir)
        .parent()
        .map(|p| p.join("skyglow"))
        .unwrap_or_else(|| PathBuf::from("data/skyglow"));
    let (ty, tm, td) = current_utc_date();
    let grid = match twilight_skyglow::dnb::DnbGrid::open(
        &cache,
        args.lat,
        args.lon,
        (ty, tm as u32, td as u32),
    ) {
        Some(g) => g,
        None => {
            eprintln!(
                "Warning: --skyglow-directional: no VIIRS Black Marble grid for this site \
                 (last 21 nights, both sensors); keeping the ISOTROPIC veil."
            );
            return None;
        }
    };
    let config = twilight_skyglow::garstang::GarstangConfig {
        observer_elevation: args.elevation,
        ..Default::default()
    };
    let directional = match twilight_skyglow::DirectionalSkyglow::from_radiance_source(
        &grid, args.lat, args.lon, config, 200.0,
    ) {
        Some(d) => d,
        None => {
            eprintln!(
                "Warning: --skyglow-directional: VIIRS grid {} shows no light sources \
                 within 200 km; keeping the ISOTROPIC veil.",
                grid.date
            );
            return None;
        }
    };

    // One-line summary at the khayt patch elevation (3 deg above the
    // horizon): veil toward the dawn azimuth vs toward the brightest
    // (city) azimuth vs the isotropic value the old path applies
    // everywhere (also the exact all-azimuth mean of the directional
    // veils, by the normalization contract).
    let elev = 3.0;
    let azs: Vec<f64> = (0..72).map(|i| i as f64 * 5.0).collect();
    if let Some(veils) =
        twilight_skyglow::directional_veils(&directional, radiance_nw, args.led_fraction, &azs, elev)
    {
        let iso_mes = veils.iter().map(|v| v.0).sum::<f64>() / veils.len() as f64;
        let (city_az, city_mes) = veils
            .iter()
            .enumerate()
            .map(|(i, v)| (azs[i], v.0))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((0.0, 0.0));
        let dawn_txt = match dawn_azimuth_deg.and_then(|az| {
            twilight_skyglow::directional_veils(
                &directional,
                radiance_nw,
                args.led_fraction,
                &[az],
                elev,
            )
            .map(|v| (az, v[0].0))
        }) {
            Some((az, v)) => format!("dawn az {:.0} deg {:.4}", az, v),
            None => "dawn az n/a".to_string(),
        };
        println!(
            "Skyglow dir: {} vs city az {:.0} deg {:.4} vs isotropic {:.4} cd/m^2 \
             (mesopic, 3 deg elev; VIIRS {}, {} source bins)",
            dawn_txt,
            city_az,
            city_mes,
            iso_mes,
            grid.date,
            directional.sources.len()
        );
    }
    Some(directional)
}

/// Assemble the pipeline input from the resolved pieces. Weather-derived
/// properties travel in the custom_* slots; manual flags go through the
/// type-based path.
#[allow(clippy::too_many_arguments)] // independently resolved inputs
fn build_input(
    args: &PrayArgs,
    date: (i32, i32, i32),
    tz_offset: f64,
    weather: Option<WeatherBlock>,
    cloud_layers: Option<Vec<CloudProperties>>,
    cloud_field: Option<twilight_data::cloud_field_builder::OwnedCloudField>,
    horizon_profile: Option<HorizonProfile>,
    skyglow: Option<SkyglowResult>,
    directional_skyglow: Option<twilight_skyglow::DirectionalSkyglow>,
    solar_f107: Option<f64>,
) -> PrayerTimeInput {
    let (year, month, day) = date;
    let (aerosol_type, cloud_type, custom_aerosol, custom_cloud, o3_du, no2_density, aod_sigma) =
        match weather {
            Some(w) => {
                let (o3, no2) = w
                    .gas
                    .map(|gc| (gc.o3_column_du, gc.no2_surface_density))
                    .unwrap_or((None, None));
                (None, None, w.aerosol, w.cloud, o3, no2, w.aod_sigma_550)
            }
            None => {
                let at = args.aerosol.to_aerosol_type();
                let ct = args.cloud.to_cloud_type();
                (at, ct, None, None, None, None, None)
            }
        };
    PrayerTimeInput {
        latitude: args.lat,
        longitude: args.lon,
        elevation: args.elevation,
        year,
        month,
        day,
        timezone: tz_offset,
        delta_t: args.delta_t,
        surface_albedo: args.albedo,
        sza_step: args.sza_step,
        aerosol_type,
        cloud_type,
        custom_aerosol,
        custom_cloud,
        aod_sigma_550: aod_sigma,
        de440_path: args.de440.clone(),
        scattering_mode: args.scattering.to_scattering_mode(),
        photons_per_wavelength: args.photons,
        horizon_profile,
        skyglow,
        directional_skyglow,
        o3_column_du: o3_du,
        no2_surface_density: no2_density,
        polarized: !args.fast,
        solar_f107,
        cloud_layers,
        cloud_field,
        verbose: args.verbose,
        ..Default::default()
    }
}

/// Timezone resolved for a location and date.
struct ResolvedTz {
    /// Engine offset [hours east of UTC], fixed for the run (the zone's
    /// offset at the date's UTC noon).
    offset_hours: f64,
    /// IANA zone for exact per-instant wall-clock conversion (None when
    /// the user forced a manual offset or the lookup failed).
    zone: Option<chrono_tz::Tz>,
    /// Human-readable provenance for the header line.
    label: String,
}

fn fmt_utc_offset(hours: f64) -> String {
    let total_min = (hours * 60.0).round() as i64;
    let sign = if total_min < 0 { '-' } else { '+' };
    let a = total_min.abs();
    format!("UTC{}{:02}:{:02}", sign, a / 60, a % 60)
}

/// Coordinates + date -> timezone. A wrong fixed offset is the largest
/// possible error in this program (a full hour, 30x anything the physics
/// can do), so the default is the IANA database: polygon lookup for the
/// zone name, then the zone's real offset for THIS date (DST-aware).
fn resolve_timezone(lat: f64, lon: f64, year: i32, month: i32, day: i32, manual: Option<f64>) -> ResolvedTz {
    if let Some(t) = manual {
        return ResolvedTz {
            offset_hours: t,
            zone: None,
            label: format!("{} (manual --tz)", fmt_utc_offset(t)),
        };
    }
    let finder = tzf_rs::DefaultFinder::default();
    let name = finder.get_tz_name(lon, lat);
    if let Ok(zone) = name.parse::<chrono_tz::Tz>() {
        use chrono::{NaiveDate, Offset, TimeZone};
        if let Some(noon_utc) = NaiveDate::from_ymd_opt(year, month as u32, day as u32)
            .and_then(|d| d.and_hms_opt(12, 0, 0))
        {
            let off =
                zone.offset_from_utc_datetime(&noon_utc).fix().local_minus_utc() as f64 / 3600.0;
            return ResolvedTz {
                offset_hours: off,
                zone: Some(zone),
                label: format!("{} ({}, IANA tzdb)", name, fmt_utc_offset(off)),
            };
        }
    }
    // Lookup failed (ocean, unparseable zone): longitude estimate, loudly.
    let off = (lon / 15.0).round();
    eprintln!(
        "Warning: timezone lookup failed for {lat:.3},{lon:.3}; using longitude estimate {} - pass --tz to override.",
        fmt_utc_offset(off)
    );
    ResolvedTz {
        offset_hours: off,
        zone: None,
        label: format!("{} (longitude estimate)", fmt_utc_offset(off)),
    }
}

/// Convert an engine-local fractional hour to the WALL-CLOCK fractional
/// hour via the zone rules at that exact instant. Identical to the input
/// almost always; differs exactly on DST-transition nights, where a fixed
/// offset cannot express the correct local time (transitions happen at
/// 02:00-03:00 - where Fajr lives).
fn wall_fractional_hour(
    h_local: f64,
    year: i32,
    month: i32,
    day: i32,
    engine_offset: f64,
    zone: Option<chrono_tz::Tz>,
) -> f64 {
    let Some(z) = zone else { return h_local };
    use chrono::{Datelike, Duration, NaiveDate, TimeZone, Timelike};
    let Some(base) = NaiveDate::from_ymd_opt(year, month as u32, day as u32) else {
        return h_local;
    };
    let utc = base.and_hms_opt(0, 0, 0).unwrap()
        - Duration::milliseconds((engine_offset * 3.6e6).round() as i64)
        + Duration::milliseconds((h_local * 3.6e6).round() as i64);
    let wall = z.from_utc_datetime(&utc);
    let days = (wall.date_naive() - base).num_days() as f64;
    let _ = wall.year();
    days * 24.0
        + wall.hour() as f64
        + wall.minute() as f64 / 60.0
        + wall.second() as f64 / 3600.0
}

/// Warn when --cloud-field data is older than this many hours: clouds
/// advect ~10 km per 10 minutes, so a 3 h old field no longer describes
/// the actual sky. A warning, not an error: computing tomorrow's Fajr
/// from tonight's export is legitimate planning.
const CLOUD_FIELD_STALE_HOURS: f64 = 3.0;

/// Footprint of a 3D cloud field as (lat_min, lat_max, lon_min, lon_max)
/// in degrees. lon_max may exceed 180 for grids crossing the
/// antimeridian; containment tests must therefore be wrap-safe.
fn cloud_field_footprint(
    f: &twilight_data::cloud_field_builder::OwnedCloudField,
) -> (f64, f64, f64, f64) {
    (
        f.lat0_deg,
        f.lat0_deg + f.nlat as f64 * f.dlat_deg,
        f.lon0_deg,
        f.lon0_deg + f.nlon as f64 * f.dlon_deg,
    )
}

/// Wrap-safe test that an observer lies inside a field footprint.
///
/// Outside the footprint the transport engine silently substitutes the
/// field's horizontal-mean background column, i.e. the wrong region's
/// clouds become a global uniform deck - so the CLI must hard-error
/// instead of accepting a field exported for another location.
fn observer_in_footprint(
    lat: f64,
    lon: f64,
    lat_min: f64,
    lat_max: f64,
    lon_min: f64,
    lon_max: f64,
) -> bool {
    if lat < lat_min || lat > lat_max {
        return false;
    }
    let lon_span = lon_max - lon_min;
    if lon_span >= 360.0 {
        return true;
    }
    (lon - lon_min).rem_euclid(360.0) <= lon_span
}

/// Parse a sidecar field timestamp: RFC 3339 ("2026-06-13T02:00:00Z")
/// or the naive "YYYY-MM-DDTHH:MM[:SS]" form the cloud3d sidecar emits,
/// which is UTC by contract.
fn parse_field_timestamp(ts: &str) -> Option<chrono::DateTime<chrono::Utc>> {
    if let Ok(t) = chrono::DateTime::parse_from_rfc3339(ts) {
        return Some(t.with_timezone(&chrono::Utc));
    }
    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"] {
        if let Ok(t) = chrono::NaiveDateTime::parse_from_str(ts, fmt) {
            return Some(t.and_utc());
        }
    }
    None
}

/// Footprint and staleness validation for a loaded --cloud-field.
/// Exits the process when the observer is outside the footprint; prints
/// a prominent warning when the field data is stale.
fn validate_cloud_field_for_observer(
    f: &twilight_data::cloud_field_builder::OwnedCloudField,
    lat: f64,
    lon: f64,
) {
    let (lat_min, lat_max, lon_min, lon_max) = cloud_field_footprint(f);
    if !observer_in_footprint(lat, lon, lat_min, lat_max, lon_min, lon_max) {
        eprintln!(
            "Error: --cloud-field footprint does not contain the observer.\n  \
             observer: lat {:.4}, lon {:.4}\n  \
             field:    lat {:.4}..{:.4}, lon {:.4}..{:.4}\n  \
             Outside the footprint the field's mean background column would \
             silently stand in for the whole sky (the wrong region's clouds \
             as a uniform deck). Re-export the field around this observer \
             (sidecar --field-out).",
            lat, lon, lat_min, lat_max, lon_min, lon_max
        );
        std::process::exit(1);
    }

    match parse_field_timestamp(&f.timestamp) {
        Some(t) => {
            let age_h = (chrono::Utc::now() - t).num_seconds() as f64 / 3600.0;
            if age_h > CLOUD_FIELD_STALE_HOURS {
                eprintln!(
                    "WARNING: --cloud-field data is {:.1} h old ({}). Clouds advect \
                     ~10 km per 10 min, so this field may no longer describe the \
                     actual sky; the times below are computed on the stale field. \
                     Re-export a fresh field when possible.",
                    age_h, f.timestamp
                );
            }
        }
        None => {
            eprintln!(
                "Note: --cloud-field timestamp {:?} is unparseable; staleness \
                 not checked.",
                f.timestamp
            );
        }
    }
}

fn cmd_pray(args: PrayArgs) {
    let (year, month, day) = resolve_date(&args.date);
    let tz = resolve_timezone(args.lat, args.lon, year, month, day, args.tz);

    println!("Twilight MCRT Prayer Time Calculator");
    println!("====================================");
    println!("Date:       {}-{:02}-{:02}", year, month, day);
    println!("Location:   {:.4}°N, {:.4}°E", args.lat, args.lon);
    println!("Elevation:  {:.0} m", args.elevation);
    println!("Timezone:   {}", tz.label);
    println!("Albedo:     {:.2}", args.albedo);
    println!("SZA step:   {:.2}°", args.sza_step);

    // The forecast sampling hour (evening twilight = SPA sunset + 45 min,
    // UTC) and the matching sun azimuth, shared by the weather block and
    // the cloud3d sidecar.
    let date_iso = format!("{}-{:02}-{:02}", year, month, day);
    let utc_spa = spa_input_for(
        args.lat,
        args.lon,
        args.elevation,
        0.0,
        args.delta_t,
        year,
        month,
        day,
    );
    let twilight_hour_utc = spa::solar_position(&SpaInput {
        hour: 12,
        ..utc_spa.clone()
    })
    .ok()
    .map(|o| (o.sunset + 0.75).rem_euclid(24.0));
    let sampling_hour_utc = twilight_hour_utc.unwrap_or(21.0);
    let sun_azimuth = sun_azimuth_at(&utc_spa, sampling_hour_utc).unwrap_or(270.0);
    let satellite_cache = Path::new(&args.dem_dir)
        .parent()
        .map(|p| p.join("satellite"))
        .unwrap_or_else(|| PathBuf::from("data/satellite"));

    let weather = if args.weather {
        Some(weather_block(
            args.lat,
            args.lon,
            &date_iso,
            twilight_hour_utc,
            sun_azimuth,
            &satellite_cache,
        ))
    } else {
        None
    };

    let cloud_field = args.cloud_field.as_deref().map(|p| {
        match twilight_weather::cloud3d::load_field(p) {
            Ok(f) => {
                println!(
                    "3D cloud field: {} voxels ({}x{}x{}), source {} @ {}",
                    f.sigma.iter().filter(|v| **v > 0.0).count(),
                    f.nz,
                    f.nlat,
                    f.nlon,
                    f.source,
                    f.timestamp
                );
                // Wrong-location fields hard-error; stale fields warn.
                validate_cloud_field_for_observer(&f, args.lat, args.lon);
                f
            }
            Err(e) => {
                eprintln!("Error: --cloud-field {}: {e}", p.display());
                std::process::exit(1);
            }
        }
    });

    let cloud_layers = if cloud_field.is_some() {
        if args.cloud3d.is_some() {
            println!("Note: --cloud-field overrides --cloud3d.");
        }
        None
    } else {
        args.cloud3d.as_deref().and_then(|spec| {
            resolve_cloud3d(
                spec,
                args.lat,
                args.lon,
                &date_iso,
                sampling_hour_utc,
                sun_azimuth,
                &satellite_cache,
            )
        })
    };

    let atm_desc = match weather.as_ref() {
        Some(w) => w.description.clone(),
        None => format_atm_desc(args.aerosol.to_aerosol_type(), args.cloud.to_cloud_type()),
    };
    println!("Atmosphere: {}", atm_desc);
    let ephemeris_label = if args.de440.is_some() {
        "JPL DE440"
    } else {
        "NREL SPA"
    };
    let method_str = match args.scattering.to_scattering_mode() {
        ScatteringMode::Single => "Single-scatter MCRT + CIE mesopic vision".to_string(),
        ScatteringMode::Multiple => format!(
            "Multiple-scatter MC ({} photons/wl) + CIE mesopic vision",
            args.photons
        ),
        ScatteringMode::Hybrid => format!(
            "Hybrid SS+MC ({} sec. rays/step) + CIE mesopic vision",
            args.photons
        ),
    };
    println!("Ephemeris:  {}", ephemeris_label);
    println!("Method:     {}", method_str);
    let pol_str = if args.fast {
        "Scalar (--fast)"
    } else {
        "Stokes [I,Q,U,V]"
    };
    println!("Polarize:   {}", pol_str);

    let horizon_profile = if args.terrain {
        resolve_terrain(args.lat, args.lon, &args.dem_dir, args.horizon_radius)
    } else {
        None
    };

    let skyglow_result = resolve_skyglow(&args, year, month, day);

    let directional_skyglow = if args.skyglow_directional {
        match &skyglow_result {
            Some(sg) => {
                // Dawn azimuth (SPA sunrise) for the summary line only;
                // the pipeline places patches at the real per-side sun
                // azimuths internally.
                let dawn_az = spa::solar_position(&SpaInput {
                    hour: 12,
                    ..utc_spa.clone()
                })
                .ok()
                .and_then(|o| sun_azimuth_at(&utc_spa, o.sunrise.rem_euclid(24.0)));
                resolve_directional_skyglow(&args, sg.integrated_radiance, dawn_az)
            }
            None => {
                eprintln!(
                    "Warning: --skyglow-directional needs a resolved --skyglow amplitude; \
                     ignoring."
                );
                None
            }
        }
    } else {
        None
    };

    // Measured solar activity for the airglow background. F10.7 is a real
    // daily-measured quantity (Penticton/NOAA SWPC) - only fetched when the
    // run is already online (weather mode); offline runs keep the mid-cycle
    // default inside the pipeline.
    let solar_f107 = if args.weather {
        match twilight_weather::f107::fetch_f107() {
            Ok(flux) => {
                println!(
                    "Solar flux: F10.7 = {:.0} sfu measured {} (90d mean {}) -> airglow input {:.0} sfu",
                    flux.latest_sfu,
                    flux.time_tag,
                    flux.ninety_day_mean_sfu
                        .map(|m| format!("{:.0}", m))
                        .unwrap_or_else(|| "n/a".to_string()),
                    flux.effective_sfu()
                );
                Some(flux.effective_sfu())
            }
            Err(e) => {
                eprintln!("Note: F10.7 fetch failed ({}); using mid-cycle 130 sfu.", e);
                None
            }
        }
    } else {
        None
    };

    println!();

    let input = build_input(
        &args,
        (year, month, day),
        tz.offset_hours,
        weather,
        cloud_layers,
        cloud_field,
        horizon_profile,
        skyglow_result,
        directional_skyglow,
        solar_f107,
    );

    // GPU initialization. GPU is the default compute backend for all modes.
    // Single-scatter benefits from batched dispatch (2.5x faster than serial).
    // MC/hybrid benefits from massive parallelism across photons/wavelengths.
    // Use --cpu to opt out.

    #[cfg(feature = "gpu")]
    let mut gpu_backend = if args.cpu {
        None
    } else if input.cloud_field.is_some() {
        // 3D cloud field: every scattering mode runs the CPU reference
        // scan until the GPU field port is re-verified (the single/mcrt
        // kernels are field-blind and would silently compute a clear
        // sky), so a GPU backend would go unused; skip the init.
        println!(
            "Note: 3D cloud field active; using the CPU reference scan \
             (GPU field port pending re-verification)."
        );
        None
    } else {
        try_init_gpu(args.gpu_backend, args.photons)
    };

    let compute_label = {
        #[cfg(feature = "gpu")]
        {
            if let Some(backend) = gpu_backend.as_ref() {
                // Honest label: a backend initialized, but individual
                // kernels may still fall back to CPU on dispatch failure.
                // Name the backend that actually initialized rather than
                // assuming Metal; on non-Apple hosts this is wgpu.
                format!(
                    "GPU ({}; per-kernel CPU fallback)",
                    backend.device_info().backend
                )
            } else {
                "CPU (rayon)".to_string()
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            // gpu feature off: the GPU flags exist but cannot take effect.
            let _ = (args.cpu, args.gpu_backend);
            "CPU (rayon)".to_string()
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

    report(&args, &input, &output, tz.zone);
}

/// All result printing for `pray`: prayer-time blocks, terrain and
/// skyglow adjustments, the khayt and legacy methods, the conventional
/// fixed-angle comparison, the verbose analysis table and the notes.
fn report(
    args: &PrayArgs,
    input: &PrayerTimeInput,
    output: &PrayerTimeOutput,
    zone: Option<chrono_tz::Tz>,
) {
    // Wall-clock conversion: exact per-instant zone rules (DST-transition
    // nights differ from the fixed engine offset precisely where Fajr
    // lives). Identity when no zone is known.
    let wall = |h: f64| -> f64 {
        wall_fractional_hour(h, input.year, input.month, input.day, input.timezone, zone)
    };
    println!();

    // Print results
    println!("Prayer Times (MCRT-derived):");
    println!("{:-<65}", "");

    // Sunrise/Sunset
    println!(
        "  Sunrise:              {}",
        output
            .sunrise_time
            .map(|h| format_fractional_hour(wall(h)))
            .unwrap_or("N/A".to_string())
    );
    println!(
        "  Sunset:               {}",
        output
            .sunset_time
            .map(|h| format_fractional_hour(wall(h)))
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

    // Threshold refloat on the measured celestial background (moonlit
    // nights, strong airglow): the pipeline computes the note, the CLI
    // reports it.
    if let Some(ref note) = output.celestial_refloat {
        println!("  {}", note);
    }

    // ── PRIMARY: the khayt al-abyad criterion (Quran 2:187) ──
    let kh = &output.khayt;
    if let (Some(time), Some(sza)) = (kh.fajr_time, kh.fajr_sza_deg) {
        println!(
            "  Fajr (khayt al-abyad): {}{}  (SZA {:.2}°, depression {:.2}°)",
            format_fractional_hour(wall(time)),
            format_uncertainty(kh.fajr_uncertainty_min),
            sza,
            sza - 90.0
        );
        println!(
            "    └ white thread distinct from black + lateral spread (2:187, mustatir)"
        );
        if let Some(kt) = kh.kadhib_time {
            println!(
                "    └ false dawn (al-fajr al-kadhib) visible from {} - do not pray Fajr yet",
                format_fractional_hour(wall(kt))
            );
        }
    } else {
        println!("  Fajr (khayt al-abyad): N/A (contrast criterion not crossed in scan)");
    }
    if let (Some(time), Some(sza)) = (kh.isha_ahmar_time, kh.isha_ahmar_sza_deg) {
        println!(
            "  Isha (shafaq ahmar):   {}{}  (SZA {:.2}°, depression {:.2}°)",
            format_fractional_hour(wall(time)),
            format_uncertainty(kh.isha_ahmar_uncertainty_min),
            sza,
            sza - 90.0
        );
        println!("    └ red band no longer distinct - Shafi'i/Maliki/Hanbali (primary)");
    } else {
        println!("  Isha (shafaq ahmar):   N/A (contrast criterion not crossed in scan)");
    }
    if let (Some(time), Some(sza)) = (kh.isha_abyad_time, kh.isha_abyad_sza_deg) {
        println!(
            "  Isha (shafaq abyad):   {}{}  (SZA {:.2}°, depression {:.2}°)",
            format_fractional_hour(wall(time)),
            format_uncertainty(kh.isha_abyad_uncertainty_min),
            sza,
            sza - 90.0
        );
        println!("    └ white band no longer distinct - Hanafi");
    }
    println!();
    println!("  Legacy absolute-threshold method (comparison):");

    // Fajr
    if let (Some(time), Some(sza), Some(dep)) = (
        output.fajr_time,
        output.fajr_sza_deg,
        output.fajr_depression_deg,
    ) {
        println!(
            "  Fajr (true dawn):     {}{}  (SZA {:.2}°, depression {:.2}°)",
            format_fractional_hour(wall(time)),
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
        println!("  Fajr (true dawn):     N/A (no night at all - midnight sun)");
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
            format_fractional_hour(wall(time)),
            format_uncertainty(output.isha_abyad_uncertainty_min),
            sza,
            dep
        );
        println!("    └ Hanafi school - white twilight disappears");
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
            format_fractional_hour(wall(time)),
            format_uncertainty(output.isha_ahmar_uncertainty_min),
            sza,
            dep
        );
        println!("    └ Shafi'i/Maliki/Hanbali - red glow disappears");
    } else {
        println!("  Isha (al-ahmar):      N/A (threshold not crossed in scan range)");
    }

    println!("{:-<65}", "");

    // Compare with conventional
    println!();
    println!("Comparison with conventional fixed-angle methods:");
    println!("{:-<65}", "");

    let base_input = spa_input_for(
        input.latitude,
        input.longitude,
        input.elevation,
        input.timezone,
        input.delta_t,
        input.year,
        input.month,
        input.day,
    );

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
            .map(|h| format_fractional_hour(wall(h)))
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
    if input.verbose {
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
    match input.scattering_mode {
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
    let has_aerosol = input.aerosol_type.is_some() || input.custom_aerosol.is_some();
    let has_cloud = input.cloud_type.is_some() || input.custom_cloud.is_some();
    if args.weather {
        println!("  - Atmosphere configured from live Open-Meteo weather data.");
        if has_aerosol {
            println!("  - Aerosol optical properties derived from measured AOD.");
        }
        if has_cloud {
            println!("  - Cloud layer derived from observed cloud cover.");
        }
        if input.o3_column_du.is_some() || input.no2_surface_density.is_some() {
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

// ===================== SQM field calibration =====================
//
// `sqm predict` produces the engine's forecast of what a zenith-pointed
// Sky Quality Meter should read through one night; `sqm compare` aligns
// a real meter log to that forecast and reports the bias binned by
// solar depression. The depression-binned bias IS the field measurement
// that calibrates the twilight thresholds (docs/SQM_CAMPAIGN.md).

/// One predicted point of the night curve.
struct SqmPoint {
    /// Unix epoch seconds (UTC) of the instant.
    epoch_utc: f64,
    /// Solar zenith angle [deg].
    sza_deg: f64,
    /// Total zenith luminance [cd/m^2]: MCRT twilight + celestial
    /// background + skyglow.
    total_cd: f64,
    /// The same, as mag/arcsec^2 (what the SQM displays).
    mag: f64,
}

/// A predicted night curve plus the metadata to label it.
struct SqmCurve {
    points: Vec<SqmPoint>,
    tz_offset: f64,
    tz_label: String,
    atm_desc: String,
    glow_desc: String,
    date: (i32, i32, i32),
    /// Local fractional hours (relative to the start date's midnight).
    sunset_h: f64,
    sunrise_h: f64,
}

/// The civil day after (y, m, d).
fn next_civil_day(year: i32, month: i32, day: i32) -> (i32, i32, i32) {
    use chrono::Datelike;
    chrono::NaiveDate::from_ymd_opt(year, month as u32, day as u32)
        .and_then(|d| d.succ_opt())
        .map(|d| (d.year(), d.month() as i32, d.day() as i32))
        .unwrap_or((year, month, day + 1))
}

/// Unix epoch seconds (UTC) of local midnight on (y, m, d) in a fixed
/// UTC offset frame.
fn local_midnight_epoch(year: i32, month: i32, day: i32, tz_offset: f64) -> f64 {
    let naive = chrono::NaiveDate::from_ymd_opt(year, month as u32, day as u32)
        .and_then(|d| d.and_hms_opt(0, 0, 0))
        .map(|dt| dt.and_utc().timestamp() as f64)
        .unwrap_or(0.0);
    naive - tz_offset * 3600.0
}

/// Epoch seconds -> "YYYY-MM-DDTHH:MM:SSZ".
fn iso_utc(epoch: f64) -> String {
    chrono::DateTime::<chrono::Utc>::from_timestamp(epoch.round() as i64, 0)
        .map(|d| d.format("%Y-%m-%dT%H:%M:%SZ").to_string())
        .unwrap_or_else(|| "invalid".to_string())
}

/// Epoch seconds -> local ISO with the fixed engine offset
/// ("YYYY-MM-DDTHH:MM:SS+02:00").
fn iso_local(epoch: f64, tz_offset: f64) -> String {
    let off = chrono::FixedOffset::east_opt((tz_offset * 3600.0).round() as i32);
    match (
        chrono::DateTime::<chrono::Utc>::from_timestamp(epoch.round() as i64, 0),
        off,
    ) {
        (Some(d), Some(o)) => d
            .with_timezone(&o)
            .format("%Y-%m-%dT%H:%M:%S%:z")
            .to_string(),
        _ => "invalid".to_string(),
    }
}

/// Parse an ISO timestamp to Unix epoch seconds. Explicit offsets
/// (Z, +02:00) are honored; bare timestamps are taken as UTC - the
/// convention of the Unihedron log's first column.
fn parse_epoch_utc(s: &str) -> Option<f64> {
    let s = s.trim();
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return Some(dt.timestamp() as f64);
    }
    for fmt in ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%d %H:%M:%S%.f"] {
        if let Ok(ndt) = chrono::NaiveDateTime::parse_from_str(s, fmt) {
            return Some(ndt.and_utc().timestamp() as f64);
        }
    }
    None
}

/// Parsed SQM readings (epoch_utc seconds, mag) plus the detected
/// format label.
type SqmLog = (Vec<(f64, f64)>, &'static str);

/// Parse an SQM log into (epoch_utc, mag) readings.
///
/// Two formats, autodetected per file (the one that parses more lines
/// wins):
/// - Unihedron SQM-LE/LU: `#` comment headers, then semicolon-separated
///   `utc_iso;local_iso;temperature;counts;Hz;mag` (UTC field first,
///   magnitude last).
/// - Simple CSV: `timestamp_iso,mag` (bare timestamps read as UTC).
fn parse_sqm_log(path: &str) -> Result<SqmLog, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("cannot read {}: {}", path, e))?;
    let mut unihedron: Vec<(f64, f64)> = Vec::new();
    let mut csv: Vec<(f64, f64)> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let semi: Vec<&str> = line.split(';').collect();
        if semi.len() >= 6 {
            if let (Some(t), Ok(m)) = (
                parse_epoch_utc(semi[0]),
                semi[semi.len() - 1].trim().parse::<f64>(),
            ) {
                unihedron.push((t, m));
                continue;
            }
        }
        let comma: Vec<&str> = line.split(',').collect();
        if comma.len() >= 2 {
            if let (Some(t), Ok(m)) = (parse_epoch_utc(comma[0]), comma[1].trim().parse::<f64>())
            {
                csv.push((t, m));
            }
        }
    }
    if unihedron.is_empty() && csv.is_empty() {
        return Err(format!(
            "no readings parsed from {} (expected Unihedron semicolon format or timestamp_iso,mag CSV)",
            path
        ));
    }
    if unihedron.len() >= csv.len() {
        Ok((unihedron, "unihedron"))
    } else {
        Ok((csv, "csv"))
    }
}

/// Artificial zenith luminance [cd/m^2] for the sqm commands, plus its
/// provenance. Priority: --radiance, --bortle, --skyglow (Falchi atlas
/// value at its epoch; the DNB temporal rescale of `pray --skyglow` is
/// deliberately skipped here - a field campaign should record the site
/// and pass --radiance or --bortle for a stable, reproducible input).
fn sqm_skyglow_cd(args: &SqmArgs) -> (f64, String) {
    let from_radiance = |r: f64, label: String| {
        let mcd = twilight_skyglow::bortle::radiance_to_zenith_luminance(r);
        (mcd * 1e-3, format!("{} -> {:.3} mcd/m^2 artificial", label, mcd))
    };
    if let Some(r) = args.radiance {
        from_radiance(r, format!("VIIRS {:.2} nW/cm^2/sr", r))
    } else if let Some(b) = args.bortle {
        from_radiance(
            twilight_skyglow::bortle::bortle_to_radiance(b),
            format!("Bortle {}", b),
        )
    } else if args.skyglow {
        let cache = repo_root().join("data/skyglow");
        match twilight_skyglow::atlas::artificial_zenith(&cache, args.lat, args.lon) {
            Some(a) => (
                a.zenith_mcd * 1e-3,
                format!(
                    "satellite atlas {} -> {:.3} mcd/m^2 artificial",
                    a.year, a.zenith_mcd
                ),
            ),
            None => {
                eprintln!(
                    "Note: no skyglow atlas data here; running without skyglow - pass --bortle or --radiance."
                );
                (0.0, "none (atlas lookup failed)".to_string())
            }
        }
    } else {
        (0.0, "none".to_string())
    }
}

/// Predict the zenith sky-brightness curve for the night starting on
/// args.date: local sunset (SPA zenith crossing at 90.8333 deg) to the
/// next morning's sunrise, in steps of --step-min.
///
/// Per step, three luminance components are summed [cd/m^2]:
/// 1. MCRT twilight sky: simulate_at_sza at view_zenith 0, reduced to
///    CIE mesopic luminance (the engine's threshold currency; at
///    twilight light levels mesopic tracks photopic, which is close to
///    the SQM's response). Skipped for SZA > 110 deg, where the solar
///    contribution sits orders of magnitude below the celestial floor.
/// 2. Celestial background: the full measured-data night-sky model
///    (airglow at the F10.7 = 130 mid-cycle default, Leinert zodiacal
///    table, Pioneer starlight map, Meeus-series moon) - NOT the
///    dark-sky constant, so moonlit calibration nights predict
///    correctly. Extinction k mirrors the pipeline: 0.16 gas + 1.2 per
///    AOD (0.05 default aerosol term without weather).
/// 3. Configured skyglow (constant across the night).
///
/// Conversion to mag/arcsec^2 via luminance_to_sqm, which takes
/// mcd/m^2: total_cd * 1e3.
fn sqm_predict_curve(args: &SqmArgs) -> SqmCurve {
    let (year, month, day) = resolve_date(&args.date);
    let tz = resolve_timezone(args.lat, args.lon, year, month, day, args.tz);

    // Night window from real SPA zenith crossings (refraction-corrected
    // bisection, not the approximate transit formula).
    let spa_d = spa_input_for(
        args.lat,
        args.lon,
        args.elevation,
        tz.offset_hours,
        args.delta_t,
        year,
        month,
        day,
    );
    let (y2, m2, d2) = next_civil_day(year, month, day);
    let spa_d2 = spa_input_for(
        args.lat,
        args.lon,
        args.elevation,
        tz.offset_hours,
        args.delta_t,
        y2,
        m2,
        d2,
    );
    let sunset = spa::find_zenith_crossing(&spa_d, 90.8333, 12.0, 24.0, 0.0001);
    let sunrise = spa::find_zenith_crossing(&spa_d2, 90.8333, 0.0, 12.0, 0.0001);
    let (Some(sunset_h), Some(sunrise_next)) = (sunset, sunrise) else {
        eprintln!(
            "Error: no sunset/sunrise crossing on {} at {:.3},{:.3} (polar day or polar night) - sqm needs a real night.",
            args.date, args.lat, args.lon
        );
        std::process::exit(1);
    };
    let sunrise_h = sunrise_next + 24.0;

    // Atmosphere: live weather or US Standard clear sky, mirroring `pray`.
    let date_iso = format!("{}-{:02}-{:02}", year, month, day);
    let (aerosol_props, cloud_props, gas, atm_desc) = if args.weather {
        let utc_spa = spa_input_for(
            args.lat,
            args.lon,
            args.elevation,
            0.0,
            args.delta_t,
            year,
            month,
            day,
        );
        let twilight_hour_utc = spa::solar_position(&SpaInput {
            hour: 12,
            ..utc_spa.clone()
        })
        .ok()
        .map(|o| (o.sunset + 0.75).rem_euclid(24.0));
        let sun_az =
            sun_azimuth_at(&utc_spa, twilight_hour_utc.unwrap_or(21.0)).unwrap_or(270.0);
        let w = weather_block(
            args.lat,
            args.lon,
            &date_iso,
            twilight_hour_utc,
            sun_az,
            Path::new("data/satellite"),
        );
        (w.aerosol, w.cloud, w.gas, w.description)
    } else {
        (
            None,
            None,
            None,
            "US Standard 1976 (clear sky)".to_string(),
        )
    };
    let atm = if let Some(ref gc) = gas {
        builder::build_full_with_gas(
            AtmosphereType::UsStandard,
            args.albedo,
            aerosol_props.as_ref(),
            cloud_props.as_ref(),
            gc.o3_column_du,
            gc.no2_surface_density,
        )
    } else {
        builder::build_full(
            AtmosphereType::UsStandard,
            args.albedo,
            aerosol_props.as_ref(),
            cloud_props.as_ref(),
        )
    };

    // Broadband V extinction for the celestial components, same recipe
    // as the prayer pipeline (gas 0.16 + aerosol term).
    let extinction_k = 0.16
        + aerosol_props
            .as_ref()
            .map(|a| 1.2 * a.aod_550)
            .unwrap_or(0.05);

    let config = SimulationConfig {
        latitude: args.lat,
        longitude: args.lon,
        elevation: args.elevation,
        // Irrelevant for a zenith view: the scattering angle is fixed
        // by the SZA alone.
        solar_azimuth: 270.0,
        view_zenith: 0.0,
        view_azimuth: None,
        apply_solar_irradiance: true,
        scattering_mode: args.scattering.to_scattering_mode(),
        photons_per_wavelength: args.photons,
        polarized: true,
        seed_salt: 0,
    };
    let threshold_config = twilight_threshold::threshold::ThresholdConfig::default();
    let (glow_cd, glow_desc) = sqm_skyglow_cd(args);
    let beam = beam_samples(args.beam_fwhm, config.solar_azimuth);
    if args.beam_fwhm > 0.0 {
        println!(
            "Beam:       {:.0} deg FWHM, {} samples per step (Gaussian response)",
            args.beam_fwhm,
            beam.len()
        );
    }

    let base_epoch = local_midnight_epoch(year, month, day, tz.offset_hours);
    let step_h = args.step_min.max(0.25) / 60.0;
    let mut points = Vec::new();
    let mut h = sunset_h;
    while h <= sunrise_h + 1e-9 {
        // SZA via SPA; hours past 24 roll to the next civil day.
        let (base, hh) = if h < 24.0 {
            (&spa_d, h)
        } else {
            (&spa_d2, h - 24.0)
        };
        let mut inp = base.clone();
        let total_s = (hh * 3600.0).round() as i32;
        inp.hour = total_s / 3600;
        inp.minute = (total_s % 3600) / 60;
        inp.second = total_s % 60;
        let Ok(pos) = spa::solar_position(&inp) else {
            h += step_h;
            continue;
        };
        let sza = pos.zenith;

        let sun_cd = if sza <= 110.0 {
            // Beam integration: one MCRT evaluation per sample, combined
            // by the meter's angular response. With --beam-fwhm 0 this is
            // exactly the historical single zenith-point evaluation.
            let mut acc = 0.0;
            for &(vz, az, w) in &beam {
                let mut c = config.clone();
                c.view_zenith = vz;
                if vz > 0.0 {
                    c.view_azimuth = Some(az);
                }
                let result = simulation::simulate_at_sza(&atm, &c, sza, None);
                let analysis = twilight_threshold::threshold::analyze_twilight(
                    sza,
                    &result.wavelengths_nm,
                    &result.radiance,
                    &threshold_config,
                );
                acc += w * analysis.luminance_mesopic;
            }
            acc
        } else {
            0.0
        };

        // Raw (un-wrapped) UTC hour relative to the START date: the
        // night-sky model's Julian-day arithmetic places hour 25+ on
        // the correct next civil day (same convention as the pipeline).
        let night_inp = twilight_threshold::night_sky::NightSkyInput {
            latitude: args.lat,
            longitude: args.lon,
            year,
            month,
            day,
            hour_utc: h - tz.offset_hours,
            view_zenith_deg: 0.0,
            view_azimuth_deg: 0.0,
            solar_f107: 130.0,
            extinction_k,
        };
        let night = twilight_threshold::night_sky::night_sky_luminance(&night_inp);

        let total_cd = sun_cd + night.total + glow_cd;
        let mag = twilight_skyglow::bortle::luminance_to_sqm(total_cd * 1e3);
        points.push(SqmPoint {
            epoch_utc: base_epoch + h * 3600.0,
            sza_deg: sza,
            total_cd,
            mag,
        });
        h += step_h;
    }

    SqmCurve {
        points,
        tz_offset: tz.offset_hours,
        tz_label: tz.label,
        atm_desc,
        glow_desc,
        date: (year, month, day),
        sunset_h,
        sunrise_h,
    }
}

/// Human summary of a predicted curve: the darkest point and the two
/// times the curve enters/leaves the 0.1-mag band above the floor
/// (predicted twilight end and dawn start as an SQM would see them).
fn sqm_summary(curve: &SqmCurve) -> String {
    let mut s = String::new();
    let (y, m, d) = curve.date;
    s.push_str(&format!(
        "Night of {}-{:02}-{:02}: sunset {} to sunrise {} ({} points)\n",
        y,
        m,
        d,
        format_fractional_hour(curve.sunset_h),
        format_fractional_hour(curve.sunrise_h),
        curve.points.len()
    ));
    let Some(darkest) = curve
        .points
        .iter()
        .max_by(|a, b| a.mag.partial_cmp(&b.mag).unwrap_or(std::cmp::Ordering::Equal))
    else {
        s.push_str("No points predicted.\n");
        return s;
    };
    let floor = darkest.mag;
    s.push_str(&format!(
        "Darkest:      {:.2} mag/arcsec^2 ({:.3e} cd/m^2) at {}\n",
        floor,
        darkest.total_cd,
        iso_local(darkest.epoch_utc, curve.tz_offset)
    ));
    let edge = floor - 0.1;
    if let Some(p) = curve.points.iter().find(|p| p.mag >= edge) {
        s.push_str(&format!(
            "Twilight end: {} (first point within 0.1 mag of the floor; SZA {:.1} deg)\n",
            iso_local(p.epoch_utc, curve.tz_offset),
            p.sza_deg
        ));
    }
    if let Some(p) = curve.points.iter().rev().find(|p| p.mag >= edge) {
        s.push_str(&format!(
            "Dawn start:   {} (last point within 0.1 mag of the floor; SZA {:.1} deg)\n",
            iso_local(p.epoch_utc, curve.tz_offset),
            p.sza_deg
        ));
    }
    s
}

fn cmd_sqm_predict(args: SqmArgs) {
    let curve = sqm_predict_curve(&args);
    let mut csv = String::new();
    csv.push_str(&format!(
        "# twilight sqm predict: date={}-{:02}-{:02} lat={} lon={} elevation={}m tz={} atmosphere=\"{}\" skyglow=\"{}\" scattering={:?} step_min={}\n",
        curve.date.0,
        curve.date.1,
        curve.date.2,
        args.lat,
        args.lon,
        args.elevation,
        curve.tz_label,
        curve.atm_desc,
        curve.glow_desc,
        args.scattering.to_scattering_mode(),
        args.step_min
    ));
    csv.push_str("time_local_iso,time_utc_iso,sza_deg,sim_total_cd_m2,sim_mag_arcsec2\n");
    for p in &curve.points {
        csv.push_str(&format!(
            "{},{},{:.3},{:.6e},{:.3}\n",
            iso_local(p.epoch_utc, curve.tz_offset),
            iso_utc(p.epoch_utc),
            p.sza_deg,
            p.total_cd,
            p.mag
        ));
    }
    let summary = sqm_summary(&curve);
    match &args.out {
        Some(path) => {
            if let Err(e) = std::fs::write(path, &csv) {
                eprintln!("Error: cannot write {}: {}", path, e);
                std::process::exit(1);
            }
            println!("Wrote {} points to {}", curve.points.len(), path);
            print!("{}", summary);
        }
        None => {
            print!("{}", csv);
            eprint!("{}", summary);
        }
    }
}

fn cmd_sqm_compare(args: SqmCompareArgs) {
    let (readings, log_format) = match parse_sqm_log(&args.log) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };
    let curve = sqm_predict_curve(&args.base);
    if curve.points.len() < 2 {
        eprintln!("Error: predicted curve has fewer than 2 points.");
        std::process::exit(1);
    }
    let pts = &curve.points;
    let t0 = pts[0].epoch_utc;
    let t1 = pts[pts.len() - 1].epoch_utc;

    // Align each reading to the predicted curve by UTC time (linear
    // interpolation), keeping only points with the sun below the
    // horizon (depression >= 0).
    let mut aligned: Vec<(f64, f64)> = Vec::new(); // (depression_deg, sim - measured)
    for &(t, meas) in &readings {
        if t < t0 || t > t1 {
            continue;
        }
        let i = pts
            .partition_point(|p| p.epoch_utc <= t)
            .clamp(1, pts.len() - 1);
        let (a, b) = (&pts[i - 1], &pts[i]);
        let span = b.epoch_utc - a.epoch_utc;
        let f = if span > 0.0 { (t - a.epoch_utc) / span } else { 0.0 };
        let sim_mag = a.mag + f * (b.mag - a.mag);
        let sza = a.sza_deg + f * (b.sza_deg - a.sza_deg);
        let depression = sza - 90.0;
        if depression < 0.0 {
            continue;
        }
        aligned.push((depression, sim_mag - meas));
    }

    println!("SQM Comparison Report");
    println!("=====================");
    println!(
        "Log:        {} (format: {}, {} readings)",
        args.log,
        log_format,
        readings.len()
    );
    println!(
        "Night:      {} to {} ({} predicted points, step {} min)",
        iso_local(t0, curve.tz_offset),
        iso_local(t1, curve.tz_offset),
        pts.len(),
        args.base.step_min
    );
    println!("Atmosphere: {}", curve.atm_desc);
    println!("Skyglow:    {}", curve.glow_desc);
    println!(
        "Aligned:    {} readings inside the night window (depression >= 0)",
        aligned.len()
    );
    if aligned.len() < 10 {
        eprintln!(
            "Error: only {} readings align with the predicted night (need >= 10).",
            aligned.len()
        );
        std::process::exit(1);
    }

    let n = aligned.len() as f64;
    let mean: f64 = aligned.iter().map(|(_, o)| o).sum::<f64>() / n;
    let rms: f64 = (aligned.iter().map(|(_, o)| o * o).sum::<f64>() / n).sqrt();
    println!();
    println!("Mean offset (sim - measured): {:+.3} mag", mean);
    println!("RMS offset:                   {:.3} mag", rms);
    println!();
    println!("Offset by solar depression (the threshold-calibration measurement):");
    let bins: [(f64, f64, &str); 4] = [
        (0.0, 6.0, "0-6 deg  (civil)"),
        (6.0, 12.0, "6-12 deg (nautical)"),
        (12.0, 18.0, "12-18 deg (astronomical)"),
        (18.0, 180.0, "18+ deg  (night floor)"),
    ];
    for (lo, hi, label) in bins {
        let sel: Vec<f64> = aligned
            .iter()
            .filter(|(d, _)| *d >= lo && *d < hi)
            .map(|(_, o)| *o)
            .collect();
        if sel.is_empty() {
            println!("  {:<26} n={:>4}  (no data)", label, 0);
        } else {
            let bn = sel.len() as f64;
            let bm = sel.iter().sum::<f64>() / bn;
            let br = (sel.iter().map(|o| o * o).sum::<f64>() / bn).sqrt();
            println!(
                "  {:<26} n={:>4}  mean {:+.3}  rms {:.3}",
                label,
                sel.len(),
                bm,
                br
            );
        }
    }
    println!();
    println!("Positive offset = engine predicts a DARKER sky than measured.");
    println!("Feed the depression-binned bias back per docs/SQM_CAMPAIGN.md.");
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
                None,
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

    // Estimator A/B switch: TWILIGHT_BDPT_CONN_OFF=1 disables the deep-SZA
    // chain-vertex connection estimator entirely (pre-connection engine),
    // for criterion-recalibration comparisons. Default: connections on.
    if std::env::var("TWILIGHT_BDPT_CONN_OFF").as_deref() == Ok("1") {
        twilight_core::photon::BDPT_CHAIN_CONN_DISABLE
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

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
        Commands::Pray(args) => {
            cmd_pray(args);
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
            cloud_tau,
            cloud_base_km,
            cloud_top_km,
            cloud_ssa,
            cloud_g,
            seed_salt,
            cloud_field,
        } => {
            // Custom uniform deck for the external slab referee (G2).
            let custom_cloud = cloud_tau.map(|tau| CloudProperties {
                base_km: cloud_base_km,
                top_km: cloud_top_km,
                optical_depth: tau,
                ssa: cloud_ssa,
                asymmetry: cloud_g,
            });
            // 3D field referee surface: mirrors `pray --cloud-field`
            // (same loader, same footprint/staleness validation).
            let cloud_field = cloud_field.as_deref().map(|p| {
                match twilight_weather::cloud3d::load_field(p) {
                    Ok(f) => {
                        eprintln!(
                            "3D cloud field: {} voxels ({}x{}x{}), source {} @ {}",
                            f.sigma.iter().filter(|v| **v > 0.0).count(),
                            f.nz,
                            f.nlat,
                            f.nlon,
                            f.source,
                            f.timestamp
                        );
                        validate_cloud_field_for_observer(&f, lat, lon);
                        f
                    }
                    Err(e) => {
                        eprintln!("Error: --cloud-field {}: {e}", p.display());
                        std::process::exit(1);
                    }
                }
            });
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
                custom_cloud,
                o3_du,
                scattering,
                photons,
                fast,
                no_refraction,
                seed_salt,
                cloud_field,
            );
        }
        Commands::Sqm { action } => match action {
            SqmCommands::Predict(args) => cmd_sqm_predict(args),
            SqmCommands::Compare(args) => cmd_sqm_compare(args),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── SQM beam response ──

    #[test]
    fn beam_zero_fwhm_is_the_zenith_point() {
        let s = beam_samples(0.0, 270.0);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].0, 0.0, "on-axis");
        assert!((s[0].2 - 1.0).abs() < 1e-12, "all weight on the point");
    }

    #[test]
    fn beam_weights_are_normalized_and_on_axis_dominates() {
        for fwhm in [20.0, 84.0] {
            let s = beam_samples(fwhm, 270.0);
            let total: f64 = s.iter().map(|(_, _, w)| w).sum();
            assert!(
                (total - 1.0).abs() < 1e-12,
                "fwhm {fwhm}: weights sum to {total}"
            );
            // Every sample must look at SKY. A sample at or past 90 deg
            // of zenith angle points into the ground, and the sky model
            // evaluated there returns a grazing radiance that swamps the
            // integral: an 84-degree beam integrated to 1.2x FWHM without
            // this clamp predicted a "darkest" night of 8.97 mag/arcsec^2,
            // brighter than daylight.
            for (vz, _, w) in &s {
                assert!(*vz >= 0.0, "vz {vz} not negative");
                assert!(*vz < 90.0, "vz {vz} must stay above the horizon");
                assert!(*vz <= 1.2 * fwhm + 1e-9, "vz {vz} inside integration radius");
                assert!(*w > 0.0, "positive weight");
            }
            // The response is peaked, so the mean offset must sit well
            // inside the FWHM: a beam whose weight ran to the rim would
            // be sampling the wrong sky.
            let mean_off: f64 = s.iter().map(|(vz, _, w)| vz * w).sum();
            assert!(
                mean_off < fwhm,
                "fwhm {fwhm}: mean offset {mean_off} should sit inside the FWHM"
            );
        }
    }

    #[test]
    fn beam_rings_sample_four_sun_relative_azimuths() {
        let s = beam_samples(20.0, 100.0);
        // Off-axis samples must come in complete azimuth quartets so the
        // sunward gradient is captured rather than assumed symmetric.
        let off: Vec<_> = s.iter().filter(|(vz, _, _)| *vz > 0.0).collect();
        assert_eq!(off.len() % 4, 0, "azimuths come in quartets");
        let rel: Vec<f64> = off.iter().take(4).map(|(_, az, _)| az - 100.0).collect();
        for want in [0.0, 90.0, 180.0, 270.0] {
            assert!(
                rel.iter().any(|r| (r - want).abs() < 1e-9),
                "missing sun-relative azimuth {want} in {rel:?}"
            );
        }
    }

    // ── --cloud-field observer/footprint validation ──

    #[test]
    fn footprint_contains_interior_and_edges() {
        // Padborg-style field: lat 54.0..55.0, lon 9.0..10.0.
        assert!(observer_in_footprint(54.5, 9.5, 54.0, 55.0, 9.0, 10.0));
        // Boundary points are inside.
        assert!(observer_in_footprint(54.0, 9.0, 54.0, 55.0, 9.0, 10.0));
        assert!(observer_in_footprint(55.0, 10.0, 54.0, 55.0, 9.0, 10.0));
    }

    #[test]
    fn footprint_rejects_outside_observers() {
        // North/south of the grid.
        assert!(!observer_in_footprint(55.5, 9.5, 54.0, 55.0, 9.0, 10.0));
        assert!(!observer_in_footprint(53.9, 9.5, 54.0, 55.0, 9.0, 10.0));
        // East/west of the grid.
        assert!(!observer_in_footprint(54.5, 10.5, 54.0, 55.0, 9.0, 10.0));
        assert!(!observer_in_footprint(54.5, 8.9, 54.0, 55.0, 9.0, 10.0));
        // A Mecca observer against a Denmark field: the exact silent
        // uniform-deck case this validation exists to stop.
        assert!(!observer_in_footprint(21.4225, 39.8262, 54.0, 55.0, 9.0, 10.0));
    }

    #[test]
    fn footprint_longitude_is_wrap_safe() {
        // Field crossing the antimeridian: lon 179..181 (= -179).
        assert!(observer_in_footprint(0.0, 179.5, -1.0, 1.0, 179.0, 181.0));
        assert!(observer_in_footprint(0.0, -179.5, -1.0, 1.0, 179.0, 181.0));
        assert!(!observer_in_footprint(0.0, 178.0, -1.0, 1.0, 179.0, 181.0));
        assert!(!observer_in_footprint(0.0, -178.0, -1.0, 1.0, 179.0, 181.0));
        // Same footprint written with negative west edge.
        assert!(observer_in_footprint(0.0, 179.5, -1.0, 1.0, -181.0, -179.0));
        // Full-circumference grids contain every longitude.
        assert!(observer_in_footprint(0.0, 123.4, -1.0, 1.0, 0.0, 360.0));
    }

    #[test]
    fn cloud_field_footprint_spans_whole_grid() {
        use twilight_data::cloud_field_builder::{field_from_layers, FieldGeometry};
        let f = field_from_layers(
            &[],
            FieldGeometry {
                center_lat_deg: 54.5,
                center_lon_deg: 9.5,
                half_extent_km: 64.0,
                res_km: 2.0,
            },
            "test",
        );
        let (lat_min, lat_max, lon_min, lon_max) = cloud_field_footprint(&f);
        // The center must sit inside, roughly centered.
        assert!(observer_in_footprint(54.5, 9.5, lat_min, lat_max, lon_min, lon_max));
        assert!((0.5 * (lat_min + lat_max) - 54.5).abs() < 0.1);
        assert!((0.5 * (lon_min + lon_max) - 9.5).abs() < 0.1);
        // A point far outside the ~128 km grid must be rejected.
        assert!(!observer_in_footprint(58.0, 9.5, lat_min, lat_max, lon_min, lon_max));
    }

    // ── --cloud-field timestamp parsing / staleness ──

    #[test]
    fn field_timestamp_accepts_sidecar_formats() {
        // RFC 3339 with Z (field header contract).
        let t = parse_field_timestamp("2026-06-13T02:00:00Z").expect("rfc3339");
        assert_eq!(t.to_rfc3339(), "2026-06-13T02:00:00+00:00");
        // Naive with and without seconds (cloud3d profile style, UTC).
        assert!(parse_field_timestamp("2026-06-12T08:30:00").is_some());
        assert!(parse_field_timestamp("2026-06-12T08:30").is_some());
        // Offset form normalizes to UTC.
        let t = parse_field_timestamp("2026-06-13T04:00:00+02:00").expect("offset");
        assert_eq!(t.to_rfc3339(), "2026-06-13T02:00:00+00:00");
    }

    #[test]
    fn field_timestamp_rejects_garbage() {
        assert!(parse_field_timestamp("").is_none());
        assert!(parse_field_timestamp("not a time").is_none());
        assert!(parse_field_timestamp("2026-13-45T99:99").is_none());
    }
}
