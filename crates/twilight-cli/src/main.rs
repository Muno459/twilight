use clap::{Parser, Subcommand};
use twilight_cpu::pipeline::{self, PrayerTimeInput};
use twilight_cpu::simulation::{self, SimulationConfig, SpectralResult};
use twilight_data::atmosphere_profiles::AtmosphereType;
use twilight_data::builder;
use twilight_solar::spa::{self, SpaInput};
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
        /// Number of photons per wavelength
        #[arg(short, long, default_value = "5000")]
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
        /// Show detailed twilight analysis
        #[arg(long)]
        verbose: bool,
    },
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

fn format_fractional_hour(h: f64) -> String {
    if h < 0.0 || h > 24.0 {
        return "N/A".to_string();
    }
    let hours = h as u32;
    let minutes = ((h - hours as f64) * 60.0) as u32;
    let seconds = ((h - hours as f64 - minutes as f64 / 60.0) * 3600.0) as u32;
    format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
}

fn cmd_solar(lat: f64, lon: f64, date: &str, tz: f64, elevation: f64, delta_t: f64) {
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
            println!("Solar Position at Local Noon:");
            println!("  Zenith:       {:.4}°", noon.zenith);
            println!("  Azimuth:      {:.4}°", noon.azimuth);
            println!("  Declination:  {:.4}°", noon.delta);
            println!("  Earth-Sun:    {:.6} AU", noon.r);
            println!("  Eq. of Time:  {:.2} min", noon.eot);
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
            .map(|h| format_fractional_hour(h))
            .unwrap_or("N/A".to_string());
        let evening_str = evening
            .map(|h| format_fractional_hour(h))
            .unwrap_or("N/A".to_string());

        println!(
            "{:<35} {:>10}  {:>10}",
            angle.name, morning_str, evening_str
        );
    }

    println!();
    println!("Note: These are CONVENTIONAL times using fixed solar depression angles.");
    println!("      Use 'twilight pray' to compute physically-based times.");
}

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
    println!("Atmosphere:   US Standard 1976 (clear sky)");
    println!("Surface:      albedo = {:.2}", albedo);
    println!(
        "View:         zenith {:.0}°, azimuth {:.0}°",
        view_zenith, solar_azimuth
    );
    println!();

    // Build atmosphere
    let atm = builder::build_clear_sky(AtmosphereType::UsStandard, albedo);

    let config = SimulationConfig {
        latitude: lat,
        longitude: lon,
        elevation: 0.0,
        solar_azimuth,
        view_zenith,
        apply_solar_irradiance: true,
    };

    println!("Running MCRT (solar-irradiance-weighted)...");
    println!();

    let start = std::time::Instant::now();
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

fn cmd_pray(
    lat: f64,
    lon: f64,
    date: &str,
    tz: f64,
    elevation: f64,
    albedo: f64,
    delta_t: f64,
    sza_step: f64,
    verbose: bool,
) {
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
    println!("Atmosphere: US Standard 1976 (clear sky)");
    println!("Method:     Single-scatter MCRT + CIE mesopic vision");
    println!();

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
        ..Default::default()
    };

    println!("Computing...");
    let output = pipeline::compute_prayer_times(&input);
    println!("Done in {} ms", output.computation_time_ms);
    println!();

    // Print results
    println!("Prayer Times (MCRT-derived):");
    println!("{:-<65}", "");

    // Sunrise/Sunset
    println!(
        "  Sunrise:              {}",
        output
            .sunrise_time
            .map(|h| format_fractional_hour(h))
            .unwrap_or("N/A".to_string())
    );
    println!(
        "  Sunset:               {}",
        output
            .sunset_time
            .map(|h| format_fractional_hour(h))
            .unwrap_or("N/A".to_string())
    );
    println!();

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
            "  Fajr (true dawn):     {}  (SZA {:.2}°, depression {:.2}°)",
            format_fractional_hour(time),
            sza,
            dep
        );
    } else if output.persistent_twilight {
        println!("  Fajr (true dawn):     N/A (persistent twilight — sky never fully dark)");
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
            "  Isha (al-abyad):      {}  (SZA {:.2}°, depression {:.2}°)",
            format_fractional_hour(time),
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
            "  Isha (al-ahmar):      {}  (SZA {:.2}°, depression {:.2}°)",
            format_fractional_hour(time),
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
            .map(|h| format_fractional_hour(h))
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
    println!("  - Current model: clear sky, US Standard 1976 atmosphere, single scattering.");
    println!("  - The 'depression' angle is the equivalent fixed angle that gives the same time.");
    println!(
        "  - Differences from conventional times reflect atmospheric conditions vs fixed angles."
    );
    println!("  - Future: clouds, aerosols, terrain, and real-time weather will improve accuracy.");
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
        } => {
            cmd_solar(lat, lon, &date, tz, elevation, delta_t);
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
            );
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
            verbose,
        } => {
            cmd_pray(
                lat, lon, &date, tz, elevation, albedo, delta_t, sza_step, verbose,
            );
        }
    }
}
