//! Garstang (1986, 1991) analytical skyglow model.
//!
//! This module implements the Garstang radiative transfer model for computing
//! the artificial sky brightness at an observer due to ground-level light
//! sources. It integrates over:
//! - Distance from observer to each light source
//! - Altitude (the scattering volume along the line of sight)
//! - Scattering by both molecular (Rayleigh) and aerosol (Mie) particles
//!
//! The model assumes single scattering, which captures 85-90% of the total
//! skyglow under typical conditions (Cinzano 2005).
//!
//! # References
//! - Garstang (1986) "Model for artificial night-sky illumination", PASP 98:364
//! - Garstang (1989) "Night-sky brightness at observatories and sites", PASP 101:306
//! - Garstang (1991) "Dust and light pollution", PASP 103:1109
//! - Cinzano et al. (2000) "The artificial night sky brightness mapped from DMSP", MNRAS
//! - Cinzano & Falchi (2005) "Quantifying light pollution", J Quant Spec Rad Trans

use std::f64::consts::PI;

/// Scale height of the molecular atmosphere (meters).
/// Standard value for Rayleigh scattering.
const H_RAYLEIGH: f64 = 8500.0;

/// Scale height of the aerosol atmosphere (meters).
/// Typical for continental boundary layer aerosols.
const H_AEROSOL: f64 = 1500.0;

/// Sea-level total Rayleigh optical depth at 550nm along the vertical.
/// tau_R = 0.0962 at 550nm (standard atmosphere).
const TAU_RAYLEIGH_550: f64 = 0.0962;

/// Sea-level total aerosol optical depth at 550nm along the vertical.
/// Typical value for moderately polluted continental atmosphere.
/// This can be overridden by weather data (AOD from Open-Meteo).
const TAU_AEROSOL_550_DEFAULT: f64 = 0.15;

/// Mean Earth radius (m) for curvature correction.
const EARTH_RADIUS: f64 = 6_371_000.0;

/// A binned ground-level light source for the Garstang integration.
#[derive(Debug, Clone)]
pub struct LightSource {
    /// Horizontal distance from observer to center of this bin (meters).
    pub distance_m: f64,
    /// Azimuth angle from observer to source (degrees, 0=N, 90=E).
    pub azimuth_deg: f64,
    /// Total upward flux of this bin (W/sr).
    /// Derived from VIIRS radiance * pixel area * emission geometry.
    pub upward_flux: f64,
}

/// Configuration for the Garstang integration.
#[derive(Debug, Clone)]
pub struct GarstangConfig {
    /// Observer elevation above sea level (meters).
    pub observer_elevation: f64,
    /// Aerosol optical depth at 550nm (vertical).
    pub aod_550: f64,
    /// Fraction of light emitted directly upward (0.0 = fully shielded).
    pub uplight_fraction: f64,
    /// Ground reflectance (for indirect uplight from reflected downlight).
    pub ground_reflectance: f64,
    /// Wavelength for the calculation (nm). Affects Rayleigh scattering.
    pub wavelength_nm: f64,
    /// Number of altitude integration steps. Default 50.
    pub altitude_steps: usize,
    /// Maximum integration altitude (m). Default 30000.
    pub max_altitude: f64,
}

impl Default for GarstangConfig {
    fn default() -> Self {
        GarstangConfig {
            observer_elevation: 0.0,
            aod_550: TAU_AEROSOL_550_DEFAULT,
            uplight_fraction: 0.10,
            ground_reflectance: 0.15,
            wavelength_nm: 550.0,
            altitude_steps: 50,
            max_altitude: 30_000.0,
        }
    }
}

/// Compute the artificial sky brightness at zenith due to a set of light sources.
///
/// This is the core Garstang single-scatter integration. For each source,
/// we integrate the scattered light along the observer's zenith line of sight.
///
/// # Arguments
/// * `sources` - Array of binned light sources with distances and fluxes
/// * `config` - Integration parameters
///
/// # Returns
/// Artificial zenith sky brightness in W/m^2/sr at the configured wavelength.
pub fn zenith_brightness(sources: &[LightSource], config: &GarstangConfig) -> f64 {
    if sources.is_empty() {
        return 0.0;
    }

    let wl = config.wavelength_nm;
    let rayleigh_tau = rayleigh_optical_depth(wl);
    let aerosol_tau = config.aod_550 * (550.0 / wl).powf(1.3); // Angstrom exponent ~1.3

    let mut total_brightness = 0.0;

    for source in sources {
        total_brightness += single_source_zenith(source, config, rayleigh_tau, aerosol_tau);
    }

    total_brightness
}

/// Compute the artificial sky brightness at a given elevation angle.
///
/// Similar to `zenith_brightness` but integrates along a line of sight
/// at the specified elevation angle above the horizon.
///
/// # Arguments
/// * `sources` - Array of binned light sources
/// * `config` - Integration parameters
/// * `elevation_deg` - Viewing angle above horizon (degrees)
///
/// # Returns
/// Artificial sky brightness in W/m^2/sr at the specified direction.
pub fn directional_brightness(
    sources: &[LightSource],
    config: &GarstangConfig,
    elevation_deg: f64,
) -> f64 {
    if sources.is_empty() {
        return 0.0;
    }

    // For directions other than zenith, we need to account for:
    // 1. Longer path through the atmosphere (airmass)
    // 2. Different scattering angle between source light and LOS
    // 3. More scattering volume at lower elevations
    //
    // The full integration is expensive, so we use the angular enhancement
    // model from angular.rs as a correction factor on the zenith computation.
    // This is the approach used by Cinzano (2005) for computational efficiency.

    let zenith = zenith_brightness(sources, config);
    let factor = crate::angular::enhancement_factor(elevation_deg);
    zenith * factor
}

/// Compute the contribution of a single source to zenith brightness.
///
/// This performs the altitude integration along the observer's zenith LOS:
///
/// I = integral_0^inf [ j_source(h) * exp(-tau_obs(h)) ] dh
///
/// where j_source(h) is the emissivity due to scattering of the source's
/// light at altitude h, and tau_obs(h) is the optical depth from h to the
/// observer along the zenith.
fn single_source_zenith(
    source: &LightSource,
    config: &GarstangConfig,
    rayleigh_tau: f64,
    aerosol_tau: f64,
) -> f64 {
    let d = source.distance_m;
    if d < 1.0 {
        return 0.0; // Degenerate
    }

    let n_steps = config.altitude_steps;
    let dh = config.max_altitude / n_steps as f64;
    let mut integral = 0.0;

    // Effective upward fraction: direct uplight + reflected downlight
    // A fraction of downward light reflects off the ground and goes up.
    // Total upward = direct_up + (1 - direct_up) * reflectance / pi * some_factor
    // Simplified: effective_up ≈ uplight_fraction + ground_reflectance * 0.5
    let effective_up = config.uplight_fraction + config.ground_reflectance * 0.5;

    // Source intensity directed toward the scattering point (W/sr)
    let source_intensity = source.upward_flux * effective_up;

    for step in 0..n_steps {
        let h = (step as f64 + 0.5) * dh; // altitude of scattering point

        // Skip altitudes below observer
        if h < config.observer_elevation {
            continue;
        }

        // Distance from source to scattering point.
        // The source is on the ground, the scattering point is at altitude h
        // directly above the observer, with horizontal distance d.
        // With Earth curvature, this is slightly more complex, but for
        // typical distances (<200km) and altitudes (<30km), the flat-Earth
        // approximation with a curvature correction is adequate.
        let r_source_to_scatter = (d * d + h * h).sqrt();

        // Elevation angle of the scattering point as seen from the source
        let _sin_chi = h / r_source_to_scatter;

        // Scattering angle: angle between incoming ray (from source to scatter point)
        // and outgoing ray (from scatter point to observer along zenith).
        // The incoming ray makes angle chi with the horizontal.
        // The outgoing ray is vertical (zenith).
        // Scattering angle = 180 - chi - elevation_of_source_from_scatter
        // For zenith observation with scatter point directly above observer:
        //   incoming direction: from horizontal_distance d, altitude h -> angle atan2(h, d)
        //   outgoing direction: straight down to observer
        //   scattering angle = PI - atan2(d, h)
        let theta_scatter = PI - (d / h).atan(); // scattering angle

        // Rayleigh scattering coefficient at this altitude
        let n_rayleigh = rayleigh_tau / H_RAYLEIGH * (-h / H_RAYLEIGH).exp();

        // Aerosol scattering coefficient at this altitude
        let n_aerosol = aerosol_tau / H_AEROSOL * (-h / H_AEROSOL).exp();

        // Total scattering coefficient (per meter)
        let sigma_total = n_rayleigh + n_aerosol;

        // Phase functions
        let p_rayleigh = rayleigh_phase(theta_scatter);
        let p_mie = henyey_greenstein_phase(theta_scatter, 0.7); // g=0.7 typical aerosol

        // Weighted average phase function
        let f_rayleigh = if sigma_total > 0.0 {
            n_rayleigh / sigma_total
        } else {
            0.5
        };
        let p_average = f_rayleigh * p_rayleigh + (1.0 - f_rayleigh) * p_mie;

        // Optical depth from source to scattering point (slant path)
        // tau_slant = tau_vertical * (1 / sin(chi)) for the molecular part,
        // with correction for the actual path through the exponential atmosphere.
        // Simplified: use the total vertical optical depth weighted by the
        // fraction of atmosphere traversed.
        let tau_source_to_h = optical_depth_slant(
            0.0, // source at ground level
            h,
            d,
            rayleigh_tau,
            aerosol_tau,
        );

        // Optical depth from scattering point to observer (along zenith, downward)
        // This is the vertical optical depth between the scattering altitude and
        // the observer's altitude.
        let tau_h_to_obs =
            optical_depth_vertical(config.observer_elevation, h, rayleigh_tau, aerosol_tau);

        // Scattered radiance contribution from this altitude step:
        // dI = (source_intensity / r^2) * sigma * P(theta) * exp(-tau_in - tau_out) * dh
        //
        // where sigma * P(theta) = (Rayleigh + Mie) * P_avg
        let extinction = (-tau_source_to_h - tau_h_to_obs).exp();
        let r2 = r_source_to_scatter * r_source_to_scatter;

        let di = source_intensity / (4.0 * PI * r2) * sigma_total * p_average * extinction * dh;

        integral += di;
    }

    integral
}

/// Rayleigh phase function.
///
/// P(theta) = (3/(16*pi)) * (1 + cos^2(theta))
fn rayleigh_phase(theta: f64) -> f64 {
    3.0 / (16.0 * PI) * (1.0 + theta.cos().powi(2))
}

/// Henyey-Greenstein phase function.
///
/// P(theta, g) = (1 - g^2) / (4*pi * (1 + g^2 - 2*g*cos(theta))^(3/2))
fn henyey_greenstein_phase(theta: f64, g: f64) -> f64 {
    let cos_theta = theta.cos();
    let denom = (1.0 + g * g - 2.0 * g * cos_theta).powf(1.5);
    if denom < 1e-30 {
        return 1.0 / (4.0 * PI);
    }
    (1.0 - g * g) / (4.0 * PI * denom)
}

/// Total Rayleigh optical depth at a given wavelength.
///
/// tau_R(lambda) = tau_R(550) * (550/lambda)^4
fn rayleigh_optical_depth(wavelength_nm: f64) -> f64 {
    TAU_RAYLEIGH_550 * (550.0 / wavelength_nm).powi(4)
}

/// Optical depth along a slant path from ground level at horizontal distance d
/// to altitude h, through an exponential atmosphere.
///
/// For an exponential density profile n(z) = n0 * exp(-z/H), the slant
/// optical depth from (0, d_horizontal) to (h, 0_horizontal_above_observer) is:
///
///   tau = tau_vertical * [integral along path]
///
/// We use the approximation valid for paths not too close to horizontal:
///   tau ≈ tau_vertical * H * (1 - exp(-h/H)) / sin(chi)
/// where chi = atan2(h, d) is the elevation angle of the path.
fn optical_depth_slant(_z_source: f64, h: f64, d: f64, rayleigh_tau: f64, aerosol_tau: f64) -> f64 {
    let path_len = (d * d + h * h).sqrt();
    if path_len < 1.0 {
        return 0.0;
    }

    // For the molecular component:
    // Integral of n0*exp(-z/H) along the slant path from ground to h.
    // If the path is from (0,0) to (h, d), parametrize by t in [0,1]:
    //   z(t) = h*t, horizontal(t) = d*t
    //   ds = sqrt(h^2 + d^2) dt = path_len * dt
    //   tau = (n0 * path_len) * integral_0^1 exp(-h*t/H) dt
    //       = n0 * path_len * H / h * (1 - exp(-h/H))    [if h > 0]
    //
    // where n0 = tau_vertical / H (sea-level extinction coefficient)

    let tau_r = if h > 1.0 {
        let n0_r = rayleigh_tau / H_RAYLEIGH;
        n0_r * path_len * H_RAYLEIGH / h * (1.0 - (-h / H_RAYLEIGH).exp())
    } else {
        rayleigh_tau / H_RAYLEIGH * path_len
    };

    let tau_a = if h > 1.0 {
        let n0_a = aerosol_tau / H_AEROSOL;
        n0_a * path_len * H_AEROSOL / h * (1.0 - (-h / H_AEROSOL).exp())
    } else {
        aerosol_tau / H_AEROSOL * path_len
    };

    tau_r + tau_a
}

/// Vertical optical depth between two altitudes.
fn optical_depth_vertical(z_low: f64, z_high: f64, rayleigh_tau: f64, aerosol_tau: f64) -> f64 {
    let tau_r = rayleigh_tau * ((-z_low / H_RAYLEIGH).exp() - (-z_high / H_RAYLEIGH).exp());
    let tau_a = aerosol_tau * ((-z_low / H_AEROSOL).exp() - (-z_high / H_AEROSOL).exp());
    tau_r + tau_a
}

/// Convert VIIRS pixel radiance to upward flux for a source bin.
///
/// # Arguments
/// * `radiance_nw` - VIIRS DNB radiance in nW/cm^2/sr
/// * `pixel_area_m2` - Area of the pixel or bin in square meters
///
/// # Returns
/// Upward flux in W/sr
pub fn viirs_to_flux(radiance_nw: f64, pixel_area_m2: f64) -> f64 {
    // VIIRS measures upward radiance at the top of atmosphere.
    // The ground-level upward flux is approximately:
    //   F = L_viirs * pi * A / T_atm
    // where T_atm is atmospheric transmittance and pi accounts for
    // the hemisphere of emission directions.
    //
    // However, for the Garstang model we need the source intensity (W/sr)
    // as seen from above. VIIRS already measures this quantity (radiance = W/m^2/sr
    // at the satellite). The ground-level intensity per pixel is:
    //   I = L_viirs * A
    // in units of W/sr (the pixel acts as a Lambertian source of area A).
    //
    // Convert nW/cm^2/sr to W/m^2/sr: multiply by 1e-5
    let radiance_si = radiance_nw * 1e-5;
    radiance_si * pixel_area_m2
}

/// Create light source bins from a radiance grid around an observer.
///
/// This function discretizes the surrounding area into distance bins,
/// summing VIIRS radiance within each bin.
///
/// # Arguments
/// * `source` - Radiance data provider
/// * `observer_lat` - Observer latitude (degrees)
/// * `observer_lon` - Observer longitude (degrees)
/// * `radius_km` - Maximum integration radius (km)
///
/// # Returns
/// Vector of light source bins for the Garstang integration.
pub fn bin_sources(
    source: &dyn crate::RadianceSource,
    observer_lat: f64,
    observer_lon: f64,
    radius_km: f64,
) -> Vec<LightSource> {
    let mut bins = Vec::new();

    // Adaptive binning by distance:
    // 0-5km:    resolution = source resolution (typically 500-750m)
    // 5-20km:   2km bins
    // 20-50km:  5km bins
    // 50-200km: 20km bins
    let bin_configs: &[(f64, f64, f64)] = &[
        // (start_km, end_km, step_km)
        (0.5, 5.0, 1.0),
        (5.0, 20.0, 2.0),
        (20.0, 50.0, 5.0),
        (50.0, radius_km, 20.0),
    ];

    let n_azimuths_base = 36; // 10-degree azimuth resolution

    for &(start, end, step) in bin_configs {
        if start >= radius_km {
            break;
        }
        let actual_end = end.min(radius_km);

        let mut dist = start;
        while dist < actual_end {
            // More azimuth bins at larger distances (covers more area)
            let n_az = (n_azimuths_base as f64 * (dist / 5.0).max(1.0).min(4.0)) as usize;
            let az_step = 360.0 / n_az as f64;

            for az_idx in 0..n_az {
                let az = az_idx as f64 * az_step;

                // Compute center of this bin in geographic coordinates
                let dist_m = dist * 1000.0;
                let (bin_lat, bin_lon) = offset_latlon(observer_lat, observer_lon, dist_m, az);

                // Query radiance at this point
                if let Some(radiance) = source.radiance_at(bin_lat, bin_lon) {
                    if radiance > 0.1 {
                        // Skip negligible sources
                        // Bin area: annular sector
                        let r_inner = dist * 1000.0;
                        let r_outer = (dist + step).min(actual_end) * 1000.0;
                        let area = PI * (r_outer * r_outer - r_inner * r_inner) / n_az as f64;

                        let flux = viirs_to_flux(radiance, area);

                        bins.push(LightSource {
                            distance_m: dist_m,
                            azimuth_deg: az,
                            upward_flux: flux,
                        });
                    }
                }
            }

            dist += step;
        }
    }

    bins
}

/// Offset a lat/lon by a distance and bearing (simple spherical approximation).
fn offset_latlon(lat: f64, lon: f64, distance_m: f64, bearing_deg: f64) -> (f64, f64) {
    let lat_rad = lat.to_radians();
    let bearing_rad = bearing_deg.to_radians();
    let delta = distance_m / EARTH_RADIUS;

    let new_lat_rad =
        (lat_rad.sin() * delta.cos() + lat_rad.cos() * delta.sin() * bearing_rad.cos()).asin();

    let new_lon_rad = lon.to_radians()
        + (bearing_rad.sin() * delta.sin() * lat_rad.cos())
            .atan2(delta.cos() - lat_rad.sin() * new_lat_rad.sin());

    (new_lat_rad.to_degrees(), new_lon_rad.to_degrees())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ConstantRadiance;

    #[test]
    fn rayleigh_phase_forward_backward_symmetric() {
        let p_fwd = rayleigh_phase(0.0);
        let p_bwd = rayleigh_phase(PI);
        assert!(
            (p_fwd - p_bwd).abs() < 1e-10,
            "Rayleigh should be symmetric: fwd={}, bwd={}",
            p_fwd,
            p_bwd
        );
    }

    #[test]
    fn rayleigh_phase_minimum_at_90() {
        let p_90 = rayleigh_phase(PI / 2.0);
        let p_0 = rayleigh_phase(0.0);
        assert!(p_90 < p_0, "Minimum should be at 90 degrees");
    }

    #[test]
    fn hg_phase_forward_peaked() {
        let p_fwd = henyey_greenstein_phase(0.0, 0.7);
        let p_bwd = henyey_greenstein_phase(PI, 0.7);
        assert!(
            p_fwd > 5.0 * p_bwd,
            "HG g=0.7 should be strongly forward: fwd={}, bwd={}",
            p_fwd,
            p_bwd
        );
    }

    #[test]
    fn hg_phase_isotropic_at_g_zero() {
        let g = 0.0;
        let p_0 = henyey_greenstein_phase(0.0, g);
        let p_90 = henyey_greenstein_phase(PI / 2.0, g);
        let p_180 = henyey_greenstein_phase(PI, g);
        let expected = 1.0 / (4.0 * PI);
        assert!((p_0 - expected).abs() < 1e-10);
        assert!((p_90 - expected).abs() < 1e-10);
        assert!((p_180 - expected).abs() < 1e-10);
    }

    #[test]
    fn rayleigh_od_wavelength_dependence() {
        let tau_400 = rayleigh_optical_depth(400.0);
        let tau_550 = rayleigh_optical_depth(550.0);
        let tau_700 = rayleigh_optical_depth(700.0);

        assert!(tau_400 > tau_550, "Blue should have larger OD");
        assert!(tau_550 > tau_700, "Green should have larger OD than red");

        // Check lambda^-4 scaling
        let ratio_expected = (550.0 / 400.0_f64).powi(4);
        let ratio_actual = tau_400 / tau_550;
        assert!(
            (ratio_actual - ratio_expected).abs() < 0.01,
            "Should follow lambda^-4: expected {}, got {}",
            ratio_expected,
            ratio_actual
        );
    }

    #[test]
    fn vertical_od_positive() {
        let tau = optical_depth_vertical(0.0, 10000.0, TAU_RAYLEIGH_550, TAU_AEROSOL_550_DEFAULT);
        assert!(tau > 0.0, "Vertical OD should be positive");
    }

    #[test]
    fn vertical_od_increases_with_height() {
        let tau_5km =
            optical_depth_vertical(0.0, 5000.0, TAU_RAYLEIGH_550, TAU_AEROSOL_550_DEFAULT);
        let tau_10km =
            optical_depth_vertical(0.0, 10000.0, TAU_RAYLEIGH_550, TAU_AEROSOL_550_DEFAULT);
        assert!(tau_10km > tau_5km, "Deeper column should have more OD");
    }

    #[test]
    fn vertical_od_full_column_matches_total() {
        // OD from 0 to infinity should equal the total vertical OD
        let tau = optical_depth_vertical(0.0, 100_000.0, TAU_RAYLEIGH_550, TAU_AEROSOL_550_DEFAULT);
        let expected = TAU_RAYLEIGH_550 + TAU_AEROSOL_550_DEFAULT;
        assert!(
            (tau - expected).abs() / expected < 0.01,
            "Full column OD should match total: got {}, expected {}",
            tau,
            expected
        );
    }

    #[test]
    fn viirs_to_flux_positive() {
        let flux = viirs_to_flux(50.0, 500.0 * 500.0);
        assert!(flux > 0.0, "Flux should be positive");
    }

    #[test]
    fn viirs_to_flux_scales_with_radiance() {
        let f1 = viirs_to_flux(10.0, 1000.0);
        let f2 = viirs_to_flux(20.0, 1000.0);
        assert!(
            (f2 / f1 - 2.0).abs() < 0.01,
            "Flux should scale linearly with radiance"
        );
    }

    #[test]
    fn viirs_to_flux_scales_with_area() {
        let f1 = viirs_to_flux(10.0, 1000.0);
        let f2 = viirs_to_flux(10.0, 2000.0);
        assert!(
            (f2 / f1 - 2.0).abs() < 0.01,
            "Flux should scale linearly with area"
        );
    }

    #[test]
    fn zenith_brightness_empty_sources() {
        let config = GarstangConfig::default();
        let b = zenith_brightness(&[], &config);
        assert_eq!(b, 0.0, "No sources should give zero brightness");
    }

    #[test]
    fn zenith_brightness_positive_for_source() {
        let source = LightSource {
            distance_m: 10_000.0, // 10km
            azimuth_deg: 0.0,
            upward_flux: 1e6, // 1 MW/sr (a small city)
        };
        let config = GarstangConfig::default();
        let b = zenith_brightness(&[source], &config);
        assert!(b > 0.0, "Should have positive brightness, got {}", b);
    }

    #[test]
    fn zenith_brightness_decreases_with_distance() {
        let config = GarstangConfig::default();

        let near = LightSource {
            distance_m: 10_000.0,
            azimuth_deg: 0.0,
            upward_flux: 1e6,
        };
        let far = LightSource {
            distance_m: 100_000.0,
            azimuth_deg: 0.0,
            upward_flux: 1e6,
        };

        let b_near = zenith_brightness(&[near], &config);
        let b_far = zenith_brightness(&[far], &config);

        assert!(
            b_near > b_far,
            "Nearby source should be brighter: near={}, far={}",
            b_near,
            b_far
        );
    }

    #[test]
    fn zenith_brightness_increases_with_flux() {
        let config = GarstangConfig::default();

        let weak = LightSource {
            distance_m: 20_000.0,
            azimuth_deg: 0.0,
            upward_flux: 1e5,
        };
        let strong = LightSource {
            distance_m: 20_000.0,
            azimuth_deg: 0.0,
            upward_flux: 1e7,
        };

        let b_weak = zenith_brightness(&[weak], &config);
        let b_strong = zenith_brightness(&[strong], &config);

        assert!(
            b_strong > b_weak,
            "Stronger source should be brighter: strong={}, weak={}",
            b_strong,
            b_weak
        );
    }

    #[test]
    fn zenith_brightness_additive() {
        let config = GarstangConfig::default();

        let s1 = LightSource {
            distance_m: 15_000.0,
            azimuth_deg: 0.0,
            upward_flux: 1e6,
        };
        let s2 = LightSource {
            distance_m: 15_000.0,
            azimuth_deg: 180.0,
            upward_flux: 1e6,
        };

        let b1 = zenith_brightness(&[s1.clone()], &config);
        let b2 = zenith_brightness(&[s2.clone()], &config);
        let b_both = zenith_brightness(&[s1, s2], &config);

        assert!(
            (b_both - (b1 + b2)).abs() / b_both < 0.01,
            "Sources should be additive: both={}, sum={}",
            b_both,
            b1 + b2
        );
    }

    #[test]
    fn bin_sources_creates_bins() {
        let source = ConstantRadiance { radiance: 50.0 };
        let bins = bin_sources(&source, 21.4225, 39.8262, 50.0);
        assert!(!bins.is_empty(), "Should create some bins");

        // Check all bins have positive distance and flux
        for bin in &bins {
            assert!(bin.distance_m > 0.0);
            assert!(bin.upward_flux > 0.0);
        }
    }

    #[test]
    fn bin_sources_empty_for_zero_radiance() {
        let source = ConstantRadiance { radiance: 0.0 };
        let bins = bin_sources(&source, 0.0, 0.0, 50.0);
        assert!(bins.is_empty(), "Zero radiance should give no bins");
    }

    #[test]
    fn offset_latlon_north() {
        let (lat2, lon2) = offset_latlon(0.0, 0.0, 111_000.0, 0.0); // ~1 degree north
        assert!(
            (lat2 - 1.0).abs() < 0.05,
            "Should be ~1 degree north, got {}",
            lat2
        );
        assert!(lon2.abs() < 0.01, "Should be same longitude, got {}", lon2);
    }

    #[test]
    fn offset_latlon_east() {
        let (lat2, lon2) = offset_latlon(0.0, 0.0, 111_000.0, 90.0); // ~1 degree east at equator
        assert!(lat2.abs() < 0.05, "Should be same latitude, got {}", lat2);
        assert!(
            (lon2 - 1.0).abs() < 0.05,
            "Should be ~1 degree east, got {}",
            lon2
        );
    }

    #[test]
    fn slant_od_zero_for_zero_distance() {
        let tau = optical_depth_slant(0.0, 10000.0, 0.0, TAU_RAYLEIGH_550, TAU_AEROSOL_550_DEFAULT);
        // When d=0, the path is purely vertical, so slant OD = vertical OD
        let tau_v = optical_depth_vertical(0.0, 10000.0, TAU_RAYLEIGH_550, TAU_AEROSOL_550_DEFAULT);
        // path_len = h when d=0, so the formula reduces to vertical
        assert!(
            (tau - tau_v).abs() / tau_v < 0.01,
            "Vertical path should match: slant={}, vert={}",
            tau,
            tau_v
        );
    }

    #[test]
    fn aod_affects_brightness() {
        // The relationship between AOD and skyglow brightness is non-trivial:
        // More aerosols = more scattering (brightens) BUT also more extinction (dims).
        //
        // For a source directly below the observer (d=0 + some offset), the zenith
        // scattering path goes straight up through the aerosol layer. More aerosol
        // means more scattering particles in the zenith column, so more light is
        // scattered toward the observer.
        //
        // At larger distances, the slant path extinction from source to the
        // scattering volume dominates, and high AOD can actually reduce brightness.
        // This is the well-known "AOD non-monotonicity" effect in skyglow modeling
        // (Kocifaj 2007, Cinzano 2005).

        let config_low_aod = GarstangConfig {
            aod_550: 0.05,
            ..GarstangConfig::default()
        };
        let config_high_aod = GarstangConfig {
            aod_550: 0.30,
            ..GarstangConfig::default()
        };
        let config_extreme_aod = GarstangConfig {
            aod_550: 1.0,
            ..GarstangConfig::default()
        };

        let source = LightSource {
            distance_m: 20_000.0,
            azimuth_deg: 0.0,
            upward_flux: 1e6,
        };

        let b_low = zenith_brightness(&[source.clone()], &config_low_aod);
        let b_high = zenith_brightness(&[source.clone()], &config_high_aod);
        let b_extreme = zenith_brightness(&[source], &config_extreme_aod);

        // All should be positive
        assert!(b_low > 0.0, "Low AOD brightness should be positive");
        assert!(b_high > 0.0, "High AOD brightness should be positive");
        assert!(b_extreme > 0.0, "Extreme AOD brightness should be positive");

        // At extreme AOD, extinction dominates and brightness drops
        // (this is the expected non-monotonic behavior)
        assert!(
            b_extreme < b_low || b_extreme < b_high,
            "Extreme AOD should show extinction effects"
        );
    }
}
