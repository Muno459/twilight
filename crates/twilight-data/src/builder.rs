//! Atmosphere builder: constructs a populated AtmosphereModel from
//! embedded profile data, Rayleigh scattering, and ozone absorption.
//!
//! This bridges twilight-data (raw atmospheric data) and twilight-core
//! (the shell-based atmosphere model that the MCRT engine consumes).

use crate::atmosphere_profiles::{self, AtmosphereType};
use crate::ozone_xsec;
use twilight_core::atmosphere::{AtmosphereModel, ShellOptics};
use twilight_core::spectrum::rayleigh_scattering_coeff;

/// Default wavelength grid for twilight simulations: 380-780nm at 10nm steps.
pub const DEFAULT_WAVELENGTHS_NM: [f64; 41] = [
    380.0, 390.0, 400.0, 410.0, 420.0, 430.0, 440.0, 450.0, 460.0, 470.0, 480.0, 490.0, 500.0,
    510.0, 520.0, 530.0, 540.0, 550.0, 560.0, 570.0, 580.0, 590.0, 600.0, 610.0, 620.0, 630.0,
    640.0, 650.0, 660.0, 670.0, 680.0, 690.0, 700.0, 710.0, 720.0, 730.0, 740.0, 750.0, 760.0,
    770.0, 780.0,
];

/// Default altitude grid for atmospheric shells (km).
/// Finer resolution in the lower atmosphere (0-30 km) where most scattering
/// occurs, coarser above.
pub const DEFAULT_ALTITUDES_KM: [f64; 51] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0,
    38.0, 40.0, 42.0, 44.0, 46.0, 48.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0,
    100.0,
];

/// Number of altitude levels (one fewer shells than levels).
pub const NUM_ALTITUDE_LEVELS: usize = 51;

/// Build a clear-sky atmosphere model with Rayleigh scattering and ozone absorption.
///
/// This creates a fully populated `AtmosphereModel` suitable for MCRT simulation.
///
/// # Arguments
/// * `profile` - Which AFGL standard atmosphere to use
/// * `surface_albedo` - Broadband surface albedo (0.0 - 1.0), applied to all wavelengths
pub fn build_clear_sky(profile: AtmosphereType, surface_albedo: f64) -> AtmosphereModel {
    let mut atm = AtmosphereModel::new(
        &DEFAULT_ALTITUDES_KM[..NUM_ALTITUDE_LEVELS],
        &DEFAULT_WAVELENGTHS_NM,
    );

    // Set surface albedo
    for w in 0..atm.num_wavelengths {
        atm.surface_albedo[w] = surface_albedo;
    }

    // Populate optical properties for each shell and wavelength
    let num_shells = atm.num_shells;
    for s in 0..num_shells {
        let alt_mid_km = atm.shells[s].altitude_mid / 1000.0; // m to km

        // Get atmospheric state at shell midpoint
        let n_density = atmosphere_profiles::number_density_at(alt_mid_km, profile);
        let o3_density = atmosphere_profiles::ozone_density_at(alt_mid_km, profile);

        // Number density: convert from molecules/cm³ to molecules/m³
        let n_density_m3 = n_density * 1e6;
        let o3_density_m3 = o3_density * 1e6;

        for w in 0..atm.num_wavelengths {
            let wl = atm.wavelengths_nm[w];

            // Rayleigh scattering coefficient [1/m]
            let beta_ray = rayleigh_scattering_coeff(wl, n_density_m3);

            // Ozone absorption coefficient [1/m]
            let sigma_o3 = ozone_xsec::o3_cross_section_at(wl); // cm²/molecule
            let beta_o3 = sigma_o3 * 1e-4 * o3_density_m3; // convert cm² to m², multiply by density

            // Total extinction = Rayleigh scattering + O3 absorption
            let extinction = beta_ray + beta_o3;

            // Single scattering albedo = scattering / total extinction
            let ssa = if extinction > 1e-30 {
                beta_ray / extinction
            } else {
                1.0
            };

            atm.optics[s][w] = ShellOptics {
                extinction,
                ssa,
                asymmetry: 0.0,         // Rayleigh is symmetric (g=0)
                rayleigh_fraction: 1.0, // Pure Rayleigh in clear sky
            };
        }
    }

    atm
}

/// Build a clear-sky atmosphere with a simple cloud layer added.
///
/// This adds a single horizontally-uniform cloud layer to the clear-sky model.
///
/// # Arguments
/// * `profile` - Which AFGL standard atmosphere to use
/// * `surface_albedo` - Broadband surface albedo
/// * `cloud_base_km` - Cloud base altitude in km
/// * `cloud_top_km` - Cloud top altitude in km
/// * `cloud_od` - Cloud optical depth (at 550nm)
/// * `cloud_ssa` - Cloud single scattering albedo (~0.999 for water clouds)
/// * `cloud_g` - Cloud asymmetry parameter (~0.85 for water droplets)
pub fn build_with_cloud_layer(
    profile: AtmosphereType,
    surface_albedo: f64,
    cloud_base_km: f64,
    cloud_top_km: f64,
    cloud_od: f64,
    cloud_ssa: f64,
    cloud_g: f64,
) -> AtmosphereModel {
    let mut atm = build_clear_sky(profile, surface_albedo);

    // Distribute cloud optical depth uniformly across shells that overlap the cloud
    let cloud_thickness_m = (cloud_top_km - cloud_base_km) * 1000.0;
    if cloud_thickness_m <= 0.0 || cloud_od <= 0.0 {
        return atm;
    }

    // Cloud extinction coefficient (uniform within cloud)
    // At 550nm: extinction = OD / thickness
    let cloud_ext_550 = cloud_od / cloud_thickness_m;

    let num_shells = atm.num_shells;
    for s in 0..num_shells {
        let shell_base_km =
            (atm.shells[s].r_inner - twilight_core::atmosphere::EARTH_RADIUS_M) / 1000.0;
        let shell_top_km =
            (atm.shells[s].r_outer - twilight_core::atmosphere::EARTH_RADIUS_M) / 1000.0;

        // Check if this shell overlaps the cloud layer
        let overlap_base = shell_base_km.max(cloud_base_km);
        let overlap_top = shell_top_km.min(cloud_top_km);
        if overlap_top <= overlap_base {
            continue; // No overlap
        }

        // Fraction of this shell occupied by cloud
        let shell_thickness_km = shell_top_km - shell_base_km;
        let overlap_fraction = (overlap_top - overlap_base) / shell_thickness_km;

        for w in 0..atm.num_wavelengths {
            let existing = &atm.optics[s][w];

            // Cloud extinction (approximate: assume wavelength-independent for now)
            // In reality, cloud extinction is nearly wavelength-independent in the visible
            let cloud_ext = cloud_ext_550 * overlap_fraction;
            let cloud_scat = cloud_ext * cloud_ssa;
            let cloud_abs = cloud_ext * (1.0 - cloud_ssa);

            let rayleigh_scat = existing.extinction * existing.ssa;
            let gas_abs = existing.extinction * (1.0 - existing.ssa);

            let total_scat = rayleigh_scat + cloud_scat;
            let total_ext = total_scat + gas_abs + cloud_abs;

            let new_ssa = if total_ext > 1e-30 {
                total_scat / total_ext
            } else {
                1.0
            };

            // Weighted asymmetry parameter
            let new_g = if total_scat > 1e-30 {
                (rayleigh_scat * 0.0 + cloud_scat * cloud_g) / total_scat
            } else {
                0.0
            };

            // Rayleigh fraction of total scattering
            let ray_frac = if total_scat > 1e-30 {
                rayleigh_scat / total_scat
            } else {
                1.0
            };

            atm.optics[s][w] = ShellOptics {
                extinction: total_ext,
                ssa: new_ssa,
                asymmetry: new_g,
                rayleigh_fraction: ray_frac,
            };
        }
    }

    atm
}

#[cfg(test)]
mod tests {
    use super::*;
    use twilight_core::atmosphere::EARTH_RADIUS_M;

    // ── build_clear_sky ──

    #[test]
    fn clear_sky_has_50_shells() {
        let atm = build_clear_sky(AtmosphereType::UsStandard, 0.15);
        assert_eq!(atm.num_shells, 50);
    }

    #[test]
    fn clear_sky_has_41_wavelengths() {
        let atm = build_clear_sky(AtmosphereType::UsStandard, 0.15);
        assert_eq!(atm.num_wavelengths, 41);
    }

    #[test]
    fn clear_sky_wavelengths_are_380_to_780() {
        let atm = build_clear_sky(AtmosphereType::UsStandard, 0.15);
        assert!((atm.wavelengths_nm[0] - 380.0).abs() < 0.01);
        assert!((atm.wavelengths_nm[40] - 780.0).abs() < 0.01);
    }

    #[test]
    fn clear_sky_surface_at_earth_radius() {
        let atm = build_clear_sky(AtmosphereType::UsStandard, 0.15);
        assert!((atm.surface_radius() - EARTH_RADIUS_M).abs() < 1.0);
    }

    #[test]
    fn clear_sky_toa_at_100km() {
        let atm = build_clear_sky(AtmosphereType::UsStandard, 0.15);
        assert!((atm.toa_radius() - (EARTH_RADIUS_M + 100_000.0)).abs() < 1.0);
    }

    #[test]
    fn clear_sky_surface_albedo_set() {
        let atm = build_clear_sky(AtmosphereType::UsStandard, 0.30);
        for w in 0..atm.num_wavelengths {
            assert!(
                (atm.surface_albedo[w] - 0.30).abs() < 1e-10,
                "Albedo at wl[{}] = {}, expected 0.30",
                w,
                atm.surface_albedo[w]
            );
        }
    }

    #[test]
    fn clear_sky_extinction_positive_at_surface() {
        let atm = build_clear_sky(AtmosphereType::UsStandard, 0.15);
        for w in 0..atm.num_wavelengths {
            assert!(
                atm.optics[0][w].extinction > 0.0,
                "Surface shell extinction at wl[{}] should be positive",
                w
            );
        }
    }

    #[test]
    fn clear_sky_extinction_decreases_with_altitude() {
        let atm = build_clear_sky(AtmosphereType::UsStandard, 0.15);
        let w = 20; // 580nm
        let ext_low = atm.optics[0][w].extinction;
        let ext_high = atm.optics[atm.num_shells - 1][w].extinction;
        assert!(
            ext_low > ext_high * 100.0,
            "Surface extinction ({:.4e}) should be >> top ({:.4e})",
            ext_low,
            ext_high
        );
    }

    #[test]
    fn clear_sky_ssa_near_one_at_surface_400nm() {
        let atm = build_clear_sky(AtmosphereType::UsStandard, 0.15);
        let w = 2; // 400nm
        assert!(
            atm.optics[0][w].ssa > 0.95,
            "SSA at 400nm surface = {}, expected > 0.95",
            atm.optics[0][w].ssa
        );
    }

    #[test]
    fn clear_sky_rayleigh_fraction_is_one() {
        let atm = build_clear_sky(AtmosphereType::UsStandard, 0.15);
        for s in 0..atm.num_shells {
            for w in 0..atm.num_wavelengths {
                assert!(
                    (atm.optics[s][w].rayleigh_fraction - 1.0).abs() < 1e-10,
                    "Rayleigh fraction at shell {}, wl {} = {}, expected 1.0",
                    s,
                    w,
                    atm.optics[s][w].rayleigh_fraction
                );
            }
        }
    }

    #[test]
    fn clear_sky_asymmetry_is_zero() {
        let atm = build_clear_sky(AtmosphereType::UsStandard, 0.15);
        for s in 0..atm.num_shells {
            for w in 0..atm.num_wavelengths {
                assert!(
                    atm.optics[s][w].asymmetry.abs() < 1e-10,
                    "Asymmetry at shell {}, wl {} = {}, expected 0.0",
                    s,
                    w,
                    atm.optics[s][w].asymmetry
                );
            }
        }
    }

    #[test]
    fn clear_sky_blue_scatters_more_than_red() {
        let atm = build_clear_sky(AtmosphereType::UsStandard, 0.15);
        let ext_380 = atm.optics[0][0].extinction;
        let ext_780 = atm.optics[0][40].extinction;
        let ratio = ext_380 / ext_780;
        assert!(
            ratio > 5.0,
            "Blue/red extinction ratio = {}, expected >> 1",
            ratio
        );
    }

    #[test]
    fn clear_sky_ozone_absorption_at_chappuis_peak() {
        let atm = build_clear_sky(AtmosphereType::UsStandard, 0.15);
        let w = 20; // 580nm
        let mut ozone_shell = None;
        for s in 0..atm.num_shells {
            let alt_km = atm.shells[s].altitude_mid / 1000.0;
            if alt_km > 18.0 && alt_km < 22.0 {
                ozone_shell = Some(s);
                break;
            }
        }
        if let Some(s) = ozone_shell {
            assert!(
                atm.optics[s][w].ssa < 0.999,
                "SSA at ozone peak shell should be < 1 due to absorption, got {}",
                atm.optics[s][w].ssa
            );
        }
    }

    #[test]
    fn clear_sky_shells_are_contiguous() {
        let atm = build_clear_sky(AtmosphereType::UsStandard, 0.15);
        for i in 0..(atm.num_shells - 1) {
            assert!(
                (atm.shells[i].r_outer - atm.shells[i + 1].r_inner).abs() < 0.1,
                "Gap between shell {} and {}",
                i,
                i + 1
            );
        }
    }

    // ── build_with_cloud_layer ──

    #[test]
    fn cloud_layer_increases_extinction() {
        let clear = build_clear_sky(AtmosphereType::UsStandard, 0.15);
        let cloudy = build_with_cloud_layer(
            AtmosphereType::UsStandard,
            0.15,
            2.0,
            4.0,
            10.0,
            0.999,
            0.85,
        );
        for s in 0..clear.num_shells {
            let alt_km = clear.shells[s].altitude_mid / 1000.0;
            if alt_km > 2.0 && alt_km < 4.0 {
                let w = 17; // 550nm
                assert!(
                    cloudy.optics[s][w].extinction > clear.optics[s][w].extinction * 2.0,
                    "Cloud should increase extinction at {}km",
                    alt_km
                );
            }
        }
    }

    #[test]
    fn cloud_layer_reduces_rayleigh_fraction() {
        let cloudy = build_with_cloud_layer(
            AtmosphereType::UsStandard,
            0.15,
            2.0,
            4.0,
            10.0,
            0.999,
            0.85,
        );
        for s in 0..cloudy.num_shells {
            let alt_km = cloudy.shells[s].altitude_mid / 1000.0;
            if alt_km > 2.0 && alt_km < 4.0 {
                let w = 17;
                assert!(
                    cloudy.optics[s][w].rayleigh_fraction < 0.9,
                    "Cloud should reduce Rayleigh fraction at {}km: got {}",
                    alt_km,
                    cloudy.optics[s][w].rayleigh_fraction
                );
            }
        }
    }

    #[test]
    fn cloud_layer_sets_nonzero_asymmetry() {
        let cloudy = build_with_cloud_layer(
            AtmosphereType::UsStandard,
            0.15,
            2.0,
            4.0,
            10.0,
            0.999,
            0.85,
        );
        for s in 0..cloudy.num_shells {
            let alt_km = cloudy.shells[s].altitude_mid / 1000.0;
            if alt_km > 2.0 && alt_km < 4.0 {
                let w = 17;
                assert!(
                    cloudy.optics[s][w].asymmetry > 0.1,
                    "Cloud should set nonzero asymmetry at {}km: got {}",
                    alt_km,
                    cloudy.optics[s][w].asymmetry
                );
            }
        }
    }

    #[test]
    fn cloud_layer_does_not_affect_other_shells() {
        let clear = build_clear_sky(AtmosphereType::UsStandard, 0.15);
        let cloudy = build_with_cloud_layer(
            AtmosphereType::UsStandard,
            0.15,
            2.0,
            4.0,
            10.0,
            0.999,
            0.85,
        );
        for s in 0..clear.num_shells {
            let alt_km = clear.shells[s].altitude_mid / 1000.0;
            if alt_km > 10.0 {
                let w = 17;
                assert!(
                    (cloudy.optics[s][w].extinction - clear.optics[s][w].extinction).abs() < 1e-20,
                    "Shell at {}km should be unaffected by 2-4km cloud",
                    alt_km
                );
            }
        }
    }

    #[test]
    fn cloud_layer_zero_od_is_clear_sky() {
        let clear = build_clear_sky(AtmosphereType::UsStandard, 0.15);
        let cloudy =
            build_with_cloud_layer(AtmosphereType::UsStandard, 0.15, 2.0, 4.0, 0.0, 0.999, 0.85);
        for s in 0..clear.num_shells {
            for w in 0..clear.num_wavelengths {
                assert!(
                    (cloudy.optics[s][w].extinction - clear.optics[s][w].extinction).abs() < 1e-20,
                    "OD=0 cloud should not change extinction",
                );
            }
        }
    }

    #[test]
    fn cloud_layer_inverted_base_top_is_clear_sky() {
        let clear = build_clear_sky(AtmosphereType::UsStandard, 0.15);
        let cloudy = build_with_cloud_layer(
            AtmosphereType::UsStandard,
            0.15,
            4.0,
            2.0,
            10.0,
            0.999,
            0.85,
        );
        for s in 0..clear.num_shells {
            for w in 0..clear.num_wavelengths {
                assert!(
                    (cloudy.optics[s][w].extinction - clear.optics[s][w].extinction).abs() < 1e-20,
                    "Inverted cloud should not change extinction",
                );
            }
        }
    }

    // ── DEFAULT_ALTITUDES_KM ──

    #[test]
    fn default_altitudes_monotonic() {
        for i in 0..(NUM_ALTITUDE_LEVELS - 1) {
            assert!(
                DEFAULT_ALTITUDES_KM[i + 1] > DEFAULT_ALTITUDES_KM[i],
                "Altitudes not monotonic at index {}: {} >= {}",
                i,
                DEFAULT_ALTITUDES_KM[i],
                DEFAULT_ALTITUDES_KM[i + 1]
            );
        }
    }

    #[test]
    fn default_altitudes_start_at_zero() {
        assert!((DEFAULT_ALTITUDES_KM[0]).abs() < 1e-10);
    }

    #[test]
    fn default_altitudes_end_at_100km() {
        assert!((DEFAULT_ALTITUDES_KM[NUM_ALTITUDE_LEVELS - 1] - 100.0).abs() < 1e-10);
    }

    #[test]
    fn default_altitudes_finer_below_30km() {
        for i in 0..(NUM_ALTITUDE_LEVELS - 1) {
            if DEFAULT_ALTITUDES_KM[i] < 30.0 && DEFAULT_ALTITUDES_KM[i + 1] <= 30.0 {
                let spacing = DEFAULT_ALTITUDES_KM[i + 1] - DEFAULT_ALTITUDES_KM[i];
                assert!(
                    spacing <= 2.01,
                    "Below 30km, spacing should be <= 2km: got {} at index {}",
                    spacing,
                    i
                );
            }
        }
    }
}
