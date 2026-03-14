//! Atmosphere builder: constructs a populated AtmosphereModel from
//! embedded profile data, Rayleigh scattering, and molecular gas absorption.
//!
//! This bridges twilight-data (raw atmospheric data) and twilight-core
//! (the shell-based atmosphere model that the MCRT engine consumes).
//!
//! Gas absorption (O3, NO2, O2, H2O, O4 CIA) is applied as the final
//! step in [`build_full`], after aerosol and cloud layers are mixed in.
//! This uses the full multi-gas model from [`twilight_core::gas_absorption`]
//! with temperature-dependent cross-sections and bilinear (P,T) interpolation.

use crate::aerosol::{self, AerosolProperties, AerosolType};
use crate::atmosphere_profiles::{self, AtmosphereType};
use crate::cloud::{self, CloudProperties, CloudType};
use twilight_core::atmosphere::{AtmosphereModel, ShellOptics};
use twilight_core::gas_absorption::{apply_gas_absorption, scale_o3_column, standard_gas_profile};
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

/// Build a clear-sky atmosphere with Rayleigh scattering only (no gas absorption).
///
/// This is the base layer for all atmosphere configurations. Gas absorption
/// (O3, NO2, O2, H2O, O4 CIA) is added later by [`build_full`], which calls
/// [`apply_gas_absorption`] as the final step after aerosol and cloud mixing.
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

    // Populate optical properties for each shell and wavelength (Rayleigh only)
    let num_shells = atm.num_shells;
    for s in 0..num_shells {
        let alt_mid_km = atm.shells[s].altitude_mid / 1000.0; // m to km

        // Get atmospheric state at shell midpoint
        let n_density = atmosphere_profiles::number_density_at(alt_mid_km, profile);

        // Number density: convert from molecules/cm^3 to molecules/m^3
        let n_density_m3 = n_density * 1e6;

        for w in 0..atm.num_wavelengths {
            let wl = atm.wavelengths_nm[w];

            // Rayleigh scattering coefficient [1/m]
            let beta_ray = rayleigh_scattering_coeff(wl, n_density_m3);

            atm.optics[s][w] = ShellOptics {
                extinction: beta_ray,
                ssa: 1.0,               // Pure scattering, no absorption yet
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

/// Build an atmosphere with tropospheric aerosols from OPAC climatology.
///
/// Starts with a clear-sky Rayleigh atmosphere, then adds aerosol
/// extinction, absorption, and forward scattering at each shell according
/// to the specified aerosol type's spectral properties and vertical profile.
///
/// **Note:** This does not apply gas absorption. For the full atmosphere
/// (Rayleigh + aerosol + gas absorption), use [`build_full`] instead.
///
/// # Arguments
/// * `profile` - Which AFGL standard atmosphere to use for gas properties
/// * `surface_albedo` - Broadband surface albedo (0.0 - 1.0)
/// * `atype` - OPAC aerosol type (determines all spectral properties)
///
/// # Example
/// ```no_run
/// use twilight_data::builder::build_with_aerosols;
/// use twilight_data::atmosphere_profiles::AtmosphereType;
/// use twilight_data::aerosol::AerosolType;
///
/// let atm = build_with_aerosols(
///     AtmosphereType::UsStandard,
///     0.15,
///     AerosolType::Urban,
/// );
/// ```
pub fn build_with_aerosols(
    profile: AtmosphereType,
    surface_albedo: f64,
    atype: AerosolType,
) -> AtmosphereModel {
    let props = aerosol::default_properties(atype);
    build_with_aerosol_properties(profile, surface_albedo, &props)
}

/// Build an atmosphere with custom aerosol properties.
///
/// Like `build_with_aerosols` but takes explicit aerosol properties instead
/// of a named type. This allows fine-tuning AOD, SSA, etc. for specific
/// conditions or for parameter sweeps.
///
/// # Arguments
/// * `profile` - Which AFGL standard atmosphere to use for gas properties
/// * `surface_albedo` - Broadband surface albedo (0.0 - 1.0)
/// * `aerosol_props` - Custom aerosol optical properties
pub fn build_with_aerosol_properties(
    profile: AtmosphereType,
    surface_albedo: f64,
    aerosol_props: &AerosolProperties,
) -> AtmosphereModel {
    let mut atm = build_clear_sky(profile, surface_albedo);

    if aerosol_props.aod_550 <= 0.0 {
        return atm;
    }

    let num_shells = atm.num_shells;
    for s in 0..num_shells {
        let alt_mid_m = atm.shells[s].altitude_mid;

        // Aerosol extinction at shell midpoint altitude
        // (negligible above ~5 scale heights)
        if alt_mid_m > aerosol_props.scale_height_m * 7.0 {
            continue;
        }

        for w in 0..atm.num_wavelengths {
            let wl = atm.wavelengths_nm[w];

            let aer_ext = aerosol::aerosol_extinction(aerosol_props, wl, alt_mid_m);
            if aer_ext < 1e-30 {
                continue;
            }

            let aer_ssa = aerosol::aerosol_ssa(aerosol_props, wl);
            let aer_g = aerosol::aerosol_asymmetry(aerosol_props, wl);

            let aer_scat = aer_ext * aer_ssa;
            let aer_abs = aer_ext * (1.0 - aer_ssa);

            let existing = &atm.optics[s][w];

            // Existing Rayleigh scattering (ssa=1 for pure Rayleigh, but
            // may be <1 in shells with O₃ absorption)
            let existing_scat = existing.extinction * existing.ssa;
            let existing_ray_scat = existing_scat * existing.rayleigh_fraction;
            let existing_nonray_scat = existing_scat * (1.0 - existing.rayleigh_fraction);
            let existing_abs = existing.extinction * (1.0 - existing.ssa);

            // Mix: total scattering = Rayleigh + existing non-Rayleigh + aerosol
            let total_scat = existing_ray_scat + existing_nonray_scat + aer_scat;
            let total_abs = existing_abs + aer_abs;
            let total_ext = total_scat + total_abs;

            let new_ssa = if total_ext > 1e-30 {
                total_scat / total_ext
            } else {
                1.0
            };

            // Scattering-weighted asymmetry parameter:
            // g_total = (g_ray × β_ray + g_existing × β_existing_nonray + g_aer × β_aer) / β_total_scat
            // Rayleigh has g=0, so its term vanishes.
            let new_g = if total_scat > 1e-30 {
                (existing_nonray_scat * existing.asymmetry + aer_scat * aer_g) / total_scat
            } else {
                0.0
            };

            // Rayleigh fraction of total scattering
            let ray_frac = if total_scat > 1e-30 {
                existing_ray_scat / total_scat
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

/// Build an atmosphere with a named cloud type.
///
/// Equivalent to calling `build_with_cloud_layer` with the default
/// properties for the given cloud type.
pub fn build_with_cloud(
    profile: AtmosphereType,
    surface_albedo: f64,
    ctype: CloudType,
) -> AtmosphereModel {
    let props = cloud::default_properties(ctype);
    build_with_cloud_layer(
        profile,
        surface_albedo,
        props.base_km,
        props.top_km,
        props.optical_depth,
        props.ssa,
        props.asymmetry,
    )
}

/// Build an atmosphere with custom cloud properties.
pub fn build_with_cloud_properties(
    profile: AtmosphereType,
    surface_albedo: f64,
    cloud_props: &CloudProperties,
) -> AtmosphereModel {
    build_with_cloud_layer(
        profile,
        surface_albedo,
        cloud_props.base_km,
        cloud_props.top_km,
        cloud_props.optical_depth,
        cloud_props.ssa,
        cloud_props.asymmetry,
    )
}

/// Build an atmosphere with both aerosols and a cloud layer.
///
/// Starts with clear sky, adds aerosol, then adds cloud on top.
/// This is the most realistic single-layer configuration.
pub fn build_with_aerosols_and_cloud(
    profile: AtmosphereType,
    surface_albedo: f64,
    atype: AerosolType,
    ctype: CloudType,
) -> AtmosphereModel {
    let aerosol_props = aerosol::default_properties(atype);
    let cloud_props = cloud::default_properties(ctype);
    build_full(
        profile,
        surface_albedo,
        Some(&aerosol_props),
        Some(&cloud_props),
    )
}

/// Build an atmosphere with optional aerosol and optional cloud.
///
/// The most general builder. Pass `None` for either to omit.
///
/// This is the primary entry point for all production paths. It applies
/// molecular gas absorption (O3, NO2, O2, H2O, O4 CIA) as the final step
/// using the full multi-gas model from [`twilight_core::gas_absorption`].
///
/// The layering order is:
/// 1. Rayleigh scattering (from `build_clear_sky`)
/// 2. Aerosol extinction + scattering (if provided)
/// 3. Cloud extinction + scattering (if provided)
/// 4. **Gas absorption** (O3, NO2, O2, H2O, O4 CIA) -- applied last
pub fn build_full(
    profile: AtmosphereType,
    surface_albedo: f64,
    aerosol_props: Option<&AerosolProperties>,
    cloud_props: Option<&CloudProperties>,
) -> AtmosphereModel {
    let mut atm = match aerosol_props {
        Some(props) => build_with_aerosol_properties(profile, surface_albedo, props),
        None => build_clear_sky(profile, surface_albedo),
    };

    if let Some(cloud_props) = cloud_props {
        if cloud_props.optical_depth > 0.0 {
            let cloud_thickness_m = (cloud_props.top_km - cloud_props.base_km) * 1000.0;
            if cloud_thickness_m > 0.0 {
                let cloud_ext = cloud_props.optical_depth / cloud_thickness_m;
                add_cloud_layer(&mut atm, cloud_props, cloud_ext);
            }
        }
    }

    // Apply molecular gas absorption as the final step.
    apply_gas_absorption_standard(&mut atm, None, None);

    atm
}

/// Build an atmosphere with gas composition overrides from observations.
///
/// Like [`build_full`] but allows overriding the O3 total column (in Dobson
/// Units) and surface NO2 density (molecules/m^3) using real-time data
/// from the weather/air quality API.
///
/// # Arguments
/// * `o3_column_du` - Target O3 total column in DU. `None` uses the standard
///   atmosphere default (~347 DU).
/// * `no2_surface_density` - Surface NO2 in molecules/m^3. `None` uses the
///   standard atmosphere default.
pub fn build_full_with_gas(
    profile: AtmosphereType,
    surface_albedo: f64,
    aerosol_props: Option<&AerosolProperties>,
    cloud_props: Option<&CloudProperties>,
    o3_column_du: Option<f64>,
    no2_surface_density: Option<f64>,
) -> AtmosphereModel {
    let mut atm = match aerosol_props {
        Some(props) => build_with_aerosol_properties(profile, surface_albedo, props),
        None => build_clear_sky(profile, surface_albedo),
    };

    if let Some(cloud_props) = cloud_props {
        if cloud_props.optical_depth > 0.0 {
            let cloud_thickness_m = (cloud_props.top_km - cloud_props.base_km) * 1000.0;
            if cloud_thickness_m > 0.0 {
                let cloud_ext = cloud_props.optical_depth / cloud_thickness_m;
                add_cloud_layer(&mut atm, cloud_props, cloud_ext);
            }
        }
    }

    apply_gas_absorption_standard(&mut atm, o3_column_du, no2_surface_density);

    atm
}

/// Apply standard gas absorption with optional column overrides.
///
/// Builds a standard gas profile, optionally scales O3 to a target column
/// and/or replaces the NO2 surface density, then applies gas absorption.
fn apply_gas_absorption_standard(
    atm: &mut AtmosphereModel,
    o3_column_du: Option<f64>,
    no2_surface_density: Option<f64>,
) {
    let mut gas_profile = standard_gas_profile(atm);

    // Scale O3 column to target DU if provided
    if let Some(target_du) = o3_column_du {
        scale_o3_column(&mut gas_profile, atm, target_du);
    }

    // Scale NO2 profile if surface density override provided.
    // We scale all shells proportionally so the shape is preserved
    // but the surface value matches the observation.
    if let Some(target_surface) = no2_surface_density {
        if gas_profile.num_shells > 0 {
            let current_surface = gas_profile.shells[0].no2_density;
            if current_surface > 1e-30 {
                let factor = target_surface / current_surface;
                for s in 0..gas_profile.num_shells {
                    gas_profile.shells[s].no2_density *= factor;
                }
            }
        }
    }

    apply_gas_absorption(atm, &gas_profile);
}

/// Add a cloud layer to an existing atmosphere model.
///
/// Extracted from `build_full` to keep the function manageable. Distributes
/// cloud optical properties across shells that overlap the cloud layer.
fn add_cloud_layer(atm: &mut AtmosphereModel, cloud_props: &CloudProperties, cloud_ext: f64) {
    let num_shells = atm.num_shells;
    for s in 0..num_shells {
        let shell_base_km =
            (atm.shells[s].r_inner - twilight_core::atmosphere::EARTH_RADIUS_M) / 1000.0;
        let shell_top_km =
            (atm.shells[s].r_outer - twilight_core::atmosphere::EARTH_RADIUS_M) / 1000.0;

        let overlap_base = shell_base_km.max(cloud_props.base_km);
        let overlap_top = shell_top_km.min(cloud_props.top_km);
        if overlap_top <= overlap_base {
            continue;
        }

        let shell_thickness_km = shell_top_km - shell_base_km;
        let overlap_fraction = (overlap_top - overlap_base) / shell_thickness_km;

        for w in 0..atm.num_wavelengths {
            let existing = &atm.optics[s][w];

            let c_ext = cloud_ext * overlap_fraction;
            let c_scat = c_ext * cloud_props.ssa;
            let c_abs = c_ext * (1.0 - cloud_props.ssa);

            let existing_scat = existing.extinction * existing.ssa;
            let existing_ray_scat = existing_scat * existing.rayleigh_fraction;
            let existing_nonray_scat = existing_scat * (1.0 - existing.rayleigh_fraction);
            let existing_abs = existing.extinction * (1.0 - existing.ssa);

            let total_scat = existing_ray_scat + existing_nonray_scat + c_scat;
            let total_abs = existing_abs + c_abs;
            let total_ext = total_scat + total_abs;

            let new_ssa = if total_ext > 1e-30 {
                total_scat / total_ext
            } else {
                1.0
            };

            let new_g = if total_scat > 1e-30 {
                (existing_nonray_scat * existing.asymmetry + c_scat * cloud_props.asymmetry)
                    / total_scat
            } else {
                0.0
            };

            let ray_frac = if total_scat > 1e-30 {
                existing_ray_scat / total_scat
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
    fn clear_sky_ssa_is_one_pure_rayleigh() {
        // build_clear_sky is now Rayleigh-only: SSA = 1.0 everywhere
        let atm = build_clear_sky(AtmosphereType::UsStandard, 0.15);
        for s in 0..atm.num_shells {
            for w in 0..atm.num_wavelengths {
                assert!(
                    (atm.optics[s][w].ssa - 1.0).abs() < 1e-10,
                    "SSA at shell {}, wl {} = {}, expected 1.0 (pure Rayleigh)",
                    s,
                    w,
                    atm.optics[s][w].ssa
                );
            }
        }
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
    fn full_build_ozone_absorption_at_chappuis_peak() {
        // Gas absorption is now applied in build_full, not build_clear_sky
        let atm = build_full(AtmosphereType::UsStandard, 0.15, None, None);
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

    // ── build_with_aerosols ──

    #[test]
    fn aerosol_increases_extinction_at_surface() {
        let clear = build_clear_sky(AtmosphereType::UsStandard, 0.15);
        let aerosol = build_with_aerosols(AtmosphereType::UsStandard, 0.15, AerosolType::Urban);
        // At the surface shell, aerosol adds extinction at all wavelengths
        for w in 0..clear.num_wavelengths {
            assert!(
                aerosol.optics[0][w].extinction > clear.optics[0][w].extinction,
                "Aerosol should increase surface extinction at wl[{}]: {:.4e} vs {:.4e}",
                w,
                aerosol.optics[0][w].extinction,
                clear.optics[0][w].extinction
            );
        }
    }

    #[test]
    fn aerosol_negligible_at_high_altitude() {
        let clear = build_clear_sky(AtmosphereType::UsStandard, 0.15);
        let aerosol = build_with_aerosols(
            AtmosphereType::UsStandard,
            0.15,
            AerosolType::ContinentalClean,
        );
        // Above ~30km, aerosol contribution should be negligible
        // (scale height is 2km, so at 30km = 15 scale heights, exp(-15) ~ 3e-7)
        for s in 0..clear.num_shells {
            let alt_km = clear.shells[s].altitude_mid / 1000.0;
            if alt_km > 30.0 {
                let w = 17; // 550nm
                let rel_change = (aerosol.optics[s][w].extinction - clear.optics[s][w].extinction)
                    / clear.optics[s][w].extinction.max(1e-30);
                assert!(
                    rel_change.abs() < 1e-4,
                    "At {}km, aerosol should be negligible: rel_change={:.4e}",
                    alt_km,
                    rel_change
                );
            }
        }
    }

    #[test]
    fn aerosol_reduces_rayleigh_fraction() {
        let aerosol = build_with_aerosols(AtmosphereType::UsStandard, 0.15, AerosolType::Urban);
        // In the boundary layer, Rayleigh fraction should be < 1
        let w = 17; // 550nm
        assert!(
            aerosol.optics[0][w].rayleigh_fraction < 0.99,
            "Aerosol should reduce Rayleigh fraction at surface: got {}",
            aerosol.optics[0][w].rayleigh_fraction
        );
    }

    #[test]
    fn aerosol_sets_nonzero_asymmetry() {
        let aerosol = build_with_aerosols(AtmosphereType::UsStandard, 0.15, AerosolType::Urban);
        // Aerosol has nonzero asymmetry, so blended value should be positive
        let w = 17;
        assert!(
            aerosol.optics[0][w].asymmetry > 0.01,
            "Aerosol should set nonzero asymmetry: got {}",
            aerosol.optics[0][w].asymmetry
        );
    }

    #[test]
    fn aerosol_reduces_ssa_for_absorbing_type() {
        let clear = build_clear_sky(AtmosphereType::UsStandard, 0.15);
        let aerosol = build_with_aerosols(AtmosphereType::UsStandard, 0.15, AerosolType::Urban);
        // Urban aerosol (SSA~0.88) should reduce the blended SSA below the
        // clear-sky value (which is ~1.0 for pure Rayleigh)
        let w = 17;
        assert!(
            aerosol.optics[0][w].ssa < clear.optics[0][w].ssa,
            "Urban aerosol should reduce SSA: aerosol={:.4} vs clear={:.4}",
            aerosol.optics[0][w].ssa,
            clear.optics[0][w].ssa
        );
    }

    #[test]
    fn aerosol_zero_aod_is_clear_sky() {
        let clear = build_clear_sky(AtmosphereType::UsStandard, 0.15);
        let zero_aerosol = build_with_aerosol_properties(
            AtmosphereType::UsStandard,
            0.15,
            &AerosolProperties {
                aod_550: 0.0,
                ssa_550: 0.90,
                asymmetry_550: 0.70,
                angstrom_exponent: 1.3,
                scale_height_m: 2000.0,
                ssa_slope: 0.0,
                g_slope: 0.0,
            },
        );
        for s in 0..clear.num_shells {
            for w in 0..clear.num_wavelengths {
                assert!(
                    (zero_aerosol.optics[s][w].extinction - clear.optics[s][w].extinction).abs()
                        < 1e-20,
                    "Zero AOD should not change extinction at shell {}, wl {}",
                    s,
                    w
                );
            }
        }
    }

    #[test]
    fn aerosol_desert_larger_effect_than_continental_clean() {
        let desert = build_with_aerosols(AtmosphereType::UsStandard, 0.15, AerosolType::Desert);
        let clean = build_with_aerosols(
            AtmosphereType::UsStandard,
            0.15,
            AerosolType::ContinentalClean,
        );
        // Desert (AOD=0.5) should have much more extinction than continental clean (AOD=0.05)
        let w = 17; // 550nm
        assert!(
            desert.optics[0][w].extinction > clean.optics[0][w].extinction * 1.5,
            "Desert ext={:.4e} should be >> continental clean ext={:.4e}",
            desert.optics[0][w].extinction,
            clean.optics[0][w].extinction
        );
    }

    #[test]
    fn aerosol_all_types_build_without_panic() {
        use crate::aerosol::ALL_AEROSOL_TYPES;
        for atype in &ALL_AEROSOL_TYPES {
            let atm = build_with_aerosols(AtmosphereType::UsStandard, 0.15, *atype);
            // Sanity: should still have 50 shells and 41 wavelengths
            assert_eq!(atm.num_shells, 50);
            assert_eq!(atm.num_wavelengths, 41);
        }
    }

    #[test]
    fn aerosol_preserves_shell_geometry() {
        let clear = build_clear_sky(AtmosphereType::UsStandard, 0.15);
        let aerosol = build_with_aerosols(AtmosphereType::UsStandard, 0.15, AerosolType::Urban);
        // Shell geometry should be identical
        for s in 0..clear.num_shells {
            assert!(
                (aerosol.shells[s].r_inner - clear.shells[s].r_inner).abs() < 1e-6,
                "r_inner mismatch at shell {}",
                s
            );
            assert!(
                (aerosol.shells[s].r_outer - clear.shells[s].r_outer).abs() < 1e-6,
                "r_outer mismatch at shell {}",
                s
            );
        }
    }

    #[test]
    fn aerosol_ssa_bounded() {
        use crate::aerosol::ALL_AEROSOL_TYPES;
        for atype in &ALL_AEROSOL_TYPES {
            let atm = build_with_aerosols(AtmosphereType::UsStandard, 0.15, *atype);
            for s in 0..atm.num_shells {
                for w in 0..atm.num_wavelengths {
                    assert!(
                        (0.0..=1.0).contains(&atm.optics[s][w].ssa),
                        "{:?}: SSA out of bounds at shell {}, wl {}: {}",
                        atype,
                        s,
                        w,
                        atm.optics[s][w].ssa
                    );
                }
            }
        }
    }

    #[test]
    fn aerosol_asymmetry_bounded() {
        use crate::aerosol::ALL_AEROSOL_TYPES;
        for atype in &ALL_AEROSOL_TYPES {
            let atm = build_with_aerosols(AtmosphereType::UsStandard, 0.15, *atype);
            for s in 0..atm.num_shells {
                for w in 0..atm.num_wavelengths {
                    assert!(
                        (-1.0..=1.0).contains(&atm.optics[s][w].asymmetry),
                        "{:?}: asymmetry out of bounds at shell {}, wl {}: {}",
                        atype,
                        s,
                        w,
                        atm.optics[s][w].asymmetry
                    );
                }
            }
        }
    }

    #[test]
    fn aerosol_rayleigh_fraction_bounded() {
        use crate::aerosol::ALL_AEROSOL_TYPES;
        for atype in &ALL_AEROSOL_TYPES {
            let atm = build_with_aerosols(AtmosphereType::UsStandard, 0.15, *atype);
            for s in 0..atm.num_shells {
                for w in 0..atm.num_wavelengths {
                    assert!(
                        (0.0..=1.0).contains(&atm.optics[s][w].rayleigh_fraction),
                        "{:?}: rayleigh_fraction out of bounds at shell {}, wl {}: {}",
                        atype,
                        s,
                        w,
                        atm.optics[s][w].rayleigh_fraction
                    );
                }
            }
        }
    }

    #[test]
    fn aerosol_extinction_positive_everywhere() {
        use crate::aerosol::ALL_AEROSOL_TYPES;
        for atype in &ALL_AEROSOL_TYPES {
            let atm = build_with_aerosols(AtmosphereType::UsStandard, 0.15, *atype);
            for s in 0..atm.num_shells {
                for w in 0..atm.num_wavelengths {
                    assert!(
                        atm.optics[s][w].extinction >= 0.0,
                        "{:?}: negative extinction at shell {}, wl {}: {}",
                        atype,
                        s,
                        w,
                        atm.optics[s][w].extinction
                    );
                }
            }
        }
    }

    #[test]
    fn aerosol_custom_high_aod() {
        // Extreme AOD should still produce valid optics
        let atm = build_with_aerosol_properties(
            AtmosphereType::UsStandard,
            0.15,
            &AerosolProperties {
                aod_550: 5.0, // Very heavy dust storm
                ssa_550: 0.85,
                asymmetry_550: 0.75,
                angstrom_exponent: 0.3,
                scale_height_m: 3000.0,
                ssa_slope: 0.0,
                g_slope: 0.0,
            },
        );
        // Extinction should be huge at the surface
        let w = 17; // 550nm
        let ext = atm.optics[0][w].extinction;
        assert!(
            ext > 1e-3,
            "Heavy dust should give large extinction: {}",
            ext
        );
        // But SSA should still be valid
        assert!((0.0..=1.0).contains(&atm.optics[0][w].ssa));
    }

    #[test]
    fn aerosol_maritime_clean_barely_absorbs() {
        let atm = build_with_aerosols(AtmosphereType::UsStandard, 0.15, AerosolType::MaritimeClean);
        // Maritime clean has SSA=0.99, so blended SSA should still be very high
        let w = 17;
        assert!(
            atm.optics[0][w].ssa > 0.95,
            "Maritime clean should have high SSA: {}",
            atm.optics[0][w].ssa
        );
    }

    #[test]
    fn aerosol_effect_strongest_at_surface() {
        let clear = build_clear_sky(AtmosphereType::UsStandard, 0.15);
        let aerosol = build_with_aerosols(AtmosphereType::UsStandard, 0.15, AerosolType::Urban);
        let w = 17; // 550nm
        let delta_surface = aerosol.optics[0][w].extinction - clear.optics[0][w].extinction;
        // Find a shell at ~5km (about 3 scale heights for urban H=1.5km)
        let mut delta_5km = 0.0;
        for s in 0..aerosol.num_shells {
            let alt_km = aerosol.shells[s].altitude_mid / 1000.0;
            if alt_km > 4.5 && alt_km < 5.5 {
                delta_5km = aerosol.optics[s][w].extinction - clear.optics[s][w].extinction;
                break;
            }
        }
        assert!(
            delta_surface > delta_5km,
            "Aerosol effect should be strongest at surface: delta_sfc={:.4e} vs delta_5km={:.4e}",
            delta_surface,
            delta_5km
        );
    }

    // ── build_with_cloud (named type) ──

    #[test]
    fn named_cloud_all_types_build() {
        use crate::cloud::ALL_CLOUD_TYPES;
        for ctype in &ALL_CLOUD_TYPES {
            let atm = build_with_cloud(AtmosphereType::UsStandard, 0.15, *ctype);
            assert_eq!(atm.num_shells, 50);
            assert_eq!(atm.num_wavelengths, 41);
        }
    }

    #[test]
    fn named_cloud_matches_manual() {
        // build_with_cloud should produce the same result as calling
        // build_with_cloud_layer with the default properties
        let named = build_with_cloud(AtmosphereType::UsStandard, 0.15, CloudType::Stratus);
        let props = cloud::default_properties(CloudType::Stratus);
        let manual = build_with_cloud_layer(
            AtmosphereType::UsStandard,
            0.15,
            props.base_km,
            props.top_km,
            props.optical_depth,
            props.ssa,
            props.asymmetry,
        );
        for s in 0..named.num_shells {
            for w in 0..named.num_wavelengths {
                assert!(
                    (named.optics[s][w].extinction - manual.optics[s][w].extinction).abs() < 1e-20,
                    "Named vs manual mismatch at shell {}, wl {}",
                    s,
                    w
                );
            }
        }
    }

    #[test]
    fn stratus_much_thicker_than_thin_cirrus() {
        let stratus = build_with_cloud(AtmosphereType::UsStandard, 0.15, CloudType::Stratus);
        let cirrus = build_with_cloud(AtmosphereType::UsStandard, 0.15, CloudType::ThinCirrus);
        let clear = build_clear_sky(AtmosphereType::UsStandard, 0.15);

        // Both should increase extinction over clear sky, but stratus much more
        let w = 17;
        let mut stratus_max_delta = 0.0f64;
        let mut cirrus_max_delta = 0.0f64;
        for s in 0..clear.num_shells {
            let sd = stratus.optics[s][w].extinction - clear.optics[s][w].extinction;
            let cd = cirrus.optics[s][w].extinction - clear.optics[s][w].extinction;
            stratus_max_delta = stratus_max_delta.max(sd);
            cirrus_max_delta = cirrus_max_delta.max(cd);
        }
        assert!(
            stratus_max_delta > cirrus_max_delta * 5.0,
            "Stratus delta ({:.4e}) should be >> thin cirrus delta ({:.4e})",
            stratus_max_delta,
            cirrus_max_delta
        );
    }

    // ── build_with_aerosols_and_cloud ──

    #[test]
    fn combined_aerosol_cloud_builds() {
        let atm = build_with_aerosols_and_cloud(
            AtmosphereType::UsStandard,
            0.15,
            AerosolType::Urban,
            CloudType::Stratus,
        );
        assert_eq!(atm.num_shells, 50);
        assert_eq!(atm.num_wavelengths, 41);
    }

    #[test]
    fn combined_more_extinction_than_either_alone() {
        let aerosol_only =
            build_with_aerosols(AtmosphereType::UsStandard, 0.15, AerosolType::Urban);
        let cloud_only = build_with_cloud(AtmosphereType::UsStandard, 0.15, CloudType::Stratus);
        let combined = build_with_aerosols_and_cloud(
            AtmosphereType::UsStandard,
            0.15,
            AerosolType::Urban,
            CloudType::Stratus,
        );

        let w = 17;
        // In the cloud layer, combined should exceed aerosol-only
        for s in 0..combined.num_shells {
            let alt_km = combined.shells[s].altitude_mid / 1000.0;
            if alt_km > 0.5 && alt_km < 1.5 {
                assert!(
                    combined.optics[s][w].extinction > aerosol_only.optics[s][w].extinction,
                    "Combined should exceed aerosol-only in cloud layer at {:.1}km",
                    alt_km
                );
            }
        }
        // At the surface (below cloud), combined should have aerosol effect
        assert!(
            combined.optics[0][w].extinction > cloud_only.optics[0][w].extinction,
            "Combined should have aerosol at surface"
        );
    }

    #[test]
    fn combined_ssa_and_asymmetry_bounded() {
        // Test a few combinations
        let aerosol_types = [AerosolType::Urban, AerosolType::Desert];
        let cloud_types = [CloudType::Stratus, CloudType::ThinCirrus];
        for atype in &aerosol_types {
            for ctype in &cloud_types {
                let atm =
                    build_with_aerosols_and_cloud(AtmosphereType::UsStandard, 0.15, *atype, *ctype);
                for s in 0..atm.num_shells {
                    for w in 0..atm.num_wavelengths {
                        assert!(
                            (0.0..=1.0).contains(&atm.optics[s][w].ssa),
                            "{:?}+{:?}: SSA out of bounds at shell {}, wl {}",
                            atype,
                            ctype,
                            s,
                            w
                        );
                        assert!(
                            (-1.0..=1.0).contains(&atm.optics[s][w].asymmetry),
                            "{:?}+{:?}: asymmetry out of bounds at shell {}, wl {}",
                            atype,
                            ctype,
                            s,
                            w
                        );
                        assert!(
                            (0.0..=1.0).contains(&atm.optics[s][w].rayleigh_fraction),
                            "{:?}+{:?}: rayleigh_fraction out of bounds at shell {}, wl {}",
                            atype,
                            ctype,
                            s,
                            w
                        );
                    }
                }
            }
        }
    }

    // ── build_full ──

    #[test]
    fn build_full_no_aerosol_no_cloud_has_gas_absorption() {
        // build_full applies gas absorption on top of build_clear_sky,
        // so extinction should be >= the Rayleigh-only clear sky
        let clear = build_clear_sky(AtmosphereType::UsStandard, 0.15);
        let full = build_full(AtmosphereType::UsStandard, 0.15, None, None);
        let mut found_absorption = false;
        for s in 0..clear.num_shells {
            for w in 0..clear.num_wavelengths {
                assert!(
                    full.optics[s][w].extinction >= clear.optics[s][w].extinction - 1e-30,
                    "Gas absorption should not decrease extinction at shell {}, wl {}",
                    s,
                    w,
                );
                if full.optics[s][w].extinction > clear.optics[s][w].extinction * 1.001 {
                    found_absorption = true;
                }
            }
        }
        assert!(
            found_absorption,
            "Gas absorption should increase extinction in at least some shells"
        );
    }

    #[test]
    fn build_full_aerosol_only_has_more_extinction() {
        // build_full adds gas absorption on top of aerosol, so extinction
        // should be >= the aerosol-only result
        let aerosol_only =
            build_with_aerosols(AtmosphereType::UsStandard, 0.15, AerosolType::Urban);
        let props = aerosol::default_properties(AerosolType::Urban);
        let full = build_full(AtmosphereType::UsStandard, 0.15, Some(&props), None);
        for s in 0..aerosol_only.num_shells {
            for w in 0..aerosol_only.num_wavelengths {
                assert!(
                    full.optics[s][w].extinction >= aerosol_only.optics[s][w].extinction - 1e-30,
                    "Gas absorption should not decrease extinction at shell {}, wl {}",
                    s,
                    w,
                );
            }
        }
    }

    #[test]
    fn build_full_cloud_only_has_more_extinction() {
        // build_full adds gas absorption on top of cloud, so extinction
        // should be >= the cloud-only result
        let cloud_only = build_with_cloud(AtmosphereType::UsStandard, 0.15, CloudType::Stratus);
        let props = cloud::default_properties(CloudType::Stratus);
        let full = build_full(AtmosphereType::UsStandard, 0.15, None, Some(&props));
        for s in 0..cloud_only.num_shells {
            for w in 0..cloud_only.num_wavelengths {
                assert!(
                    full.optics[s][w].extinction >= cloud_only.optics[s][w].extinction - 1e-30,
                    "Gas absorption should not decrease extinction at shell {}, wl {}",
                    s,
                    w,
                );
            }
        }
    }

    // ── Gas absorption integration tests ──

    #[test]
    fn build_full_ssa_bounded_with_gas_absorption() {
        // SSA must remain in [0, 1] after gas absorption is applied
        let atm = build_full(AtmosphereType::UsStandard, 0.15, None, None);
        for s in 0..atm.num_shells {
            for w in 0..atm.num_wavelengths {
                assert!(
                    (0.0..=1.0).contains(&atm.optics[s][w].ssa),
                    "SSA out of bounds at shell {}, wl {}: {}",
                    s,
                    w,
                    atm.optics[s][w].ssa
                );
            }
        }
    }

    #[test]
    fn build_full_extinction_positive_with_gas_absorption() {
        let atm = build_full(AtmosphereType::UsStandard, 0.15, None, None);
        for s in 0..atm.num_shells {
            for w in 0..atm.num_wavelengths {
                assert!(
                    atm.optics[s][w].extinction >= 0.0,
                    "Negative extinction at shell {}, wl {}: {}",
                    s,
                    w,
                    atm.optics[s][w].extinction
                );
            }
        }
    }

    #[test]
    fn build_full_gas_absorption_reduces_ssa_in_ozone_layer() {
        // In the ozone layer (20-25 km), gas absorption should noticeably
        // reduce SSA at Chappuis band wavelengths (500-650 nm)
        let atm = build_full(AtmosphereType::UsStandard, 0.15, None, None);
        let w_550 = 17; // 550nm, near Chappuis peak
        for s in 0..atm.num_shells {
            let alt_km = atm.shells[s].altitude_mid / 1000.0;
            if alt_km > 20.0 && alt_km < 26.0 {
                assert!(
                    atm.optics[s][w_550].ssa < 0.95,
                    "SSA at {:.1} km, 550nm should be < 0.95 due to O3: got {:.4}",
                    alt_km,
                    atm.optics[s][w_550].ssa
                );
                return; // Found it
            }
        }
        panic!("No shell found in 20-26 km range");
    }

    #[test]
    fn build_full_preserves_rayleigh_fraction_clear_sky() {
        // Gas absorption does not scatter, so rayleigh_fraction should
        // still be 1.0 in a clear-sky (no aerosol/cloud) atmosphere
        let atm = build_full(AtmosphereType::UsStandard, 0.15, None, None);
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
    fn build_full_preserves_asymmetry_clear_sky() {
        // Gas absorption does not scatter, so asymmetry should remain 0.0
        // in a clear-sky atmosphere
        let atm = build_full(AtmosphereType::UsStandard, 0.15, None, None);
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
    fn build_full_gas_absorption_stronger_at_chappuis_than_green() {
        // O3 Chappuis band absorption peaks around 600nm and is weaker at
        // 500nm green. In the ozone layer, SSA at 600nm should be lower
        // than at 500nm due to stronger O3 absorption.
        let atm = build_full(AtmosphereType::UsStandard, 0.15, None, None);
        for s in 0..atm.num_shells {
            let alt_km = atm.shells[s].altitude_mid / 1000.0;
            if alt_km > 20.0 && alt_km < 26.0 {
                let ssa_500 = atm.optics[s][12].ssa; // 500nm
                let ssa_600 = atm.optics[s][22].ssa; // 600nm
                assert!(
                    ssa_600 < ssa_500,
                    "O3 Chappuis at 600nm (SSA={:.4}) should reduce SSA more than 500nm (SSA={:.4})",
                    ssa_600,
                    ssa_500
                );
                return;
            }
        }
    }

    #[test]
    fn build_full_all_combinations_valid() {
        // All aerosol + cloud combinations should produce valid optics
        // with gas absorption applied
        use crate::aerosol::ALL_AEROSOL_TYPES;
        use crate::cloud::ALL_CLOUD_TYPES;

        for atype in &ALL_AEROSOL_TYPES {
            let aprops = aerosol::default_properties(*atype);
            for ctype in &ALL_CLOUD_TYPES {
                let cprops = cloud::default_properties(*ctype);
                let atm = build_full(
                    AtmosphereType::UsStandard,
                    0.15,
                    Some(&aprops),
                    Some(&cprops),
                );
                for s in 0..atm.num_shells {
                    for w in 0..atm.num_wavelengths {
                        assert!(
                            atm.optics[s][w].extinction >= 0.0,
                            "{:?}+{:?}: negative extinction at shell {}, wl {}",
                            atype,
                            ctype,
                            s,
                            w,
                        );
                        assert!(
                            (0.0..=1.0).contains(&atm.optics[s][w].ssa),
                            "{:?}+{:?}: SSA out of bounds at shell {}, wl {}",
                            atype,
                            ctype,
                            s,
                            w,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn build_full_scattering_coefficient_preserved() {
        // Gas absorption preserves the scattering coefficient:
        // scat_before = ext_before * ssa_before should equal ext_after * ssa_after
        let clear = build_clear_sky(AtmosphereType::UsStandard, 0.15);
        let full = build_full(AtmosphereType::UsStandard, 0.15, None, None);
        for s in 0..clear.num_shells {
            for w in 0..clear.num_wavelengths {
                let scat_before = clear.optics[s][w].extinction * clear.optics[s][w].ssa;
                let scat_after = full.optics[s][w].extinction * full.optics[s][w].ssa;
                let rel_err = if scat_before > 1e-30 {
                    (scat_after - scat_before).abs() / scat_before
                } else {
                    0.0
                };
                assert!(
                    rel_err < 1e-10,
                    "Scattering coefficient not preserved at shell {}, wl {}: before={:.6e} after={:.6e}",
                    s, w, scat_before, scat_after,
                );
            }
        }
    }

    // ── build_full_with_gas tests ───────────────────────────────────────

    #[test]
    fn build_full_with_gas_none_overrides_matches_build_full() {
        // When both overrides are None, build_full_with_gas should produce
        // identical results to build_full (same standard gas profile).
        let full = build_full(AtmosphereType::UsStandard, 0.15, None, None);
        let full_gas =
            build_full_with_gas(AtmosphereType::UsStandard, 0.15, None, None, None, None);
        for s in 0..full.num_shells {
            for w in 0..full.num_wavelengths {
                let ext_a = full.optics[s][w].extinction;
                let ext_b = full_gas.optics[s][w].extinction;
                let rel = if ext_a > 1e-30 {
                    (ext_b - ext_a).abs() / ext_a
                } else {
                    0.0
                };
                assert!(
                    rel < 1e-10,
                    "Shell {} wl {}: build_full={:.6e} vs build_full_with_gas={:.6e}",
                    s,
                    w,
                    ext_a,
                    ext_b,
                );
            }
        }
    }

    #[test]
    fn build_full_with_gas_o3_override_changes_extinction() {
        // Doubling O3 column (standard is ~347 DU) should increase extinction
        // in the Chappuis band (~600 nm) in the ozone layer (20-25 km).
        let standard =
            build_full_with_gas(AtmosphereType::UsStandard, 0.15, None, None, None, None);
        let high_o3 = build_full_with_gas(
            AtmosphereType::UsStandard,
            0.15,
            None,
            None,
            Some(600.0),
            None,
        );
        // Find Chappuis wavelength index (~600 nm)
        let mut chappuis_idx = 0;
        for w in 0..standard.num_wavelengths {
            if (standard.wavelengths_nm[w] - 600.0).abs() < 15.0 {
                chappuis_idx = w;
                break;
            }
        }
        // Check ozone layer shells (~15-35 km). O3 absorption adds to
        // Rayleigh extinction, so the relative increase in total extinction
        // is modest (~5-10% going from 347 to 600 DU). We only require >1%.
        let mut found_increase = false;
        for s in 0..standard.num_shells {
            let alt = standard.shells[s].altitude_mid;
            if alt > 15_000.0 && alt < 35_000.0 {
                let ext_std = standard.optics[s][chappuis_idx].extinction;
                let ext_high = high_o3.optics[s][chappuis_idx].extinction;
                if ext_std > 1e-30 && ext_high > ext_std * 1.01 {
                    found_increase = true;
                }
            }
        }
        assert!(
            found_increase,
            "600 DU O3 column should increase Chappuis extinction over default ~347 DU"
        );
    }

    #[test]
    fn build_full_with_gas_low_o3_reduces_extinction() {
        // A low O3 column (220 DU, ozone hole) should reduce extinction
        // relative to the default ~347 DU.
        let standard =
            build_full_with_gas(AtmosphereType::UsStandard, 0.15, None, None, None, None);
        let low_o3 = build_full_with_gas(
            AtmosphereType::UsStandard,
            0.15,
            None,
            None,
            Some(220.0),
            None,
        );
        let mut chappuis_idx = 0;
        for w in 0..standard.num_wavelengths {
            if (standard.wavelengths_nm[w] - 600.0).abs() < 15.0 {
                chappuis_idx = w;
                break;
            }
        }
        let mut found_decrease = false;
        for s in 0..standard.num_shells {
            let alt = standard.shells[s].altitude_mid;
            if alt > 15_000.0 && alt < 35_000.0 {
                let ext_std = standard.optics[s][chappuis_idx].extinction;
                let ext_low = low_o3.optics[s][chappuis_idx].extinction;
                if ext_std > 1e-30 && ext_low < ext_std * 0.95 {
                    found_decrease = true;
                }
            }
        }
        assert!(
            found_decrease,
            "220 DU O3 column should reduce Chappuis extinction below default ~347 DU"
        );
    }

    #[test]
    fn build_full_with_gas_no2_override_changes_extinction() {
        // High NO2 (urban, ~5e17 molecules/m3) should increase extinction
        // at short wavelengths (400 nm, Huggins region) near the surface.
        let standard =
            build_full_with_gas(AtmosphereType::UsStandard, 0.15, None, None, None, None);
        let high_no2 = build_full_with_gas(
            AtmosphereType::UsStandard,
            0.15,
            None,
            None,
            None,
            Some(5.0e17),
        );
        // Find 400nm index
        let mut uv_idx = 0;
        for w in 0..standard.num_wavelengths {
            if (standard.wavelengths_nm[w] - 400.0).abs() < 15.0 {
                uv_idx = w;
                break;
            }
        }
        // Surface shell should show increased extinction
        let ext_std = standard.optics[0][uv_idx].extinction;
        let ext_high = high_no2.optics[0][uv_idx].extinction;
        assert!(
            ext_high > ext_std,
            "High NO2 should increase surface extinction at 400nm: std={:.6e} high={:.6e}",
            ext_std,
            ext_high,
        );
    }

    #[test]
    fn build_full_with_gas_preserves_scattering_coefficient() {
        // Gas absorption is purely absorptive: scattering coefficient must
        // be preserved regardless of O3/NO2 overrides.
        let clear = build_clear_sky(AtmosphereType::UsStandard, 0.15);
        let with_gas = build_full_with_gas(
            AtmosphereType::UsStandard,
            0.15,
            None,
            None,
            Some(400.0),
            Some(1.0e18),
        );
        for s in 0..clear.num_shells {
            for w in 0..clear.num_wavelengths {
                let scat_before = clear.optics[s][w].extinction * clear.optics[s][w].ssa;
                let scat_after = with_gas.optics[s][w].extinction * with_gas.optics[s][w].ssa;
                let rel_err = if scat_before > 1e-30 {
                    (scat_after - scat_before).abs() / scat_before
                } else {
                    0.0
                };
                assert!(
                    rel_err < 1e-10,
                    "Scattering coefficient changed at shell {} wl {}: before={:.6e} after={:.6e}",
                    s,
                    w,
                    scat_before,
                    scat_after,
                );
            }
        }
    }

    #[test]
    fn build_full_with_gas_ssa_bounded() {
        // SSA must stay in [0, 1] even with extreme overrides.
        let atm = build_full_with_gas(
            AtmosphereType::UsStandard,
            0.15,
            None,
            None,
            Some(600.0),
            Some(1.0e18),
        );
        for s in 0..atm.num_shells {
            for w in 0..atm.num_wavelengths {
                let ssa = atm.optics[s][w].ssa;
                assert!(
                    ssa >= 0.0 && ssa <= 1.0,
                    "SSA out of bounds at shell {} wl {}: {}",
                    s,
                    w,
                    ssa,
                );
            }
        }
    }
}
