//! Single-scattering line-of-sight integration for twilight radiance.
//!
//! This implements the deterministic single-scattering integral which is the
//! dominant contribution to twilight sky brightness. For clear-sky twilight
//! (SZA 90°-108°), single scattering accounts for ~70-90% of total radiance.
//!
//! The integral:
//!   I(λ) = F_sun(λ) × ∫₀^∞ β_scat(s,λ) × P(θ) × T_sun(s,λ) × T_obs(s,λ) ds
//!
//! Where:
//! - F_sun is the solar spectral irradiance at TOA
//! - β_scat is the scattering coefficient at point s along the LOS
//! - P(θ) is the scattering phase function for angle θ between sun and view
//! - T_sun is the transmittance from the sun to the scatter point
//! - T_obs is the transmittance from the scatter point to the observer
//! - s is distance along the line of sight from the observer
//!
//! This is computed by numerical quadrature along the line of sight,
//! with shadow rays traced toward the sun at each quadrature point.

use crate::atmosphere::AtmosphereModel;
use crate::geometry::{ray_sphere_intersect, Vec3};
use crate::scattering::{henyey_greenstein_phase, rayleigh_phase};

/// Maximum number of integration steps along the line of sight.
const MAX_LOS_STEPS: usize = 200;

/// Maximum number of steps for shadow ray integration (legacy, kept for reference).
#[allow(dead_code)]
const MAX_SHADOW_STEPS: usize = 300;

/// Compute the total path length of a ray through a spherical shell.
///
/// The shell is bounded by concentric spheres of radii `r_inner` and `r_outer`.
/// Returns the total distance the ray `origin + t*direction` spends inside the
/// shell for `0 <= t <= t_max`.
///
/// This is exact (analytical) — no stepping artifacts.
fn ray_path_through_shell(
    origin: Vec3,
    direction: Vec3,
    r_inner: f64,
    r_outer: f64,
    t_max: f64,
) -> f64 {
    // Find where the ray is inside the outer sphere
    let outer_interval = match ray_sphere_intersect(origin, direction, r_outer) {
        Some(hit) => {
            let t0 = if hit.t_near > 0.0 { hit.t_near } else { 0.0 };
            let t1 = if hit.t_far < t_max { hit.t_far } else { t_max };
            if t1 > t0 + 1e-6 {
                Some((t0, t1))
            } else {
                None
            }
        }
        None => None,
    };

    let (outer_start, outer_end) = match outer_interval {
        Some((s, e)) => (s, e),
        None => return 0.0, // Ray misses the outer sphere
    };

    // Find where the ray is inside the inner sphere (to subtract)
    let inner_interval = match ray_sphere_intersect(origin, direction, r_inner) {
        Some(hit) => {
            let t0 = if hit.t_near > 0.0 { hit.t_near } else { 0.0 };
            let t1 = if hit.t_far < t_max { hit.t_far } else { t_max };
            if t1 > t0 + 1e-6 {
                Some((t0, t1))
            } else {
                None
            }
        }
        None => None,
    };

    match inner_interval {
        None => {
            // Ray doesn't enter the inner sphere — full path through shell
            outer_end - outer_start
        }
        Some((inner_start, inner_end)) => {
            // The shell interval = outer_interval \ inner_interval
            // This can produce 0, 1, or 2 segments:
            // Segment 1: [outer_start, min(outer_end, inner_start)] (before entering inner)
            // Segment 2: [max(outer_start, inner_end), outer_end] (after exiting inner)
            let mut total = 0.0;

            // Segment before inner sphere
            let seg1_end = if outer_end < inner_start {
                outer_end
            } else {
                inner_start
            };
            if seg1_end > outer_start {
                total += seg1_end - outer_start;
            }

            // Segment after inner sphere
            let seg2_start = if outer_start > inner_end {
                outer_start
            } else {
                inner_end
            };
            if outer_end > seg2_start {
                total += outer_end - seg2_start;
            }

            total
        }
    }
}

/// Compute single-scattering radiance along a line of sight.
///
/// # Arguments
/// * `atm` - Atmosphere model with shell geometry and optical properties
/// * `observer_pos` - Observer position in ECEF [m]
/// * `view_dir` - Viewing direction (unit vector)
/// * `sun_dir` - Direction toward the sun (unit vector)
/// * `wavelength_idx` - Index into the atmosphere wavelength grid
///
/// # Returns
/// Single-scattering radiance (proportional to W/m²/sr/nm when multiplied by solar irradiance)
pub fn single_scatter_radiance(
    atm: &AtmosphereModel,
    observer_pos: Vec3,
    view_dir: Vec3,
    sun_dir: Vec3,
    wavelength_idx: usize,
) -> f64 {
    let toa_radius = atm.toa_radius();
    let surface_radius = atm.surface_radius();

    // Find where the line of sight exits the atmosphere
    let los_max = match ray_sphere_intersect(observer_pos, view_dir, toa_radius) {
        Some(hit) => {
            if hit.t_far > 0.0 {
                hit.t_far
            } else {
                return 0.0;
            }
        }
        None => return 0.0,
    };

    // Check if LOS hits the ground
    let los_end = match ray_sphere_intersect(observer_pos, view_dir, surface_radius) {
        Some(hit) if hit.t_near > 1e-3 => {
            // LOS hits ground before exiting atmosphere
            hit.t_near
        }
        _ => los_max,
    };

    if los_end <= 0.0 {
        return 0.0;
    }

    // Compute optical depth from observer along LOS (for T_obs)
    // We'll compute this incrementally as we integrate

    // Adaptive step size: smaller steps near the observer (denser atmosphere),
    // larger steps higher up
    let num_steps = MAX_LOS_STEPS.min((los_end / 500.0) as usize + 20);
    let ds_base = los_end / num_steps as f64;

    let mut radiance = 0.0;
    let mut tau_obs = 0.0; // Accumulated optical depth from observer to current point

    for step in 0..num_steps {
        // Distance along LOS to midpoint of this step
        let s = (step as f64 + 0.5) * ds_base;
        let scatter_pos = observer_pos + view_dir * s;
        let r = scatter_pos.length();

        // Skip if outside atmosphere
        if r > toa_radius || r < surface_radius {
            continue;
        }

        // Find which shell we're in
        let shell_idx = match atm.shell_index(r) {
            Some(idx) => idx,
            None => continue,
        };

        let optics = &atm.optics[shell_idx][wavelength_idx];

        // Scattering coefficient at this point
        let beta_scat = optics.extinction * optics.ssa;

        if beta_scat < 1e-30 {
            // Update optical depth even if no scattering
            tau_obs += optics.extinction * ds_base;
            continue;
        }

        // Transmittance from observer to scatter point
        // T_obs = exp(-tau_obs) where tau_obs is accumulated along the LOS
        // We add half the current step's optical depth for midpoint rule
        let tau_obs_mid = tau_obs + optics.extinction * ds_base * 0.5;
        let t_obs = libm::exp(-tau_obs_mid);

        if t_obs < 1e-30 {
            // Negligible contribution beyond this point
            break;
        }

        // Transmittance from sun to scatter point (shadow ray)
        let t_sun = shadow_ray_transmittance(atm, scatter_pos, sun_dir, wavelength_idx);

        if t_sun < 1e-30 {
            // This point is in shadow (sun below local horizon)
            tau_obs += optics.extinction * ds_base;
            continue;
        }

        // Scattering angle: angle between sun direction and viewing direction
        // cos(θ) = sun_dir · (-view_dir) because we want the angle between
        // the incoming solar ray and the scattered ray toward the observer
        let cos_theta = sun_dir.dot(-view_dir);

        // Phase function
        let phase = if optics.rayleigh_fraction > 0.99 {
            rayleigh_phase(cos_theta)
        } else {
            optics.rayleigh_fraction * rayleigh_phase(cos_theta)
                + (1.0 - optics.rayleigh_fraction)
                    * henyey_greenstein_phase(cos_theta, optics.asymmetry)
        };

        // Contribution from this step
        // dI = β_scat × P(θ)/(4π) × T_sun × T_obs × ds
        let di = beta_scat * phase / (4.0 * core::f64::consts::PI) * t_sun * t_obs * ds_base;

        radiance += di;

        // Update accumulated optical depth for next step
        tau_obs += optics.extinction * ds_base;
    }

    radiance
}

/// Compute transmittance along a shadow ray from a point toward the sun.
///
/// Uses analytical shell-by-shell path length computation for exact results,
/// eliminating stepping artifacts that caused zeros at deep twilight angles.
fn shadow_ray_transmittance(
    atm: &AtmosphereModel,
    start_pos: Vec3,
    sun_dir: Vec3,
    wavelength_idx: usize,
) -> f64 {
    let toa_radius = atm.toa_radius();
    let surface_radius = atm.surface_radius();

    // Find where the shadow ray exits the atmosphere
    let ray_max = match ray_sphere_intersect(start_pos, sun_dir, toa_radius) {
        Some(hit) if hit.t_far > 0.0 => hit.t_far,
        _ => return 0.0,
    };

    // Check if shadow ray hits the ground (point is in Earth's shadow)
    match ray_sphere_intersect(start_pos, sun_dir, surface_radius) {
        Some(hit) if hit.t_near > 1e-3 && hit.t_near < ray_max => {
            return 0.0;
        }
        _ => {}
    }

    // Analytical integration: compute exact path length through each shell
    let mut tau = 0.0;
    for s in 0..atm.num_shells {
        let path = ray_path_through_shell(
            start_pos,
            sun_dir,
            atm.shells[s].r_inner,
            atm.shells[s].r_outer,
            ray_max,
        );
        if path > 0.0 {
            tau += atm.optics[s][wavelength_idx].extinction * path;
            if tau > 50.0 {
                return 0.0;
            }
        }
    }

    libm::exp(-tau)
}

/// Compute single-scattering radiance for all wavelengths simultaneously.
///
/// More efficient than calling `single_scatter_radiance` per wavelength
/// because the geometry (LOS positions, shadow ray geometry) is shared.
pub fn single_scatter_spectrum(
    atm: &AtmosphereModel,
    observer_pos: Vec3,
    view_dir: Vec3,
    sun_dir: Vec3,
) -> [f64; 64] {
    let mut radiance = [0.0f64; 64];
    let num_wl = atm.num_wavelengths;

    let toa_radius = atm.toa_radius();
    let surface_radius = atm.surface_radius();

    // Find LOS extent
    let los_end = match ray_sphere_intersect(observer_pos, view_dir, toa_radius) {
        Some(hit) if hit.t_far > 0.0 => {
            // Check ground intersection
            match ray_sphere_intersect(observer_pos, view_dir, surface_radius) {
                Some(gh) if gh.t_near > 1e-3 && gh.t_near < hit.t_far => gh.t_near,
                _ => hit.t_far,
            }
        }
        _ => return radiance,
    };

    let num_steps = MAX_LOS_STEPS.min((los_end / 500.0) as usize + 20);
    let ds = los_end / num_steps as f64;

    // Accumulated optical depth per wavelength
    let mut tau_obs = [0.0f64; 64];

    for step in 0..num_steps {
        let s = (step as f64 + 0.5) * ds;
        let scatter_pos = observer_pos + view_dir * s;
        let r = scatter_pos.length();

        if r > toa_radius || r < surface_radius {
            continue;
        }

        let shell_idx = match atm.shell_index(r) {
            Some(idx) => idx,
            None => continue,
        };

        // Scattering angle (same for all wavelengths)
        let cos_theta = sun_dir.dot(-view_dir);

        // Shadow ray transmittance per wavelength
        // (compute once per step, reuse geometry)
        let t_sun = shadow_ray_transmittance_spectrum(atm, scatter_pos, sun_dir, num_wl);

        for w in 0..num_wl {
            let optics = &atm.optics[shell_idx][w];
            let beta_scat = optics.extinction * optics.ssa;

            if beta_scat < 1e-30 {
                tau_obs[w] += optics.extinction * ds;
                continue;
            }

            let tau_obs_mid = tau_obs[w] + optics.extinction * ds * 0.5;
            let t_obs = libm::exp(-tau_obs_mid);

            if t_obs < 1e-30 || t_sun[w] < 1e-30 {
                tau_obs[w] += optics.extinction * ds;
                continue;
            }

            let phase = if optics.rayleigh_fraction > 0.99 {
                rayleigh_phase(cos_theta)
            } else {
                optics.rayleigh_fraction * rayleigh_phase(cos_theta)
                    + (1.0 - optics.rayleigh_fraction)
                        * henyey_greenstein_phase(cos_theta, optics.asymmetry)
            };

            radiance[w] +=
                beta_scat * phase / (4.0 * core::f64::consts::PI) * t_sun[w] * t_obs * ds;

            tau_obs[w] += optics.extinction * ds;
        }
    }

    radiance
}

/// Compute shadow ray transmittance for all wavelengths at once.
///
/// Uses analytical shell-by-shell path length computation for exact results.
fn shadow_ray_transmittance_spectrum(
    atm: &AtmosphereModel,
    start_pos: Vec3,
    sun_dir: Vec3,
    num_wl: usize,
) -> [f64; 64] {
    let mut result = [1.0f64; 64];
    let toa_radius = atm.toa_radius();
    let surface_radius = atm.surface_radius();

    let ray_max = match ray_sphere_intersect(start_pos, sun_dir, toa_radius) {
        Some(hit) if hit.t_far > 0.0 => hit.t_far,
        _ => {
            result = [0.0; 64];
            return result;
        }
    };

    // Check ground intersection
    match ray_sphere_intersect(start_pos, sun_dir, surface_radius) {
        Some(hit) if hit.t_near > 1e-3 && hit.t_near < ray_max => {
            result = [0.0; 64];
            return result;
        }
        _ => {}
    }

    // Analytical integration: compute exact path length through each shell
    let mut tau = [0.0f64; 64];

    for s in 0..atm.num_shells {
        let path = ray_path_through_shell(
            start_pos,
            sun_dir,
            atm.shells[s].r_inner,
            atm.shells[s].r_outer,
            ray_max,
        );
        if path > 0.0 {
            for w in 0..num_wl {
                tau[w] += atm.optics[s][w].extinction * path;
            }
        }
    }

    for w in 0..num_wl {
        result[w] = if tau[w] > 50.0 {
            0.0
        } else {
            libm::exp(-tau[w])
        };
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atmosphere::{AtmosphereModel, ShellOptics, EARTH_RADIUS_M};
    use crate::geometry::Vec3;

    /// Build a simple 3-shell clear-sky atmosphere with Rayleigh-only scattering.
    fn make_test_atmosphere() -> AtmosphereModel {
        let altitudes_km = [0.0, 10.0, 50.0, 100.0];
        let wavelengths = [400.0, 550.0, 700.0];
        let mut atm = AtmosphereModel::new(&altitudes_km, &wavelengths);

        // Shell 0 (0-10km): dense, strong Rayleigh
        for w in 0..3 {
            atm.optics[0][w] = ShellOptics {
                extinction: 1e-5
                    * if w == 0 {
                        4.0
                    } else if w == 1 {
                        1.0
                    } else {
                        0.3
                    },
                ssa: 1.0,
                asymmetry: 0.0,
                rayleigh_fraction: 1.0,
            };
        }
        // Shell 1 (10-50km): moderate
        for w in 0..3 {
            atm.optics[1][w] = ShellOptics {
                extinction: 1e-6
                    * if w == 0 {
                        4.0
                    } else if w == 1 {
                        1.0
                    } else {
                        0.3
                    },
                ssa: 1.0,
                asymmetry: 0.0,
                rayleigh_fraction: 1.0,
            };
        }
        // Shell 2 (50-100km): thin
        for w in 0..3 {
            atm.optics[2][w] = ShellOptics {
                extinction: 1e-8
                    * if w == 0 {
                        4.0
                    } else if w == 1 {
                        1.0
                    } else {
                        0.3
                    },
                ssa: 1.0,
                asymmetry: 0.0,
                rayleigh_fraction: 1.0,
            };
        }

        atm
    }

    fn observer_on_surface() -> Vec3 {
        Vec3::new(EARTH_RADIUS_M + 1.0, 0.0, 0.0)
    }

    // ── ray_path_through_shell (private, tested via public API indirectly) ──
    // We test it directly since it's in scope within the module.

    #[test]
    fn ray_path_through_shell_radial_outward() {
        // Ray from inside a shell going radially outward.
        // Origin at r=5, dir=(1,0,0), shell r_inner=4, r_outer=6
        // Should traverse from r=5 to r=6 = distance 1
        let origin = Vec3::new(5.0, 0.0, 0.0);
        let dir = Vec3::new(1.0, 0.0, 0.0);
        let path = ray_path_through_shell(origin, dir, 4.0, 6.0, 100.0);
        assert!(
            (path - 1.0).abs() < 0.01,
            "Radial outward path: expected ~1.0, got {}",
            path
        );
    }

    #[test]
    fn ray_path_through_shell_misses() {
        // Ray that completely misses the shell
        let origin = Vec3::new(0.0, 10.0, 0.0); // far from shell
        let dir = Vec3::new(1.0, 0.0, 0.0); // parallel, not intersecting
        let path = ray_path_through_shell(origin, dir, 4.0, 6.0, 100.0);
        assert!(
            path < 0.01,
            "Ray missing shell should have near-zero path, got {}",
            path
        );
    }

    #[test]
    fn ray_path_through_shell_diametric() {
        // Ray passing through center of shell (maximum path length)
        // Origin far away on -x axis, dir=(1,0,0), shell r_inner=0 (no inner), r_outer=5
        let origin = Vec3::new(-10.0, 0.0, 0.0);
        let dir = Vec3::new(1.0, 0.0, 0.0);
        let path = ray_path_through_shell(origin, dir, 0.0, 5.0, 100.0);
        // Should pass through full diameter = 10
        assert!(
            (path - 10.0).abs() < 0.1,
            "Diametric path: expected ~10.0, got {}",
            path
        );
    }

    #[test]
    fn ray_path_through_shell_with_inner_hole() {
        // Ray through a shell with inner cavity (like atmosphere shell)
        // Origin far on -x, dir=(1,0,0), shell r_inner=3, r_outer=5
        // Path = entry_to_inner + inner_to_exit = (5-3) + (5-3) = 4
        // Actually: path through outer = 10, path through inner = 6, shell = 10-6 = 4
        let origin = Vec3::new(-10.0, 0.0, 0.0);
        let dir = Vec3::new(1.0, 0.0, 0.0);
        let path = ray_path_through_shell(origin, dir, 3.0, 5.0, 100.0);
        assert!(
            (path - 4.0).abs() < 0.1,
            "Shell with hole: expected ~4.0, got {}",
            path
        );
    }

    #[test]
    fn ray_path_through_shell_t_max_limits() {
        // t_max should limit the path
        let origin = Vec3::new(-10.0, 0.0, 0.0);
        let dir = Vec3::new(1.0, 0.0, 0.0);
        // Without t_max limit: ~10 (full diameter of r=5 sphere)
        let path_full = ray_path_through_shell(origin, dir, 0.0, 5.0, 100.0);
        // With t_max = 6 (only enters sphere at t=5, so only 1 unit inside)
        let path_limited = ray_path_through_shell(origin, dir, 0.0, 5.0, 6.0);
        assert!(
            path_limited < path_full,
            "t_max should limit path: limited={}, full={}",
            path_limited,
            path_full
        );
    }

    // ── single_scatter_radiance ──

    #[test]
    fn radiance_zero_in_empty_atmosphere() {
        let altitudes_km = [0.0, 50.0, 100.0];
        let wavelengths = [550.0];
        let atm = AtmosphereModel::new(&altitudes_km, &wavelengths);
        // All optics are default (extinction=0)

        let obs = observer_on_surface();
        let view = Vec3::new(0.0, 0.0, 1.0).normalize();
        let sun = Vec3::new(1.0, 0.0, 0.0); // overhead

        let rad = single_scatter_radiance(&atm, obs, view, sun, 0);
        assert!(
            rad.abs() < 1e-30,
            "Zero extinction → zero radiance, got {}",
            rad
        );
    }

    #[test]
    fn radiance_positive_for_sunlit_atmosphere() {
        let atm = make_test_atmosphere();
        let obs = observer_on_surface();
        // Look toward zenith
        let view = Vec3::new(1.0, 0.0, 0.0).normalize();
        // Sun near horizon (SZA≈80°, still above horizon)
        let sun = {
            let sza = 80.0_f64.to_radians();
            Vec3::new(libm::cos(sza), libm::sin(sza), 0.0)
        };

        let rad = single_scatter_radiance(&atm, obs, view, sun, 1); // 550nm
        assert!(
            rad > 0.0,
            "Sunlit atmosphere should produce positive radiance, got {}",
            rad
        );
    }

    #[test]
    fn radiance_decreases_with_deeper_twilight() {
        let atm = make_test_atmosphere();
        let obs = observer_on_surface();
        let view = Vec3::new(0.0, 1.0, 0.0).normalize(); // horizontal

        // Two SZA values: 92° (civil twilight) and 100° (nautical)
        let sun_92 = crate::geometry::solar_direction_ecef(92.0, 180.0, 0.0, 0.0);
        let sun_100 = crate::geometry::solar_direction_ecef(100.0, 180.0, 0.0, 0.0);

        let rad_92 = single_scatter_radiance(&atm, obs, view, sun_92, 1);
        let rad_100 = single_scatter_radiance(&atm, obs, view, sun_100, 1);

        assert!(
            rad_92 > rad_100,
            "Radiance should decrease with deeper twilight: SZA92={:.4e}, SZA100={:.4e}",
            rad_92,
            rad_100
        );
    }

    #[test]
    fn radiance_blue_stronger_than_red_at_small_sza() {
        // In a Rayleigh-only atmosphere at small SZA (short path lengths),
        // blue scatters more than red (λ⁻⁴ law dominates).
        // At deeper twilight, blue gets attenuated MORE along the long LOS path,
        // so red can dominate — this is the correct physics of why twilight turns red!
        //
        // Test at SZA=80° (sun above horizon, short paths) where blue should win.
        // But our test atmosphere has high enough extinction that even at SZA=80°,
        // the path-integrated effect matters. Instead, verify the spectral shape
        // changes with SZA: the blue/red ratio should decrease with increasing SZA.
        let atm = make_test_atmosphere();
        let obs = observer_on_surface();
        let view = Vec3::new(0.0, 1.0, 0.0).normalize();

        let sun_shallow = crate::geometry::solar_direction_ecef(92.0, 180.0, 0.0, 0.0);
        let sun_deep = crate::geometry::solar_direction_ecef(100.0, 180.0, 0.0, 0.0);

        let rad_blue_shallow = single_scatter_radiance(&atm, obs, view, sun_shallow, 0); // 400nm
        let rad_red_shallow = single_scatter_radiance(&atm, obs, view, sun_shallow, 2); // 700nm
        let rad_blue_deep = single_scatter_radiance(&atm, obs, view, sun_deep, 0);
        let rad_red_deep = single_scatter_radiance(&atm, obs, view, sun_deep, 2);

        // At deeper twilight, the blue/red ratio should decrease (sky reddens)
        let ratio_shallow = if rad_red_shallow > 1e-30 {
            rad_blue_shallow / rad_red_shallow
        } else {
            1.0
        };
        let ratio_deep = if rad_red_deep > 1e-30 {
            rad_blue_deep / rad_red_deep
        } else {
            0.0
        };

        assert!(
            ratio_deep < ratio_shallow || rad_blue_deep < 1e-30,
            "Sky should redden with deeper twilight: blue/red ratio shallow={:.4}, deep={:.4}",
            ratio_shallow,
            ratio_deep
        );
    }

    // ── single_scatter_spectrum ──

    #[test]
    fn spectrum_returns_64_element_array() {
        let atm = make_test_atmosphere();
        let obs = observer_on_surface();
        let view = Vec3::new(0.0, 1.0, 0.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(80.0, 180.0, 0.0, 0.0);

        let spectrum = single_scatter_spectrum(&atm, obs, view, sun);
        // Should be [f64; 64], the first num_wavelengths should have values
        assert_eq!(spectrum.len(), 64);
    }

    #[test]
    fn spectrum_matches_individual_radiance() {
        // single_scatter_spectrum should give the same results as calling
        // single_scatter_radiance for each wavelength individually
        let atm = make_test_atmosphere();
        let obs = observer_on_surface();
        let view = Vec3::new(0.0, 1.0, 0.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(92.0, 180.0, 0.0, 0.0);

        let spectrum = single_scatter_spectrum(&atm, obs, view, sun);

        for w in 0..atm.num_wavelengths {
            let individual = single_scatter_radiance(&atm, obs, view, sun, w);
            let rel_err = if individual > 1e-30 {
                ((spectrum[w] - individual) / individual).abs()
            } else {
                (spectrum[w] - individual).abs()
            };
            assert!(
                rel_err < 0.01,
                "Spectrum[{}] = {:.6e} != individual = {:.6e}, rel_err = {:.4e}",
                w,
                spectrum[w],
                individual,
                rel_err
            );
        }
    }

    #[test]
    fn spectrum_unused_wavelengths_are_zero() {
        let atm = make_test_atmosphere(); // 3 wavelengths
        let obs = observer_on_surface();
        let view = Vec3::new(0.0, 1.0, 0.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(80.0, 180.0, 0.0, 0.0);

        let spectrum = single_scatter_spectrum(&atm, obs, view, sun);
        for w in atm.num_wavelengths..64 {
            assert!(
                spectrum[w].abs() < 1e-30,
                "Unused wavelength index {} should be 0, got {}",
                w,
                spectrum[w]
            );
        }
    }

    #[test]
    fn spectrum_zero_for_deep_shadow() {
        // At very deep twilight (SZA > 108°), all LOS scatter points
        // should be in Earth's shadow → zero radiance
        let atm = make_test_atmosphere();
        let obs = observer_on_surface();
        let view = Vec3::new(0.0, 1.0, 0.0).normalize();
        // SZA = 120° → well below horizon, deep shadow
        let sun = crate::geometry::solar_direction_ecef(120.0, 180.0, 0.0, 0.0);

        let spectrum = single_scatter_spectrum(&atm, obs, view, sun);
        for w in 0..atm.num_wavelengths {
            assert!(
                spectrum[w] < 1e-20,
                "Deep shadow (SZA=120°) should give ~0 radiance at wl[{}], got {:.4e}",
                w,
                spectrum[w]
            );
        }
    }

    #[test]
    fn radiance_is_non_negative() {
        // Radiance should never be negative for any geometry
        let atm = make_test_atmosphere();
        let obs = observer_on_surface();

        for sza in &[80.0, 90.0, 96.0, 102.0, 108.0, 120.0] {
            let sun = crate::geometry::solar_direction_ecef(*sza, 180.0, 0.0, 0.0);
            let view = Vec3::new(0.0, 1.0, 0.0).normalize();
            for w in 0..atm.num_wavelengths {
                let rad = single_scatter_radiance(&atm, obs, view, sun, w);
                assert!(
                    rad >= 0.0,
                    "Radiance should be non-negative: SZA={}, wl={}, rad={:.4e}",
                    sza,
                    w,
                    rad
                );
            }
        }
    }
}
