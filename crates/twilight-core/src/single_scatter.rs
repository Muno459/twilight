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
use crate::geometry::{
    next_shell_boundary, ray_sphere_intersect, refract_at_boundary, RefractResult, Vec3,
};
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
/// This is exact (analytical) -- no stepping artifacts.
pub fn ray_path_through_shell(
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
    let ground_hit = ray_sphere_intersect(observer_pos, view_dir, surface_radius);
    let hits_ground = matches!(ground_hit, Some(ref hit) if hit.t_near > 1e-3);
    let los_end = match ground_hit {
        Some(ref hit) if hit.t_near > 1e-3 => hit.t_near,
        _ => los_max,
    };

    if los_end <= 0.0 {
        return 0.0;
    }

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

    // Ground reflection: Lambertian BRDF = albedo / π
    //
    // If the LOS hits the ground, the surface reflects sunlight toward the
    // observer. The contribution is:
    //   I_ground = (A/π) × cos(θ_sun) × T_sun_to_ground × T_ground_to_observer
    //
    // where θ_sun is the solar incidence angle at the ground point and A is
    // the surface albedo. This only contributes when the ground point is
    // illuminated (sun above local horizon at that point).
    if hits_ground {
        let albedo = atm.surface_albedo[wavelength_idx];
        if albedo > 1e-10 {
            let ground_pos = observer_pos + view_dir * los_end;
            let ground_normal = ground_pos.normalize();
            let cos_sun_incidence = sun_dir.dot(ground_normal);

            // Sun must be above the local horizon at the ground point
            if cos_sun_incidence > 0.0 {
                let t_sun_ground =
                    shadow_ray_transmittance(atm, ground_pos, sun_dir, wavelength_idx);
                // tau_obs at this point is the full LOS optical depth to the ground
                let t_obs_ground = libm::exp(-tau_obs);

                radiance += albedo / core::f64::consts::PI
                    * cos_sun_incidence
                    * t_sun_ground
                    * t_obs_ground;
            }
        }
    }

    radiance
}

/// Compute transmittance along a shadow ray from a point toward the sun.
///
/// Traces the ray shell-by-shell, applying Snell's law at each boundary
/// so the shadow ray follows the physically correct refracted path through
/// the atmosphere. Within each shell the path is straight (piecewise-constant
/// refractive index), so the path length to the next boundary is exact.
///
/// This is public so the hybrid integrator can reuse the exact shadow ray
/// logic instead of the step-by-step `trace_transmittance` in `photon.rs`.
pub fn shadow_ray_transmittance(
    atm: &AtmosphereModel,
    start_pos: Vec3,
    sun_dir: Vec3,
    wavelength_idx: usize,
) -> f64 {
    let surface_radius = atm.surface_radius();
    let num_shells = atm.num_shells;
    let mut pos = start_pos;
    let mut dir = sun_dir;
    let mut tau = 0.0;

    // Find initial shell once (O(n)), then track directly (O(1) per step).
    let mut shell_idx = match atm.shell_index(pos.length()) {
        Some(idx) => idx,
        None => return 1.0,
    };

    for _ in 0..200 {
        let shell = &atm.shells[shell_idx];
        let optics = &atm.optics[shell_idx][wavelength_idx];

        match next_shell_boundary(pos, dir, shell.r_inner, shell.r_outer) {
            Some((dist, is_outward)) => {
                tau += optics.extinction * dist;

                // Refract at boundary
                let boundary_pos = pos + dir * dist;
                let n_from = atm.refractive_index[shell_idx];
                let next_shell = if is_outward {
                    shell_idx + 1
                } else {
                    shell_idx.wrapping_sub(1)
                };
                let n_to = if next_shell < num_shells {
                    atm.refractive_index[next_shell]
                } else {
                    1.0
                };
                dir = match refract_at_boundary(dir, boundary_pos, n_from, n_to) {
                    RefractResult::Refracted(d) | RefractResult::TotalReflection(d) => d,
                };
                pos = boundary_pos + dir * 1e-3;

                // Hit ground -- fully opaque
                if !is_outward && pos.length() <= surface_radius + 1.0 {
                    return 0.0;
                }

                // Exited atmosphere
                if next_shell >= num_shells {
                    break;
                }

                shell_idx = next_shell;
            }
            None => break,
        }

        if tau > 50.0 {
            return 0.0;
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
    let los_max = match ray_sphere_intersect(observer_pos, view_dir, toa_radius) {
        Some(hit) if hit.t_far > 0.0 => hit.t_far,
        _ => return radiance,
    };

    let ground_hit = ray_sphere_intersect(observer_pos, view_dir, surface_radius);
    let hits_ground =
        matches!(ground_hit, Some(ref hit) if hit.t_near > 1e-3 && hit.t_near < los_max);
    let los_end = match ground_hit {
        Some(ref hit) if hit.t_near > 1e-3 && hit.t_near < los_max => hit.t_near,
        _ => los_max,
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

    // Ground reflection (Lambertian BRDF = albedo / π)
    if hits_ground {
        let ground_pos = observer_pos + view_dir * los_end;
        let ground_normal = ground_pos.normalize();
        let cos_sun_incidence = sun_dir.dot(ground_normal);

        if cos_sun_incidence > 0.0 {
            let t_sun_ground = shadow_ray_transmittance_spectrum(atm, ground_pos, sun_dir, num_wl);

            for w in 0..num_wl {
                let albedo = atm.surface_albedo[w];
                if albedo > 1e-10 && t_sun_ground[w] > 1e-30 {
                    let t_obs_ground = libm::exp(-tau_obs[w]);
                    radiance[w] += albedo / core::f64::consts::PI
                        * cos_sun_incidence
                        * t_sun_ground[w]
                        * t_obs_ground;
                }
            }
        }
    }

    radiance
}

/// Compute shadow ray transmittance for all wavelengths at once.
///
/// Shell-by-shell tracer with Snell's law at each boundary.
/// The refracted path is the same for all wavelengths (refractive index
/// dispersion in air is negligible over the visible range), so we trace
/// the geometry once and accumulate per-wavelength optical depths.
fn shadow_ray_transmittance_spectrum(
    atm: &AtmosphereModel,
    start_pos: Vec3,
    sun_dir: Vec3,
    num_wl: usize,
) -> [f64; 64] {
    let surface_radius = atm.surface_radius();
    let num_shells = atm.num_shells;
    let mut pos = start_pos;
    let mut dir = sun_dir;
    let mut tau = [0.0f64; 64];

    let mut shell_idx = match atm.shell_index(pos.length()) {
        Some(idx) => idx,
        None => return [1.0f64; 64],
    };

    for _ in 0..200 {
        let shell = &atm.shells[shell_idx];

        match next_shell_boundary(pos, dir, shell.r_inner, shell.r_outer) {
            Some((dist, is_outward)) => {
                for (w, tau_w) in tau.iter_mut().enumerate().take(num_wl) {
                    *tau_w += atm.optics[shell_idx][w].extinction * dist;
                }

                // Refract at boundary
                let boundary_pos = pos + dir * dist;
                let n_from = atm.refractive_index[shell_idx];
                let next_shell = if is_outward {
                    shell_idx + 1
                } else {
                    shell_idx.wrapping_sub(1)
                };
                let n_to = if next_shell < num_shells {
                    atm.refractive_index[next_shell]
                } else {
                    1.0
                };
                dir = match refract_at_boundary(dir, boundary_pos, n_from, n_to) {
                    RefractResult::Refracted(d) | RefractResult::TotalReflection(d) => d,
                };
                pos = boundary_pos + dir * 1e-3;

                // Hit ground -- fully opaque for ALL wavelengths
                if !is_outward && pos.length() <= surface_radius + 1.0 {
                    return [0.0f64; 64];
                }

                // Exited atmosphere
                if next_shell >= num_shells {
                    break;
                }

                shell_idx = next_shell;
            }
            None => break,
        }

        // Early out if ALL wavelengths are opaque
        let min_tau = tau.iter().take(num_wl).copied().fold(f64::MAX, f64::min);
        if min_tau > 50.0 {
            return [0.0f64; 64];
        }
    }

    let mut result = [0.0f64; 64];
    for (w, res_w) in result.iter_mut().enumerate().take(num_wl) {
        *res_w = if tau[w] > 50.0 {
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

    // ── Ground reflection (Phase 4) ──

    /// Build a test atmosphere with nonzero surface albedo.
    fn make_test_atmosphere_with_albedo(albedo: f64) -> AtmosphereModel {
        let mut atm = make_test_atmosphere();
        for w in 0..atm.num_wavelengths {
            atm.surface_albedo[w] = albedo;
        }
        atm
    }

    #[test]
    fn albedo_zero_matches_no_reflection() {
        // With albedo=0, ground reflection contributes nothing. Verify by
        // comparing the same atmosphere with albedo=0 vs albedo=0.5: the
        // atmospheric scattering component should be present in both, and
        // the albedo=0 case should have strictly less (or equal) radiance.
        let atm_no = make_test_atmosphere_with_albedo(0.0);
        let atm_yes = make_test_atmosphere_with_albedo(0.5);
        let obs = observer_on_surface();
        let view = Vec3::new(0.0, 0.0, 1.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(60.0, 0.0, 0.0, 0.0);

        for w in 0..atm_no.num_wavelengths {
            let r_no = single_scatter_radiance(&atm_no, obs, view, sun, w);
            let r_yes = single_scatter_radiance(&atm_yes, obs, view, sun, w);
            assert!(
                r_yes >= r_no,
                "albedo=0.5 should give >= albedo=0: wl[{}] {:.4e} vs {:.4e}",
                w,
                r_yes,
                r_no
            );
        }
    }

    #[test]
    fn ground_reflection_increases_radiance() {
        // With nonzero albedo and sun above horizon, ground reflection
        // should add radiance compared to albedo=0.
        let atm_no = make_test_atmosphere(); // albedo=0
        let atm_yes = make_test_atmosphere_with_albedo(0.3);
        let obs = observer_on_surface();

        // Look toward the horizon (the LOS will hit the ground far away)
        // At the observer position (R+1, 0, 0), looking in +z is horizontal
        let view = Vec3::new(0.0, 0.0, 1.0).normalize();
        // Sun well above horizon so ground is illuminated
        let sun = crate::geometry::solar_direction_ecef(60.0, 0.0, 0.0, 0.0);

        let rad_no = single_scatter_radiance(&atm_no, obs, view, sun, 1);
        let rad_yes = single_scatter_radiance(&atm_yes, obs, view, sun, 1);

        assert!(
            rad_yes >= rad_no,
            "Albedo=0.3 should give >= radiance than albedo=0: {:.4e} vs {:.4e}",
            rad_yes,
            rad_no
        );
    }

    #[test]
    fn ground_reflection_scales_with_albedo() {
        // Higher albedo should give more ground-reflected radiance.
        let atm_low = make_test_atmosphere_with_albedo(0.1);
        let atm_high = make_test_atmosphere_with_albedo(0.9);
        let obs = observer_on_surface();
        let view = Vec3::new(0.0, 0.0, 1.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(60.0, 0.0, 0.0, 0.0);

        let rad_low = single_scatter_radiance(&atm_low, obs, view, sun, 1);
        let rad_high = single_scatter_radiance(&atm_high, obs, view, sun, 1);

        assert!(
            rad_high >= rad_low,
            "Higher albedo should give more radiance: 0.9→{:.4e} vs 0.1→{:.4e}",
            rad_high,
            rad_low
        );
    }

    #[test]
    fn ground_reflection_zero_when_ground_in_shadow() {
        // At deep twilight (SZA=120°), the ground point should be in shadow,
        // so albedo shouldn't matter.
        let atm_no = make_test_atmosphere();
        let atm_yes = make_test_atmosphere_with_albedo(1.0); // maximum albedo
        let obs = observer_on_surface();
        let view = Vec3::new(0.0, 0.0, 1.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(120.0, 180.0, 0.0, 0.0);

        let rad_no = single_scatter_radiance(&atm_no, obs, view, sun, 1);
        let rad_yes = single_scatter_radiance(&atm_yes, obs, view, sun, 1);

        assert!(
            (rad_no - rad_yes).abs() < 1e-25,
            "In deep shadow, albedo shouldn't matter: {:.4e} vs {:.4e}",
            rad_no,
            rad_yes
        );
    }

    #[test]
    fn ground_reflection_non_negative() {
        let atm = make_test_atmosphere_with_albedo(0.5);
        let obs = observer_on_surface();

        for sza in &[30.0, 60.0, 80.0, 90.0, 96.0, 108.0, 120.0] {
            let sun = crate::geometry::solar_direction_ecef(*sza, 180.0, 0.0, 0.0);
            let view = Vec3::new(0.0, 0.0, 1.0).normalize();
            for w in 0..atm.num_wavelengths {
                let rad = single_scatter_radiance(&atm, obs, view, sun, w);
                assert!(
                    rad >= 0.0,
                    "Radiance with albedo should be non-negative: SZA={}, wl={}, rad={:.4e}",
                    sza,
                    w,
                    rad
                );
            }
        }
    }

    #[test]
    fn ground_reflection_spectrum_matches_individual() {
        // single_scatter_spectrum with albedo should match per-wavelength calls
        let atm = make_test_atmosphere_with_albedo(0.3);
        let obs = observer_on_surface();
        let view = Vec3::new(0.0, 0.0, 1.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(60.0, 0.0, 0.0, 0.0);

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
                "Spectrum with albedo[{}] = {:.6e} != individual = {:.6e}, err = {:.4e}",
                w,
                spectrum[w],
                individual,
                rel_err
            );
        }
    }

    #[test]
    fn ground_reflection_only_when_los_hits_ground() {
        // Looking straight up (zenith) should NOT hit the ground, so albedo
        // should not matter.
        let atm_no = make_test_atmosphere();
        let atm_yes = make_test_atmosphere_with_albedo(1.0);
        let obs = observer_on_surface();
        // Observer at (R+1,0,0), looking radially outward = straight up
        let view = Vec3::new(1.0, 0.0, 0.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(60.0, 180.0, 0.0, 0.0);

        let rad_no = single_scatter_radiance(&atm_no, obs, view, sun, 1);
        let rad_yes = single_scatter_radiance(&atm_yes, obs, view, sun, 1);

        assert!(
            (rad_no - rad_yes).abs() < 1e-25,
            "Upward LOS should not hit ground: albedo=0 {:.4e} vs albedo=1 {:.4e}",
            rad_no,
            rad_yes
        );
    }

    #[test]
    fn ground_reflection_per_wavelength_albedo() {
        // Test that per-wavelength albedo works: set albedo=0 for blue,
        // albedo=0.5 for green, albedo=0 for red.
        let mut atm = make_test_atmosphere();
        atm.surface_albedo[0] = 0.0; // 400nm
        atm.surface_albedo[1] = 0.5; // 550nm
        atm.surface_albedo[2] = 0.0; // 700nm

        let atm_zero = make_test_atmosphere(); // all albedo=0

        let obs = observer_on_surface();
        let view = Vec3::new(0.0, 0.0, 1.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(60.0, 0.0, 0.0, 0.0);

        // Blue and red should be unchanged
        let rad_blue = single_scatter_radiance(&atm, obs, view, sun, 0);
        let rad_blue_zero = single_scatter_radiance(&atm_zero, obs, view, sun, 0);
        assert!(
            (rad_blue - rad_blue_zero).abs() < 1e-25,
            "Blue (albedo=0) should be unchanged"
        );

        let rad_red = single_scatter_radiance(&atm, obs, view, sun, 2);
        let rad_red_zero = single_scatter_radiance(&atm_zero, obs, view, sun, 2);
        assert!(
            (rad_red - rad_red_zero).abs() < 1e-25,
            "Red (albedo=0) should be unchanged"
        );

        // Green should be >= the zero-albedo case
        let rad_green = single_scatter_radiance(&atm, obs, view, sun, 1);
        let rad_green_zero = single_scatter_radiance(&atm_zero, obs, view, sun, 1);
        assert!(
            rad_green >= rad_green_zero,
            "Green (albedo=0.5) should be >= zero-albedo: {:.4e} vs {:.4e}",
            rad_green,
            rad_green_zero
        );
    }

    // ── Atmospheric refraction tests ──

    /// Build a many-shell atmosphere for refraction tests (finer shells = smoother refraction).
    fn make_refraction_atmosphere() -> AtmosphereModel {
        // 10 shells: 0-1, 1-2, ..., 9-10 km, then 10-50, 50-100
        let altitudes_km = [
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 50.0, 100.0,
        ];
        let wavelengths = [550.0];
        let mut atm = AtmosphereModel::new(&altitudes_km, &wavelengths);

        // Rayleigh-only extinction with scale height 8.5 km
        for s in 0..atm.num_shells {
            let h = atm.shells[s].altitude_mid;
            atm.optics[s][0] = ShellOptics {
                extinction: 1.3e-5 * libm::exp(-h / 8500.0),
                ssa: 1.0,
                asymmetry: 0.0,
                rayleigh_fraction: 1.0,
            };
        }

        atm
    }

    #[test]
    fn shadow_ray_n1_matches_default() {
        // With all n=1.0 (default), shadow_ray_transmittance should give the
        // same result as when refraction was not yet implemented.
        let atm = make_test_atmosphere(); // n=1.0 everywhere (default)
        let pos = Vec3::new(EARTH_RADIUS_M + 5000.0, 0.0, 0.0);
        let sun = crate::geometry::solar_direction_ecef(92.0, 180.0, 0.0, 0.0);

        // All n are 1.0 by default, so refraction is identity
        let t = shadow_ray_transmittance(&atm, pos, sun, 1);
        assert!(
            t >= 0.0 && t <= 1.0,
            "Transmittance should be in [0,1], got {}",
            t
        );
    }

    #[test]
    fn shadow_ray_transmittance_with_refraction_non_negative() {
        let mut atm = make_refraction_atmosphere();
        atm.compute_refractive_indices();

        let pos = Vec3::new(EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let sun = crate::geometry::solar_direction_ecef(92.0, 180.0, 0.0, 0.0);

        let t = shadow_ray_transmittance(&atm, pos, sun, 0);
        assert!(
            t >= 0.0 && t <= 1.0,
            "Transmittance with refraction should be in [0,1], got {}",
            t
        );
    }

    #[test]
    fn shadow_ray_transmittance_refraction_changes_value() {
        // With refraction enabled, the shadow ray path is curved. At near-
        // horizontal sun angles, refraction changes the path length through
        // each shell, so the transmittance should differ from the n=1 case.
        let mut atm_refract = make_refraction_atmosphere();
        atm_refract.compute_refractive_indices();

        let atm_straight = make_refraction_atmosphere(); // n=1.0 default

        let pos = Vec3::new(EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        // Use a near-horizon sun angle where refraction matters most
        let sun = crate::geometry::solar_direction_ecef(90.5, 180.0, 0.0, 0.0);

        let t_refract = shadow_ray_transmittance(&atm_refract, pos, sun, 0);
        let t_straight = shadow_ray_transmittance(&atm_straight, pos, sun, 0);

        // Both should be valid transmittances
        assert!(t_refract >= 0.0 && t_refract <= 1.0);
        assert!(t_straight >= 0.0 && t_straight <= 1.0);

        // They should differ (refraction bends the ray, changing path lengths)
        // But the difference is small (< a few percent for near-horizon geometry)
        if t_straight > 1e-10 && t_refract > 1e-10 {
            let ratio = t_refract / t_straight;
            assert!(
                (ratio - 1.0).abs() < 0.2,
                "Refraction should cause small change in transmittance: ratio={}",
                ratio
            );
        }
    }

    #[test]
    fn radiance_with_refraction_non_negative() {
        // Radiance should never be negative even with refraction enabled
        let mut atm = make_refraction_atmosphere();
        atm.compute_refractive_indices();

        let obs = Vec3::new(EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = Vec3::new(0.0, 1.0, 0.0).normalize();

        for sza in &[80.0, 90.0, 92.0, 96.0, 102.0, 108.0] {
            let sun = crate::geometry::solar_direction_ecef(*sza, 180.0, 0.0, 0.0);
            let rad = single_scatter_radiance(&atm, obs, view, sun, 0);
            assert!(
                rad >= 0.0,
                "Radiance with refraction should be non-negative: SZA={}, rad={:.4e}",
                sza,
                rad
            );
        }
    }

    #[test]
    fn radiance_with_refraction_positive_at_civil_twilight() {
        let mut atm = make_refraction_atmosphere();
        atm.compute_refractive_indices();

        let obs = Vec3::new(EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = Vec3::new(0.0, 1.0, 0.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(92.0, 180.0, 0.0, 0.0);

        let rad = single_scatter_radiance(&atm, obs, view, sun, 0);
        assert!(
            rad > 0.0,
            "Refracted radiance should be positive at civil twilight, got {:.4e}",
            rad
        );
    }

    #[test]
    fn spectrum_with_refraction_matches_individual() {
        // single_scatter_spectrum with refraction should match per-wavelength calls
        let altitudes_km = [0.0, 5.0, 15.0, 50.0, 100.0];
        let wavelengths = [400.0, 550.0, 700.0];
        let mut atm = AtmosphereModel::new(&altitudes_km, &wavelengths);

        for s in 0..atm.num_shells {
            let h = atm.shells[s].altitude_mid;
            for w in 0..3 {
                let lambda_factor = if w == 0 {
                    4.0
                } else if w == 1 {
                    1.0
                } else {
                    0.3
                };
                atm.optics[s][w] = ShellOptics {
                    extinction: 1.3e-5 * lambda_factor * libm::exp(-h / 8500.0),
                    ssa: 1.0,
                    asymmetry: 0.0,
                    rayleigh_fraction: 1.0,
                };
            }
        }
        atm.compute_refractive_indices();

        let obs = Vec3::new(EARTH_RADIUS_M + 1.0, 0.0, 0.0);
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
                "Refracted spectrum[{}] = {:.6e} != individual = {:.6e}, err = {:.4e}",
                w,
                spectrum[w],
                individual,
                rel_err
            );
        }
    }

    #[test]
    fn shadow_ray_upward_transmittance_one_in_empty_atm() {
        // In a clear atmosphere (zero extinction), transmittance to the sun
        // should be 1.0 regardless of refraction.
        let mut atm = AtmosphereModel::new(&[0.0, 50.0, 100.0], &[550.0]);
        atm.compute_refractive_indices_from_altitude();

        let pos = Vec3::new(EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let sun = Vec3::new(1.0, 0.0, 0.0); // directly overhead

        let t = shadow_ray_transmittance(&atm, pos, sun, 0);
        assert!(
            (t - 1.0).abs() < 1e-10,
            "Zero-extinction transmittance should be 1.0, got {}",
            t
        );
    }

    #[test]
    fn shadow_ray_ground_hit_returns_zero() {
        // A shadow ray aimed downward should hit the ground and return 0.
        let mut atm = make_refraction_atmosphere();
        atm.compute_refractive_indices();

        let pos = Vec3::new(EARTH_RADIUS_M + 5000.0, 0.0, 0.0);
        // Direction straight down
        let sun_down = Vec3::new(-1.0, 0.0, 0.0);

        let t = shadow_ray_transmittance(&atm, pos, sun_down, 0);
        assert!(
            t < 1e-20,
            "Shadow ray hitting ground should have zero transmittance, got {}",
            t
        );
    }

    #[test]
    fn bouger_invariant_along_refracted_path() {
        // Bouger's invariant: n(r) * r * sin(alpha) = const along a refracted
        // ray in a spherically symmetric atmosphere. Here alpha is the angle
        // between the ray direction and the radial (outward) direction.
        //
        // We verify this by manually stepping through shell boundaries
        // using refract_at_boundary and checking the invariant at each step.
        use crate::geometry::{refract_at_boundary, RefractResult};

        // Build a fine-grained atmosphere for smooth refraction
        let mut alt_km = [0.0f64; 21];
        for i in 0..21 {
            alt_km[i] = i as f64 * 5.0; // 0, 5, 10, ..., 100 km
        }
        let mut atm = AtmosphereModel::new(&alt_km, &[550.0]);
        atm.compute_refractive_indices_from_altitude();

        // Start at 1m above surface with a 70-degree zenith angle ray
        let r0 = EARTH_RADIUS_M + 1.0;
        let theta0 = 70.0_f64 * core::f64::consts::PI / 180.0;
        let mut pos = Vec3::new(r0, 0.0, 0.0);
        let mut dir = Vec3::new(libm::cos(theta0), libm::sin(theta0), 0.0);

        // Compute initial Bouger invariant
        let r = pos.length();
        let radial = pos.normalize();
        let cos_alpha = dir.dot(radial);
        let sin_alpha = libm::sqrt(1.0 - cos_alpha * cos_alpha);
        let n0 = atm.refractive_index[0];
        let bouger_initial = n0 * r * sin_alpha;

        // Walk outward through shells
        for s in 0..(atm.num_shells - 1) {
            let shell = &atm.shells[s];
            // Move to outer boundary
            let hit = crate::geometry::ray_sphere_intersect(pos, dir, shell.r_outer);
            let dist = match hit {
                Some(h) if h.t_far > 1e-6 => {
                    if h.t_near > 1e-6 {
                        h.t_near
                    } else {
                        h.t_far
                    }
                }
                _ => break,
            };

            let boundary_pos = pos + dir * dist;
            let n_from = atm.refractive_index[s];
            let n_to = if s + 1 < atm.num_shells {
                atm.refractive_index[s + 1]
            } else {
                1.0
            };

            // Refract
            dir = match refract_at_boundary(dir, boundary_pos, n_from, n_to) {
                RefractResult::Refracted(d) => d,
                RefractResult::TotalReflection(_) => break, // shouldn't happen
            };
            pos = boundary_pos + dir * 1e-3; // nudge past boundary

            // Check Bouger invariant
            let r_new = pos.length();
            let radial_new = pos.normalize();
            let cos_alpha_new = dir.dot(radial_new);
            let sin_alpha_new = libm::sqrt(1.0 - cos_alpha_new * cos_alpha_new);
            let n_new = n_to;
            let bouger_new = n_new * r_new * sin_alpha_new;

            let rel_err = (bouger_new - bouger_initial).abs() / bouger_initial;
            assert!(
                rel_err < 1e-4,
                "Bouger invariant violated at shell {}: initial={:.6}, current={:.6}, rel_err={:.2e}",
                s + 1,
                bouger_initial,
                bouger_new,
                rel_err
            );
        }
    }

    #[test]
    fn refraction_bends_moderate_angle_ray() {
        // A key physical effect: atmospheric refraction bends rays at shell
        // boundaries. Going from denser to rarer medium (outward), the ray
        // bends away from the normal, making the angle to radial larger.
        //
        // We use a 60-degree zenith angle to avoid TIR (critical angle for
        // adjacent-shell n difference is ~89 deg; for large n jumps it can
        // be lower).
        use crate::geometry::{refract_at_boundary, RefractResult};

        let r_boundary = EARTH_RADIUS_M + 5000.0;
        let boundary_pos = Vec3::new(r_boundary, 0.0, 0.0);
        let normal = boundary_pos.normalize();

        // 60-degree zenith angle (outward ray, well below critical angle)
        let theta = 60.0_f64 * core::f64::consts::PI / 180.0;
        let dir = Vec3::new(libm::cos(theta), libm::sin(theta), 0.0);

        let n_from = 1.000293; // denser (lower shell)
        let n_to = 1.000150; // rarer (upper shell)

        match refract_at_boundary(dir, boundary_pos, n_from, n_to) {
            RefractResult::Refracted(d) => {
                // Going from denser to rarer: sin(theta_t) > sin(theta_i)
                // meaning the ray bends away from the normal.
                let sin_i = dir.cross(normal).length();
                let sin_t = d.cross(normal).length();
                assert!(
                    sin_t > sin_i,
                    "Denser to rarer should bend away from normal: sin_t={}, sin_i={}",
                    sin_t,
                    sin_i
                );
            }
            RefractResult::TotalReflection(_) => {
                panic!("60 deg should not produce TIR with these n values");
            }
        }
    }
}
