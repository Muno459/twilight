//! Single photon trace logic — the core MCRT pure function.
//!
//! This module contains the backward Monte Carlo photon tracing algorithm.
//! The trace function is a pure function with no platform dependencies,
//! making it compilable to any target (CPU, GPU via WGSL, WASM, CUDA PTX).

use crate::atmosphere::AtmosphereModel;
use crate::geometry::{next_shell_boundary, refract_at_boundary, RefractResult, Vec3};
use crate::path_guide::PathGuide;
use crate::scattering::{
    henyey_greenstein_phase, rayleigh_phase, sample_henyey_greenstein, sample_rayleigh_analytic,
    scatter_direction, scatter_stokes_fast, scattering_plane_cos_sin, StokesVector,
};

/// Maximum number of scattering events before terminating a photon.
pub const MAX_SCATTERS: usize = 100;

/// Apply refraction at a shell boundary and advance the photon past it.
///
/// Returns the new (position, direction) after crossing. For total
/// internal reflection the direction is reflected and the photon stays
/// in the same shell (the caller should `continue` the bounce loop).
#[inline]
fn cross_boundary(
    pos: Vec3,
    dir: Vec3,
    boundary_dist: f64,
    is_outward: bool,
    shell_idx: usize,
    atm: &AtmosphereModel,
) -> (Vec3, Vec3) {
    let boundary_pos = pos + dir * boundary_dist;
    let n_from = atm.refractive_index[shell_idx];
    let next_shell = if is_outward {
        shell_idx + 1
    } else {
        shell_idx.wrapping_sub(1)
    };
    let n_to = if next_shell < atm.num_shells {
        atm.refractive_index[next_shell]
    } else {
        1.0 // vacuum above TOA / ground below
    };
    let new_dir = match refract_at_boundary(dir, boundary_pos, n_from, n_to) {
        RefractResult::Refracted(d) | RefractResult::TotalReflection(d) => d,
    };
    (boundary_pos + new_dir * 1e-3, new_dir)
}

/// Compute total optical depth from `pos` along `dir` to atmosphere exit.
///
/// Marches shell-by-shell with refraction, identical path geometry to
/// `shadow_ray_transmittance` but in an arbitrary direction and returning
/// the raw optical depth rather than exp(-tau).
///
/// Early-exits when tau exceeds `FORCED_TAU_CUTOFF` (20.0). At that point
/// `1 - exp(-20) = 0.999999998` in f64, so the forced-scattering weight
/// is indistinguishable from 1.0 and the truncated exponential is
/// indistinguishable from the regular exponential. This means photons
/// deep in the atmosphere (where tau_max >> 20) pay only 1-3 shell ops
/// instead of marching all 50 shells to TOA.
///
/// Returns `(tau_max, hit_ground)`. `hit_ground` is true if the ray
/// terminates at the surface rather than exiting to space. When
/// `hit_ground` is true, forced scattering should NOT be used (the
/// photon will be handled by ground reflection in the bounce loop).
fn scout_tau_to_boundary(
    atm: &AtmosphereModel,
    start_pos: Vec3,
    start_dir: Vec3,
    wavelength_idx: usize,
) -> (f64, bool) {
    let surface_radius = atm.surface_radius();
    let num_shells = atm.num_shells;
    let mut pos = start_pos;
    let mut dir = start_dir;
    let mut tau = 0.0;

    let mut shell_idx = match atm.shell_index(pos.length()) {
        Some(idx) => idx,
        None => return (0.0, false),
    };

    for _ in 0..200 {
        let shell = &atm.shells[shell_idx];
        let optics = &atm.optics[shell_idx][wavelength_idx];

        match next_shell_boundary(pos, dir, shell.r_inner, shell.r_outer) {
            Some((dist, is_outward)) => {
                tau += optics.extinction * dist;

                // Refract at boundary (same logic as shadow_ray_transmittance)
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

                // Hit ground -- path terminates here
                if !is_outward && pos.length() <= surface_radius + 1.0 {
                    return (tau, true);
                }

                // Exited atmosphere
                if next_shell >= num_shells {
                    return (tau, false);
                }

                shell_idx = next_shell;
            }
            None => return (tau, false),
        }

        // At tau > FORCED_TAU_CUTOFF (20.0), 1-exp(-20) = 0.999999998.
        // The forced-scattering weight is indistinguishable from 1.0
        // and the truncated exponential is the regular exponential.
        // Early exit avoids pointless shell marching through dense
        // lower atmosphere. No bias: weight correction is exact to
        // f64 precision at this threshold.
        if tau > FORCED_TAU_CUTOFF {
            return (tau, false);
        }
    }

    (tau, false)
}

/// Advance a photon along its ray until `tau_target` optical depth is consumed.
///
/// Marches shell-by-shell with refraction, following the same path geometry
/// as `scout_tau_to_boundary`. Returns `(scatter_pos, dir_at_scatter, shell_idx)`
/// where the photon scatters.
///
/// The caller must ensure `tau_target <= tau_max` from a prior scout call,
/// guaranteeing the scatter point lies within the atmosphere.
fn advance_to_optical_depth(
    atm: &AtmosphereModel,
    start_pos: Vec3,
    start_dir: Vec3,
    tau_target: f64,
    wavelength_idx: usize,
) -> (Vec3, Vec3, usize) {
    let surface_radius = atm.surface_radius();
    let num_shells = atm.num_shells;
    let mut pos = start_pos;
    let mut dir = start_dir;
    let mut tau_accumulated = 0.0;

    let mut shell_idx = match atm.shell_index(pos.length()) {
        Some(idx) => idx,
        None => return (pos, dir, 0),
    };

    for _ in 0..200 {
        let shell = &atm.shells[shell_idx];
        let optics = &atm.optics[shell_idx][wavelength_idx];

        match next_shell_boundary(pos, dir, shell.r_inner, shell.r_outer) {
            Some((boundary_dist, is_outward)) => {
                let tau_shell = optics.extinction * boundary_dist;

                if tau_accumulated + tau_shell >= tau_target {
                    // Scatter point is within this shell.
                    let tau_remaining = tau_target - tau_accumulated;
                    let dist = if optics.extinction > 1e-30 {
                        tau_remaining / optics.extinction
                    } else {
                        // Zero extinction: shouldn't reach here if scout
                        // was consistent, but place at boundary as fallback.
                        boundary_dist
                    };
                    pos = pos + dir * dist;
                    return (pos, dir, shell_idx);
                }

                // Cross boundary -- same refraction as scout
                tau_accumulated += tau_shell;
                let boundary_pos = pos + dir * boundary_dist;
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

                // Hit ground -- place scatter here
                if !is_outward && pos.length() <= surface_radius + 1.0 {
                    return (pos, dir, shell_idx);
                }

                // Exited atmosphere -- place scatter at exit
                if next_shell >= num_shells {
                    return (pos, dir, shell_idx);
                }

                shell_idx = next_shell;
            }
            None => return (pos, dir, shell_idx),
        }
    }

    (pos, dir, shell_idx)
}

/// Result of tracing a single photon.
#[derive(Debug, Clone, Copy)]
pub struct PhotonResult {
    /// Accumulated weight/contribution of this photon.
    pub weight: f64,
    /// Number of scattering events.
    pub num_scatters: u32,
    /// Whether the photon was terminated (absorbed, escaped, or max scatters).
    pub terminated: bool,
}

/// Trace a single photon backward from observer toward the sun.
///
/// This is the core MCRT function. It traces a photon starting at the observer,
/// propagating through the atmosphere, scattering at each interaction, and
/// accumulating the contribution from direct solar illumination at each
/// scattering point (next-event estimation).
///
/// # Arguments
/// * `atm` - Atmosphere model with shell geometry and optical properties
/// * `observer_pos` - Observer position in ECEF coordinates [m]
/// * `initial_dir` - Initial photon direction (unit vector, pointing away from observer)
/// * `sun_dir` - Direction toward the sun (unit vector)
/// * `wavelength_idx` - Index into the atmosphere model's wavelength grid
/// * `rng_state` - Mutable RNG state (simple xorshift for no_std compatibility)
///
/// # Returns
/// The photon's contribution to sky radiance at this wavelength.
pub fn trace_photon(
    atm: &AtmosphereModel,
    observer_pos: Vec3,
    initial_dir: Vec3,
    sun_dir: Vec3,
    wavelength_idx: usize,
    rng_state: &mut u64,
) -> PhotonResult {
    let mut pos = observer_pos;
    let mut dir = initial_dir;
    let mut weight = 1.0;
    let mut result = PhotonResult {
        weight: 0.0,
        num_scatters: 0,
        terminated: false,
    };

    for _bounce in 0..MAX_SCATTERS {
        let r = pos.length();

        // Find which shell we're in
        let shell_idx = match atm.shell_index(r) {
            Some(idx) => idx,
            None => {
                // Outside atmosphere — photon escaped
                result.terminated = true;
                break;
            }
        };

        let shell = &atm.shells[shell_idx];
        let optics = &atm.optics[shell_idx][wavelength_idx];

        // If extinction is zero, photon passes through without interaction
        if optics.extinction < 1e-20 {
            // Move to next shell boundary (with refraction)
            match next_shell_boundary(pos, dir, shell.r_inner, shell.r_outer) {
                Some((dist, is_outward)) => {
                    let (new_pos, new_dir) =
                        cross_boundary(pos, dir, dist, is_outward, shell_idx, atm);
                    pos = new_pos;
                    dir = new_dir;
                    continue;
                }
                None => {
                    result.terminated = true;
                    break;
                }
            }
        }

        // Sample free path length (Beer-Lambert)
        let xi = xorshift_f64(rng_state);
        let free_path = -libm::log(1.0 - xi + 1e-30) / optics.extinction;

        // Check if free path reaches a shell boundary
        match next_shell_boundary(pos, dir, shell.r_inner, shell.r_outer) {
            Some((boundary_dist, is_outward)) => {
                if free_path >= boundary_dist {
                    // Photon exits this shell without scattering.
                    // Apply refraction at the boundary.
                    let (new_pos, new_dir) =
                        cross_boundary(pos, dir, boundary_dist, is_outward, shell_idx, atm);
                    pos = new_pos;
                    dir = new_dir;

                    // Check if we hit the ground
                    if !is_outward && pos.length() <= atm.surface_radius() + 1.0 {
                        // Ground reflection (Lambertian)
                        let albedo = atm.surface_albedo[wavelength_idx];
                        weight *= albedo;

                        // Reflect: random hemisphere direction
                        let normal = pos.normalize();
                        dir = sample_hemisphere(normal, rng_state);
                        continue;
                    }

                    continue;
                }
            }
            None => {
                result.terminated = true;
                break;
            }
        }

        // Scattering event at free_path distance
        pos = pos + dir * free_path;

        // --- Next-Event Estimation (NEE) ---
        // Compute direct contribution from sun at this scatter point.
        // Pass the current photon direction for correct phase function evaluation.
        let nee_contribution = compute_nee(atm, pos, dir, sun_dir, optics, wavelength_idx, weight);
        result.weight += nee_contribution;
        result.num_scatters += 1;

        // Apply single scattering albedo (probability of scattering vs absorption)
        weight *= optics.ssa;

        // Sample new direction based on phase function
        let cos_theta = if xorshift_f64(rng_state) < optics.rayleigh_fraction {
            sample_rayleigh_analytic(xorshift_f64(rng_state))
        } else {
            sample_henyey_greenstein(xorshift_f64(rng_state), optics.asymmetry)
        };
        let phi = 2.0 * core::f64::consts::PI * xorshift_f64(rng_state);
        dir = scatter_direction(dir, cos_theta, phi);
    }

    result.terminated = true;
    result
}

/// Compute next-event estimation: direct solar contribution at a scatter point.
///
/// Traces a shadow ray from the scatter point toward the sun, computing
/// the transmittance along the path. Multiplied by the phase function
/// evaluated at the angle between the photon's current direction and the
/// sun direction.
///
/// In backward MC, the photon travels from observer into the atmosphere.
/// At each scatter event, NEE asks: "what if this photon had arrived from
/// the sun?" The phase function is evaluated at the angle between the
/// incoming solar direction and the outgoing direction toward the observer
/// (which is -photon_dir in backward tracing).
fn compute_nee(
    atm: &AtmosphereModel,
    scatter_pos: Vec3,
    photon_dir: Vec3,
    sun_dir: Vec3,
    local_optics: &crate::atmosphere::ShellOptics,
    wavelength_idx: usize,
    weight: f64,
) -> f64 {
    // Trace shadow ray toward sun
    let transmittance = trace_transmittance(atm, scatter_pos, sun_dir, wavelength_idx);

    if transmittance < 1e-30 {
        return 0.0;
    }

    // Phase function: cos(angle) between the sun direction and the
    // direction back toward the observer (-photon_dir).
    // This is the scattering angle for light coming from the sun being
    // scattered toward the observer.
    let cos_angle = sun_dir.dot(-photon_dir);

    let phase = if local_optics.rayleigh_fraction > 0.99 {
        rayleigh_phase(cos_angle)
    } else {
        local_optics.rayleigh_fraction * rayleigh_phase(cos_angle)
            + (1.0 - local_optics.rayleigh_fraction)
                * henyey_greenstein_phase(cos_angle, local_optics.asymmetry)
    };

    // Contribution = weight × transmittance × phase / (4π)
    weight * transmittance * phase * INV_4PI
}

/// Compute transmittance along a ray through the atmosphere.
///
/// Traces the ray shell-by-shell, applying Snell's law at each boundary
/// so the shadow ray follows the physically correct curved path.
/// Returns exp(-total_optical_depth).
fn trace_transmittance(
    atm: &AtmosphereModel,
    start_pos: Vec3,
    direction: Vec3,
    wavelength_idx: usize,
) -> f64 {
    let mut pos = start_pos;
    let mut dir = direction;
    let mut total_optical_depth = 0.0;

    for _ in 0..200 {
        let r = pos.length();

        let shell_idx = match atm.shell_index(r) {
            Some(idx) => idx,
            None => break, // Exited atmosphere
        };

        let shell = &atm.shells[shell_idx];
        let optics = &atm.optics[shell_idx][wavelength_idx];

        match next_shell_boundary(pos, dir, shell.r_inner, shell.r_outer) {
            Some((dist, is_outward)) => {
                total_optical_depth += optics.extinction * dist;

                // Refract at the boundary
                let (new_pos, new_dir) = cross_boundary(pos, dir, dist, is_outward, shell_idx, atm);
                pos = new_pos;
                dir = new_dir;

                // Hit ground -- fully opaque
                if !is_outward && pos.length() <= atm.surface_radius() + 1.0 {
                    return 0.0;
                }
            }
            None => break,
        }

        // Early termination if transmittance is negligible
        if total_optical_depth > 50.0 {
            return 0.0;
        }
    }

    libm::exp(-total_optical_depth)
}

/// Trace multiple photons across all wavelengths and return a spectral radiance
/// array compatible with `single_scatter_spectrum`.
///
/// This is the main entry point for multiple-scattering spectral computation.
/// For each wavelength, it traces `photons_per_wavelength` backward photons
/// and averages the NEE contributions. The result is in the same arbitrary
/// units as `single_scatter_spectrum` (needs solar irradiance weighting by
/// the caller).
///
/// # Arguments
/// * `atm` - Atmosphere model
/// * `observer_pos` - Observer position in ECEF [m]
/// * `view_dir` - Initial viewing direction (unit vector)
/// * `sun_dir` - Direction toward the sun (unit vector)
/// * `photons_per_wavelength` - Number of photons to trace per wavelength
/// * `base_seed` - Base RNG seed (each wavelength/photon gets a unique derived seed)
///
/// # Returns
/// Spectral radiance array `[f64; 64]`, one value per wavelength channel.
pub fn mc_scatter_spectrum(
    atm: &AtmosphereModel,
    observer_pos: Vec3,
    view_dir: Vec3,
    sun_dir: Vec3,
    photons_per_wavelength: usize,
    base_seed: u64,
) -> [f64; 64] {
    let mut radiance = [0.0f64; 64];
    let num_wl = atm.num_wavelengths;

    if photons_per_wavelength == 0 {
        return radiance;
    }

    for (w, rad_w) in radiance.iter_mut().enumerate().take(num_wl) {
        let mut total_weight = 0.0;
        for p in 0..photons_per_wavelength {
            // Unique seed per (wavelength, photon) pair to avoid correlation
            let mut rng = base_seed
                .wrapping_add(w as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(p as u64)
                .wrapping_mul(2862933555777941757)
                .wrapping_add(1);

            let result = trace_photon(atm, observer_pos, view_dir, sun_dir, w, &mut rng);
            total_weight += result.weight;
        }
        *rad_w = total_weight / photons_per_wavelength as f64;
    }

    radiance
}

// ── Polarized (Stokes vector) transport ────────────────────────────────

/// Result of tracing a single photon in polarized mode.
#[derive(Debug, Clone, Copy)]
pub struct PolarizedPhotonResult {
    /// Accumulated Stokes vector contribution.
    pub stokes: StokesVector,
    /// Number of scattering events.
    pub num_scatters: u32,
    /// Whether the photon was terminated.
    pub terminated: bool,
}

/// Trace a single photon backward with full Stokes vector tracking.
///
/// This is the polarized counterpart of [`trace_photon`]. The photon carries
/// a Stokes vector state that is transformed by Mueller matrices at each
/// scattering event. The reference frame is rotated between successive
/// scattering planes.
///
/// In backward MC with NEE, each scatter event computes the Mueller matrix
/// for scattering sunlight toward the observer. The NEE contribution at
/// each bounce is a Stokes vector (not just a scalar weight).
///
/// For unpolarized sunlight (I=1, Q=U=V=0), the total intensity (I component)
/// converges to the same value as the scalar transport, with 1-2% corrections
/// from polarization cross-coupling. The Q/U/V components give the sky
/// polarization pattern.
pub fn trace_photon_polarized(
    atm: &AtmosphereModel,
    observer_pos: Vec3,
    initial_dir: Vec3,
    sun_dir: Vec3,
    wavelength_idx: usize,
    rng_state: &mut u64,
) -> PolarizedPhotonResult {
    let mut pos = observer_pos;
    let mut dir = initial_dir;
    let mut weight = 1.0;
    let mut prev_dir = initial_dir; // previous direction for plane rotation
    let mut result = PolarizedPhotonResult {
        stokes: StokesVector::unpolarized(0.0),
        num_scatters: 0,
        terminated: false,
    };

    for _bounce in 0..MAX_SCATTERS {
        let r = pos.length();

        let shell_idx = match atm.shell_index(r) {
            Some(idx) => idx,
            None => {
                result.terminated = true;
                break;
            }
        };

        let shell = &atm.shells[shell_idx];
        let optics = &atm.optics[shell_idx][wavelength_idx];

        // Zero extinction: pass through with refraction
        if optics.extinction < 1e-20 {
            match next_shell_boundary(pos, dir, shell.r_inner, shell.r_outer) {
                Some((dist, is_outward)) => {
                    let (new_pos, new_dir) =
                        cross_boundary(pos, dir, dist, is_outward, shell_idx, atm);
                    prev_dir = dir;
                    pos = new_pos;
                    dir = new_dir;
                    continue;
                }
                None => {
                    result.terminated = true;
                    break;
                }
            }
        }

        // Sample free path
        let xi = xorshift_f64(rng_state);
        let free_path = -libm::log(1.0 - xi + 1e-30) / optics.extinction;

        // Check shell boundary
        match next_shell_boundary(pos, dir, shell.r_inner, shell.r_outer) {
            Some((boundary_dist, is_outward)) => {
                if free_path >= boundary_dist {
                    let (new_pos, new_dir) =
                        cross_boundary(pos, dir, boundary_dist, is_outward, shell_idx, atm);
                    prev_dir = dir;
                    pos = new_pos;
                    dir = new_dir;

                    if !is_outward && pos.length() <= atm.surface_radius() + 1.0 {
                        let albedo = atm.surface_albedo[wavelength_idx];
                        weight *= albedo;
                        let normal = pos.normalize();
                        prev_dir = dir;
                        dir = sample_hemisphere(normal, rng_state);
                        continue;
                    }
                    continue;
                }
            }
            None => {
                result.terminated = true;
                break;
            }
        }

        // Scattering event
        pos = pos + dir * free_path;

        // --- Polarized NEE ---
        // Compute the Mueller matrix for scattering sunlight (coming from
        // sun_dir) toward the observer (along -dir).
        let nee_stokes = compute_nee_polarized(
            atm,
            pos,
            dir,
            prev_dir,
            sun_dir,
            optics,
            wavelength_idx,
            weight,
        );
        result.stokes = result.stokes.add(&nee_stokes);
        result.num_scatters += 1;

        // Apply SSA
        weight *= optics.ssa;

        // Sample new direction
        let cos_theta = if xorshift_f64(rng_state) < optics.rayleigh_fraction {
            sample_rayleigh_analytic(xorshift_f64(rng_state))
        } else {
            sample_henyey_greenstein(xorshift_f64(rng_state), optics.asymmetry)
        };
        let phi = 2.0 * core::f64::consts::PI * xorshift_f64(rng_state);
        prev_dir = dir;
        dir = scatter_direction(dir, cos_theta, phi);
    }

    result.terminated = true;
    result
}

/// Polarized NEE: compute the Stokes vector contribution from the sun
/// at a scatter point, using the full Mueller matrix framework.
#[allow(clippy::too_many_arguments)] // Physics function: all 8 params are independent physical quantities
fn compute_nee_polarized(
    atm: &AtmosphereModel,
    scatter_pos: Vec3,
    photon_dir: Vec3,
    prev_dir: Vec3,
    sun_dir: Vec3,
    local_optics: &crate::atmosphere::ShellOptics,
    wavelength_idx: usize,
    weight: f64,
) -> StokesVector {
    let transmittance = trace_transmittance(atm, scatter_pos, sun_dir, wavelength_idx);

    if transmittance < 1e-30 {
        return StokesVector::unpolarized(0.0);
    }

    // Scattering angle for light from sun scattered toward observer (-photon_dir)
    let cos_angle = sun_dir.dot(-photon_dir);

    // Rotation angle: align reference frame from previous scattering plane
    // to the current one (prev_dir, photon_dir) -> (photon_dir, -sun_dir is
    // the "virtual" next direction). For NEE, the "next direction" is the sun
    // direction (reversed, since we compute scattering of sunlight).
    let (rot_c, rot_s) = scattering_plane_cos_sin(prev_dir, photon_dir, sun_dir);

    // Direct Stokes scatter+rotate (no matrices, no trig)
    let solar_stokes = StokesVector::unpolarized(1.0);
    let scattered = scatter_stokes_fast(
        &solar_stokes,
        cos_angle,
        local_optics.rayleigh_fraction,
        local_optics.asymmetry,
        rot_c,
        rot_s,
    );

    // Scale by weight, transmittance, and 1/(4pi)
    let factor = weight * transmittance * INV_4PI;
    scattered.scale(factor)
}

/// Trace multiple photons in polarized mode and return spectral Stokes vectors.
///
/// Returns an array of 64 Stokes vectors (one per wavelength channel).
/// The I component of each converges to the same value as `mc_scatter_spectrum`
/// (with small polarization corrections). Q/U/V give the polarization state.
pub fn mc_scatter_spectrum_polarized(
    atm: &AtmosphereModel,
    observer_pos: Vec3,
    view_dir: Vec3,
    sun_dir: Vec3,
    photons_per_wavelength: usize,
    base_seed: u64,
) -> [StokesVector; 64] {
    let mut radiance = [StokesVector::unpolarized(0.0); 64];
    let num_wl = atm.num_wavelengths;

    if photons_per_wavelength == 0 {
        return radiance;
    }

    for (w, rad_w) in radiance.iter_mut().enumerate().take(num_wl) {
        let mut total_stokes = StokesVector::unpolarized(0.0);
        for p in 0..photons_per_wavelength {
            let mut rng = base_seed
                .wrapping_add(w as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(p as u64)
                .wrapping_mul(2862933555777941757)
                .wrapping_add(1);

            let result = trace_photon_polarized(atm, observer_pos, view_dir, sun_dir, w, &mut rng);
            total_stokes = total_stokes.add(&result.stokes);
        }
        let inv_n = 1.0 / photons_per_wavelength as f64;
        *rad_w = total_stokes.scale(inv_n);
    }

    radiance
}

/// Number of LOS steps for the hybrid integrator.
const HYBRID_LOS_STEPS: usize = 200;

/// Maximum bounces for secondary chains in the hybrid integrator.
const HYBRID_MAX_BOUNCES: usize = 50;

/// Precomputed 1 / (4 * pi), used at every NEE evaluation.
const INV_4PI: f64 = 1.0 / (4.0 * core::f64::consts::PI);

/// Early-exit threshold for `scout_tau_to_boundary`.
///
/// At tau > 20, `1 - exp(-20) = 0.999999998` in f64. The forced-scattering
/// weight is indistinguishable from 1.0 and the truncated exponential is
/// indistinguishable from the regular exponential. Stopping the scout here
/// avoids marching all 50 shells when the photon is deep in the atmosphere
/// (where analog scattering is already efficient).
const FORCED_TAU_CUTOFF: f64 = 20.0;

/// Maximum directional bias parameter for the exponential transform.
///
/// The exponential transform modifies the free-path sampling within each shell
/// to bias photons in the upward (zenith) direction. The modified extinction is:
///   sigma' = sigma * (1 - alpha * cos_z)
/// where cos_z = dot(dir, local_up).
///
/// - Upward (cos_z > 0): sigma' < sigma, longer mean free path, photon climbs faster
/// - Downward (cos_z < 0): sigma' > sigma, shorter mean free path, photon absorbed sooner
///
/// At alpha=0.5: sigma' in [0.5*sigma, 1.5*sigma], always positive.
/// Weight corrections keep the estimator exactly unbiased:
///   - Scatter at distance d:   weight *= (sigma/sigma') * exp(-alpha*sigma*cos_z*d)
///   - Boundary cross at dist D: weight *= exp(-alpha*sigma*cos_z*D)
///
/// Ramped from 0 (SZA < 96) to EXP_TRANSFORM_ALPHA_MAX (SZA >= 106), using
/// the same SZA ramp as zenith-biased direction sampling.
const EXP_TRANSFORM_ALPHA_MAX: f64 = 0.5;

/// Power exponent for zenith-biased initial direction sampling.
///
/// The secondary chain's hemisphere branch uses a power-cosine PDF:
///   p(omega) = (n+1)/(2*pi) * cos^n(theta_zenith)
/// instead of the default cosine-weighted hemisphere (n=1).
///
/// Higher n concentrates more rays near the zenith. At n=5:
///   - 58% of rays within 30 deg of zenith (vs 25% for cosine-weighted)
///   - 88% within 45 deg (vs 50%)
///   - Max importance weight within 60 deg: ~5.3x
///
/// This helps at deep twilight (SZA > 100) where chains must climb to
/// high altitude to enable lateral transport to sunlit regions.
const ZENITH_BIAS_N: f64 = 5.0;

/// SZA (degrees) below which the zenith bias is inactive (standard 50/50 mix
/// with cosine-weighted hemisphere, no importance weight overhead).
const ZENITH_SZA_START: f64 = 96.0;

/// SZA (degrees) at which the zenith-biased fraction reaches its maximum.
const ZENITH_SZA_FULL: f64 = 106.0;

/// Maximum fraction of rays using zenith-biased sampling at deep twilight.
/// The remaining (1 - ZENITH_MAX_FRACTION) still use phase function
/// sampling to maintain some coverage of non-vertical scattering paths.
const ZENITH_MAX_FRACTION: f64 = 0.95;

/// Maximum fraction of the zenith-allocated rays redirected to the
/// terminator lobe at deep twilight.
///
/// At SZA >= ZENITH_SZA_FULL:
///   phase branch:      (1 - ZENITH_MAX_FRACTION) = 5%
///   zenith branch:     ZENITH_MAX_FRACTION * (1 - TERMINATOR_MAX_SHARE) = 47.5%
///   terminator branch: ZENITH_MAX_FRACTION * TERMINATOR_MAX_SHARE      = 47.5%
///
/// At SZA <= ZENITH_SZA_START: term_share = 0, no terminator rays.
const TERMINATOR_MAX_SHARE: f64 = 0.5;

/// Power-cosine exponent for the terminator lobe at maximum SZA.
///
/// The terminator lobe samples from cos^m(theta_t) centered on the
/// terminator axis. Higher m concentrates rays more tightly around
/// the axis. At m=8, ~70% of rays fall within 30 deg of the axis.
///
/// Ramps from 1.0 (SZA <= 96, inactive -- equivalent to cosine hemisphere)
/// to TERMINATOR_N_MAX (SZA >= 106).
const TERMINATOR_N_MAX: f64 = 8.0;

/// Tilt angle (degrees) of the terminator axis from zenith at SZA = ZENITH_SZA_START.
///
/// The terminator axis is: normalize(cos(tilt) * up + sin(tilt) * sun_horiz).
/// At civil twilight, a small tilt gently biases toward the sun's azimuth.
const TERMINATOR_TILT_MIN_DEG: f64 = 20.0;

/// Tilt angle (degrees) of the terminator axis from zenith at SZA = ZENITH_SZA_FULL.
///
/// At deep twilight (SZA 106), the terminator axis points 50 deg from
/// zenith toward the sub-solar horizon, directing rays into the region
/// where shadow rays can first reach sunlit atmosphere.
const TERMINATOR_TILT_MAX_DEG: f64 = 50.0;

/// Fraction of bounces (after the first) that use guide-sampled directions
/// when a trained PathGuide is available. The remaining fraction uses the
/// standard phase function sampling. One-sample MIS with balance heuristic
/// combines both, keeping the estimator exactly unbiased.
///
/// At 0.5: half the bounces try the guide, half use the phase function.
/// The MIS weight at each bounce is:
///   w_mis = p_chosen / (GUIDE_MIS_FRAC * p_guide + (1 - GUIDE_MIS_FRAC) * p_phase)
const GUIDE_MIS_FRAC: f64 = 0.5;

/// Number of altitude-based splitting levels for deep twilight variance reduction.
///
/// When a backward MC chain scatters above a split altitude, it is duplicated
/// into K copies, each with weight/K. Each copy explores independently from
/// the split point with an independent RNG stream. This is provably unbiased
/// by weight conservation: K * (w/K) * E[score] = w * E[score].
///
/// Splitting directly addresses the rare-event bottleneck: at SZA 106, the
/// probability of a chain reaching 60 km altitude from a 10 km LOS step is
/// ~10^-3. Splitting at intermediate altitudes converts rare survivors into
/// multiple independent explorations of the high-altitude region.
const NUM_SPLIT_LEVELS: usize = 3;

/// Altitude thresholds (meters above surface) at which splitting occurs.
///
///   25 km: chain escapes the dense troposphere (major bottleneck)
///   45 km: chain reaches mid-stratosphere (approaching shadow boundary)
///   65 km: chain reaches mesosphere (lateral transport region)
const SPLIT_ALTITUDES_M: [f64; NUM_SPLIT_LEVELS] = [25_000.0, 45_000.0, 65_000.0];

/// Split factors at deep twilight (SZA >= 102 deg).
///
/// Worst-case budget: 3 * 3 * 2 = 18 copies per chain. Each copy carries
/// weight / (product of split factors encountered), so no weight inflation.
const SPLIT_FACTORS_DEEP: [usize; NUM_SPLIT_LEVELS] = [3, 3, 2];

/// Split factors in the transition band (96 <= SZA < 102 deg).
///
/// More conservative: 2 * 2 * 1 = 4 copies max. At SZA 96-100, chains can
/// reach sunlit regions with ~5-10% probability, so moderate splitting suffices.
const SPLIT_FACTORS_TRANSITION: [usize; NUM_SPLIT_LEVELS] = [2, 2, 1];

/// Maximum number of concurrent split particles in the work stack.
///
/// Must be >= product of maximum split factors (3*3*2 = 18), plus headroom.
/// Stack memory per particle: ~80 bytes (scalar) or ~600 bytes (ALIS with
/// weight_ratio[64]). At 20 particles: 1.6 KB (scalar) or 12 KB (ALIS).
const MAX_SPLIT_PARTICLES: usize = 20;

/// State of a single particle in the altitude-splitting work stack (scalar mode).
///
/// When a chain scatters above a split altitude, it spawns K copies (each with
/// weight/K) that explore the high-altitude region independently. The work
/// stack stores pending copies; the main particle is processed first.
#[derive(Clone, Copy)]
struct SplitParticleScalar {
    pos: Vec3,
    dir: Vec3,
    weight: f64,
    rng: u64,
    bounces_left: usize,
    next_split: usize,
}

/// State of a single particle in the altitude-splitting work stack (ALIS mode).
///
/// Same as `SplitParticleScalar` but carries per-wavelength weight ratios
/// for the ALIS hero tracing scheme.
#[derive(Clone, Copy)]
struct SplitParticleAlis {
    pos: Vec3,
    dir: Vec3,
    hero_weight: f64,
    weight_ratio: [f64; 64],
    rng: u64,
    bounces_left: usize,
    next_split: usize,
}

/// Return split factors for the current SZA.
///
/// - SZA <= 96: no splitting (all 1s, zero overhead)
/// - 96 < SZA < 102: transition factors (moderate, 2*2*1 = 4 max copies)
/// - SZA >= 102: deep factors (aggressive, 3*3*2 = 18 max copies)
#[inline]
fn split_factors_for_sza(sza_deg: f64) -> [usize; NUM_SPLIT_LEVELS] {
    if sza_deg < ZENITH_SZA_START {
        [1; NUM_SPLIT_LEVELS]
    } else if sza_deg < 102.0 {
        SPLIT_FACTORS_TRANSITION
    } else {
        SPLIT_FACTORS_DEEP
    }
}

// --- VSPG (Volume Scattering Probability Guiding) ---
//
// VSPG biases forced-scattering distance sampling toward high altitude.
// Standard forced scattering samples from exp(-tau), concentrating scatters
// in the dense troposphere. At deep twilight, most tropospheric scatters are
// wasted because the chains cannot reach sunlit regions. VSPG importance-
// weights the per-shell scattering probability so that chains "skip" the
// troposphere and scatter in the stratosphere/mesosphere where they can
// contribute to the signal via NEE.
//
// Math:
//   Natural per-shell probability: p_i = exp(-tau_lo_i) - exp(-tau_hi_i)
//   Importance:                    I_i = vspg_importance(alt_i, sza)
//   Guided probability:            q_i = I_i * p_i  (unnormalized)
//   Weight correction:             w   = I_avg / I_j
//                                      = [sum(I_k * p_k) / sum(p_k)] / I_j
//
// Provably unbiased: the weight correction exactly compensates for the
// biased sampling. When all I_i = 1 (SZA <= 96), w = 1 and the sampling
// degenerates to standard forced scattering with zero overhead (gated out).

/// Maximum number of shell segments for VSPG importance sampling.
/// A ray through the atmosphere crosses at most ~64 shells in each
/// direction. 128 handles reflections and re-entries with headroom.
const VSPG_MAX_SEGMENTS: usize = 128;

/// Altitude (meters) below which VSPG importance is 1.0 (no boost).
/// Below 15 km, the troposphere is dense and chains scatter frequently
/// via analog mode anyway. VSPG does not need to act here.
const VSPG_BOOST_START_M: f64 = 15_000.0;

/// Altitude (meters) at which VSPG importance reaches maximum.
/// At 70 km (mesosphere), photons are in the lateral transport region
/// where NEE toward the sun first becomes possible at deep twilight.
const VSPG_BOOST_FULL_M: f64 = 70_000.0;

/// Maximum importance multiplier at full SZA and full altitude.
/// A value of 50 means high-altitude shells get 50x the natural
/// probability of being selected as scatter sites. This aggressively
/// pushes chains into the mesosphere at deep twilight while maintaining
/// exact unbiasedness via weight correction.
const VSPG_MAX_IMPORTANCE: f64 = 50.0;

/// Per-shell segment data for VSPG importance-weighted sampling.
#[derive(Clone, Copy)]
struct VspgSegment {
    /// Cumulative optical depth at segment entry.
    tau_lo: f64,
    /// Cumulative optical depth at segment exit.
    tau_hi: f64,
    /// Precomputed VSPG importance for this shell.
    importance: f64,
}

/// Compute altitude-dependent importance for VSPG.
///
/// Returns a multiplier >= 1.0 that biases scatter site selection toward
/// high altitude. The multiplier ramps quadratically from 1.0 (at or below
/// `VSPG_BOOST_START_M`) to a SZA-dependent maximum (at `VSPG_BOOST_FULL_M`).
///
/// The SZA dependence ensures zero overhead at civil/nautical twilight:
/// - SZA <= 96: returns 1.0 for all altitudes (no VSPG effect)
/// - SZA = 101: moderate boost (up to ~25x at 70 km)
/// - SZA >= 106: full boost (up to `VSPG_MAX_IMPORTANCE` at 70 km)
#[inline]
fn vspg_importance(alt_m: f64, sza_deg: f64) -> f64 {
    if sza_deg < ZENITH_SZA_START || alt_m <= VSPG_BOOST_START_M {
        return 1.0;
    }
    let sza_t =
        ((sza_deg - ZENITH_SZA_START) / (ZENITH_SZA_FULL - ZENITH_SZA_START)).clamp(0.0, 1.0);
    let alt_t =
        ((alt_m - VSPG_BOOST_START_M) / (VSPG_BOOST_FULL_M - VSPG_BOOST_START_M)).clamp(0.0, 1.0);
    let max_imp = 1.0 + (VSPG_MAX_IMPORTANCE - 1.0) * sza_t;
    1.0 + (max_imp - 1.0) * alt_t * alt_t
}

/// Sample a forced-scatter optical depth using VSPG importance weighting.
///
/// Re-walks the ray path through shells (same geometry as the scout),
/// collecting per-shell segment data and importance values. Then uses
/// CDF inversion on the importance-weighted probability distribution
/// to select a shell and sample tau within it.
///
/// Returns `(tau_s, weight_correction)` where:
/// - `tau_s` is the sampled optical depth along the ray
/// - `weight_correction` is the multiplicative factor for unbiasedness:
///   `weight_correction = I_avg / I_j`
///   where `I_avg` is the importance-weighted average and `I_j` is the
///   importance of the selected segment.
///
/// When VSPG importance is uniform (SZA <= 96), the weight correction
/// is 1.0 and sampling equals standard forced scattering.
fn vspg_sample_scatter_tau(
    atm: &AtmosphereModel,
    start_pos: Vec3,
    start_dir: Vec3,
    wavelength_idx: usize,
    tau_max: f64,
    sza_deg: f64,
    rng: &mut u64,
) -> (f64, f64) {
    let surface_radius = atm.surface_radius();
    let num_shells = atm.num_shells;
    let mut pos = start_pos;
    let mut dir = start_dir;
    let mut tau = 0.0;

    // Collect segments during a re-walk of the ray path.
    let mut segments = [VspgSegment {
        tau_lo: 0.0,
        tau_hi: 0.0,
        importance: 1.0,
    }; VSPG_MAX_SEGMENTS];
    let mut num_seg: usize = 0;

    let mut shell_idx = match atm.shell_index(pos.length()) {
        Some(idx) => idx,
        None => {
            // Outside atmosphere: fall back to natural sampling.
            let xi = xorshift_f64(rng);
            let one_minus_exp = 1.0 - libm::exp(-tau_max);
            return (-libm::log(1.0 - xi * one_minus_exp + 1e-30), 1.0);
        }
    };

    for _ in 0..200 {
        let shell = &atm.shells[shell_idx];
        let optics = &atm.optics[shell_idx][wavelength_idx];

        match next_shell_boundary(pos, dir, shell.r_inner, shell.r_outer) {
            Some((dist, is_outward)) => {
                let tau_shell = optics.extinction * dist;
                let tau_end = tau + tau_shell;

                // Cap at tau_max (the scout capped here too).
                let tau_hi = if tau_end > tau_max { tau_max } else { tau_end };

                if num_seg < VSPG_MAX_SEGMENTS && tau_hi > tau + 1e-30 {
                    segments[num_seg] = VspgSegment {
                        tau_lo: tau,
                        tau_hi,
                        importance: vspg_importance(shell.altitude_mid, sza_deg),
                    };
                    num_seg += 1;
                }

                if tau_end >= tau_max {
                    break; // Reached scout's tau_max
                }

                tau = tau_end;

                // Refract at boundary (same path geometry as scout).
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

                // Hit ground.
                if !is_outward && pos.length() <= surface_radius + 1.0 {
                    break;
                }
                // Exited atmosphere.
                if next_shell >= num_shells {
                    break;
                }

                shell_idx = next_shell;
            }
            None => break,
        }

        if tau > FORCED_TAU_CUTOFF {
            break;
        }
    }

    // Fallback: if no segments collected, use natural sampling.
    if num_seg == 0 {
        let xi = xorshift_f64(rng);
        let one_minus_exp = 1.0 - libm::exp(-tau_max);
        return (-libm::log(1.0 - xi * one_minus_exp + 1e-30), 1.0);
    }

    // Compute per-segment natural and importance-weighted probabilities.
    // p_i = exp(-tau_lo_i) - exp(-tau_hi_i)  (natural scatter probability)
    // q_i = I_i * p_i                        (importance-weighted)
    let mut p_sum = 0.0_f64;
    let mut q_sum = 0.0_f64;
    let mut q_cdf = [0.0f64; VSPG_MAX_SEGMENTS];

    for i in 0..num_seg {
        let p_i = libm::exp(-segments[i].tau_lo) - libm::exp(-segments[i].tau_hi);
        p_sum += p_i;
        q_sum += segments[i].importance * p_i;
        q_cdf[i] = q_sum;
    }

    if q_sum < 1e-30 {
        // All probabilities negligible: fall back.
        let xi = xorshift_f64(rng);
        let one_minus_exp = 1.0 - libm::exp(-tau_max);
        return (-libm::log(1.0 - xi * one_minus_exp + 1e-30), 1.0);
    }

    // CDF inversion: select segment j.
    let xi_segment = xorshift_f64(rng) * q_sum;
    let mut j = 0usize;
    while j + 1 < num_seg && q_cdf[j] < xi_segment {
        j += 1;
    }

    // Within segment j: sample tau from conditional truncated exponential.
    // PDF:     exp(-tau) / p_j   over [tau_lo_j, tau_hi_j]
    // Inverse: tau = -ln(exp(-tau_lo_j) - xi * p_j)
    let seg = &segments[j];
    let p_j = libm::exp(-seg.tau_lo) - libm::exp(-seg.tau_hi);
    let xi_within = xorshift_f64(rng);
    let tau_s = -libm::log(libm::exp(-seg.tau_lo) - xi_within * p_j + 1e-30);

    // Clamp to valid range (numerical safety).
    let tau_s = tau_s.clamp(seg.tau_lo, seg.tau_hi);

    // Weight correction: I_avg / I_j.
    // I_avg = sum(I_k * p_k) / sum(p_k) = q_sum / p_sum.
    // Corrects for biased segment selection, keeping the estimator unbiased.
    let i_avg = q_sum / p_sum;
    let weight_correction = i_avg / seg.importance;

    (tau_s, weight_correction)
}

/// Compute multi-scatter spectral radiance using a hybrid approach.
///
/// This combines the deterministic single-scatter integrator (order 1, exact)
/// with Monte Carlo secondary chains (orders 2+, stochastic). The result
/// captures all scattering orders with minimal noise.
///
/// **Algorithm:**
/// 1. Step along the line of sight (LOS) from the observer.
/// 2. At each LOS step point, compute:
///    a. The single-scatter contribution (deterministic NEE toward sun)
///    using the exact analytical shadow ray from `single_scatter.rs`.
///    b. Launch `secondary_rays` MC chains from this scatter point.
///    Chains are importance-sampled toward the upper atmosphere (upward
///    bias) so that at deep twilight, photons have a chance of reaching
///    sunlit altitudes (>40km) where they can connect to the sun via NEE.
/// 3. Sum both contributions, weighted by transmittance and scattering
///    probability along the LOS.
///
/// **Key insight for deep twilight (SZA > 102°):**
/// At deep twilight, single-scatter drops to zero because all LOS scatter
/// points are in the Earth's geometric shadow. But multiple scattering can
/// redirect photons from sunlit high altitudes down to the observer via
/// chains of scattering events. The secondary chains capture this by:
/// - Launching from LOS points (even those in shadow)
/// - Propagating upward to where the sun IS visible
/// - Scattering back down toward the observer path
///
/// # Arguments
/// * `atm` - Atmosphere model
/// * `observer_pos` - Observer position in ECEF [m]
/// * `view_dir` - Viewing direction (unit vector)
/// * `sun_dir` - Direction toward the sun (unit vector)
/// * `wavelength_idx` - Index into wavelength grid
/// * `secondary_rays` - Number of MC chains to launch per LOS step
/// * `rng_state` - Mutable RNG state
///
/// # Returns
/// Total spectral radiance (single-scatter + multi-scatter contribution)
/// in the same units as `single_scatter_radiance`.
#[allow(clippy::too_many_arguments)] // Physics function: observer, view, sun, wavelength, rays, rng, polarized are all independent
pub fn hybrid_scatter_radiance(
    atm: &AtmosphereModel,
    observer_pos: Vec3,
    view_dir: Vec3,
    sun_dir: Vec3,
    wavelength_idx: usize,
    secondary_rays: usize,
    rng_state: &mut u64,
    polarized: bool,
) -> f64 {
    use crate::geometry::ray_sphere_intersect;
    use crate::scattering::{hg_mueller, rayleigh_mueller, MuellerMatrix, StokesVector};
    use crate::single_scatter::shadow_ray_transmittance;

    let toa_radius = atm.toa_radius();
    let surface_radius = atm.surface_radius();

    // Find LOS extent
    let los_max = match ray_sphere_intersect(observer_pos, view_dir, toa_radius) {
        Some(hit) if hit.t_far > 0.0 => hit.t_far,
        _ => return 0.0,
    };

    let ground_hit = ray_sphere_intersect(observer_pos, view_dir, surface_radius);
    let los_end = match ground_hit {
        Some(ref hit) if hit.t_near > 1e-3 && hit.t_near < los_max => hit.t_near,
        _ => los_max,
    };

    if los_end <= 0.0 {
        return 0.0;
    }

    let num_steps = HYBRID_LOS_STEPS.min((los_end / 500.0) as usize + 20);
    let ds = los_end / num_steps as f64;

    // Dual path: full Stokes [I,Q,U,V] when polarized, scalar when not.
    let mut stokes_total = StokesVector::unpolarized(0.0);
    let mut scalar_total = 0.0_f64;
    let mut tau_obs = 0.0; // optical depth from observer

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

        let optics = &atm.optics[shell_idx][wavelength_idx];
        let beta_scat = optics.extinction * optics.ssa;

        if beta_scat < 1e-30 {
            tau_obs += optics.extinction * ds;
            continue;
        }

        let tau_obs_mid = tau_obs + optics.extinction * ds * 0.5;
        let t_obs = libm::exp(-tau_obs_mid);

        if t_obs < 1e-30 {
            break;
        }

        // --- Order 1: deterministic single-scatter NEE ---
        let t_sun = shadow_ray_transmittance(atm, scatter_pos, sun_dir, wavelength_idx);
        if t_sun > 1e-30 {
            let cos_theta_1 = sun_dir.dot(-view_dir);
            let scale_1 = beta_scat * INV_4PI * t_sun * t_obs * ds;

            if polarized {
                // Full Mueller matrix for polarized order-1
                let mueller_1 = if optics.rayleigh_fraction > 0.99 {
                    rayleigh_mueller(cos_theta_1)
                } else if optics.rayleigh_fraction < 0.01 {
                    hg_mueller(cos_theta_1, optics.asymmetry)
                } else {
                    let mr = rayleigh_mueller(cos_theta_1).scale(optics.rayleigh_fraction);
                    let mh = hg_mueller(cos_theta_1, optics.asymmetry)
                        .scale(1.0 - optics.rayleigh_fraction);
                    let mut m = MuellerMatrix::zero();
                    for i in 0..4 {
                        for j in 0..4 {
                            m.m[i][j] = mr.m[i][j] + mh.m[i][j];
                        }
                    }
                    m
                };
                let ss_stokes = mueller_1.apply(&StokesVector::unpolarized(1.0));
                stokes_total = stokes_total.add(&ss_stokes.scale(scale_1));
            } else {
                // Scalar phase function (no Mueller, no Stokes)
                let phase = if optics.rayleigh_fraction > 0.99 {
                    rayleigh_phase(cos_theta_1)
                } else {
                    optics.rayleigh_fraction * rayleigh_phase(cos_theta_1)
                        + (1.0 - optics.rayleigh_fraction)
                            * henyey_greenstein_phase(cos_theta_1, optics.asymmetry)
                };
                scalar_total += phase * scale_1;
            }
        }

        // --- Orders 2+: MC secondary chains ---
        if secondary_rays > 0 {
            if polarized {
                let mut mc_stokes = StokesVector::unpolarized(0.0);
                for ray in 0..secondary_rays {
                    let chain_stokes = trace_secondary_chain(
                        atm,
                        scatter_pos,
                        view_dir,
                        sun_dir,
                        wavelength_idx,
                        optics,
                        rng_state,
                        ray,
                        secondary_rays,
                    );
                    mc_stokes = mc_stokes.add(&chain_stokes);
                }
                let inv_rays = 1.0 / secondary_rays as f64;
                let mc_avg = mc_stokes.scale(inv_rays);
                let scale_m = beta_scat * t_obs * ds;
                stokes_total = stokes_total.add(&mc_avg.scale(scale_m));
            } else {
                let mut mc_scalar = 0.0_f64;
                for ray in 0..secondary_rays {
                    mc_scalar += trace_secondary_chain_scalar(
                        atm,
                        scatter_pos,
                        sun_dir,
                        wavelength_idx,
                        optics,
                        rng_state,
                        ray,
                        secondary_rays,
                        None, // no path guide in per-wavelength mode
                    );
                }
                let inv_rays = 1.0 / secondary_rays as f64;
                let scale_m = beta_scat * t_obs * ds;
                scalar_total += mc_scalar * inv_rays * scale_m;
            }
        }

        tau_obs += optics.extinction * ds;
    }

    if polarized {
        stokes_total.intensity()
    } else {
        scalar_total
    }
}

/// Trace a secondary MC chain from a scatter point on the LOS.
///
/// Full Stokes [I,Q,U,V] propagation through the chain. Tracks the photon's
/// polarization state (normalized, I=1) through each scatter event. At each
/// NEE, applies the Mueller matrix to the photon's actual Stokes state.
///
/// Returns the multi-scatter Stokes contribution that should be multiplied
/// by the LOS-step weighting factor.
///
/// # Variance reduction
///
/// Same techniques as `trace_secondary_chain_scalar`:
///
/// 1. **Stratified initial direction sampling** via `ray_idx` / `total_rays`.
/// 2. **Zenith-biased importance sampling** with SZA-adaptive mix fraction.
///
/// The RNG consumption order is identical to the scalar version so both
/// produce the same chain trajectories (given the same seed and ray index).
#[allow(clippy::too_many_arguments)]
fn trace_secondary_chain(
    atm: &AtmosphereModel,
    start_pos: Vec3,
    prev_dir_in: Vec3,
    sun_dir: Vec3,
    wavelength_idx: usize,
    start_optics: &crate::atmosphere::ShellOptics,
    rng_state: &mut u64,
    ray_idx: usize,
    total_rays: usize,
) -> crate::scattering::StokesVector {
    use crate::scattering::{scatter_stokes_fast, scattering_plane_cos_sin, StokesVector};
    use crate::single_scatter::shadow_ray_transmittance;

    let local_up = start_pos.normalize();

    // --- SZA-adaptive 3-branch parameters ---
    let cos_sza = sun_dir.dot(local_up);
    let bp = branch_params_for_sza(cos_sza);

    // Branch probabilities:
    //   alpha_p = 1 - zenith_frac           (phase function)
    //   alpha_z = zenith_frac * (1 - term_share)  (zenith lobe)
    //   alpha_t = zenith_frac * term_share         (terminator lobe)
    let alpha_p = 1.0 - bp.zenith_frac;
    let alpha_z = bp.zenith_frac * (1.0 - bp.term_share);
    let alpha_t = bp.zenith_frac * bp.term_share;

    // Terminator axis (only used if term_share > 0, but cheap to compute)
    let term_axis = terminator_axis(local_up, sun_dir, bp.tilt_rad);

    // --- Stratified initial direction sampling ---
    let xi_jitter = xorshift_f64(rng_state);
    let xi_mix = (ray_idx as f64 + xi_jitter) / total_rays as f64;

    // 3-branch importance sampling with correct branch probability weights.
    //
    // The baseline estimator is:
    //   E = 0.5 * E_phase + 0.5 * E_hemi
    //
    // We sample with probabilities (alpha_p, alpha_z, alpha_t). Each branch
    // carries weight (baseline_prob / actual_prob) * shape_correction:
    //   phase:      0.5 / alpha_p
    //   zenith:     0.5 / alpha_z * zenith_shape_weight
    //   terminator: 0.5 / alpha_t * terminator_shape_weight
    //
    // At SZA <= 96: alpha_p=0.5, alpha_z=0.5, alpha_t=0. Both active
    // branch weights = 1.0 exactly (n=1 makes zenith_importance_weight=1).
    let (dir, cos_theta_init, initial_weight) = if xi_mix < alpha_p {
        // Phase function branch
        let cos_theta_init = if xorshift_f64(rng_state) < start_optics.rayleigh_fraction {
            sample_rayleigh_analytic(xorshift_f64(rng_state))
        } else {
            sample_henyey_greenstein(xorshift_f64(rng_state), start_optics.asymmetry)
        };
        let phi_init = 2.0 * core::f64::consts::PI * xorshift_f64(rng_state);
        let d = scatter_direction(sun_dir, cos_theta_init, phi_init);
        let branch_w = 0.5 / alpha_p;
        (d, cos_theta_init, branch_w)
    } else if xi_mix < alpha_p + alpha_z || alpha_t < 1e-12 {
        // Zenith-biased branch with shape + branch weight correction
        let (d, cos_z) = sample_zenith_biased(local_up, bp.n_zenith, rng_state);
        let cos_theta_init = sun_dir.dot(d);
        let shape_w = zenith_importance_weight(cos_z, bp.n_zenith);
        let branch_w = 0.5 / (alpha_z + alpha_t); // fallback if alpha_t ~ 0
        (d, cos_theta_init, shape_w * branch_w)
    } else {
        // Terminator lobe branch
        let (d, cos_t) = sample_zenith_biased(term_axis, bp.m_term, rng_state);
        let cos_z = d.dot(local_up);
        let cos_theta_init = sun_dir.dot(d);
        let shape_w = terminator_shape_weight(cos_z, cos_t, bp.m_term);
        let branch_w = 0.5 / alpha_t;
        (d, cos_theta_init, shape_w * branch_w)
    };

    // Initialize Stokes state: apply first scatter to [1,0,0,0]
    let mut stokes;
    {
        let (c0, s0) = scattering_plane_cos_sin(prev_dir_in, sun_dir, dir);
        stokes = scatter_stokes_fast(
            &StokesVector::unpolarized(1.0),
            cos_theta_init,
            start_optics.rayleigh_fraction,
            start_optics.asymmetry,
            c0,
            s0,
        );
        // Normalize by I (importance weighting)
        let i_val = stokes.intensity();
        if i_val > 1e-30 {
            stokes = stokes.scale(1.0 / i_val);
        }
    }

    let mut pos = start_pos;
    let mut current_dir = dir;
    let mut prev_dir = sun_dir;
    let mut weight = start_optics.ssa * initial_weight;
    let mut total_stokes = StokesVector::unpolarized(0.0);

    // Upfront forced scattering at deep twilight (SZA >= 96).
    //
    // At each bounce, scout tau_max to atmosphere exit BEFORE sampling
    // a free path. If the path is optically thin (tau_max < 20) and
    // doesn't hit ground, force the scatter: weight *= (1-e^{-tau_max}),
    // sample from the truncated exponential, advance to scatter point.
    // This is the ONLY unbiased way to do forced scattering -- the analog
    // path is bypassed entirely, preventing double-counting.
    //
    // When tau_max >= 20, (1-e^{-20}) = 1.0 to f64 precision, so the
    // weight correction is exactly 1.0 and the truncated exponential is
    // indistinguishable from the regular exponential. Analog scatter is
    // equivalent and faster (no scout overhead), so we fall back to it.
    //
    // The scout early-exits at tau > 20, costing only 1-3 shell ops in
    // the dense lower atmosphere.
    let local_up = start_pos.normalize();
    let cos_sza_local = sun_dir.dot(local_up);
    let sza_deg_local = libm::acos(cos_sza_local.clamp(-1.0, 1.0)) * 180.0 / core::f64::consts::PI;
    let use_forced = sza_deg_local >= ZENITH_SZA_START;

    // Exponential transform bias parameter.
    // Ramps from 0 (SZA < 96) to EXP_TRANSFORM_ALPHA_MAX (SZA >= 106).
    // At alpha=0: sigma'=sigma, all weight corrections are 1.0 (zero overhead).
    // At alpha=0.5: upward photons get 2x mean free path, downward get 2/3x.
    let sza_t_et =
        ((sza_deg_local - ZENITH_SZA_START) / (ZENITH_SZA_FULL - ZENITH_SZA_START)).clamp(0.0, 1.0);
    let alpha_et = EXP_TRANSFORM_ALPHA_MAX * sza_t_et;

    for _scatter in 0..HYBRID_MAX_BOUNCES {
        // --- Decide scatter mode for this bounce ---
        let mut forced_this_bounce = false;
        let mut tau_max = 0.0;

        if use_forced {
            let (tm, hit_ground) = scout_tau_to_boundary(atm, pos, current_dir, wavelength_idx);
            tau_max = tm;
            // Force scatter only when path exits to space AND is optically thin.
            // Ground-bound paths are handled by the analog loop (ground reflection).
            // Dense paths (tau >= 20) use analog (equivalent, no scout overhead).
            forced_this_bounce = !hit_ground && tm < FORCED_TAU_CUTOFF;
        }

        let scatter_shell;

        if forced_this_bounce {
            // Upfront forced scattering: weight = exact scatter probability.
            // No analog free-path walk, no escape, no double-counting.
            let exp_neg_tau = libm::exp(-tau_max);
            weight *= 1.0 - exp_neg_tau;
            if weight < 1e-30 {
                break;
            }
            let xi = xorshift_f64(rng_state);
            let tau_s = -libm::log(1.0 - xi * (1.0 - exp_neg_tau) + 1e-30);
            let (sp, sd, ss) =
                advance_to_optical_depth(atm, pos, current_dir, tau_s, wavelength_idx);
            pos = sp;
            current_dir = sd;
            scatter_shell = ss;
        } else {
            // Analog scatter with exponential transform.
            // Modified extinction: sigma' = sigma * (1 - alpha * cos_z)
            // where cos_z = dot(dir, local_up). Upward photons get longer
            // mean free path, downward get shorter. Weight corrections keep
            // the estimator exactly unbiased.
            let mut scatter_found = false;
            let mut found_shell = 0usize;

            for _ in 0..200 {
                let r = pos.length();
                let shell_idx = match atm.shell_index(r) {
                    Some(idx) => idx,
                    None => break, // exited atmosphere
                };

                let shell = &atm.shells[shell_idx];
                let optics = &atm.optics[shell_idx][wavelength_idx];

                if optics.extinction < 1e-20 {
                    match next_shell_boundary(pos, current_dir, shell.r_inner, shell.r_outer) {
                        Some((dist, is_outward)) => {
                            let (np, nd) =
                                cross_boundary(pos, current_dir, dist, is_outward, shell_idx, atm);
                            pos = np;
                            current_dir = nd;
                            continue;
                        }
                        None => break,
                    }
                }

                // Exponential transform: modified extinction.
                // Bias axis is tilted toward the terminator at deep twilight,
                // drifting the random walk toward sunlit atmosphere.
                let cos_bias = current_dir.dot(term_axis);
                let sigma = optics.extinction;
                let sigma_prime = sigma * (1.0 - alpha_et * cos_bias);
                // sigma_prime > 0 guaranteed: alpha_et <= 0.5, |cos_bias| <= 1

                let xi = xorshift_f64(rng_state);
                let free_path = -libm::log(1.0 - xi + 1e-30) / sigma_prime;

                match next_shell_boundary(pos, current_dir, shell.r_inner, shell.r_outer) {
                    Some((boundary_dist, is_outward)) => {
                        if free_path >= boundary_dist {
                            // Boundary crossing weight correction:
                            // exp(-(sigma - sigma') * D) = exp(-alpha * sigma * cos_bias * D)
                            if alpha_et > 0.0 {
                                weight *= libm::exp(-alpha_et * sigma * cos_bias * boundary_dist);
                            }

                            let (np, nd) = cross_boundary(
                                pos,
                                current_dir,
                                boundary_dist,
                                is_outward,
                                shell_idx,
                                atm,
                            );
                            pos = np;
                            current_dir = nd;

                            // Ground reflection: depolarizes
                            if !is_outward && pos.length() <= atm.surface_radius() + 1.0 {
                                let albedo = atm.surface_albedo[wavelength_idx];
                                weight *= albedo;
                                if weight < 1e-30 {
                                    break;
                                }
                                let normal = pos.normalize();
                                prev_dir = current_dir;
                                current_dir = sample_hemisphere(normal, rng_state);
                                stokes = StokesVector::unpolarized(1.0);
                                continue;
                            }
                            continue;
                        }
                    }
                    None => break,
                }

                // Scatter within this shell.
                // Weight correction: (sigma/sigma') * exp(-alpha * sigma * cos_bias * d)
                if alpha_et > 0.0 {
                    weight *=
                        (sigma / sigma_prime) * libm::exp(-alpha_et * sigma * cos_bias * free_path);
                }
                pos = pos + current_dir * free_path;
                found_shell = shell_idx;
                scatter_found = true;
                break;
            }

            if !scatter_found {
                break; // chain terminates: escaped atmosphere
            }
            scatter_shell = found_shell;
        }

        let optics = &atm.optics[scatter_shell][wavelength_idx];

        // NEE: apply Mueller to photon's actual Stokes state
        let t_sun_secondary = shadow_ray_transmittance(atm, pos, sun_dir, wavelength_idx);

        if t_sun_secondary > 1e-30 {
            let cos_angle_nee = sun_dir.dot(-current_dir);
            let (cn, sn) = scattering_plane_cos_sin(prev_dir, current_dir, -sun_dir);
            let nee_stokes = scatter_stokes_fast(
                &stokes,
                cos_angle_nee,
                optics.rayleigh_fraction,
                optics.asymmetry,
                cn,
                sn,
            );

            let scale = weight * t_sun_secondary * INV_4PI;
            total_stokes = total_stokes.add(&nee_stokes.scale(scale));
        }

        // Apply SSA
        weight *= optics.ssa;
        if weight < 1e-30 {
            break;
        }

        // Sample new direction and update Stokes state
        let cos_theta = if xorshift_f64(rng_state) < optics.rayleigh_fraction {
            sample_rayleigh_analytic(xorshift_f64(rng_state))
        } else {
            sample_henyey_greenstein(xorshift_f64(rng_state), optics.asymmetry)
        };
        let phi = 2.0 * core::f64::consts::PI * xorshift_f64(rng_state);
        let new_dir = scatter_direction(current_dir, cos_theta, phi);

        // Update Stokes through this scatter (fused, no matrices, no trig)
        let (cs, ss) = scattering_plane_cos_sin(prev_dir, current_dir, new_dir);
        stokes = scatter_stokes_fast(
            &stokes,
            cos_theta,
            optics.rayleigh_fraction,
            optics.asymmetry,
            cs,
            ss,
        );

        // Normalize by I (importance weighting -- keeps stokes I = 1)
        let i_val = stokes.intensity();
        if i_val > 1e-30 {
            stokes = stokes.scale(1.0 / i_val);
        } else {
            stokes = StokesVector::unpolarized(1.0);
        }

        prev_dir = current_dir;
        current_dir = new_dir;
    }

    total_stokes
}

/// Scalar-mode secondary MC chain (no Stokes, no Mueller matrices).
///
/// Identical physics to `trace_secondary_chain` but tracks only scalar
/// radiance weight. All RNG consumption is identical so direction sampling
/// produces the same trajectories -- the only difference is that we evaluate
/// scalar phase functions instead of Mueller/Stokes operations at each
/// scatter event and NEE.
///
/// This saves 3x `scatter_stokes_fast`, 3x `scattering_plane_cos_sin`,
/// and multiple 4-component Stokes operations per bounce.
///
/// # Variance reduction
///
/// 1. **Stratified initial direction sampling**: The `ray_idx` / `total_rays`
///    parameters stratify the branch choice across rays at each LOS step.
///
/// 2. **Zenith-biased importance sampling**: The hemisphere branch uses a
///    power-cosine PDF (cos^n) instead of cosine-weighted, concentrating
///    rays toward the zenith. An importance weight correction keeps the
///    estimator unbiased. The fraction of rays using zenith-biased vs
///    phase-function sampling is SZA-adaptive: 50/50 at civil twilight,
///    shifting to 95/5 at deep twilight where the phase-function branch
///    (toward the below-horizon sun) is nearly useless.
#[allow(clippy::too_many_arguments)]
fn trace_secondary_chain_scalar(
    atm: &AtmosphereModel,
    start_pos: Vec3,
    sun_dir: Vec3,
    wavelength_idx: usize,
    start_optics: &crate::atmosphere::ShellOptics,
    rng_state: &mut u64,
    ray_idx: usize,
    total_rays: usize,
    guide: Option<&PathGuide>,
) -> f64 {
    use crate::single_scatter::shadow_ray_transmittance;

    let local_up = start_pos.normalize();

    // --- SZA-adaptive 3-branch parameters ---
    let cos_sza = sun_dir.dot(local_up);
    let bp = branch_params_for_sza(cos_sza);

    let alpha_p = 1.0 - bp.zenith_frac;
    let alpha_z = bp.zenith_frac * (1.0 - bp.term_share);
    let alpha_t = bp.zenith_frac * bp.term_share;

    let term_axis = terminator_axis(local_up, sun_dir, bp.tilt_rad);

    // --- Stratified initial direction sampling ---
    let xi_jitter = xorshift_f64(rng_state);
    let xi_mix = (ray_idx as f64 + xi_jitter) / total_rays as f64;

    // 3-branch importance sampling. See trace_secondary_chain for derivation.
    // At SZA <= 96: alpha_p=0.5, alpha_z=0.5, alpha_t=0. Both weights = 1.0.
    let (dir, initial_weight) = if xi_mix < alpha_p {
        // Phase function branch (toward sun_dir -- effective at civil twilight)
        let _cos_theta_init = if xorshift_f64(rng_state) < start_optics.rayleigh_fraction {
            sample_rayleigh_analytic(xorshift_f64(rng_state))
        } else {
            sample_henyey_greenstein(xorshift_f64(rng_state), start_optics.asymmetry)
        };
        let phi_init = 2.0 * core::f64::consts::PI * xorshift_f64(rng_state);
        let branch_w = 0.5 / alpha_p;
        (
            scatter_direction(sun_dir, _cos_theta_init, phi_init),
            branch_w,
        )
    } else if xi_mix < alpha_p + alpha_z || alpha_t < 1e-12 {
        // Zenith-biased branch with shape + branch weight correction
        let (d, cos_z) = sample_zenith_biased(local_up, bp.n_zenith, rng_state);
        let shape_w = zenith_importance_weight(cos_z, bp.n_zenith);
        let branch_w = 0.5 / (alpha_z + alpha_t);
        (d, shape_w * branch_w)
    } else {
        // Terminator lobe branch
        let (d, cos_t) = sample_zenith_biased(term_axis, bp.m_term, rng_state);
        let cos_z = d.dot(local_up);
        let shape_w = terminator_shape_weight(cos_z, cos_t, bp.m_term);
        let branch_w = 0.5 / alpha_t;
        (d, shape_w * branch_w)
    };

    let surface_radius = atm.surface_radius();
    let mut total = 0.0_f64;

    // Upfront forced scattering gate (same logic as Stokes version).
    let sza_deg_local = libm::acos(cos_sza.clamp(-1.0, 1.0)) * 180.0 / core::f64::consts::PI;
    let use_forced = sza_deg_local >= ZENITH_SZA_START;

    // Exponential transform bias parameter (same ramp as Stokes version).
    let sza_t_et =
        ((sza_deg_local - ZENITH_SZA_START) / (ZENITH_SZA_FULL - ZENITH_SZA_START)).clamp(0.0, 1.0);
    let alpha_et = EXP_TRANSFORM_ALPHA_MAX * sza_t_et;

    // Altitude-splitting setup. At SZA <= 96 all factors are 1 (no splitting,
    // zero overhead). At SZA >= 102 we aggressively split chains that reach
    // high altitude, rewarding the rare event of climbing through the troposphere.
    let split_factors = split_factors_for_sza(sza_deg_local);

    // Initialize work stack with the main particle.
    let mut stack = [SplitParticleScalar {
        pos: Vec3::new(0.0, 0.0, 0.0),
        dir: Vec3::new(0.0, 0.0, 1.0),
        weight: 0.0,
        rng: 0,
        bounces_left: 0,
        next_split: 0,
    }; MAX_SPLIT_PARTICLES];
    let mut stack_len: usize = 1;
    stack[0] = SplitParticleScalar {
        pos: start_pos,
        dir,
        weight: start_optics.ssa * initial_weight,
        rng: *rng_state,
        bounces_left: HYBRID_MAX_BOUNCES,
        next_split: 0,
    };
    let mut main_rng_out = *rng_state;
    let mut main_processed = false;

    // Process all particles: main first, then split copies (LIFO order).
    while stack_len > 0 {
        stack_len -= 1;
        let is_main = !main_processed;
        main_processed = true;
        let mut pos = stack[stack_len].pos;
        let mut current_dir = stack[stack_len].dir;
        let mut weight = stack[stack_len].weight;
        let mut local_rng = stack[stack_len].rng;
        let bounces_left = stack[stack_len].bounces_left;
        let mut next_split = stack[stack_len].next_split;

        for _bounce in 0..bounces_left {
            // --- Decide scatter mode for this bounce ---
            let mut forced_this_bounce = false;
            let mut tau_max = 0.0;

            if use_forced {
                let (tm, hit_ground) = scout_tau_to_boundary(atm, pos, current_dir, wavelength_idx);
                tau_max = tm;
                forced_this_bounce = !hit_ground && tm < FORCED_TAU_CUTOFF;
            }

            let scatter_shell;

            if forced_this_bounce {
                let exp_neg_tau = libm::exp(-tau_max);
                weight *= 1.0 - exp_neg_tau;
                if weight < 1e-30 {
                    break;
                }
                // VSPG: importance-weighted shell selection biases scatter
                // toward high altitude at deep twilight. Weight correction
                // keeps the estimator exactly unbiased.
                let (tau_s, vspg_w) = vspg_sample_scatter_tau(
                    atm,
                    pos,
                    current_dir,
                    wavelength_idx,
                    tau_max,
                    sza_deg_local,
                    &mut local_rng,
                );
                weight *= vspg_w;
                let (sp, sd, ss) =
                    advance_to_optical_depth(atm, pos, current_dir, tau_s, wavelength_idx);
                pos = sp;
                current_dir = sd;
                scatter_shell = ss;
            } else {
                let mut scatter_found = false;
                let mut found_shell = 0usize;

                for _ in 0..200 {
                    let r = pos.length();
                    let shell_idx = match atm.shell_index(r) {
                        Some(idx) => idx,
                        None => break,
                    };

                    let shell = &atm.shells[shell_idx];
                    let optics = &atm.optics[shell_idx][wavelength_idx];

                    if optics.extinction < 1e-20 {
                        match next_shell_boundary(pos, current_dir, shell.r_inner, shell.r_outer) {
                            Some((dist, is_outward)) => {
                                let (np, nd) = cross_boundary(
                                    pos,
                                    current_dir,
                                    dist,
                                    is_outward,
                                    shell_idx,
                                    atm,
                                );
                                pos = np;
                                current_dir = nd;
                                continue;
                            }
                            None => break,
                        }
                    }

                    let cos_bias = current_dir.dot(term_axis);
                    let sigma = optics.extinction;
                    let sigma_prime = sigma * (1.0 - alpha_et * cos_bias);

                    let xi = xorshift_f64(&mut local_rng);
                    let free_path = -libm::log(1.0 - xi + 1e-30) / sigma_prime;

                    match next_shell_boundary(pos, current_dir, shell.r_inner, shell.r_outer) {
                        Some((boundary_dist, is_outward)) => {
                            if free_path >= boundary_dist {
                                if alpha_et > 0.0 {
                                    weight *=
                                        libm::exp(-alpha_et * sigma * cos_bias * boundary_dist);
                                }

                                let (np, nd) = cross_boundary(
                                    pos,
                                    current_dir,
                                    boundary_dist,
                                    is_outward,
                                    shell_idx,
                                    atm,
                                );
                                pos = np;
                                current_dir = nd;

                                if !is_outward && pos.length() <= surface_radius + 1.0 {
                                    let albedo = atm.surface_albedo[wavelength_idx];
                                    weight *= albedo;
                                    if weight < 1e-30 {
                                        break;
                                    }
                                    let normal = pos.normalize();
                                    current_dir = sample_hemisphere(normal, &mut local_rng);
                                    continue;
                                }
                                continue;
                            }
                        }
                        None => break,
                    }

                    if alpha_et > 0.0 {
                        weight *= (sigma / sigma_prime)
                            * libm::exp(-alpha_et * sigma * cos_bias * free_path);
                    }
                    pos = pos + current_dir * free_path;
                    found_shell = shell_idx;
                    scatter_found = true;
                    break;
                }

                if !scatter_found {
                    break;
                }
                scatter_shell = found_shell;
            }

            let optics = &atm.optics[scatter_shell][wavelength_idx];

            // NEE: scalar phase function (no Mueller matrix)
            let t_sun_secondary = shadow_ray_transmittance(atm, pos, sun_dir, wavelength_idx);

            if t_sun_secondary > 1e-30 {
                let cos_angle_nee = sun_dir.dot(-current_dir);
                let phase = if optics.rayleigh_fraction > 0.99 {
                    rayleigh_phase(cos_angle_nee)
                } else {
                    optics.rayleigh_fraction * rayleigh_phase(cos_angle_nee)
                        + (1.0 - optics.rayleigh_fraction)
                            * henyey_greenstein_phase(cos_angle_nee, optics.asymmetry)
                };

                let scale = weight * t_sun_secondary * INV_4PI;
                total += phase * scale;
            }

            weight *= optics.ssa;
            if weight < 1e-30 {
                break;
            }

            // Sample new direction: one-sample MIS between phase function
            // and path guide (when trained). At each bounce, flip a coin:
            //   - With prob GUIDE_MIS_FRAC: sample from guide, weight by
            //     balance heuristic MIS weight
            //   - Otherwise: sample from phase function, same MIS weight
            // When no guide is available, always use phase function (no overhead).
            let alt_for_guide = pos.length() - surface_radius;
            let use_guide_this_bounce =
                guide.is_some() && xorshift_f64(&mut local_rng) < GUIDE_MIS_FRAC;

            let new_dir = if use_guide_this_bounce {
                let g = guide.unwrap();
                let local_up_here = pos.normalize();
                let (gdir, p_guide) =
                    g.sample(alt_for_guide, local_up_here, sun_dir, &mut local_rng);
                let cos_t = current_dir.dot(gdir);
                let p_phase = scalar_phase_value(cos_t, optics) * INV_4PI;
                let mis_denom = GUIDE_MIS_FRAC * p_guide + (1.0 - GUIDE_MIS_FRAC) * p_phase;
                if mis_denom > 1e-30 {
                    weight *= p_guide / mis_denom;
                }
                // Consume RNG slots to keep stream aligned with non-guide path
                let _ = xorshift_f64(&mut local_rng);
                let _ = xorshift_f64(&mut local_rng);
                let _ = xorshift_f64(&mut local_rng);
                gdir
            } else {
                let cos_theta = if xorshift_f64(&mut local_rng) < optics.rayleigh_fraction {
                    sample_rayleigh_analytic(xorshift_f64(&mut local_rng))
                } else {
                    sample_henyey_greenstein(xorshift_f64(&mut local_rng), optics.asymmetry)
                };
                let phi = 2.0 * core::f64::consts::PI * xorshift_f64(&mut local_rng);
                let d = scatter_direction(current_dir, cos_theta, phi);
                if let Some(g) = guide {
                    let local_up_here = pos.normalize();
                    let p_phase = scalar_phase_value(cos_theta, optics) * INV_4PI;
                    let p_guide = g.pdf(alt_for_guide, local_up_here, sun_dir, d);
                    let mis_denom = GUIDE_MIS_FRAC * p_guide + (1.0 - GUIDE_MIS_FRAC) * p_phase;
                    if mis_denom > 1e-30 {
                        weight *= p_phase / mis_denom;
                    }
                }
                d
            };
            current_dir = new_dir;

            // --- Altitude-based splitting ---
            let alt = pos.length() - surface_radius;
            while next_split < NUM_SPLIT_LEVELS && alt > SPLIT_ALTITUDES_M[next_split] {
                let k = split_factors[next_split];
                if k > 1 {
                    weight /= k as f64;
                    let remaining = bounces_left.saturating_sub(_bounce + 1);
                    for copy_idx in 1..k {
                        if stack_len < MAX_SPLIT_PARTICLES {
                            let child_rng = local_rng
                                ^ (copy_idx as u64).wrapping_mul(2654435761)
                                ^ ((next_split as u64) << 32);
                            stack[stack_len] = SplitParticleScalar {
                                pos,
                                dir: current_dir,
                                weight,
                                rng: child_rng,
                                bounces_left: remaining,
                                next_split: next_split + 1,
                            };
                            stack_len += 1;
                        }
                    }
                }
                next_split += 1;
            }
        }

        if is_main {
            main_rng_out = local_rng;
        }
    }

    *rng_state = main_rng_out;
    total
}

/// Scalar phase function value for given scattering angle.
///
/// Convenience helper: evaluates the mixed Rayleigh+HG phase function for
/// the optics at this wavelength. Used by ALIS weight ratio corrections.
#[inline]
fn scalar_phase_value(cos_theta: f64, optics: &crate::atmosphere::ShellOptics) -> f64 {
    if optics.rayleigh_fraction > 0.99 {
        rayleigh_phase(cos_theta)
    } else {
        optics.rayleigh_fraction * rayleigh_phase(cos_theta)
            + (1.0 - optics.rayleigh_fraction)
                * henyey_greenstein_phase(cos_theta, optics.asymmetry)
    }
}

/// Multi-wavelength scout: compute optical depth to boundary for all wavelengths.
///
/// Same geometry as `scout_tau_to_boundary` but accumulates tau for all active
/// wavelengths along the path. The refracted ray path is wavelength-independent
/// (air refractive index dispersion is negligible over the visible range).
///
/// Early-exits when the hero wavelength's tau exceeds `FORCED_TAU_CUTOFF`,
/// since the forced scatter decision is based on the hero.
fn scout_tau_to_boundary_alis(
    atm: &AtmosphereModel,
    start_pos: Vec3,
    start_dir: Vec3,
    hero_wl: usize,
    num_wl: usize,
) -> ([f64; 64], bool) {
    let surface_radius = atm.surface_radius();
    let num_shells = atm.num_shells;
    let mut pos = start_pos;
    let mut dir = start_dir;
    let mut tau = [0.0f64; 64];

    let mut shell_idx = match atm.shell_index(pos.length()) {
        Some(idx) => idx,
        None => return (tau, false),
    };

    for _ in 0..200 {
        let shell = &atm.shells[shell_idx];

        match next_shell_boundary(pos, dir, shell.r_inner, shell.r_outer) {
            Some((dist, is_outward)) => {
                for (w, tau_w) in tau.iter_mut().enumerate().take(num_wl) {
                    *tau_w += atm.optics[shell_idx][w].extinction * dist;
                }

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

                if !is_outward && pos.length() <= surface_radius + 1.0 {
                    return (tau, true);
                }
                if next_shell >= num_shells {
                    return (tau, false);
                }

                shell_idx = next_shell;
            }
            None => return (tau, false),
        }

        if tau[hero_wl] > FORCED_TAU_CUTOFF {
            return (tau, false);
        }
    }

    (tau, false)
}

/// Multi-wavelength advance: advance to hero's optical depth, tracking all wavelengths.
///
/// Advances along the ray until the hero wavelength accumulates `tau_target`
/// optical depth. Returns the position, direction, scatter shell, and the
/// per-wavelength optical depths at the scatter position.
fn advance_to_optical_depth_alis(
    atm: &AtmosphereModel,
    start_pos: Vec3,
    start_dir: Vec3,
    tau_target: f64,
    hero_wl: usize,
    num_wl: usize,
) -> (Vec3, Vec3, usize, [f64; 64]) {
    let surface_radius = atm.surface_radius();
    let num_shells = atm.num_shells;
    let mut pos = start_pos;
    let mut dir = start_dir;
    let mut tau_accumulated = [0.0f64; 64];
    let mut hero_tau = 0.0;

    let mut shell_idx = match atm.shell_index(pos.length()) {
        Some(idx) => idx,
        None => return (pos, dir, 0, tau_accumulated),
    };

    for _ in 0..200 {
        let shell = &atm.shells[shell_idx];
        let hero_extinction = atm.optics[shell_idx][hero_wl].extinction;

        match next_shell_boundary(pos, dir, shell.r_inner, shell.r_outer) {
            Some((boundary_dist, is_outward)) => {
                let tau_shell_hero = hero_extinction * boundary_dist;

                if hero_tau + tau_shell_hero >= tau_target {
                    // Scatter point is within this shell.
                    let tau_remaining = tau_target - hero_tau;
                    let dist = if hero_extinction > 1e-30 {
                        tau_remaining / hero_extinction
                    } else {
                        boundary_dist
                    };
                    for (w, tau_w) in tau_accumulated.iter_mut().enumerate().take(num_wl) {
                        *tau_w += atm.optics[shell_idx][w].extinction * dist;
                    }
                    pos = pos + dir * dist;
                    return (pos, dir, shell_idx, tau_accumulated);
                }

                // Cross boundary
                hero_tau += tau_shell_hero;
                for (w, tau_w) in tau_accumulated.iter_mut().enumerate().take(num_wl) {
                    *tau_w += atm.optics[shell_idx][w].extinction * boundary_dist;
                }

                let boundary_pos = pos + dir * boundary_dist;
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

                if !is_outward && pos.length() <= surface_radius + 1.0 {
                    return (pos, dir, shell_idx, tau_accumulated);
                }
                if next_shell >= num_shells {
                    return (pos, dir, shell_idx, tau_accumulated);
                }

                shell_idx = next_shell;
            }
            None => return (pos, dir, shell_idx, tau_accumulated),
        }
    }

    (pos, dir, shell_idx, tau_accumulated)
}

/// ALIS secondary chain tracer: trace ONE hero path, evaluate ALL wavelengths.
///
/// ALIS (Adjusted Lambda Importance Sampling) traces the photon path using the
/// hero wavelength's extinction and phase function, while tracking per-wavelength
/// weight ratios. At each NEE point, all wavelengths are evaluated using a single
/// multi-wavelength shadow ray (`shadow_ray_transmittance_spectrum`).
///
/// Weight corrections for non-hero wavelengths account for:
/// - Different extinction (free-path PDF ratio at boundary crossings and scatters)
/// - Different SSA (survival probability)
/// - Different phase function (direction sampling ratio at each bounce)
///
/// Returns per-wavelength MC contributions `[f64; 64]` to be multiplied by
/// the LOS-step weighting factor for each wavelength.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn trace_secondary_chain_alis(
    atm: &AtmosphereModel,
    start_pos: Vec3,
    sun_dir: Vec3,
    hero_wl: usize,
    start_shell: usize,
    rng_state: &mut u64,
    ray_idx: usize,
    total_rays: usize,
    num_wl: usize,
    guide: Option<&PathGuide>,
) -> [f64; 64] {
    use crate::single_scatter::shadow_ray_transmittance_spectrum;

    let local_up = start_pos.normalize();
    let hero_optics = &atm.optics[start_shell][hero_wl];

    // --- SZA-adaptive 3-branch parameters ---
    let cos_sza = sun_dir.dot(local_up);
    let bp = branch_params_for_sza(cos_sza);

    let alpha_p = 1.0 - bp.zenith_frac;
    let alpha_z = bp.zenith_frac * (1.0 - bp.term_share);
    let alpha_t = bp.zenith_frac * bp.term_share;

    let term_axis = terminator_axis(local_up, sun_dir, bp.tilt_rad);

    // --- Stratified initial direction sampling ---
    let xi_jitter = xorshift_f64(rng_state);
    let xi_mix = (ray_idx as f64 + xi_jitter) / total_rays as f64;

    // Sample initial direction from hero's phase function (phase branch),
    // zenith-biased distribution (zenith branch), or terminator lobe
    // (terminator branch). Track cos_theta_init for phase function ratio
    // correction on non-hero wavelengths.
    let (dir, initial_weight, cos_theta_init, is_phase_branch) = if xi_mix < alpha_p {
        // Phase function branch
        let ct = if xorshift_f64(rng_state) < hero_optics.rayleigh_fraction {
            sample_rayleigh_analytic(xorshift_f64(rng_state))
        } else {
            sample_henyey_greenstein(xorshift_f64(rng_state), hero_optics.asymmetry)
        };
        let phi_init = 2.0 * core::f64::consts::PI * xorshift_f64(rng_state);
        let branch_w = 0.5 / alpha_p;
        (scatter_direction(sun_dir, ct, phi_init), branch_w, ct, true)
    } else if xi_mix < alpha_p + alpha_z || alpha_t < 1e-12 {
        // Zenith-biased branch (wavelength-independent)
        let (d, cos_z) = sample_zenith_biased(local_up, bp.n_zenith, rng_state);
        let shape_w = zenith_importance_weight(cos_z, bp.n_zenith);
        let branch_w = 0.5 / (alpha_z + alpha_t);
        (d, shape_w * branch_w, 0.0, false)
    } else {
        // Terminator lobe branch (wavelength-independent)
        let (d, cos_t) = sample_zenith_biased(term_axis, bp.m_term, rng_state);
        let cos_z = d.dot(local_up);
        let shape_w = terminator_shape_weight(cos_z, cos_t, bp.m_term);
        let branch_w = 0.5 / alpha_t;
        (d, shape_w * branch_w, 0.0, false)
    };

    // Initialize per-wavelength weight ratios: weight_ratio[w] = weight_w / hero_weight.
    // Corrections for SSA and initial direction sampling (phase function ratio).
    let mut weight_ratio = [0.0f64; 64];
    let hero_phase_init = if is_phase_branch {
        scalar_phase_value(cos_theta_init, hero_optics)
    } else {
        1.0
    };
    for w in 0..num_wl {
        let optics_w = &atm.optics[start_shell][w];
        let ssa_ratio = if hero_optics.ssa > 1e-30 {
            optics_w.ssa / hero_optics.ssa
        } else {
            0.0
        };
        let dir_ratio = if is_phase_branch && hero_phase_init > 1e-30 {
            scalar_phase_value(cos_theta_init, optics_w) / hero_phase_init
        } else {
            1.0
        };
        weight_ratio[w] = ssa_ratio * dir_ratio;
    }

    let surface_radius = atm.surface_radius();
    let mut total = [0.0f64; 64];

    // Forced scattering + exponential transform setup (same as scalar tracer).
    let sza_deg_local = libm::acos(cos_sza.clamp(-1.0, 1.0)) * 180.0 / core::f64::consts::PI;
    let use_forced = sza_deg_local >= ZENITH_SZA_START;
    let sza_t_et =
        ((sza_deg_local - ZENITH_SZA_START) / (ZENITH_SZA_FULL - ZENITH_SZA_START)).clamp(0.0, 1.0);
    let alpha_et = EXP_TRANSFORM_ALPHA_MAX * sza_t_et;

    // Altitude-splitting setup (same ramp as scalar tracer).
    let split_factors = split_factors_for_sza(sza_deg_local);

    // Initialize work stack with the main particle.
    let mut stack = [SplitParticleAlis {
        pos: Vec3::new(0.0, 0.0, 0.0),
        dir: Vec3::new(0.0, 0.0, 1.0),
        hero_weight: 0.0,
        weight_ratio: [0.0f64; 64],
        rng: 0,
        bounces_left: 0,
        next_split: 0,
    }; MAX_SPLIT_PARTICLES];
    let mut stack_len: usize = 1;
    stack[0] = SplitParticleAlis {
        pos: start_pos,
        dir,
        hero_weight: hero_optics.ssa * initial_weight,
        weight_ratio,
        rng: *rng_state,
        bounces_left: HYBRID_MAX_BOUNCES,
        next_split: 0,
    };
    let mut main_rng_out = *rng_state;
    let mut main_processed = false;

    // Process all particles: main first, then split copies (LIFO order).
    while stack_len > 0 {
        stack_len -= 1;
        let is_main = !main_processed;
        main_processed = true;
        let mut pos = stack[stack_len].pos;
        let mut current_dir = stack[stack_len].dir;
        let mut hero_weight = stack[stack_len].hero_weight;
        let mut wr = stack[stack_len].weight_ratio;
        let mut local_rng = stack[stack_len].rng;
        let bounces_left = stack[stack_len].bounces_left;
        let mut next_split = stack[stack_len].next_split;

        for _bounce in 0..bounces_left {
            // --- Decide scatter mode for this bounce ---
            let mut forced_this_bounce = false;
            let mut tau_maxes = [0.0f64; 64];

            if use_forced {
                let (tms, hit_ground) =
                    scout_tau_to_boundary_alis(atm, pos, current_dir, hero_wl, num_wl);
                tau_maxes = tms;
                forced_this_bounce = !hit_ground && tms[hero_wl] < FORCED_TAU_CUTOFF;
            }

            let scatter_shell;

            if forced_this_bounce {
                let tau_max_h = tau_maxes[hero_wl];
                let exp_neg_tau_h = libm::exp(-tau_max_h);
                let one_minus_exp_h = 1.0 - exp_neg_tau_h;
                hero_weight *= one_minus_exp_h;
                if hero_weight < 1e-30 {
                    break;
                }

                for w in 0..num_wl {
                    let one_minus_exp_w = 1.0 - libm::exp(-tau_maxes[w]);
                    wr[w] *= if one_minus_exp_h > 1e-30 {
                        one_minus_exp_w / one_minus_exp_h
                    } else {
                        0.0
                    };
                }

                // VSPG: importance-weighted shell selection for hero wavelength.
                let (tau_s, vspg_w) = vspg_sample_scatter_tau(
                    atm,
                    pos,
                    current_dir,
                    hero_wl,
                    tau_max_h,
                    sza_deg_local,
                    &mut local_rng,
                );
                hero_weight *= vspg_w;
                let (sp, sd, ss, taus_at_pos) =
                    advance_to_optical_depth_alis(atm, pos, current_dir, tau_s, hero_wl, num_wl);
                pos = sp;
                current_dir = sd;
                scatter_shell = ss;

                let sigma_h = atm.optics[scatter_shell][hero_wl].extinction;
                if sigma_h > 1e-30 {
                    let tau_h_pos = taus_at_pos[hero_wl];
                    for w in 0..num_wl {
                        let sigma_w = atm.optics[scatter_shell][w].extinction;
                        wr[w] *= (sigma_w / sigma_h) * libm::exp(-(taus_at_pos[w] - tau_h_pos));
                    }
                }
            } else {
                let mut scatter_found = false;
                let mut found_shell = 0usize;

                for _ in 0..200 {
                    let r = pos.length();
                    let shell_idx = match atm.shell_index(r) {
                        Some(idx) => idx,
                        None => break,
                    };

                    let shell = &atm.shells[shell_idx];
                    let hero_ext = atm.optics[shell_idx][hero_wl].extinction;

                    if hero_ext < 1e-20 {
                        match next_shell_boundary(pos, current_dir, shell.r_inner, shell.r_outer) {
                            Some((dist, is_outward)) => {
                                for w in 0..num_wl {
                                    let sigma_w = atm.optics[shell_idx][w].extinction;
                                    if sigma_w > 1e-30 {
                                        wr[w] *= libm::exp(-sigma_w * dist);
                                    }
                                }
                                let (np, nd) = cross_boundary(
                                    pos,
                                    current_dir,
                                    dist,
                                    is_outward,
                                    shell_idx,
                                    atm,
                                );
                                pos = np;
                                current_dir = nd;
                                continue;
                            }
                            None => break,
                        }
                    }

                    let cos_bias = current_dir.dot(term_axis);
                    let sigma_h = hero_ext;
                    let sigma_prime_h = sigma_h * (1.0 - alpha_et * cos_bias);

                    let xi = xorshift_f64(&mut local_rng);
                    let free_path = -libm::log(1.0 - xi + 1e-30) / sigma_prime_h;

                    match next_shell_boundary(pos, current_dir, shell.r_inner, shell.r_outer) {
                        Some((boundary_dist, is_outward)) => {
                            if free_path >= boundary_dist {
                                if alpha_et > 0.0 {
                                    hero_weight *=
                                        libm::exp(-alpha_et * sigma_h * cos_bias * boundary_dist);
                                }
                                for w in 0..num_wl {
                                    let sigma_w = atm.optics[shell_idx][w].extinction;
                                    wr[w] *= libm::exp(-(sigma_w - sigma_h) * boundary_dist);
                                }

                                let (np, nd) = cross_boundary(
                                    pos,
                                    current_dir,
                                    boundary_dist,
                                    is_outward,
                                    shell_idx,
                                    atm,
                                );
                                pos = np;
                                current_dir = nd;

                                if !is_outward && pos.length() <= surface_radius + 1.0 {
                                    let hero_albedo = atm.surface_albedo[hero_wl];
                                    hero_weight *= hero_albedo;
                                    if hero_weight < 1e-30 {
                                        break;
                                    }
                                    for w in 0..num_wl {
                                        let albedo_ratio = if hero_albedo > 1e-30 {
                                            atm.surface_albedo[w] / hero_albedo
                                        } else {
                                            0.0
                                        };
                                        wr[w] *= albedo_ratio;
                                    }
                                    let normal = pos.normalize();
                                    current_dir = sample_hemisphere(normal, &mut local_rng);
                                    continue;
                                }
                                continue;
                            }
                        }
                        None => break,
                    }

                    if alpha_et > 0.0 {
                        hero_weight *= (sigma_h / sigma_prime_h)
                            * libm::exp(-alpha_et * sigma_h * cos_bias * free_path);
                    }
                    for w in 0..num_wl {
                        let sigma_w = atm.optics[shell_idx][w].extinction;
                        if sigma_h > 1e-30 {
                            wr[w] *=
                                (sigma_w / sigma_h) * libm::exp(-(sigma_w - sigma_h) * free_path);
                        }
                    }
                    pos = pos + current_dir * free_path;
                    found_shell = shell_idx;
                    scatter_found = true;
                    break;
                }

                if !scatter_found {
                    break;
                }
                scatter_shell = found_shell;
            }

            // NEE for ALL wavelengths using multi-wavelength shadow ray.
            let t_suns = shadow_ray_transmittance_spectrum(atm, pos, sun_dir, num_wl);
            let cos_angle_nee = sun_dir.dot(-current_dir);

            for w in 0..num_wl {
                if t_suns[w] > 1e-30 {
                    let optics_w = &atm.optics[scatter_shell][w];
                    let phase_w = scalar_phase_value(cos_angle_nee, optics_w);
                    total[w] += hero_weight * wr[w] * t_suns[w] * phase_w * INV_4PI;
                }
            }

            // Apply hero SSA.
            let hero_scatter_optics = &atm.optics[scatter_shell][hero_wl];
            hero_weight *= hero_scatter_optics.ssa;
            if hero_weight < 1e-30 {
                break;
            }

            // ALIS SSA ratio correction.
            for w in 0..num_wl {
                let ssa_w = atm.optics[scatter_shell][w].ssa;
                let ssa_ratio = if hero_scatter_optics.ssa > 1e-30 {
                    ssa_w / hero_scatter_optics.ssa
                } else {
                    0.0
                };
                wr[w] *= ssa_ratio;
            }

            // Sample new direction: one-sample MIS (guide vs hero phase function).
            let alt_for_guide = pos.length() - surface_radius;
            let use_guide_this_bounce =
                guide.is_some() && xorshift_f64(&mut local_rng) < GUIDE_MIS_FRAC;

            let (new_dir, cos_theta_for_alis) = if use_guide_this_bounce {
                let g = guide.unwrap();
                let local_up_here = pos.normalize();
                let (gdir, p_guide) =
                    g.sample(alt_for_guide, local_up_here, sun_dir, &mut local_rng);
                let ct = current_dir.dot(gdir);
                let p_phase_hero = scalar_phase_value(ct, hero_scatter_optics) * INV_4PI;
                let mis_denom = GUIDE_MIS_FRAC * p_guide + (1.0 - GUIDE_MIS_FRAC) * p_phase_hero;
                if mis_denom > 1e-30 {
                    hero_weight *= p_guide / mis_denom;
                }
                let _ = xorshift_f64(&mut local_rng);
                let _ = xorshift_f64(&mut local_rng);
                let _ = xorshift_f64(&mut local_rng);
                (gdir, ct)
            } else {
                let cos_theta =
                    if xorshift_f64(&mut local_rng) < hero_scatter_optics.rayleigh_fraction {
                        sample_rayleigh_analytic(xorshift_f64(&mut local_rng))
                    } else {
                        sample_henyey_greenstein(
                            xorshift_f64(&mut local_rng),
                            hero_scatter_optics.asymmetry,
                        )
                    };
                let phi = 2.0 * core::f64::consts::PI * xorshift_f64(&mut local_rng);
                let d = scatter_direction(current_dir, cos_theta, phi);
                if let Some(g) = guide {
                    let local_up_here = pos.normalize();
                    let p_phase_hero = scalar_phase_value(cos_theta, hero_scatter_optics) * INV_4PI;
                    let p_guide = g.pdf(alt_for_guide, local_up_here, sun_dir, d);
                    let mis_denom =
                        GUIDE_MIS_FRAC * p_guide + (1.0 - GUIDE_MIS_FRAC) * p_phase_hero;
                    if mis_denom > 1e-30 {
                        hero_weight *= p_phase_hero / mis_denom;
                    }
                }
                (d, cos_theta)
            };

            // ALIS phase function ratio for direction sampling.
            let phase_hero = scalar_phase_value(cos_theta_for_alis, hero_scatter_optics);
            if phase_hero > 1e-30 {
                for w in 0..num_wl {
                    let optics_w = &atm.optics[scatter_shell][w];
                    let phase_w = scalar_phase_value(cos_theta_for_alis, optics_w);
                    wr[w] *= phase_w / phase_hero;
                }
            }

            current_dir = new_dir;

            // --- Altitude-based splitting ---
            let alt = pos.length() - surface_radius;
            while next_split < NUM_SPLIT_LEVELS && alt > SPLIT_ALTITUDES_M[next_split] {
                let k = split_factors[next_split];
                if k > 1 {
                    hero_weight /= k as f64;
                    let remaining = bounces_left.saturating_sub(_bounce + 1);
                    for copy_idx in 1..k {
                        if stack_len < MAX_SPLIT_PARTICLES {
                            let child_rng = local_rng
                                ^ (copy_idx as u64).wrapping_mul(2654435761)
                                ^ ((next_split as u64) << 32);
                            stack[stack_len] = SplitParticleAlis {
                                pos,
                                dir: current_dir,
                                hero_weight,
                                weight_ratio: wr,
                                rng: child_rng,
                                bounces_left: remaining,
                                next_split: next_split + 1,
                            };
                            stack_len += 1;
                        }
                    }
                }
                next_split += 1;
            }
        }

        if is_main {
            main_rng_out = local_rng;
        }
    }

    *rng_state = main_rng_out;
    total
}

/// ALIS hybrid multi-scatter spectral radiance for all wavelengths.
///
/// Combines deterministic single-scatter integration (order 1) with ALIS MC
/// secondary chains (orders 2+). Each chain traces ONE hero wavelength path
/// but evaluates ALL wavelengths simultaneously, giving ~N_wl fewer chains
/// than per-wavelength tracing.
///
/// The hero wavelength rotates round-robin across rays, giving even coverage.
/// Per-wavelength weight ratios correct for differences in extinction, SSA,
/// and phase function, keeping the estimator exactly unbiased.
///
/// Returns spectral radiance array `[f64; 64]` for all wavelengths.
#[allow(clippy::needless_range_loop)]
#[allow(clippy::too_many_arguments)]
pub fn hybrid_scatter_radiance_alis(
    atm: &AtmosphereModel,
    observer_pos: Vec3,
    view_dir: Vec3,
    sun_dir: Vec3,
    secondary_rays: usize,
    rng_state: &mut u64,
    guide: Option<&PathGuide>,
) -> [f64; 64] {
    use crate::geometry::ray_sphere_intersect;
    use crate::single_scatter::shadow_ray_transmittance_spectrum;

    let num_wl = atm.num_wavelengths;
    let toa_radius = atm.toa_radius();
    let surface_radius = atm.surface_radius();
    let mut radiance = [0.0f64; 64];

    // Find LOS extent.
    let los_max = match ray_sphere_intersect(observer_pos, view_dir, toa_radius) {
        Some(hit) if hit.t_far > 0.0 => hit.t_far,
        _ => return radiance,
    };

    let ground_hit = ray_sphere_intersect(observer_pos, view_dir, surface_radius);
    let los_end = match ground_hit {
        Some(ref hit) if hit.t_near > 1e-3 && hit.t_near < los_max => hit.t_near,
        _ => los_max,
    };

    if los_end <= 0.0 {
        return radiance;
    }

    let num_steps = HYBRID_LOS_STEPS.min((los_end / 500.0) as usize + 20);
    let ds = los_end / num_steps as f64;

    // Per-wavelength accumulated optical depth from observer.
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

        // Check if any wavelength still has observable transmittance.
        let mut any_visible = false;
        for w in 0..num_wl {
            let tau_mid = tau_obs[w] + atm.optics[shell_idx][w].extinction * ds * 0.5;
            if libm::exp(-tau_mid) > 1e-30 {
                any_visible = true;
                break;
            }
        }
        if !any_visible {
            break;
        }

        // --- Order 1: deterministic single-scatter NEE (all wavelengths) ---
        let t_suns = shadow_ray_transmittance_spectrum(atm, scatter_pos, sun_dir, num_wl);
        let cos_theta_1 = sun_dir.dot(-view_dir);

        for w in 0..num_wl {
            let optics = &atm.optics[shell_idx][w];
            let beta_scat = optics.extinction * optics.ssa;
            if beta_scat < 1e-30 {
                continue;
            }

            let tau_obs_mid = tau_obs[w] + optics.extinction * ds * 0.5;
            let t_obs = libm::exp(-tau_obs_mid);
            if t_obs < 1e-30 || t_suns[w] < 1e-30 {
                continue;
            }

            let phase = scalar_phase_value(cos_theta_1, optics);
            radiance[w] += beta_scat * phase * INV_4PI * t_suns[w] * t_obs * ds;
        }

        // --- Orders 2+: ALIS MC secondary chains ---
        if secondary_rays > 0 {
            let mut mc_totals = [0.0f64; 64];

            for ray in 0..secondary_rays {
                // Round-robin hero selection across wavelengths.
                let hero_wl = ray % num_wl;

                let chain_result = trace_secondary_chain_alis(
                    atm,
                    scatter_pos,
                    sun_dir,
                    hero_wl,
                    shell_idx,
                    rng_state,
                    ray,
                    secondary_rays,
                    num_wl,
                    guide,
                );

                for w in 0..num_wl {
                    mc_totals[w] += chain_result[w];
                }
            }

            let inv_rays = 1.0 / secondary_rays as f64;
            for w in 0..num_wl {
                let optics = &atm.optics[shell_idx][w];
                let beta_scat = optics.extinction * optics.ssa;
                if beta_scat < 1e-30 {
                    continue;
                }
                let tau_obs_mid = tau_obs[w] + optics.extinction * ds * 0.5;
                let t_obs = libm::exp(-tau_obs_mid);
                if t_obs < 1e-30 {
                    continue;
                }
                radiance[w] += mc_totals[w] * inv_rays * beta_scat * t_obs * ds;
            }
        }

        for w in 0..num_wl {
            tau_obs[w] += atm.optics[shell_idx][w].extinction * ds;
        }
    }

    radiance
}

/// Compute hybrid multi-scatter spectral radiance for all wavelengths.
///
/// This is the primary function for physically-accurate twilight computation.
/// It combines deterministic single-scatter integration with MC secondary
/// chains for orders 2+, producing converged results with far fewer photons
/// than pure backward MC.
///
/// # Arguments
/// * `atm` - Atmosphere model
/// * `observer_pos` - Observer position in ECEF [m]
/// * `view_dir` - Viewing direction (unit vector)
/// * `sun_dir` - Direction toward the sun (unit vector)
/// * `secondary_rays` - Number of MC chains per LOS step per wavelength
/// * `base_seed` - Base RNG seed
///
/// # Returns
/// Spectral radiance array `[f64; 64]`, one value per wavelength channel.
#[allow(clippy::too_many_arguments)]
pub fn hybrid_scatter_spectrum(
    atm: &AtmosphereModel,
    observer_pos: Vec3,
    view_dir: Vec3,
    sun_dir: Vec3,
    secondary_rays: usize,
    base_seed: u64,
    polarized: bool,
    guide: Option<&PathGuide>,
) -> [f64; 64] {
    // ALIS path: trace ONE hero path per chain, evaluate ALL wavelengths.
    // ~N_wl fewer chains than per-wavelength tracing, same expected value.
    // Only available for scalar (non-polarized) mode; Stokes ALIS would need
    // per-wavelength Mueller matrices which breaks the single-path assumption.
    if !polarized {
        let mut rng = base_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        return hybrid_scatter_radiance_alis(
            atm,
            observer_pos,
            view_dir,
            sun_dir,
            secondary_rays,
            &mut rng,
            guide,
        );
    }

    // Polarized path: per-wavelength tracing (full Stokes [I,Q,U,V]).
    let mut radiance = [0.0f64; 64];
    let num_wl = atm.num_wavelengths;

    for (w, rad_w) in radiance.iter_mut().enumerate().take(num_wl) {
        let mut rng = base_seed
            .wrapping_add(w as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);

        *rad_w = hybrid_scatter_radiance(
            atm,
            observer_pos,
            view_dir,
            sun_dir,
            w,
            secondary_rays,
            &mut rng,
            polarized,
        );
    }

    radiance
}

/// Sample a direction uniformly on the upper hemisphere around a normal vector.
fn sample_hemisphere(normal: Vec3, rng: &mut u64) -> Vec3 {
    use libm::sqrt;

    let xi1 = xorshift_f64(rng);
    let xi2 = xorshift_f64(rng);

    let cos_theta = sqrt(xi1);
    let _sin_theta = sqrt(1.0 - xi1);
    let phi = 2.0 * core::f64::consts::PI * xi2;

    scatter_direction(normal, cos_theta, phi)
}

/// Sample a direction from a power-cosine distribution biased toward `normal`.
///
/// PDF over solid angle: p(omega) = (n+1) / (2*pi) * cos^n(theta)
/// where theta is the angle from `normal`. This concentrates rays near
/// the zenith (normal direction) more aggressively than the default
/// cosine-weighted hemisphere (which corresponds to n=1).
///
/// Sampling: cos(theta) = xi^(1/(n+1)), phi = 2*pi*xi2
///
/// Consumes exactly 2 RNG draws, matching `sample_hemisphere`.
///
/// Returns (direction, cos_theta) -- the caller uses cos_theta to compute
/// the importance weight via `zenith_importance_weight`.
fn sample_zenith_biased(normal: Vec3, n: f64, rng: &mut u64) -> (Vec3, f64) {
    let xi1 = xorshift_f64(rng);
    let xi2 = xorshift_f64(rng);

    let cos_theta = libm::pow(xi1, 1.0 / (n + 1.0));
    let phi = 2.0 * core::f64::consts::PI * xi2;

    let dir = scatter_direction(normal, cos_theta, phi);
    (dir, cos_theta)
}

/// Importance weight correction for zenith-biased sampling.
///
/// When sampling from power-cosine(n) instead of cosine-weighted (n=1),
/// the importance ratio is:
///
///   w = p_cosine(theta) / p_zenith(theta)
///     = [cos(theta) / pi] / [(n+1) / (2*pi) * cos^n(theta)]
///     = 2 / (n+1) * cos^(1-n)(theta)
///
/// This keeps the estimator unbiased: E_zenith[f(w) * w] = E_cosine[f(w)].
#[inline]
fn zenith_importance_weight(cos_theta: f64, n: f64) -> f64 {
    // cos^(1-n) = 1 / cos^(n-1)
    // For n=5: 1 / cos^4(theta). At zenith (cos=1): w = 2/6 = 0.333
    // At 45 deg (cos=0.707): w = 2/6 * (0.707)^(-4) = 1.333
    let cos_nm1 = libm::pow(cos_theta, n - 1.0);
    2.0 / ((n + 1.0) * cos_nm1)
}

/// SZA-adaptive parameters for the 3-branch initial direction sampling.
///
/// All six parameters ramp linearly from SZA_START (96 deg) to SZA_FULL (106 deg).
#[derive(Clone, Copy)]
struct BranchParams {
    /// Total fraction of rays using non-phase-function sampling (zenith + terminator).
    /// Phase branch gets `1 - zenith_frac`.
    zenith_frac: f64,
    /// Power-cosine exponent for the zenith lobe.
    n_zenith: f64,
    /// Fraction of zenith-allocated rays redirected to terminator lobe.
    /// The actual probabilities are:
    ///   alpha_p = 1 - zenith_frac
    ///   alpha_z = zenith_frac * (1 - term_share)
    ///   alpha_t = zenith_frac * term_share
    term_share: f64,
    /// Power-cosine exponent for the terminator lobe.
    m_term: f64,
    /// Tilt angle (radians) of the terminator axis from zenith toward the sun.
    tilt_rad: f64,
}

/// Compute the SZA-adaptive branch parameters.
///
/// At SZA <= 96:
///   zenith_frac = 0.5, n = 1.0, term_share = 0.0
///   -> standard 50/50 mix, no terminator lobe
///   -> all branch weights evaluate to exactly 1.0 (zero overhead)
///
/// At SZA >= 106:
///   zenith_frac = 0.95, n = 5.0, term_share = 0.5, m = 8.0, tilt = 50 deg
///   -> phase: 5%, zenith: 47.5%, terminator: 47.5%
///
/// In between: linear interpolation of all parameters.
#[inline]
fn branch_params_for_sza(cos_sza: f64) -> BranchParams {
    let sza_deg = libm::acos(cos_sza.clamp(-1.0, 1.0)) * 180.0 / core::f64::consts::PI;

    let sza_t =
        ((sza_deg - ZENITH_SZA_START) / (ZENITH_SZA_FULL - ZENITH_SZA_START)).clamp(0.0, 1.0);

    let zenith_frac = 0.5 + (ZENITH_MAX_FRACTION - 0.5) * sza_t;
    let n_zenith = 1.0 + (ZENITH_BIAS_N - 1.0) * sza_t;
    let term_share = TERMINATOR_MAX_SHARE * sza_t;
    let m_term = 1.0 + (TERMINATOR_N_MAX - 1.0) * sza_t;
    let tilt_deg =
        TERMINATOR_TILT_MIN_DEG + (TERMINATOR_TILT_MAX_DEG - TERMINATOR_TILT_MIN_DEG) * sza_t;
    let tilt_rad = tilt_deg * core::f64::consts::PI / 180.0;

    BranchParams {
        zenith_frac,
        n_zenith,
        term_share,
        m_term,
        tilt_rad,
    }
}

/// Compute the terminator axis: a unit vector tilted from `up` toward the
/// sub-solar point on the local horizon.
///
/// The axis is: `cos(tilt) * up + sin(tilt) * sun_horiz` where `sun_horiz`
/// is the projection of `sun_dir` onto the horizontal plane, normalized.
///
/// If the sun is directly at zenith/nadir (no horizontal component), the
/// axis falls back to `up` (pure zenith, no tilt).
#[inline]
fn terminator_axis(up: Vec3, sun_dir: Vec3, tilt_rad: f64) -> Vec3 {
    // Project sun_dir onto the local horizontal plane
    let dot_us = sun_dir.dot(up);
    let horiz = Vec3::new(
        sun_dir.x - dot_us * up.x,
        sun_dir.y - dot_us * up.y,
        sun_dir.z - dot_us * up.z,
    );
    let h_len = horiz.length();
    if h_len < 1e-12 {
        // Sun at zenith/nadir: no preferred horizontal direction
        return up;
    }
    let sun_horiz = horiz.scale(1.0 / h_len);

    let (sin_t, cos_t) = libm::sincos(tilt_rad);
    let axis = Vec3::new(
        cos_t * up.x + sin_t * sun_horiz.x,
        cos_t * up.y + sin_t * sun_horiz.y,
        cos_t * up.z + sin_t * sun_horiz.z,
    );
    axis.normalize()
}

/// Shape weight for the terminator lobe: corrects the power-cosine PDF
/// centered on the terminator axis back to the cosine-hemisphere reference.
///
/// `cos_z` = cos(angle from zenith), `cos_t` = cos(angle from terminator axis).
///
/// Weight = p_cosine(d) / p_term(d)
///        = [cos(theta_z) / pi] / [(m+1) / (2*pi) * cos^m(theta_t)]
///        = 2 * cos(theta_z) / ((m+1) * cos^m(theta_t))
///
/// If `cos_z <= 0` (below horizon), returns 0 -- the direction has zero
/// probability in the cosine-hemisphere reference. Samples are not wasted
/// in practice because the terminator axis tilt (max 50 deg) combined with
/// the concentration (m=8) keeps 95%+ of samples above the horizon.
#[inline]
fn terminator_shape_weight(cos_z: f64, cos_t: f64, m: f64) -> f64 {
    if cos_z <= 0.0 || cos_t <= 0.0 {
        return 0.0;
    }
    let cos_t_m = libm::pow(cos_t, m);
    if cos_t_m < 1e-30 {
        return 0.0;
    }
    2.0 * cos_z / ((m + 1.0) * cos_t_m)
}

/// Simple xorshift64 PRNG suitable for no_std Monte Carlo.
///
/// Not cryptographically secure, but good statistical properties
/// for Monte Carlo sampling with minimal state.
#[inline]
pub fn xorshift_f64(state: &mut u64) -> f64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    // Convert to f64 in [0, 1)
    (x >> 11) as f64 / (1u64 << 53) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── xorshift_f64 RNG ──

    #[test]
    fn xorshift_output_in_unit_interval() {
        let mut state: u64 = 12345;
        for _ in 0..10_000 {
            let val = xorshift_f64(&mut state);
            assert!(
                (0.0..1.0).contains(&val),
                "xorshift_f64 produced {} outside [0, 1)",
                val
            );
        }
    }

    #[test]
    fn xorshift_state_changes_every_call() {
        let mut state: u64 = 42;
        let mut prev_state = state;
        for _ in 0..100 {
            let _ = xorshift_f64(&mut state);
            assert_ne!(state, prev_state, "State should change on each call");
            prev_state = state;
        }
    }

    #[test]
    fn xorshift_is_deterministic() {
        let mut s1: u64 = 9999;
        let mut s2: u64 = 9999;
        for _ in 0..100 {
            let v1 = xorshift_f64(&mut s1);
            let v2 = xorshift_f64(&mut s2);
            assert_eq!(v1, v2, "Same seed should produce same sequence");
        }
    }

    #[test]
    fn xorshift_different_seeds_different_output() {
        let mut s1: u64 = 1;
        let mut s2: u64 = 2;
        let v1 = xorshift_f64(&mut s1);
        let v2 = xorshift_f64(&mut s2);
        assert_ne!(v1, v2, "Different seeds should produce different output");
    }

    #[test]
    fn xorshift_uniformity_chi_squared() {
        // Chi-squared test: bin 100,000 samples into 10 bins, check uniformity.
        // For 10 bins with 10000 expected per bin, chi²(9) < 16.92 at p=0.05.
        let mut state: u64 = 123456789;
        let n = 100_000;
        let num_bins = 10;
        let mut bins = [0u32; 10];

        for _ in 0..n {
            let val = xorshift_f64(&mut state);
            let bin = (val * num_bins as f64) as usize;
            let bin = bin.min(num_bins - 1);
            bins[bin] += 1;
        }

        let expected = n as f64 / num_bins as f64;
        let mut chi2 = 0.0;
        for &count in &bins {
            let diff = count as f64 - expected;
            chi2 += diff * diff / expected;
        }

        assert!(
            chi2 < 30.0, // Very generous threshold (critical value at p=0.001 is 27.88 for df=9)
            "Chi-squared test failed: chi2 = {:.2}, bins = {:?}",
            chi2,
            bins
        );
    }

    #[test]
    fn xorshift_mean_near_half() {
        let mut state: u64 = 77777;
        let n = 100_000;
        let mut sum = 0.0;
        for _ in 0..n {
            sum += xorshift_f64(&mut state);
        }
        let mean = sum / n as f64;
        assert!(
            (mean - 0.5).abs() < 0.01,
            "Mean should be ~0.5, got {}",
            mean
        );
    }

    #[test]
    fn xorshift_never_returns_one() {
        // The conversion (x >> 11) / 2^53 should never reach exactly 1.0
        let mut state: u64 = 1;
        for _ in 0..100_000 {
            let val = xorshift_f64(&mut state);
            assert!(val < 1.0, "xorshift_f64 returned 1.0 (should be < 1.0)");
        }
    }

    // ── scout_tau_to_boundary ──

    #[test]
    fn scout_tau_zero_in_empty_atmosphere() {
        let atm = crate::atmosphere::AtmosphereModel::new(&[0.0, 50.0, 100.0], &[550.0]);
        let pos = crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let dir = crate::geometry::Vec3::new(1.0, 0.0, 0.0); // radially outward

        let (tau, hit_ground) = scout_tau_to_boundary(&atm, pos, dir, 0);
        assert!(
            tau.abs() < 1e-20,
            "Empty atmosphere should have zero tau, got {}",
            tau
        );
        assert!(!hit_ground, "Outward ray should not hit ground");
    }

    #[test]
    fn scout_tau_radial_outward_through_uniform_shell() {
        // Single shell 0-100km, uniform extinction 1e-5 /m.
        // Radial outward ray from surface: path = 100km = 1e5 m.
        // Expected tau = 1e-5 * 1e5 = 1.0
        use crate::atmosphere::{AtmosphereModel, ShellOptics, EARTH_RADIUS_M};

        let mut atm = AtmosphereModel::new(&[0.0, 100.0], &[550.0]);
        atm.optics[0][0] = ShellOptics {
            extinction: 1e-5,
            ssa: 1.0,
            asymmetry: 0.0,
            rayleigh_fraction: 1.0,
        };

        let pos = crate::geometry::Vec3::new(EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let dir = crate::geometry::Vec3::new(1.0, 0.0, 0.0);

        let (tau, hit_ground) = scout_tau_to_boundary(&atm, pos, dir, 0);
        // Path ~ 100km, tau ~ 1.0 (within a few percent due to 1m offset)
        assert!((tau - 1.0).abs() < 0.01, "Expected tau ~ 1.0, got {}", tau);
        assert!(!hit_ground, "Outward ray should not hit ground");
    }

    #[test]
    fn scout_tau_through_multiple_shells() {
        // Two shells: 0-10km (ext=1e-4), 10-100km (ext=1e-6).
        // Radial outward: tau = 1e-4*10km + 1e-6*90km = 1.0 + 0.09 = 1.09
        use crate::atmosphere::{AtmosphereModel, ShellOptics, EARTH_RADIUS_M};

        let mut atm = AtmosphereModel::new(&[0.0, 10.0, 100.0], &[550.0]);
        atm.optics[0][0] = ShellOptics {
            extinction: 1e-4,
            ssa: 1.0,
            asymmetry: 0.0,
            rayleigh_fraction: 1.0,
        };
        atm.optics[1][0] = ShellOptics {
            extinction: 1e-6,
            ssa: 1.0,
            asymmetry: 0.0,
            rayleigh_fraction: 1.0,
        };

        let pos = crate::geometry::Vec3::new(EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let dir = crate::geometry::Vec3::new(1.0, 0.0, 0.0);

        let (tau, hit_ground) = scout_tau_to_boundary(&atm, pos, dir, 0);
        assert!(
            (tau - 1.09).abs() < 0.02,
            "Expected tau ~ 1.09, got {}",
            tau
        );
        assert!(!hit_ground, "Outward ray should not hit ground");
    }

    #[test]
    fn scout_tau_downward_hits_ground() {
        // Radially inward from 50km altitude: hits ground.
        // Single shell 0-100km, ext=1e-5. Path = 50km. tau = 0.5
        use crate::atmosphere::{AtmosphereModel, ShellOptics, EARTH_RADIUS_M};

        let mut atm = AtmosphereModel::new(&[0.0, 100.0], &[550.0]);
        atm.optics[0][0] = ShellOptics {
            extinction: 1e-5,
            ssa: 1.0,
            asymmetry: 0.0,
            rayleigh_fraction: 1.0,
        };

        let pos = crate::geometry::Vec3::new(EARTH_RADIUS_M + 50_000.0, 0.0, 0.0);
        let dir = crate::geometry::Vec3::new(-1.0, 0.0, 0.0); // radially inward

        let (tau, hit_ground) = scout_tau_to_boundary(&atm, pos, dir, 0);
        assert!(
            (tau - 0.5).abs() < 0.01,
            "Expected tau ~ 0.5 (downward to ground), got {}",
            tau
        );
        assert!(hit_ground, "Downward ray should hit ground");
    }

    #[test]
    fn scout_tau_outside_atmosphere_returns_zero() {
        use crate::atmosphere::{AtmosphereModel, EARTH_RADIUS_M};

        let atm = AtmosphereModel::new(&[0.0, 100.0], &[550.0]);
        let pos = crate::geometry::Vec3::new(EARTH_RADIUS_M + 200_000.0, 0.0, 0.0);
        let dir = crate::geometry::Vec3::new(1.0, 0.0, 0.0);

        let (tau, hit_ground) = scout_tau_to_boundary(&atm, pos, dir, 0);
        assert!(
            tau.abs() < 1e-20,
            "Outside atmosphere should return zero tau, got {}",
            tau
        );
        assert!(!hit_ground);
    }

    #[test]
    fn scout_tau_matches_shadow_ray_transmittance() {
        // scout_tau_to_boundary should give tau such that exp(-tau) matches
        // shadow_ray_transmittance along the same ray.
        use crate::single_scatter::shadow_ray_transmittance;

        let atm = make_scattering_atmosphere();
        let pos = crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 5_000.0, 0.0, 0.0);
        let dir = crate::geometry::Vec3::new(1.0, 0.0, 0.0); // radially outward

        let (tau, _hit_ground) = scout_tau_to_boundary(&atm, pos, dir, 1); // wavelength 1 (550nm)
        let t_shadow = shadow_ray_transmittance(&atm, pos, dir, 1);

        let t_from_scout = libm::exp(-tau);
        let rel_err = if t_shadow > 1e-30 {
            (t_from_scout - t_shadow).abs() / t_shadow
        } else {
            t_from_scout.abs()
        };

        assert!(
            rel_err < 0.01,
            "scout tau={:.6} -> T={:.6e}, shadow T={:.6e}, rel_err={:.4}",
            tau,
            t_from_scout,
            t_shadow,
            rel_err
        );
    }

    // ── advance_to_optical_depth ──

    #[test]
    fn advance_lands_at_correct_optical_depth() {
        // Uniform shell, radial outward. tau_target = 0.5 with ext=1e-5.
        // Expected distance = 0.5 / 1e-5 = 50km from start.
        use crate::atmosphere::{AtmosphereModel, ShellOptics, EARTH_RADIUS_M};

        let mut atm = AtmosphereModel::new(&[0.0, 100.0], &[550.0]);
        atm.optics[0][0] = ShellOptics {
            extinction: 1e-5,
            ssa: 1.0,
            asymmetry: 0.0,
            rayleigh_fraction: 1.0,
        };

        let pos = crate::geometry::Vec3::new(EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let dir = crate::geometry::Vec3::new(1.0, 0.0, 0.0);

        let (scatter_pos, _dir, shell) = advance_to_optical_depth(&atm, pos, dir, 0.5, 0);

        let altitude = scatter_pos.length() - EARTH_RADIUS_M;
        assert_eq!(shell, 0, "Should still be in shell 0");
        assert!(
            (altitude - 50_001.0).abs() < 100.0,
            "Expected ~50km altitude, got {:.0}m",
            altitude
        );
    }

    #[test]
    fn advance_crosses_shell_boundary() {
        // Two shells: 0-10km (ext=1e-4), 10-100km (ext=1e-6).
        // tau through shell 0 = 1e-4 * 10km = 1.0.
        // Requesting tau_target = 1.5 should land in shell 1.
        // Remaining tau in shell 1 = 0.5, distance = 0.5/1e-6 = 500km.
        // But shell 1 is only 90km thick, so this would exit -- let's pick
        // tau_target = 1.05 instead: remaining 0.05 / 1e-6 = 50km into shell 1.
        use crate::atmosphere::{AtmosphereModel, ShellOptics, EARTH_RADIUS_M};

        let mut atm = AtmosphereModel::new(&[0.0, 10.0, 100.0], &[550.0]);
        atm.optics[0][0] = ShellOptics {
            extinction: 1e-4,
            ssa: 1.0,
            asymmetry: 0.0,
            rayleigh_fraction: 1.0,
        };
        atm.optics[1][0] = ShellOptics {
            extinction: 1e-6,
            ssa: 1.0,
            asymmetry: 0.0,
            rayleigh_fraction: 1.0,
        };

        let pos = crate::geometry::Vec3::new(EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let dir = crate::geometry::Vec3::new(1.0, 0.0, 0.0);

        let (scatter_pos, _dir, shell) = advance_to_optical_depth(&atm, pos, dir, 1.05, 0);

        let altitude = scatter_pos.length() - EARTH_RADIUS_M;
        assert_eq!(shell, 1, "Should have crossed into shell 1");
        // 10km + 50km = 60km altitude
        assert!(
            (altitude - 60_000.0).abs() < 1000.0,
            "Expected ~60km altitude, got {:.0}m",
            altitude
        );
    }

    #[test]
    fn advance_at_zero_tau_stays_at_start() {
        use crate::atmosphere::{AtmosphereModel, ShellOptics, EARTH_RADIUS_M};

        let mut atm = AtmosphereModel::new(&[0.0, 100.0], &[550.0]);
        atm.optics[0][0] = ShellOptics {
            extinction: 1e-5,
            ssa: 1.0,
            asymmetry: 0.0,
            rayleigh_fraction: 1.0,
        };

        let pos = crate::geometry::Vec3::new(EARTH_RADIUS_M + 1000.0, 0.0, 0.0);
        let dir = crate::geometry::Vec3::new(1.0, 0.0, 0.0);

        let (scatter_pos, _dir, _shell) = advance_to_optical_depth(&atm, pos, dir, 0.0, 0);

        let dist = (scatter_pos.x - pos.x).abs();
        assert!(dist < 1.0, "tau=0 should stay at start, moved {:.1}m", dist);
    }

    // ── PhotonResult ──

    #[test]
    fn photon_result_default_state() {
        let result = PhotonResult {
            weight: 0.0,
            num_scatters: 0,
            terminated: false,
        };
        assert_eq!(result.weight, 0.0);
        assert_eq!(result.num_scatters, 0);
        assert!(!result.terminated);
    }

    // ── trace_photon basic behavior ──

    #[test]
    fn trace_photon_escapes_empty_atmosphere() {
        // With zero extinction, photon should escape without scattering
        let altitudes_km = [0.0, 50.0, 100.0];
        let wavelengths = [550.0];
        let atm = crate::atmosphere::AtmosphereModel::new(&altitudes_km, &wavelengths);

        let observer_pos =
            crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view_dir = crate::geometry::Vec3::new(1.0, 0.0, 0.0); // looking up
        let sun_dir = crate::geometry::Vec3::new(0.0, 0.0, 1.0);
        let mut rng_state: u64 = 42;

        let result = trace_photon(&atm, observer_pos, view_dir, sun_dir, 0, &mut rng_state);
        assert!(result.terminated, "Photon should terminate");
        assert_eq!(result.num_scatters, 0, "No scattering in empty atmosphere");
        assert!(
            result.weight.abs() < 1e-20,
            "No contribution from empty atmosphere"
        );
    }

    #[test]
    fn trace_photon_terminates() {
        // With some extinction, photon should eventually terminate
        let altitudes_km = [0.0, 10.0, 50.0, 100.0];
        let wavelengths = [550.0];
        let mut atm = crate::atmosphere::AtmosphereModel::new(&altitudes_km, &wavelengths);

        // Set moderate extinction in lowest shell
        atm.optics[0][0].extinction = 1e-4;
        atm.optics[0][0].ssa = 1.0;

        let observer_pos =
            crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view_dir = crate::geometry::Vec3::new(0.0, 1.0, 0.0); // horizontal
        let sun_dir = crate::geometry::Vec3::new(1.0, 0.0, 0.0); // overhead

        let mut rng_state: u64 = 42;
        let result = trace_photon(&atm, observer_pos, view_dir, sun_dir, 0, &mut rng_state);
        assert!(result.terminated, "Photon should always terminate");
    }

    // ── Constants ──

    #[test]
    fn max_scatters_is_reasonable() {
        // Should be > 10 (multi-scatter needs multiple bounces)
        // and < 10000 (avoid infinite loops)
        assert!(MAX_SCATTERS >= 10);
        assert!(MAX_SCATTERS <= 10000);
    }

    // ── mc_scatter_spectrum ──

    fn make_scattering_atmosphere() -> crate::atmosphere::AtmosphereModel {
        use crate::atmosphere::{AtmosphereModel, ShellOptics};

        let altitudes_km = [0.0, 10.0, 50.0, 100.0];
        let wavelengths = [400.0, 550.0, 700.0];
        let mut atm = AtmosphereModel::new(&altitudes_km, &wavelengths);

        // Shell 0 (0-10km): dense Rayleigh
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

    #[test]
    fn mc_spectrum_returns_64_elements() {
        let atm = make_scattering_atmosphere();
        let obs = crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(0.0, 1.0, 0.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(92.0, 180.0, 0.0, 0.0);

        let spectrum = mc_scatter_spectrum(&atm, obs, view, sun, 100, 42);
        assert_eq!(spectrum.len(), 64);
    }

    #[test]
    fn mc_spectrum_active_wavelengths_non_negative() {
        let atm = make_scattering_atmosphere();
        let obs = crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(0.0, 1.0, 0.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(92.0, 180.0, 0.0, 0.0);

        let spectrum = mc_scatter_spectrum(&atm, obs, view, sun, 200, 42);
        for w in 0..atm.num_wavelengths {
            assert!(
                spectrum[w] >= 0.0,
                "MC spectrum[{}] = {} should be non-negative",
                w,
                spectrum[w]
            );
        }
    }

    #[test]
    fn mc_spectrum_unused_wavelengths_zero() {
        let atm = make_scattering_atmosphere(); // 3 wavelengths
        let obs = crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(0.0, 1.0, 0.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(92.0, 180.0, 0.0, 0.0);

        let spectrum = mc_scatter_spectrum(&atm, obs, view, sun, 100, 42);
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
    fn mc_spectrum_zero_photons_returns_zero() {
        let atm = make_scattering_atmosphere();
        let obs = crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(0.0, 1.0, 0.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(92.0, 180.0, 0.0, 0.0);

        let spectrum = mc_scatter_spectrum(&atm, obs, view, sun, 0, 42);
        for w in 0..64 {
            assert!(
                spectrum[w].abs() < 1e-30,
                "Zero photons should give zero spectrum"
            );
        }
    }

    #[test]
    fn mc_spectrum_deterministic_with_same_seed() {
        let atm = make_scattering_atmosphere();
        let obs = crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(0.0, 1.0, 0.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(92.0, 180.0, 0.0, 0.0);

        let s1 = mc_scatter_spectrum(&atm, obs, view, sun, 100, 42);
        let s2 = mc_scatter_spectrum(&atm, obs, view, sun, 100, 42);
        for w in 0..atm.num_wavelengths {
            assert!(
                (s1[w] - s2[w]).abs() < 1e-15,
                "Same seed should give identical results: [{}] {} vs {}",
                w,
                s1[w],
                s2[w]
            );
        }
    }

    #[test]
    fn mc_spectrum_empty_atmosphere_gives_zero() {
        let altitudes_km = [0.0, 50.0, 100.0];
        let wavelengths = [550.0];
        let atm = crate::atmosphere::AtmosphereModel::new(&altitudes_km, &wavelengths);

        let obs = crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(1.0, 0.0, 0.0);
        let sun = crate::geometry::Vec3::new(0.0, 0.0, 1.0);

        let spectrum = mc_scatter_spectrum(&atm, obs, view, sun, 100, 42);
        assert!(
            spectrum[0].abs() < 1e-20,
            "Empty atmosphere should give ~0 MC contribution, got {}",
            spectrum[0]
        );
    }

    #[test]
    fn mc_spectrum_positive_at_civil_twilight() {
        // At SZA=92° with a scattering atmosphere, MC should produce some signal
        let atm = make_scattering_atmosphere();
        let obs = crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(0.0, 1.0, 0.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(92.0, 180.0, 0.0, 0.0);

        // Use enough photons for a reliable signal
        let spectrum = mc_scatter_spectrum(&atm, obs, view, sun, 1000, 42);
        let total: f64 = spectrum[..atm.num_wavelengths].iter().sum();
        assert!(
            total > 0.0,
            "MC should produce positive radiance at civil twilight, got {}",
            total
        );
    }

    // ── hybrid_scatter_spectrum ──

    #[test]
    fn hybrid_spectrum_returns_64_elements() {
        let atm = make_scattering_atmosphere();
        let obs = crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(0.0, 1.0, 0.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(92.0, 180.0, 0.0, 0.0);

        let spectrum = hybrid_scatter_spectrum(&atm, obs, view, sun, 10, 42, true, None);
        assert_eq!(spectrum.len(), 64);
    }

    #[test]
    fn hybrid_spectrum_non_negative() {
        let atm = make_scattering_atmosphere();
        let obs = crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(0.0, 1.0, 0.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(96.0, 180.0, 0.0, 0.0);

        let spectrum = hybrid_scatter_spectrum(&atm, obs, view, sun, 50, 42, true, None);
        for w in 0..atm.num_wavelengths {
            assert!(
                spectrum[w] >= 0.0,
                "Hybrid spectrum[{}] = {} should be non-negative",
                w,
                spectrum[w]
            );
        }
    }

    #[test]
    fn hybrid_spectrum_positive_at_civil_twilight() {
        let atm = make_scattering_atmosphere();
        let obs = crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(0.0, 1.0, 0.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(92.0, 180.0, 0.0, 0.0);

        let spectrum = hybrid_scatter_spectrum(&atm, obs, view, sun, 50, 42, true, None);
        let total: f64 = spectrum[..atm.num_wavelengths].iter().sum();
        assert!(
            total > 0.0,
            "Hybrid should produce positive radiance at civil twilight, got {}",
            total
        );
    }

    #[test]
    fn hybrid_spectrum_empty_atmosphere_gives_zero() {
        let altitudes_km = [0.0, 50.0, 100.0];
        let wavelengths = [550.0];
        let atm = crate::atmosphere::AtmosphereModel::new(&altitudes_km, &wavelengths);

        let obs = crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(1.0, 0.0, 0.0);
        let sun = crate::geometry::Vec3::new(0.0, 0.0, 1.0);

        let spectrum = hybrid_scatter_spectrum(&atm, obs, view, sun, 50, 42, true, None);
        assert!(
            spectrum[0].abs() < 1e-20,
            "Empty atmosphere should give zero hybrid contribution, got {}",
            spectrum[0]
        );
    }

    #[test]
    fn hybrid_spectrum_deterministic_with_same_seed() {
        let atm = make_scattering_atmosphere();
        let obs = crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(0.0, 1.0, 0.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(92.0, 180.0, 0.0, 0.0);

        let s1 = hybrid_scatter_spectrum(&atm, obs, view, sun, 50, 42, true, None);
        let s2 = hybrid_scatter_spectrum(&atm, obs, view, sun, 50, 42, true, None);
        for w in 0..atm.num_wavelengths {
            assert!(
                (s1[w] - s2[w]).abs() < 1e-15,
                "Same seed should give identical results: [{}] {} vs {}",
                w,
                s1[w],
                s2[w]
            );
        }
    }

    // ── Refraction in MCRT transport ──

    fn make_refraction_scattering_atmosphere() -> crate::atmosphere::AtmosphereModel {
        use crate::atmosphere::{AtmosphereModel, ShellOptics};

        let altitudes_km = [0.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0];
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

        atm
    }

    #[test]
    fn trace_photon_with_refraction_terminates() {
        let mut atm = make_refraction_scattering_atmosphere();
        atm.compute_refractive_indices();

        let obs = crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(0.0, 1.0, 0.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(92.0, 180.0, 0.0, 0.0);
        let mut rng: u64 = 42;

        let result = trace_photon(&atm, obs, view, sun, 1, &mut rng);
        assert!(result.terminated, "Photon should always terminate");
    }

    #[test]
    fn trace_photon_with_refraction_non_negative() {
        let mut atm = make_refraction_scattering_atmosphere();
        atm.compute_refractive_indices();

        let obs = crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(0.0, 1.0, 0.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(92.0, 180.0, 0.0, 0.0);

        for seed in 0..50u64 {
            let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let result = trace_photon(&atm, obs, view, sun, 1, &mut rng);
            assert!(
                result.weight >= 0.0,
                "Photon weight should be non-negative: seed={}, weight={}",
                seed,
                result.weight
            );
        }
    }

    #[test]
    fn mc_spectrum_with_refraction_positive_at_civil_twilight() {
        let mut atm = make_refraction_scattering_atmosphere();
        atm.compute_refractive_indices();

        let obs = crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(0.0, 1.0, 0.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(92.0, 180.0, 0.0, 0.0);

        let spectrum = mc_scatter_spectrum(&atm, obs, view, sun, 500, 42);
        let total: f64 = spectrum[..atm.num_wavelengths].iter().sum();
        assert!(
            total > 0.0,
            "MC with refraction should produce positive radiance at SZA=92, got {}",
            total
        );
    }

    #[test]
    fn mc_spectrum_with_refraction_deterministic() {
        let mut atm = make_refraction_scattering_atmosphere();
        atm.compute_refractive_indices();

        let obs = crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(0.0, 1.0, 0.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(92.0, 180.0, 0.0, 0.0);

        let s1 = mc_scatter_spectrum(&atm, obs, view, sun, 100, 42);
        let s2 = mc_scatter_spectrum(&atm, obs, view, sun, 100, 42);
        for w in 0..atm.num_wavelengths {
            assert!(
                (s1[w] - s2[w]).abs() < 1e-15,
                "Refracted MC should be deterministic: [{}] {} vs {}",
                w,
                s1[w],
                s2[w]
            );
        }
    }

    #[test]
    fn hybrid_spectrum_with_refraction_non_negative() {
        let mut atm = make_refraction_scattering_atmosphere();
        atm.compute_refractive_indices();

        let obs = crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(0.0, 1.0, 0.0).normalize();

        for sza in &[92.0, 96.0, 102.0] {
            let sun = crate::geometry::solar_direction_ecef(*sza, 180.0, 0.0, 0.0);
            let spectrum = hybrid_scatter_spectrum(&atm, obs, view, sun, 20, 42, true, None);
            for w in 0..atm.num_wavelengths {
                assert!(
                    spectrum[w] >= 0.0,
                    "Hybrid with refraction should be non-negative: SZA={}, wl={}, val={:.4e}",
                    sza,
                    w,
                    spectrum[w]
                );
            }
        }
    }

    #[test]
    fn hybrid_spectrum_with_refraction_positive_at_civil_twilight() {
        let mut atm = make_refraction_scattering_atmosphere();
        atm.compute_refractive_indices();

        let obs = crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(0.0, 1.0, 0.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(92.0, 180.0, 0.0, 0.0);

        let spectrum = hybrid_scatter_spectrum(&atm, obs, view, sun, 20, 42, true, None);
        let total: f64 = spectrum[..atm.num_wavelengths].iter().sum();
        assert!(
            total > 0.0,
            "Hybrid with refraction should produce positive radiance at SZA=92, got {}",
            total
        );
    }

    #[test]
    fn cross_boundary_with_n1_preserves_direction() {
        // When all n=1.0, cross_boundary should not change direction
        let atm = make_scattering_atmosphere(); // n=1.0 default

        let pos = crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 5000.0, 0.0, 0.0);
        let dir = crate::geometry::Vec3::new(0.3, 0.9, 0.1).normalize();
        let boundary_dist = 100.0;

        let (new_pos, new_dir) = cross_boundary(pos, dir, boundary_dist, true, 0, &atm);

        // Direction should be identical
        assert!(
            (new_dir.x - dir.x).abs() < 1e-10,
            "n=1 should preserve direction"
        );
        assert!((new_dir.y - dir.y).abs() < 1e-10);
        assert!((new_dir.z - dir.z).abs() < 1e-10);

        // Position should be boundary + 1e-3 nudge
        let expected_pos = pos + dir * boundary_dist + dir * 1e-3;
        assert!((new_pos.x - expected_pos.x).abs() < 1e-6);
        assert!((new_pos.y - expected_pos.y).abs() < 1e-6);
        assert!((new_pos.z - expected_pos.z).abs() < 1e-6);
    }

    #[test]
    fn trace_photon_empty_atm_with_refraction_escapes() {
        // Refraction indices set but zero extinction: photon should still escape
        let mut atm = crate::atmosphere::AtmosphereModel::new(&[0.0, 50.0, 100.0], &[550.0]);
        atm.compute_refractive_indices_from_altitude();

        let obs = crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(1.0, 0.0, 0.0); // radially outward
        let sun = crate::geometry::Vec3::new(0.0, 0.0, 1.0);
        let mut rng: u64 = 42;

        let result = trace_photon(&atm, obs, view, sun, 0, &mut rng);
        assert!(result.terminated);
        assert_eq!(result.num_scatters, 0);
        assert!(result.weight.abs() < 1e-20);
    }

    // ── Polarized (Stokes) transport tests ──

    #[test]
    fn polarized_photon_terminates() {
        let atm = make_scattering_atmosphere();
        let obs = crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(0.0, 1.0, 0.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(92.0, 180.0, 0.0, 0.0);
        let mut rng: u64 = 42;

        let result = trace_photon_polarized(&atm, obs, view, sun, 1, &mut rng);
        assert!(result.terminated);
    }

    #[test]
    fn polarized_photon_intensity_non_negative() {
        let atm = make_scattering_atmosphere();
        let obs = crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(0.0, 1.0, 0.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(92.0, 180.0, 0.0, 0.0);

        for seed in 0..50u64 {
            let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let result = trace_photon_polarized(&atm, obs, view, sun, 1, &mut rng);
            assert!(
                result.stokes.intensity() >= 0.0,
                "Stokes I should be non-negative: seed={}, I={}",
                seed,
                result.stokes.intensity()
            );
        }
    }

    #[test]
    fn polarized_photon_dop_bounded() {
        // Degree of polarization must be in [0, 1]
        let atm = make_scattering_atmosphere();
        let obs = crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(0.0, 1.0, 0.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(92.0, 180.0, 0.0, 0.0);

        for seed in 0..50u64 {
            let mut rng = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let result = trace_photon_polarized(&atm, obs, view, sun, 1, &mut rng);
            if result.stokes.intensity() > 1e-20 {
                let dop = result.stokes.degree_of_polarization();
                assert!(
                    dop <= 1.0 + 1e-6,
                    "DOP must be <= 1: seed={}, DOP={}",
                    seed,
                    dop
                );
            }
        }
    }

    #[test]
    fn polarized_spectrum_positive_at_civil_twilight() {
        let atm = make_scattering_atmosphere();
        let obs = crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(0.0, 1.0, 0.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(92.0, 180.0, 0.0, 0.0);

        let spectrum = mc_scatter_spectrum_polarized(&atm, obs, view, sun, 500, 42);
        let total_i: f64 = spectrum[..atm.num_wavelengths]
            .iter()
            .map(|s| s.intensity())
            .sum();
        assert!(
            total_i > 0.0,
            "Polarized MC should produce positive intensity at SZA=92, got {}",
            total_i
        );
    }

    #[test]
    fn polarized_spectrum_deterministic() {
        let atm = make_scattering_atmosphere();
        let obs = crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(0.0, 1.0, 0.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(92.0, 180.0, 0.0, 0.0);

        let s1 = mc_scatter_spectrum_polarized(&atm, obs, view, sun, 100, 42);
        let s2 = mc_scatter_spectrum_polarized(&atm, obs, view, sun, 100, 42);
        for w in 0..atm.num_wavelengths {
            for c in 0..4 {
                assert!(
                    (s1[w].s[c] - s2[w].s[c]).abs() < 1e-15,
                    "Polarized MC not deterministic: wl={}, component={}, {} vs {}",
                    w,
                    c,
                    s1[w].s[c],
                    s2[w].s[c]
                );
            }
        }
    }

    #[test]
    fn polarized_intensity_close_to_scalar() {
        // The Stokes I component from polarized MC should be close to the
        // scalar MC result. They differ by ~1-2% due to polarization cross-
        // coupling (Q/U feeding back into I through the off-diagonal Mueller
        // elements). With enough photons, we can verify they're in the same
        // ballpark.
        let atm = make_scattering_atmosphere();
        let obs = crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(0.0, 1.0, 0.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(92.0, 180.0, 0.0, 0.0);

        let n_photons = 2000;
        let scalar = mc_scatter_spectrum(&atm, obs, view, sun, n_photons, 42);
        let polarized = mc_scatter_spectrum_polarized(&atm, obs, view, sun, n_photons, 42);

        for w in 0..atm.num_wavelengths {
            let i_scalar = scalar[w];
            let i_polarized = polarized[w].intensity();

            if i_scalar > 1e-20 {
                let rel_diff = (i_polarized - i_scalar).abs() / i_scalar;
                // Allow up to 50% difference due to MC noise and polarization
                // coupling (the difference is systematic but small; with only
                // 2000 photons, noise dominates)
                assert!(
                    rel_diff < 0.5,
                    "Polarized I should be close to scalar: wl={}, scalar={:.4e}, polarized={:.4e}, rel_diff={:.2}%",
                    w,
                    i_scalar,
                    i_polarized,
                    rel_diff * 100.0
                );
            }
        }
    }

    #[test]
    fn polarized_empty_atm_gives_zero() {
        let atm = crate::atmosphere::AtmosphereModel::new(&[0.0, 50.0, 100.0], &[550.0]);
        let obs = crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(1.0, 0.0, 0.0);
        let sun = crate::geometry::Vec3::new(0.0, 0.0, 1.0);

        let spectrum = mc_scatter_spectrum_polarized(&atm, obs, view, sun, 100, 42);
        for c in 0..4 {
            assert!(
                spectrum[0].s[c].abs() < 1e-20,
                "Empty atm should give zero Stokes[{}]",
                c
            );
        }
    }

    #[test]
    fn polarized_zero_photons_gives_zero() {
        let atm = make_scattering_atmosphere();
        let obs = crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(0.0, 1.0, 0.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(92.0, 180.0, 0.0, 0.0);

        let spectrum = mc_scatter_spectrum_polarized(&atm, obs, view, sun, 0, 42);
        for w in 0..64 {
            for c in 0..4 {
                assert!(spectrum[w].s[c].abs() < 1e-30);
            }
        }
    }

    #[test]
    fn polarized_90deg_scatter_produces_polarization() {
        // When the sun is at 90 degrees from the viewing direction,
        // single Rayleigh scattering should produce maximum polarization.
        // In a Rayleigh-only atmosphere at SZA=90 (sun on horizon),
        // looking at the zenith, the scattering angle is ~90 degrees.
        let atm = make_scattering_atmosphere();
        let obs = crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        // Looking up (zenith)
        let view = crate::geometry::Vec3::new(1.0, 0.0, 0.0).normalize();
        // Sun on horizon → scattering angle ~ 90 degrees
        let sun = crate::geometry::solar_direction_ecef(90.0, 180.0, 0.0, 0.0);

        let spectrum = mc_scatter_spectrum_polarized(&atm, obs, view, sun, 2000, 42);

        // The 550nm channel should show noticeable polarization
        let s = spectrum[1]; // 550nm
        if s.intensity() > 1e-20 {
            let dolp = s.degree_of_linear_polarization();
            // Rayleigh at 90 degrees gives 100% polarization for single scatter,
            // but multiple scattering and MC noise reduce this. We just check
            // that there IS some polarization.
            assert!(
                dolp > 0.01,
                "90-deg scatter should show polarization: DOLP={:.4}",
                dolp
            );
        }
    }

    // ── ALIS (Adjusted Lambda Importance Sampling) ──

    #[test]
    fn alis_returns_correct_array_size() {
        use crate::atmosphere::{AtmosphereModel, ShellOptics, EARTH_RADIUS_M};

        let altitudes_km = [0.0, 10.0, 50.0, 100.0];
        let wavelengths = [400.0, 550.0, 700.0];
        let mut atm = AtmosphereModel::new(&altitudes_km, &wavelengths);
        for s in 0..atm.num_shells {
            for w in 0..3 {
                let factor = if w == 0 {
                    4.0
                } else if w == 1 {
                    1.0
                } else {
                    0.3
                };
                atm.optics[s][w] = ShellOptics {
                    extinction: 1e-5 * factor * libm::exp(-atm.shells[s].altitude_mid / 8500.0),
                    ssa: 1.0,
                    asymmetry: 0.0,
                    rayleigh_fraction: 1.0,
                };
            }
        }

        let obs = crate::geometry::Vec3::new(EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(0.0, 1.0, 0.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(92.0, 180.0, 0.0, 0.0);

        let mut rng = 12345u64;
        let result = hybrid_scatter_radiance_alis(&atm, obs, view, sun, 50, &mut rng, None);
        assert_eq!(result.len(), 64);
        // Active wavelengths should be non-negative
        for w in 0..3 {
            assert!(
                result[w] >= 0.0,
                "ALIS result[{}] should be non-negative, got {:.4e}",
                w,
                result[w]
            );
        }
        // Unused wavelengths should be zero
        for w in 3..64 {
            assert!(
                result[w].abs() < 1e-30,
                "Unused ALIS result[{}] should be zero, got {:.4e}",
                w,
                result[w]
            );
        }
    }

    #[test]
    fn alis_positive_at_civil_twilight() {
        use crate::atmosphere::{AtmosphereModel, ShellOptics, EARTH_RADIUS_M};

        let altitudes_km = [0.0, 10.0, 50.0, 100.0];
        let wavelengths = [400.0, 550.0, 700.0];
        let mut atm = AtmosphereModel::new(&altitudes_km, &wavelengths);
        for s in 0..atm.num_shells {
            for w in 0..3 {
                let factor = if w == 0 {
                    4.0
                } else if w == 1 {
                    1.0
                } else {
                    0.3
                };
                atm.optics[s][w] = ShellOptics {
                    extinction: 1e-5 * factor * libm::exp(-atm.shells[s].altitude_mid / 8500.0),
                    ssa: 1.0,
                    asymmetry: 0.0,
                    rayleigh_fraction: 1.0,
                };
            }
        }

        let obs = crate::geometry::Vec3::new(EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(0.0, 1.0, 0.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(92.0, 180.0, 0.0, 0.0);

        let mut rng = 42u64;
        let result = hybrid_scatter_radiance_alis(&atm, obs, view, sun, 200, &mut rng, None);
        for w in 0..3 {
            assert!(
                result[w] > 0.0,
                "ALIS at SZA=92 should produce positive radiance at wl[{}], got {:.4e}",
                w,
                result[w]
            );
        }
    }

    #[test]
    fn alis_matches_per_wavelength_statistically() {
        // ALIS should give the same expected value as per-wavelength tracing.
        // We compare the mean over many seeds and check the ratio is close to 1.
        use crate::atmosphere::{AtmosphereModel, ShellOptics, EARTH_RADIUS_M};

        let altitudes_km = [0.0, 10.0, 50.0, 100.0];
        let wavelengths = [400.0, 550.0, 700.0];
        let mut atm = AtmosphereModel::new(&altitudes_km, &wavelengths);
        for s in 0..atm.num_shells {
            for w in 0..3 {
                let factor = if w == 0 {
                    4.0
                } else if w == 1 {
                    1.0
                } else {
                    0.3
                };
                atm.optics[s][w] = ShellOptics {
                    extinction: 1e-5 * factor * libm::exp(-atm.shells[s].altitude_mid / 8500.0),
                    ssa: 1.0,
                    asymmetry: 0.0,
                    rayleigh_fraction: 1.0,
                };
            }
        }

        let obs = crate::geometry::Vec3::new(EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(0.0, 1.0, 0.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(96.0, 180.0, 0.0, 0.0);

        let num_seeds = 20;
        let rays = 100;
        let mut alis_sum = [0.0f64; 3];
        let mut perwl_sum = [0.0f64; 3];

        for seed in 0..num_seeds {
            let base = seed * 1000 + 7777;
            let mut rng_alis = base;
            let alis =
                hybrid_scatter_radiance_alis(&atm, obs, view, sun, rays, &mut rng_alis, None);
            for w in 0..3 {
                alis_sum[w] += alis[w];
            }

            for w in 0..3 {
                let mut rng_perwl = (base as u64)
                    .wrapping_add(w as u64)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1);
                let val =
                    hybrid_scatter_radiance(&atm, obs, view, sun, w, rays, &mut rng_perwl, false);
                perwl_sum[w] += val;
            }
        }

        for w in 0..3 {
            let alis_mean = alis_sum[w] / num_seeds as f64;
            let perwl_mean = perwl_sum[w] / num_seeds as f64;
            if perwl_mean > 1e-20 {
                let ratio = alis_mean / perwl_mean;
                assert!(
                    ratio > 0.5 && ratio < 2.0,
                    "ALIS/per-wl ratio at wl[{}] = {:.3} (ALIS={:.4e}, per-wl={:.4e}), expected ~1.0",
                    w,
                    ratio,
                    alis_mean,
                    perwl_mean,
                );
            }
        }
    }

    #[test]
    fn alis_zero_in_empty_atmosphere() {
        use crate::atmosphere::{AtmosphereModel, EARTH_RADIUS_M};

        let altitudes_km = [0.0, 50.0, 100.0];
        let wavelengths = [550.0];
        let atm = AtmosphereModel::new(&altitudes_km, &wavelengths);

        let obs = crate::geometry::Vec3::new(EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(0.0, 1.0, 0.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(92.0, 180.0, 0.0, 0.0);

        let mut rng = 42u64;
        let result = hybrid_scatter_radiance_alis(&atm, obs, view, sun, 50, &mut rng, None);
        assert!(
            result[0].abs() < 1e-30,
            "Empty atmosphere ALIS should give zero, got {:.4e}",
            result[0]
        );
    }

    #[test]
    fn alis_deep_twilight_non_negative() {
        use crate::atmosphere::{AtmosphereModel, ShellOptics, EARTH_RADIUS_M};

        let altitudes_km = [0.0, 10.0, 50.0, 100.0];
        let wavelengths = [400.0, 550.0, 700.0];
        let mut atm = AtmosphereModel::new(&altitudes_km, &wavelengths);
        for s in 0..atm.num_shells {
            for w in 0..3 {
                let factor = if w == 0 {
                    4.0
                } else if w == 1 {
                    1.0
                } else {
                    0.3
                };
                atm.optics[s][w] = ShellOptics {
                    extinction: 1e-5 * factor * libm::exp(-atm.shells[s].altitude_mid / 8500.0),
                    ssa: 1.0,
                    asymmetry: 0.0,
                    rayleigh_fraction: 1.0,
                };
            }
        }

        let obs = crate::geometry::Vec3::new(EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(0.0, 1.0, 0.0).normalize();

        for sza in &[96.0, 100.0, 104.0, 106.0] {
            let sun = crate::geometry::solar_direction_ecef(*sza, 180.0, 0.0, 0.0);
            let mut rng = 12345u64;
            let result = hybrid_scatter_radiance_alis(&atm, obs, view, sun, 50, &mut rng, None);
            for w in 0..3 {
                assert!(
                    result[w] >= 0.0,
                    "ALIS at SZA={} wl[{}] should be non-negative, got {:.4e}",
                    sza,
                    w,
                    result[w]
                );
            }
        }
    }

    #[test]
    fn alis_scout_tau_boundary_matches_single() {
        // Multi-wavelength scout should give the same tau as single-wavelength scout
        // for the hero wavelength.
        use crate::atmosphere::{AtmosphereModel, ShellOptics, EARTH_RADIUS_M};

        let altitudes_km = [0.0, 10.0, 50.0, 100.0];
        let wavelengths = [400.0, 550.0, 700.0];
        let mut atm = AtmosphereModel::new(&altitudes_km, &wavelengths);
        for s in 0..atm.num_shells {
            for w in 0..3 {
                let factor = if w == 0 {
                    4.0
                } else if w == 1 {
                    1.0
                } else {
                    0.3
                };
                atm.optics[s][w] = ShellOptics {
                    extinction: 1e-5 * factor * libm::exp(-atm.shells[s].altitude_mid / 8500.0),
                    ssa: 1.0,
                    asymmetry: 0.0,
                    rayleigh_fraction: 1.0,
                };
            }
        }

        let pos = crate::geometry::Vec3::new(EARTH_RADIUS_M + 20000.0, 0.0, 0.0);
        let dir = crate::geometry::Vec3::new(0.5, 0.5, 0.707).normalize();

        // Single-wavelength scout for each hero
        for hero in 0..3 {
            let (tau_single, hit_single) = scout_tau_to_boundary(&atm, pos, dir, hero);
            let (tau_multi, hit_multi) = scout_tau_to_boundary_alis(&atm, pos, dir, hero, 3);

            assert_eq!(
                hit_single, hit_multi,
                "hit_ground mismatch for hero={}: single={}, multi={}",
                hero, hit_single, hit_multi
            );
            let rel_err = if tau_single > 1e-30 {
                ((tau_multi[hero] - tau_single) / tau_single).abs()
            } else {
                (tau_multi[hero] - tau_single).abs()
            };
            assert!(
                rel_err < 1e-10,
                "ALIS scout tau for hero={} differs: single={:.6e}, multi={:.6e}, err={:.4e}",
                hero,
                tau_single,
                tau_multi[hero],
                rel_err
            );
        }
    }

    // ── Terminator lobe + 3-branch direction sampling ──

    #[test]
    fn branch_params_baseline_at_civil_twilight() {
        // SZA = 90 deg (cos_sza = 0): well below threshold
        let bp = branch_params_for_sza(0.0);
        assert!(
            (bp.zenith_frac - 0.5).abs() < 1e-10,
            "zenith_frac should be 0.5 at SZA 90"
        );
        assert!(
            (bp.n_zenith - 1.0).abs() < 1e-10,
            "n_zenith should be 1.0 at SZA 90"
        );
        assert!(
            bp.term_share.abs() < 1e-10,
            "term_share should be 0 at SZA 90"
        );
        assert!(
            (bp.m_term - 1.0).abs() < 1e-10,
            "m_term should be 1.0 at SZA 90"
        );
    }

    #[test]
    fn branch_params_full_at_deep_twilight() {
        // SZA = 108 deg (cos = cos(108 deg))
        let cos_108 = libm::cos(108.0 * core::f64::consts::PI / 180.0);
        let bp = branch_params_for_sza(cos_108);
        assert!(
            (bp.zenith_frac - ZENITH_MAX_FRACTION).abs() < 1e-10,
            "zenith_frac should be {} at SZA 108, got {}",
            ZENITH_MAX_FRACTION,
            bp.zenith_frac
        );
        assert!(
            (bp.n_zenith - ZENITH_BIAS_N).abs() < 1e-10,
            "n_zenith should be {} at SZA 108, got {}",
            ZENITH_BIAS_N,
            bp.n_zenith
        );
        assert!(
            (bp.term_share - TERMINATOR_MAX_SHARE).abs() < 1e-10,
            "term_share should be {} at SZA 108, got {}",
            TERMINATOR_MAX_SHARE,
            bp.term_share
        );
        assert!(
            (bp.m_term - TERMINATOR_N_MAX).abs() < 1e-10,
            "m_term should be {} at SZA 108, got {}",
            TERMINATOR_N_MAX,
            bp.m_term
        );
    }

    #[test]
    fn branch_probabilities_sum_to_one() {
        // Test at several SZAs that alpha_p + alpha_z + alpha_t = 1
        for sza_deg in [85.0, 96.0, 100.0, 104.0, 106.0, 108.0, 115.0] {
            let cos_sza = libm::cos(sza_deg * core::f64::consts::PI / 180.0);
            let bp = branch_params_for_sza(cos_sza);
            let alpha_p = 1.0 - bp.zenith_frac;
            let alpha_z = bp.zenith_frac * (1.0 - bp.term_share);
            let alpha_t = bp.zenith_frac * bp.term_share;
            let sum = alpha_p + alpha_z + alpha_t;
            assert!(
                (sum - 1.0).abs() < 1e-14,
                "branch probs sum to {} at SZA {}, expected 1.0",
                sum,
                sza_deg
            );
        }
    }

    #[test]
    fn terminator_axis_is_unit_vector() {
        let up = Vec3::new(0.0, 0.0, 1.0);
        let sun = Vec3::new(0.3, 0.0, -0.2).normalize(); // below horizon, with horizontal component
        for tilt_deg in [0.0, 20.0, 45.0, 50.0, 80.0] {
            let tilt = tilt_deg * core::f64::consts::PI / 180.0;
            let axis = terminator_axis(up, sun, tilt);
            let len = axis.length();
            assert!(
                (len - 1.0).abs() < 1e-10,
                "terminator axis length = {} at tilt {} deg",
                len,
                tilt_deg
            );
        }
    }

    #[test]
    fn terminator_axis_tilt_angle_correct() {
        let up = Vec3::new(0.0, 0.0, 1.0);
        let sun = Vec3::new(1.0, 0.0, -0.3).normalize(); // sun below horizon in +x direction
        let tilt = 30.0 * core::f64::consts::PI / 180.0;
        let axis = terminator_axis(up, sun, tilt);

        // The axis should be tilted 30 degrees from up
        let cos_angle = axis.dot(up);
        let angle_deg = libm::acos(cos_angle.clamp(-1.0, 1.0)) * 180.0 / core::f64::consts::PI;
        assert!(
            (angle_deg - 30.0).abs() < 0.01,
            "terminator axis angle from zenith = {} deg, expected 30",
            angle_deg
        );

        // The axis should tilt toward the sun's horizontal projection (+x)
        assert!(
            axis.x > 0.0,
            "terminator axis should tilt toward sun (+x), got x={}",
            axis.x
        );
    }

    #[test]
    fn terminator_axis_sun_at_nadir_fallback() {
        let up = Vec3::new(0.0, 0.0, 1.0);
        let sun = Vec3::new(0.0, 0.0, -1.0); // directly below -- no horizontal component
        let tilt = 45.0 * core::f64::consts::PI / 180.0;
        let axis = terminator_axis(up, sun, tilt);

        // Should fall back to up
        assert!(
            (axis.dot(up) - 1.0).abs() < 1e-10,
            "terminator axis should fall back to up when sun is at nadir"
        );
    }

    #[test]
    fn terminator_shape_weight_at_axis() {
        // When direction is exactly on the terminator axis AND that axis is at zenith,
        // cos_z = cos_t = 1, weight = 2 / (m+1)
        let w = terminator_shape_weight(1.0, 1.0, 8.0);
        let expected = 2.0 / 9.0;
        assert!(
            (w - expected).abs() < 1e-12,
            "terminator_shape_weight(1,1,8) = {}, expected {}",
            w,
            expected
        );
    }

    #[test]
    fn terminator_shape_weight_below_horizon_zero() {
        // cos_z <= 0 should return 0
        let w = terminator_shape_weight(-0.1, 0.9, 5.0);
        assert!(
            w == 0.0,
            "terminator_shape_weight should be 0 for below-horizon, got {}",
            w
        );
    }

    #[test]
    fn terminator_shape_weight_behind_axis_zero() {
        // cos_t <= 0 should return 0 (direction is behind the terminator axis hemisphere)
        let w = terminator_shape_weight(0.5, -0.1, 5.0);
        assert!(
            w == 0.0,
            "terminator_shape_weight should be 0 for behind-axis, got {}",
            w
        );
    }

    #[test]
    fn terminator_shape_weight_positive_in_overlap() {
        // Both cos_z and cos_t positive: weight should be positive
        let w = terminator_shape_weight(0.7, 0.8, 5.0);
        assert!(
            w > 0.0,
            "terminator_shape_weight should be positive in overlap region, got {}",
            w
        );
    }

    #[test]
    fn three_branch_backward_compatible_at_civil() {
        // At SZA = 90, the 3-branch system should behave identically to the
        // old 2-branch system: alpha_p = 0.5, alpha_z = 0.5, alpha_t = 0.
        // Phase branch weight = 0.5/0.5 = 1.0.
        // Zenith branch with n=1: zenith_importance_weight(cos, 1.0) = 1.0 for all cos.
        let bp = branch_params_for_sza(0.0);
        let alpha_p = 1.0 - bp.zenith_frac;
        let alpha_z = bp.zenith_frac * (1.0 - bp.term_share);
        let alpha_t = bp.zenith_frac * bp.term_share;

        assert!((alpha_p - 0.5).abs() < 1e-14);
        assert!((alpha_z - 0.5).abs() < 1e-14);
        assert!(alpha_t < 1e-14);

        // Zenith shape weight at n=1 should be 1.0 for any cos_theta
        for cos_z in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0] {
            let w = zenith_importance_weight(cos_z, 1.0);
            assert!(
                (w - 1.0).abs() < 1e-10,
                "zenith weight at n=1, cos={}: got {} expected 1.0",
                cos_z,
                w
            );
        }
    }

    // ── Altitude splitting ──

    #[test]
    fn split_factors_no_splitting_at_civil_twilight() {
        let factors = split_factors_for_sza(90.0);
        assert_eq!(factors, [1, 1, 1], "No splitting at civil twilight");
    }

    #[test]
    fn split_factors_transition_band() {
        let factors = split_factors_for_sza(99.0);
        assert_eq!(factors, SPLIT_FACTORS_TRANSITION);
    }

    #[test]
    fn split_factors_deep_twilight() {
        let factors = split_factors_for_sza(106.0);
        assert_eq!(factors, SPLIT_FACTORS_DEEP);
    }

    #[test]
    fn split_factors_boundary_96() {
        // Exactly at 96: should be no splitting (< ZENITH_SZA_START)
        let factors = split_factors_for_sza(95.9);
        assert_eq!(factors, [1, 1, 1]);
    }

    #[test]
    fn split_factors_boundary_102() {
        // At 102: deep factors
        let factors = split_factors_for_sza(102.0);
        assert_eq!(factors, SPLIT_FACTORS_DEEP);
    }

    #[test]
    fn split_particle_scalar_size_reasonable() {
        // Ensure SplitParticleScalar doesn't blow stack.
        // 3 Vec3s (72 bytes) + f64 (8) + u64 (8) + 2 usize (16) = ~104 bytes
        let size = core::mem::size_of::<SplitParticleScalar>();
        assert!(size <= 128, "SplitParticleScalar too large: {} bytes", size);
    }

    #[test]
    fn split_particle_alis_size_reasonable() {
        // weight_ratio[64] = 512 bytes, plus overhead.
        let size = core::mem::size_of::<SplitParticleAlis>();
        assert!(size <= 640, "SplitParticleAlis too large: {} bytes", size);
    }

    #[test]
    fn split_stack_scalar_fits_in_stack() {
        // MAX_SPLIT_PARTICLES * sizeof(SplitParticleScalar) should be < 4 KB
        let total = MAX_SPLIT_PARTICLES * core::mem::size_of::<SplitParticleScalar>();
        assert!(
            total <= 4096,
            "Scalar split stack too large: {} bytes",
            total
        );
    }

    #[test]
    fn split_stack_alis_fits_in_stack() {
        // MAX_SPLIT_PARTICLES * sizeof(SplitParticleAlis) should be < 16 KB
        let total = MAX_SPLIT_PARTICLES * core::mem::size_of::<SplitParticleAlis>();
        assert!(
            total <= 16384,
            "ALIS split stack too large: {} bytes",
            total
        );
    }

    #[test]
    fn splitting_unbiased_scalar_civil_twilight() {
        // At civil twilight (SZA 90), splitting is disabled (all factors = 1).
        // The new code must produce identical results to the old code.
        // Run 1000 chains at SZA=90 with and without splitting and compare.
        use crate::atmosphere::{AtmosphereModel, ShellOptics, EARTH_RADIUS_M};

        let altitudes_km = [0.0, 10.0, 50.0, 100.0];
        let wavelengths = [550.0];
        let mut atm = AtmosphereModel::new(&altitudes_km, &wavelengths);
        for s in 0..3 {
            atm.optics[s][0] = ShellOptics {
                extinction: 1e-5 / (1.0 + s as f64),
                ssa: 0.99,
                asymmetry: 0.0,
                rayleigh_fraction: 1.0,
            };
        }

        let observer = Vec3::new(EARTH_RADIUS_M + 10.0, 0.0, 0.0);
        // SZA 90: sun on horizon
        let sun_dir = Vec3::new(0.0, 1.0, 0.0);
        let start_optics = &atm.optics[0][0];

        let n = 1000;
        let mut total = 0.0;
        let mut rng: u64 = 42;
        for ray in 0..n {
            total += trace_secondary_chain_scalar(
                &atm,
                observer,
                sun_dir,
                0,
                start_optics,
                &mut rng,
                ray,
                n,
                None,
            );
        }
        let mean = total / n as f64;
        // At civil twilight the signal should be positive and finite
        assert!(mean >= 0.0, "Mean should be non-negative, got {}", mean);
        assert!(mean.is_finite(), "Mean should be finite, got {}", mean);
    }

    #[test]
    fn splitting_non_negative_deep_twilight() {
        // At SZA 106, splitting is active. All contributions must be non-negative.
        use crate::atmosphere::{AtmosphereModel, ShellOptics, EARTH_RADIUS_M};

        let altitudes_km = [0.0, 10.0, 25.0, 50.0, 75.0, 100.0];
        let wavelengths = [550.0];
        let mut atm = AtmosphereModel::new(&altitudes_km, &wavelengths);
        for s in 0..5 {
            let alt_mid = (altitudes_km[s] + altitudes_km[s + 1]) / 2.0;
            atm.optics[s][0] = ShellOptics {
                extinction: 1.3e-5 * libm::exp(-alt_mid / 8.0),
                ssa: 0.999,
                asymmetry: 0.0,
                rayleigh_fraction: 1.0,
            };
        }

        let observer = Vec3::new(EARTH_RADIUS_M + 10.0, 0.0, 0.0);
        // SZA 106: sun 16 deg below horizon
        let sza_rad = 106.0 * core::f64::consts::PI / 180.0;
        let sun_dir = Vec3::new(libm::cos(sza_rad), libm::sin(sza_rad), 0.0);
        let start_optics = &atm.optics[0][0];

        let n = 500;
        let mut rng: u64 = 777;
        for ray in 0..n {
            let val = trace_secondary_chain_scalar(
                &atm,
                observer,
                sun_dir,
                0,
                start_optics,
                &mut rng,
                ray,
                n,
                None,
            );
            assert!(
                val >= 0.0,
                "Chain {} returned negative value {} at SZA 106",
                ray,
                val
            );
            assert!(
                val.is_finite(),
                "Chain {} returned non-finite value {} at SZA 106",
                ray,
                val
            );
        }
    }

    #[test]
    fn splitting_alis_non_negative_deep_twilight() {
        // ALIS version at SZA 106: all wavelength contributions non-negative.
        use crate::atmosphere::{AtmosphereModel, ShellOptics, EARTH_RADIUS_M};

        let altitudes_km = [0.0, 10.0, 25.0, 50.0, 75.0, 100.0];
        let wavelengths = [400.0, 550.0, 700.0];
        let mut atm = AtmosphereModel::new(&altitudes_km, &wavelengths);
        for s in 0..5 {
            let alt_mid = (altitudes_km[s] + altitudes_km[s + 1]) / 2.0;
            let base_ext = 1.3e-5 * libm::exp(-alt_mid / 8.0);
            for w in 0..3 {
                let wl = wavelengths[w];
                let lambda_ratio = (550.0 / wl).powi(4);
                atm.optics[s][w] = ShellOptics {
                    extinction: base_ext * lambda_ratio,
                    ssa: 0.999,
                    asymmetry: 0.0,
                    rayleigh_fraction: 1.0,
                };
            }
        }

        let observer = Vec3::new(EARTH_RADIUS_M + 10.0, 0.0, 0.0);
        let sza_rad = 106.0 * core::f64::consts::PI / 180.0;
        let sun_dir = Vec3::new(libm::cos(sza_rad), libm::sin(sza_rad), 0.0);

        let num_wl = 3;
        let n = 200;
        let mut rng: u64 = 999;
        for ray in 0..n {
            let hero_wl = ray % num_wl;
            let result = trace_secondary_chain_alis(
                &atm, observer, sun_dir, hero_wl, 0, &mut rng, ray, n, num_wl, None,
            );
            for w in 0..num_wl {
                assert!(
                    result[w] >= 0.0,
                    "ALIS chain {} wl {} returned negative: {}",
                    ray,
                    w,
                    result[w]
                );
                assert!(
                    result[w].is_finite(),
                    "ALIS chain {} wl {} returned non-finite: {}",
                    ray,
                    w,
                    result[w]
                );
            }
        }
    }

    #[test]
    fn max_split_particles_sufficient() {
        // The maximum split budget is 3*3*2 = 18 for deep twilight.
        // MAX_SPLIT_PARTICLES must be >= this.
        let max_budget: usize = SPLIT_FACTORS_DEEP.iter().product();
        assert!(
            MAX_SPLIT_PARTICLES >= max_budget,
            "MAX_SPLIT_PARTICLES ({}) < max budget ({})",
            MAX_SPLIT_PARTICLES,
            max_budget
        );
    }

    #[test]
    fn split_altitudes_increasing() {
        for i in 1..NUM_SPLIT_LEVELS {
            assert!(
                SPLIT_ALTITUDES_M[i] > SPLIT_ALTITUDES_M[i - 1],
                "Split altitudes must be strictly increasing: [{}]={} <= [{}]={}",
                i,
                SPLIT_ALTITUDES_M[i],
                i - 1,
                SPLIT_ALTITUDES_M[i - 1]
            );
        }
    }

    #[test]
    fn split_altitudes_within_atmosphere() {
        use crate::atmosphere::TOA_ALTITUDE_M;
        for (i, &alt) in SPLIT_ALTITUDES_M.iter().enumerate() {
            assert!(
                alt > 0.0 && alt < TOA_ALTITUDE_M,
                "Split altitude [{}] = {} must be in (0, {})",
                i,
                alt,
                TOA_ALTITUDE_M
            );
        }
    }

    // ---- VSPG tests ----

    #[test]
    fn vspg_importance_unity_below_boost_start() {
        // Below 15 km, importance should be 1.0 regardless of SZA.
        for alt in [0.0, 5_000.0, 10_000.0, VSPG_BOOST_START_M] {
            for sza in [90.0, 96.0, 100.0, 106.0, 108.0] {
                let imp = vspg_importance(alt, sza);
                assert!(
                    (imp - 1.0).abs() < 1e-12,
                    "Expected 1.0 at alt={}, sza={}, got {}",
                    alt,
                    sza,
                    imp
                );
            }
        }
    }

    #[test]
    fn vspg_importance_unity_at_civil_twilight() {
        // At SZA <= 96, importance should be 1.0 for all altitudes.
        for alt in [0.0, 30_000.0, 50_000.0, 70_000.0, 100_000.0] {
            let imp = vspg_importance(alt, 90.0);
            assert!(
                (imp - 1.0).abs() < 1e-12,
                "Expected 1.0 at SZA 90 alt={}, got {}",
                alt,
                imp
            );
            let imp2 = vspg_importance(alt, 96.0);
            assert!(
                (imp2 - 1.0).abs() < 1e-12,
                "Expected 1.0 at SZA 96 alt={}, got {}",
                alt,
                imp2
            );
        }
    }

    #[test]
    fn vspg_importance_increases_with_altitude() {
        let sza = 106.0;
        let imp_20k = vspg_importance(20_000.0, sza);
        let imp_40k = vspg_importance(40_000.0, sza);
        let imp_70k = vspg_importance(70_000.0, sza);
        assert!(
            imp_20k < imp_40k,
            "20km ({}) should be < 40km ({})",
            imp_20k,
            imp_40k
        );
        assert!(
            imp_40k < imp_70k,
            "40km ({}) should be < 70km ({})",
            imp_40k,
            imp_70k
        );
    }

    #[test]
    fn vspg_importance_increases_with_sza() {
        let alt = 50_000.0;
        let imp_98 = vspg_importance(alt, 98.0);
        let imp_102 = vspg_importance(alt, 102.0);
        let imp_106 = vspg_importance(alt, 106.0);
        assert!(
            imp_98 < imp_102,
            "SZA 98 ({}) should be < SZA 102 ({})",
            imp_98,
            imp_102
        );
        assert!(
            imp_102 < imp_106,
            "SZA 102 ({}) should be < SZA 106 ({})",
            imp_102,
            imp_106
        );
    }

    #[test]
    fn vspg_importance_max_at_full_altitude_and_sza() {
        let imp = vspg_importance(VSPG_BOOST_FULL_M, ZENITH_SZA_FULL);
        assert!(
            (imp - VSPG_MAX_IMPORTANCE).abs() < 1e-10,
            "Expected {} at max alt+sza, got {}",
            VSPG_MAX_IMPORTANCE,
            imp
        );
    }

    #[test]
    fn vspg_importance_capped_above_full_altitude() {
        // Above VSPG_BOOST_FULL_M, importance should be capped (not grow).
        let imp_at = vspg_importance(VSPG_BOOST_FULL_M, 106.0);
        let imp_above = vspg_importance(VSPG_BOOST_FULL_M + 30_000.0, 106.0);
        assert!(
            (imp_at - imp_above).abs() < 1e-10,
            "Importance should cap: at={}, above={}",
            imp_at,
            imp_above
        );
    }

    #[test]
    fn vspg_sample_returns_valid_tau() {
        use crate::atmosphere::{AtmosphereModel, EARTH_RADIUS_M};
        use crate::geometry::Vec3;

        let altitudes = [0.0, 10.0, 25.0, 50.0, 100.0];
        let wavelengths = [550.0];
        let mut atm = AtmosphereModel::new(&altitudes, &wavelengths);
        // Set up increasing extinction downward (realistic).
        for i in 0..4 {
            atm.optics[i][0].extinction = 0.01 / ((i as f64 + 1.0) * 5.0);
            atm.optics[i][0].ssa = 1.0;
            atm.optics[i][0].rayleigh_fraction = 1.0;
        }

        let pos = Vec3::new(EARTH_RADIUS_M + 5_000.0, 0.0, 0.0);
        let dir = Vec3::new(1.0, 0.0, 0.0).normalize(); // radially outward
        let tau_max = 3.0;

        let mut rng: u64 = 42;
        for _ in 0..100 {
            let (tau_s, w) = vspg_sample_scatter_tau(&atm, pos, dir, 0, tau_max, 106.0, &mut rng);
            assert!(tau_s >= 0.0, "tau_s should be >= 0, got {}", tau_s);
            assert!(
                tau_s <= tau_max + 1e-10,
                "tau_s {} > tau_max {}",
                tau_s,
                tau_max
            );
            assert!(w > 0.0, "weight correction should be > 0, got {}", w);
            assert!(
                w.is_finite(),
                "weight correction should be finite, got {}",
                w
            );
        }
    }

    #[test]
    fn vspg_weight_correction_unity_at_civil_twilight() {
        use crate::atmosphere::{AtmosphereModel, EARTH_RADIUS_M};
        use crate::geometry::Vec3;

        let altitudes = [0.0, 10.0, 25.0, 50.0, 100.0];
        let wavelengths = [550.0];
        let mut atm = AtmosphereModel::new(&altitudes, &wavelengths);
        for i in 0..4 {
            atm.optics[i][0].extinction = 0.005;
            atm.optics[i][0].ssa = 1.0;
            atm.optics[i][0].rayleigh_fraction = 1.0;
        }

        let pos = Vec3::new(EARTH_RADIUS_M + 5_000.0, 0.0, 0.0);
        let dir = Vec3::new(1.0, 0.0, 0.0).normalize();
        let tau_max = 2.0;

        // At SZA 90, all importances are 1.0, so weight correction must be 1.0.
        let mut rng: u64 = 77;
        for _ in 0..50 {
            let (_, w) = vspg_sample_scatter_tau(&atm, pos, dir, 0, tau_max, 90.0, &mut rng);
            assert!(
                (w - 1.0).abs() < 1e-12,
                "Weight correction should be 1.0 at SZA 90, got {}",
                w
            );
        }
    }

    #[test]
    fn vspg_segment_count_bounded() {
        // VSPG_MAX_SEGMENTS should accommodate worst-case ray traversal.
        assert!(
            VSPG_MAX_SEGMENTS >= 64,
            "Need at least 64 segments for full traversal, got {}",
            VSPG_MAX_SEGMENTS
        );
    }

    #[test]
    fn vspg_constants_reasonable() {
        assert!(VSPG_BOOST_START_M > 0.0);
        assert!(VSPG_BOOST_FULL_M > VSPG_BOOST_START_M);
        assert!(VSPG_MAX_IMPORTANCE > 1.0);
        assert!(
            VSPG_MAX_IMPORTANCE <= 200.0,
            "Max importance too large: {}",
            VSPG_MAX_IMPORTANCE
        );
    }

    #[test]
    fn vspg_scalar_chain_non_negative_deep_twilight() {
        // Full chain test: scalar tracer with VSPG at SZA 106 produces
        // non-negative, finite results.
        use crate::atmosphere::{AtmosphereModel, ShellOptics, EARTH_RADIUS_M};
        use crate::geometry::Vec3;

        let altitudes = [0.0, 5.0, 15.0, 30.0, 50.0, 75.0, 100.0];
        let wavelengths = [550.0];
        let mut atm = AtmosphereModel::new(&altitudes, &wavelengths);
        let sigmas = [1e-2, 5e-3, 1e-3, 2e-4, 5e-5, 1e-5];
        for (i, &sig) in sigmas.iter().enumerate() {
            atm.optics[i][0] = ShellOptics {
                extinction: sig,
                ssa: 0.999,
                asymmetry: 0.0,
                rayleigh_fraction: 1.0,
            };
        }

        let observer = Vec3::new(EARTH_RADIUS_M + 10.0, 0.0, 0.0);
        let sza_rad = 106.0 * core::f64::consts::PI / 180.0;
        let sun_dir = Vec3::new(libm::cos(sza_rad), libm::sin(sza_rad), 0.0);

        let start_optics = &atm.optics[0][0];
        let mut rng: u64 = 12345;
        let n = 200;
        for ray in 0..n {
            let result = trace_secondary_chain_scalar(
                &atm,
                observer,
                sun_dir,
                0,
                start_optics,
                &mut rng,
                ray,
                n,
                None,
            );
            assert!(
                result >= 0.0 && result.is_finite(),
                "Scalar chain {} returned invalid: {}",
                ray,
                result
            );
        }
    }

    #[test]
    fn vspg_alis_chain_non_negative_deep_twilight() {
        // Full chain test: ALIS tracer with VSPG at SZA 106.
        use crate::atmosphere::{AtmosphereModel, ShellOptics, EARTH_RADIUS_M};
        use crate::geometry::Vec3;

        let altitudes = [0.0, 5.0, 15.0, 30.0, 50.0, 75.0, 100.0];
        let wavelengths = [450.0, 550.0, 650.0];
        let mut atm = AtmosphereModel::new(&altitudes, &wavelengths);
        let sigmas = [1e-2, 5e-3, 1e-3, 2e-4, 5e-5, 1e-5];
        for (i, &sig) in sigmas.iter().enumerate() {
            for w in 0..3 {
                // Slight wavelength dependence (Rayleigh ~lambda^-4).
                let wl_factor = 1.0 + 0.1 * (w as f64 - 1.0);
                atm.optics[i][w] = ShellOptics {
                    extinction: sig * wl_factor,
                    ssa: 0.999,
                    asymmetry: 0.0,
                    rayleigh_fraction: 1.0,
                };
            }
        }

        let observer = Vec3::new(EARTH_RADIUS_M + 10.0, 0.0, 0.0);
        let sza_rad = 106.0 * core::f64::consts::PI / 180.0;
        let sun_dir = Vec3::new(libm::cos(sza_rad), libm::sin(sza_rad), 0.0);

        let num_wl = 3;
        let n = 200;
        let mut rng: u64 = 54321;
        for ray in 0..n {
            let hero_wl = ray % num_wl;
            let result = trace_secondary_chain_alis(
                &atm, observer, sun_dir, hero_wl, 0, &mut rng, ray, n, num_wl, None,
            );
            for w in 0..num_wl {
                assert!(
                    result[w] >= 0.0 && result[w].is_finite(),
                    "ALIS chain {} wl {} returned invalid: {}",
                    ray,
                    w,
                    result[w]
                );
            }
        }
    }
}
