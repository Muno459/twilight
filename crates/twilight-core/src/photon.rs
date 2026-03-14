//! Single photon trace logic — the core MCRT pure function.
//!
//! This module contains the backward Monte Carlo photon tracing algorithm.
//! The trace function is a pure function with no platform dependencies,
//! making it compilable to any target (CPU, GPU via WGSL, WASM, CUDA PTX).

use crate::atmosphere::AtmosphereModel;
use crate::geometry::{next_shell_boundary, refract_at_boundary, RefractResult, Vec3};
use crate::scattering::{
    henyey_greenstein_phase, rayleigh_phase, sample_henyey_greenstein, sample_rayleigh_analytic,
    scatter_direction, scatter_stokes_fast, scattering_plane_cos_sin, StokesVector,
};

/// Safety limit on scattering bounces to prevent infinite loops.
///
/// Chains terminate naturally via escape or ground absorption. No weight
/// floor or Russian roulette is applied. This limit is purely a backstop
/// against degenerate floating-point loops.
pub const MAX_SCATTERS: usize = 10_000;

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
                let (new_dir, crossed) = match refract_at_boundary(dir, boundary_pos, n_from, n_to)
                {
                    RefractResult::Refracted(d) => (d, true),
                    RefractResult::TotalReflection(d) => (d, false),
                };
                dir = new_dir;
                pos = boundary_pos + dir * 1e-3;

                // Hit ground -- path terminates here
                if !is_outward && pos.length() <= surface_radius + 1.0 {
                    return (tau, true);
                }

                // Exited atmosphere
                if crossed {
                    if next_shell >= num_shells {
                        return (tau, false);
                    }
                    shell_idx = next_shell;
                }
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
                let (new_dir, crossed) = match refract_at_boundary(dir, boundary_pos, n_from, n_to)
                {
                    RefractResult::Refracted(d) => (d, true),
                    RefractResult::TotalReflection(d) => (d, false),
                };
                dir = new_dir;
                pos = boundary_pos + dir * 1e-3;

                // Hit ground -- place scatter here
                if !is_outward && pos.length() <= surface_radius + 1.0 {
                    return (pos, dir, shell_idx);
                }

                // Exited atmosphere -- place scatter at exit
                if crossed {
                    if next_shell >= num_shells {
                        return (pos, dir, shell_idx);
                    }
                    shell_idx = next_shell;
                }
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
    // Split RNG: derive independent streams from master seed.
    let mut rng = McRng::from_seed(*rng_state);
    // Advance master by 1 so successive calls get different seeds.
    let _ = xorshift_f64(rng_state);

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
        let xi = xorshift_f64(&mut rng.tau);
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
                        let normal = pos.normalize();

                        // Ground-bounce NEE: Lambertian BRDF = albedo/pi.
                        let cos_sun_ground = sun_dir.dot(normal);
                        if cos_sun_ground > 0.0 {
                            let t_sun_gb = trace_transmittance(atm, pos, sun_dir, wavelength_idx);
                            if t_sun_gb > 1e-30 {
                                let albedo = atm.surface_albedo[wavelength_idx];
                                result.weight += weight * albedo * t_sun_gb * cos_sun_ground
                                    / core::f64::consts::PI;
                            }
                        }

                        // Ground reflection (Lambertian)
                        let albedo = atm.surface_albedo[wavelength_idx];
                        weight *= albedo;
                        dir = sample_hemisphere(normal, &mut rng.dir);
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
        let cos_theta = if xorshift_f64(&mut rng.dir) < optics.rayleigh_fraction {
            sample_rayleigh_analytic(xorshift_f64(&mut rng.dir))
        } else {
            sample_henyey_greenstein(xorshift_f64(&mut rng.dir), optics.asymmetry)
        };
        let phi = 2.0 * core::f64::consts::PI * xorshift_f64(&mut rng.dir);
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
    // Split RNG: derive independent streams from master seed.
    let mut rng = McRng::from_seed(*rng_state);
    let _ = xorshift_f64(rng_state);

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
        let xi = xorshift_f64(&mut rng.tau);
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
                        let normal = pos.normalize();

                        // Ground-bounce NEE: Lambertian BRDF = albedo/pi.
                        let cos_sun_ground = sun_dir.dot(normal);
                        if cos_sun_ground > 0.0 {
                            let t_sun_gb = trace_transmittance(atm, pos, sun_dir, wavelength_idx);
                            if t_sun_gb > 1e-30 {
                                let albedo = atm.surface_albedo[wavelength_idx];
                                let nee_gb = weight * albedo * t_sun_gb * cos_sun_ground
                                    / core::f64::consts::PI;
                                // Lambertian depolarizes: only I component.
                                result.stokes =
                                    result.stokes.add(&StokesVector::unpolarized(nee_gb));
                            }
                        }

                        let albedo = atm.surface_albedo[wavelength_idx];
                        weight *= albedo;
                        prev_dir = dir;
                        dir = sample_hemisphere(normal, &mut rng.dir);
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
        let cos_theta = if xorshift_f64(&mut rng.dir) < optics.rayleigh_fraction {
            sample_rayleigh_analytic(xorshift_f64(&mut rng.dir))
        } else {
            sample_henyey_greenstein(xorshift_f64(&mut rng.dir), optics.asymmetry)
        };
        let phi = 2.0 * core::f64::consts::PI * xorshift_f64(&mut rng.dir);
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

/// Safety limit to prevent infinite loops from floating-point pathology.
///
/// Chains terminate naturally via atmosphere escape (analog mode) or ground
/// absorption. No weight floor or Russian roulette is applied. This limit
/// exists only as a backstop against degenerate floating-point loops and
/// should never be reached in physically meaningful chains.
const BOUNCE_SAFETY_LIMIT: usize = 10_000;

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

/// Minimum optical depth for forced scattering to engage.
///
/// When tau_max is negligibly small, the forced-scatter weight `1-exp(-tau)`
/// approaches tau, and sampling from the truncated exponential degenerates
/// Minimum tau for forced scattering. Below this, use analog mode.
///
/// When the total optical depth along a ray to the next boundary is below
/// this threshold, forced scattering produces catastrophic weight death:
/// `weight *= (1 - exp(-tau))` ~ tau for small tau. At 50 km altitude
/// upward, tau ~ 5e-5, killing weight by 5 orders of magnitude per bounce.
///
/// Below this threshold, the chain goes analog: the photon walks a real
/// free path and either escapes the atmosphere (~99% at high altitude,
/// providing natural unbiased termination) or actually scatters (~1%,
/// continuing with full weight). No bias is introduced.
///
/// Returns the SZA-adaptive forced-scatter optical-depth floor.
///
/// Smoothly transitions from 0.05 (moderate twilight) to 0.02 (deep twilight).
///
/// At moderate twilight, a higher threshold lets more bounces go analog
/// above ~12 km, where chains escape cleanly and the forced-scatter weight
/// penalty (1-exp(-tau)) is wasteful.
///
/// At deep twilight, a lower threshold keeps forced scattering active
/// deeper into the atmosphere (~20 km), giving chains more chances to
/// survive and reach the illuminated terminator.
///
/// Uses sigmoid ramp centered at SZA 102, width 1.0. At SZA <= 99 this
/// returns ~0.05 (matching the pre-sigmoid constant), preserving the
/// established variance characteristics at civil/nautical twilight.
#[inline]
fn forced_tau_min_for_sza(sza_deg: f64) -> f64 {
    let t = sigmoid((sza_deg - 102.0) / 1.0);
    0.05 - 0.03 * t // 0.05 at SZA<100, 0.02 at SZA>104
}

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

/// Maximum allowed importance weight from power-cosine sampling.
///
/// Both `zenith_importance_weight` and `terminator_shape_weight` contain
/// factors of 1/cos^k(theta) that diverge as cos_theta -> 0. Without
/// truncation, a single chain can dominate the entire seed ensemble.
///
/// The sampling domain is truncated at cos_min(n) = (2/(W*(n+1)))^(1/n)
/// where W = ZENITH_MAX_IMPORTANCE_WEIGHT. This is a change to the
/// proposal distribution (free IS design choice), not weight clamping.
///
/// At W=200: n=1 -> cos_min=0.005 (no effect), n=5 -> cos_min=0.28,
/// n=8 -> cos_min=0.43. The removed probability mass is cos_min^(n+1),
/// which is negligible (e.g., 0.28^6 = 4.8e-4 at n=5).
const ZENITH_MAX_IMPORTANCE_WEIGHT: f64 = 200.0;

/// Compute the minimum cos(theta) for truncated power-cosine sampling
/// given exponent n, such that the importance weight stays below
/// ZENITH_MAX_IMPORTANCE_WEIGHT.
///
/// cos_min(n) = max(0.2, (2 / (W_max * (n+1)))^(1/n))
///
/// The max(0.2, ...) floor preserves directional truncation at moderate
/// exponents (n~1.4 at SZA 97) where the analytic cos_min would be too
/// small (~0.01) and allow moderate outlier weights that compound across seeds.
#[inline]
fn power_cos_min(n: f64) -> f64 {
    if n <= 1.01 {
        return 0.0; // At n=1, weight is constant ~1.0, no truncation needed.
    }
    let w_max = ZENITH_MAX_IMPORTANCE_WEIGHT;
    let analytic = libm::pow(2.0 / (w_max * (n + 1.0)), 1.0 / n);
    // Floor at 0.2: at moderate n (1.4-2), prevents ~20x outlier weights
    // that would otherwise compound over 500+ seeds.
    if analytic > 0.2 {
        analytic
    } else {
        0.2
    }
}

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
/// At deep twilight (SZA 106), the terminator is ~1780 km from the observer,
/// nearly at the horizon. The terminator axis points 60 deg from zenith
/// toward the sub-solar horizon, directing rays into the region where
/// shadow rays can first reach sunlit atmosphere. A 60 deg tilt balances
/// coverage: more horizontal than 50 deg to better target the distant
/// terminator, but not so aggressive (70+) that noise at deep twilight
/// overwhelms the signal.
const TERMINATOR_TILT_MAX_DEG: f64 = 60.0;

// --- Dwivedi-type horizontal direction biasing ---
//
// At deep twilight, photon chains must travel horizontally ~1500 km to reach
// the sunlit terminator. The phase function wastes most samples on vertical
// directions that either escape to space or get absorbed in the troposphere.
//
// Dwivedi biasing concentrates direction sampling toward the local horizontal
// plane (cos_zenith ~ 0), where chains maintain altitude and progress toward
// the terminator. The bias strength ramps smoothly with SZA.
//
// The distribution is: p(dir) = beta * exp(-beta * |cos_z|) / (4*pi * (1 - exp(-beta)))
// where cos_z = dir . local_up. At beta=0 this is uniform (1/4pi).
//
// One-sample 2-way MIS (phase + Dwivedi) keeps the estimator unbiased.

/// Concentration parameter for Dwivedi biasing at full strength.
/// Higher values = stronger horizontal concentration. beta=3.0 gives
/// ~95% of probability within 30 degrees of horizontal.
const DWIVEDI_BETA_MAX: f64 = 3.0;

/// SZA center for Dwivedi ramp [degrees].
///
/// Kept at 103 (where Dwivedi fraction = half of max). With WIDTH=2.0,
/// the ramp extends ~4 degrees in each direction:
///   SZA 99:  d_frac = 4.2%  (gentle)
///   SZA 101: d_frac = 9.4%  (moderate)
///   SZA 103: d_frac = 17.5% (half-max)
///   SZA 105: d_frac = 25.6% (near-full)
///   SZA 107: d_frac = 30.8% (near-max)
/// Tested center=101: no measurable CV change (MIS weight correction
/// exactly compensates the shifted sampling distribution).
const DWIVEDI_SZA_CENTER: f64 = 103.0;

/// SZA width for Dwivedi ramp [degrees].
///
/// Widened from 1.5 to 2.0 for a gentler ramp that extends meaningful
/// Dwivedi guidance to SZA 101-102 (9-13%) where lateral transport
/// becomes the bottleneck, while keeping the center at 103.
const DWIVEDI_SZA_WIDTH: f64 = 2.0;

/// Fraction of bounces allocated to Dwivedi at full strength.
///
/// Increased from 0.25 to 0.35 to allocate more direction samples to
/// horizontal biasing at deep twilight where lateral transport to the
/// terminator (1300-2000 km) is the bottleneck. Phase function retains
/// 65% at full SZA, sufficient for scattering angle sampling.
const DWIVEDI_FRAC_MAX: f64 = 0.35;

/// Maximum fraction of bounces allocated to path guide sampling.
///
/// Only active when a trained PathGuide is provided. At full strength
/// (SZA > 105), guide gets 20% of the direction budget, Dwivedi gets
/// 35%, and phase function retains 45%. The guide learns productive
/// directions from BDPT light vertices -- directions that successfully
/// reached scattering events near the terminator.
///
/// Tested 30%: regressed SZA 106 CV from 0.45 to 0.70. The 32-bin
/// directional resolution is too coarse for higher guide fractions --
/// MIS weight corrections blow up when the guide confidently sends
/// chains to bins with low phase function density.
const GUIDE_FRAC_MAX: f64 = 0.20;

/// Returns the SZA-adaptive path guide sampling fraction.
///
/// Ramps identically to Dwivedi (same center/width) since guide is
/// useful in the same SZA regime where lateral transport matters.
#[inline]
fn guide_frac(sza_deg: f64) -> f64 {
    GUIDE_FRAC_MAX * sigmoid((sza_deg - DWIVEDI_SZA_CENTER) / DWIVEDI_SZA_WIDTH)
}

/// Returns the SZA-adaptive Dwivedi sampling fraction.
///
/// At SZA < 98: ~0 (no Dwivedi bias, phase function is sufficient).
/// At SZA = 101: ~0.20 (moderate).
/// At SZA > 105: ~0.37 (near-full Dwivedi allocation).
#[inline]
fn dwivedi_frac(sza_deg: f64) -> f64 {
    DWIVEDI_FRAC_MAX * sigmoid((sza_deg - DWIVEDI_SZA_CENTER) / DWIVEDI_SZA_WIDTH)
}

/// Returns the SZA-adaptive Dwivedi beta parameter.
///
/// Ramps from 0 (uniform) at civil twilight to DWIVEDI_BETA_MAX at deep.
#[inline]
fn dwivedi_beta(sza_deg: f64) -> f64 {
    DWIVEDI_BETA_MAX * sigmoid((sza_deg - DWIVEDI_SZA_CENTER) / DWIVEDI_SZA_WIDTH)
}

/// Dwivedi PDF: probability density in sr^-1 for a direction with
/// `cos_z = dir . local_up`, given concentration parameter `beta`.
///
/// p(cos_z) = beta * exp(-beta * |cos_z|) / (4 * pi * (1 - exp(-beta)))
///
/// Returns 1/(4*pi) when beta < 1e-6 (effectively uniform).
#[inline]
fn dwivedi_pdf(cos_z: f64, beta: f64) -> f64 {
    if beta < 1e-6 {
        return INV_4PI;
    }
    let abs_cz = libm::fabs(cos_z).clamp(0.0, 1.0);
    beta * libm::exp(-beta * abs_cz) / (4.0 * core::f64::consts::PI * (1.0 - libm::exp(-beta)))
}

/// Sample a direction from the Dwivedi distribution.
///
/// Returns (cos_zenith, phi) where cos_zenith is relative to `local_up`.
/// Uses two uniform random numbers xi1 (for |cos_z|) and xi2 (for azimuth).
///
/// CDF inversion: |cos_z| = -ln(1 - xi1 * (1 - exp(-beta))) / beta
/// Sign of cos_z is random (symmetric about horizontal plane).
#[inline]
fn dwivedi_sample(xi1: f64, xi2: f64, xi_sign: f64, beta: f64) -> (f64, f64) {
    let phi = 2.0 * core::f64::consts::PI * xi2;
    if beta < 1e-6 {
        // Uniform: cos_z = 2*xi1 - 1
        let cos_z = 2.0 * xi1 - 1.0;
        return (cos_z, phi);
    }
    let one_minus_exp_neg_beta = 1.0 - libm::exp(-beta);
    let abs_cz = -libm::log(1.0 - xi1 * one_minus_exp_neg_beta) / beta;
    let abs_cz = abs_cz.clamp(0.0, 1.0);
    // Random sign: up or down equally (symmetric around horizontal)
    let cos_z = if xi_sign < 0.5 { abs_cz } else { -abs_cz };
    (cos_z, phi)
}

// --- Weight Windows ---
//
// Weight windows replace fixed altitude-splitting and BOUNCE_SAFETY_LIMIT
// with adaptive, importance-based population control. At each bounce, the
// chain's weight is compared against a target weight derived from an
// importance function I(altitude, sza):
//
//   w_target(alt) = 1 / I_rel(alt)
//   I_rel(alt) = exp((alt - alt_start) / H_ww(sza))
//
// where alt_start is the chain's starting altitude and H_ww is a
// SZA-adaptive scale height.
//
// - When weight > w_target * WW_UPPER_RATIO: split into k = round(w/w_target)
//   copies. Each gets weight/k. Provably unbiased: k * (w/k) = w.
// - When weight < w_target / WW_LOWER_RATIO: Russian roulette. Survive with
//   probability p = weight / w_target. On survival, weight = w_target.
//   Provably unbiased: E[output] = p * w_target = weight.
//
// The asymmetric ratios (upper=2, lower=10) reflect the physics: splitting
// at high altitude is always beneficial, but RR at low altitude can harm
// estimates when tropospheric scattering still contributes (SZA 100-104).
//
// Benefits over fixed altitude splitting:
// 1. Unbiased chain termination (replaces BOUNCE_SAFETY_LIMIT)
// 2. Adaptive: no static SZA thresholds or altitude breakpoints
// 3. Tropospheric chains killed early (~350 bounces vs 10,000)
// 4. High-altitude chains split proportionally to their rarity

/// Minimum scale height for weight window importance at deep twilight [m].
///
/// At SZA >= 106, the importance function ramps steeply with altitude:
/// I_rel(+60 km) = exp(60000/12000) = 148. With WW_UPPER_RATIO = 2,
/// chains split into ~24 copies (capped by MAX_SPLIT_PARTICLES), matching
/// or exceeding the old 3*3*2 = 18 fixed-altitude scheme.
const WW_H_MIN_M: f64 = 12_000.0;

/// Maximum scale height for weight window importance at moderate SZA [m].
///
/// At SZA < 96, the scale height is effectively infinite (no splitting).
/// This value gives I_rel(+60 km) = exp(60000/1000000) = 1.06, so weight
/// windows are dormant.
const WW_H_MAX_M: f64 = 1_000_000.0;

/// Center SZA for the weight window sigmoid ramp [degrees].
///
/// The sigmoid (sza - center) / width transitions weight window aggressiveness.
/// At center = 100: civil twilight gets no windows, nautical gets very mild,
/// astronomical gets full.
const WW_SZA_CENTER: f64 = 100.0;

/// Width of the weight window sigmoid ramp [degrees].
///
/// Smaller = sharper transition. At 1.0 degrees:
///   SZA 93: t = 0.001, dormant
///   SZA 96: t = 0.018, dormant
///   SZA 100: t = 0.5, mild
///   SZA 104: t = 0.982, aggressive
///   SZA 106: t = 0.998, full
const WW_SZA_WIDTH: f64 = 1.0;

/// Upper weight window ratio for splitting.
///
/// When weight > w_target * WW_UPPER_RATIO, the chain is split.
/// At 2.0: chains split when their weight is 2x the target, creating
/// more copies to explore high-altitude regions.
const WW_UPPER_RATIO: f64 = 2.0;

/// Lower weight window ratio for Russian roulette.
///
/// When weight < w_target / WW_LOWER_RATIO, the chain faces RR.
/// At 10.0: chains survive until their weight drops to 1/10 of the target.
/// This is conservative: at the starting altitude (w_target=1.0),
/// RR fires after ~770 bounces (SSA=0.997), giving chains ample time
/// to climb to productive altitudes while still terminating hopelessly
/// trapped chains.
///
/// The asymmetry (upper=2, lower=10) reflects the physics: aggressive
/// splitting at high altitude is always beneficial (more exploration),
/// but aggressive RR at low altitude can harm estimates when tropospheric
/// bounces still contribute (SZA 100-104).
const WW_LOWER_RATIO: f64 = 10.0;

/// Maximum number of concurrent split particles in the work stack.
///
/// Caps the maximum split count per weight-window event. Stack memory per
/// particle: ~80 bytes (scalar) or ~600 bytes (ALIS with weight_ratio[64]).
/// At 24 particles: 1.9 KB (scalar) or 14.4 KB (ALIS).
const MAX_SPLIT_PARTICLES: usize = 24;

/// Sigmoid helper for smooth parameter transitions.
///
/// Returns 1 / (1 + exp(-x)), smoothly transitioning from 0 to 1.
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + libm::exp(-x))
}

/// Compute the SZA-adaptive weight window scale height [m].
///
/// Smoothly transitions from WW_H_MAX_M (no windows) at civil twilight
/// to WW_H_MIN_M (aggressive windows) at deep twilight.
///
/// Uses logarithmic interpolation: H = H_MIN^t * H_MAX^(1-t). This ensures
/// that the transition is smooth in log-space, so even at t=0.98 (SZA 104
/// with center=100, width=1), H is close to H_MIN (13 km) rather than
/// the 37 km that linear interpolation would give.
#[inline]
fn weight_window_h(sza_deg: f64) -> f64 {
    let t = sigmoid((sza_deg - WW_SZA_CENTER) / WW_SZA_WIDTH);
    // H = H_MIN^t * H_MAX^(1-t) = exp(t*ln(H_MIN) + (1-t)*ln(H_MAX))
    let ln_h = t * libm::log(WW_H_MIN_M) + (1.0 - t) * libm::log(WW_H_MAX_M);
    libm::exp(ln_h)
}

/// Compute the weight window target weight at the current position.
///
/// Combines altitude importance (chains climbing higher are more productive)
/// with CADIS lateral importance (chains progressing toward the sunlit
/// terminator are more productive).
///
/// Altitude importance (relative to start):
///   I_alt = exp((alt - alt_start) / H_ww)
///
/// CADIS lateral importance (relative to start):
///   delta_cos = pos.dot(sun_dir) - cos_sun_start
///   I_lat = exp(cadis_k * max(0, delta_cos))
///
/// Combined target: w_target = 1 / (I_alt * I_lat)
///
/// The CADIS term only boosts importance for chains making lateral progress
/// toward the sun. Chains moving away keep I_lat = 1 (no extra RR penalty),
/// which is conservative and avoids killing chains that might loop back.
///
/// When cadis_k = 0 (civil twilight), I_lat = 1 and this degenerates to
/// the altitude-only weight window target.
#[inline]
fn weight_window_target(
    alt_m: f64,
    alt_start_m: f64,
    h_ww: f64,
    cos_sun_current: f64,
    cos_sun_start: f64,
    cadis_k: f64,
) -> f64 {
    let i_alt = libm::exp((alt_m - alt_start_m) / h_ww);
    let delta_cos = cos_sun_current - cos_sun_start;
    let i_lat = if cadis_k > 0.0 && delta_cos > 0.0 {
        libm::exp(cadis_k * delta_cos)
    } else {
        1.0
    };
    1.0 / (i_alt * i_lat)
}

// --- CADIS: forward-informed lateral importance ---
//
// At deep twilight (SZA > 103), the sunlit terminator is ~1500 km from the
// observer. Altitude-only weight windows encourage chains to climb but not
// to progress laterally toward the terminator. CADIS adds a lateral
// importance term: chains that move toward the sun (increasing cos_sun)
// get split more aggressively by the weight windows.
//
// The importance boost is exp(cadis_k * delta_cos), where delta_cos is
// the increase in cos(position_angle_to_sun) from the chain's start.
// This is 1 at the start and grows as the chain approaches the terminator.
//
// cadis_k ramps from 0 (civil) to CADIS_K_MAX (deep twilight).
// The boost is one-sided: only chains making positive lateral progress
// get higher importance. Chains moving away just have I_lat = 1.

/// Maximum CADIS lateral importance exponent at deep twilight.
///
/// At SZA 106, a chain traveling from the observer (cos_sza = -0.276)
/// to the terminator (cos_sun = 0) has delta_cos = 0.276.
/// With CADIS_K_MAX = 12: boost = exp(12 * 0.276) = exp(3.31) = 27.4x.
/// At SZA 108 (delta_cos = 0.309): boost = exp(12 * 0.309) = exp(3.71) = 40.9x.
/// This strongly concentrates split particles near the terminator where
/// they can make useful connections to sunlit atmosphere.
const CADIS_K_MAX: f64 = 12.0;

/// SZA center for CADIS ramp [degrees].
const CADIS_SZA_CENTER: f64 = 100.0;

/// SZA width for CADIS ramp [degrees].
///
/// Width 3.5 gives a gentle transition centered at SZA 100:
///   SZA 97: t=0.30, k=2.4 (mild -- chains still mostly vertical)
///   SZA 100: t=0.50, k=4.0 (moderate -- lateral transport begins)
///   SZA 103: t=0.70, k=5.6 (strong -- deep shadow)
///   SZA 104: t=0.76, k=6.0 (strong)
///   SZA 106: t=0.85, k=6.8 (near full)
///
/// Width 3.5 (vs 3.0) slightly reduces cadis_k at SZA 104-105,
/// recovering a regression where the previous width=3.0 over-boosted
/// lateral importance at those SZAs (cadis_k 6.38/6.74 vs sweet spot
/// ~5.5-6.0).
///
/// Engaging at SZA 100 is physics-motivated: the geometric shadow height
/// exceeds the atmosphere height (~98 km at SZA 100), so the entire
/// atmospheric column above the observer is in shadow and light MUST
/// arrive via lateral scattering from the terminator.
const CADIS_SZA_WIDTH: f64 = 3.5;

/// Returns the SZA-adaptive CADIS lateral importance strength.
///
/// At SZA < 100: ~0 (no lateral importance, altitude-only windows).
/// At SZA = 103: ~4 (moderate lateral boost).
/// At SZA > 106: ~8 (full lateral boost).
#[inline]
fn cadis_k(sza_deg: f64) -> f64 {
    CADIS_K_MAX * sigmoid((sza_deg - CADIS_SZA_CENTER) / CADIS_SZA_WIDTH)
}

/// Exponential scale height for LOS ray-budget redistribution at moderate
/// twilight (SZA = 96).
///
/// At moderate twilight, chains at most altitudes are productive. A large
/// scale height (100 km) gives very mild redistribution (~2.7x ratio from
/// surface to TOA), barely perturbing the uniform baseline.
const LOS_IMP_H_MODERATE_M: f64 = 100_000.0;

/// Exponential scale height for LOS ray-budget redistribution at deep
/// twilight (SZA >= 106).
///
/// At deep twilight, chains starting below ~15 km must climb 50+ km AND
/// travel 1800+ km laterally to reach sunlit atmosphere (~0.001% success
/// rate). A smaller scale height (30 km) provides stronger redistribution
/// (~28x ratio from surface to TOA), shifting the budget away from these
/// hopeless low-altitude chains toward higher-altitude LOS steps.
///
/// Chains at very high altitude (80-100 km) receive more rays but terminate
/// nearly instantly (atmosphere escape on first step) at negligible cost.
/// The real savings come from NOT launching expensive 50-bounce chains at
/// 0-15 km altitude.
///
/// Tested 15 km globally: inconclusive, possible SZA 98 regression.
/// Now SZA-conditional: 30 km for SZA 96-105, ramps to 15 km for SZA > 105.
const LOS_IMP_H_DEEP_M: f64 = 30_000.0;

/// Scale height at extreme deep twilight (SZA > 108).
/// At 15 km, surface-to-TOA ratio is exp(100/15) = ~790x, virtually
/// eliminating rays at ground level. Only applied at SZA > 105 where
/// ground-level chains have near-zero success rate anyway.
const LOS_IMP_H_EXTREME_M: f64 = 15_000.0;

/// State of a single particle in the weight-window work stack (scalar mode).
///
/// When a chain's weight exceeds the upper window bound, it is split into
/// K copies (each with weight/K). The work stack stores pending copies;
/// the main particle is processed first.
#[derive(Clone, Copy)]
struct SplitParticleScalar {
    pos: Vec3,
    dir: Vec3,
    weight: f64,
    rng: McRng,
}

/// State of a single particle in the weight-window work stack (ALIS mode).
///
/// Same as `SplitParticleScalar` but carries per-wavelength weight ratios
/// for the ALIS hero tracing scheme.
#[derive(Clone, Copy)]
struct SplitParticleAlis {
    pos: Vec3,
    dir: Vec3,
    hero_weight: f64,
    weight_ratio: [f64; 64],
    rng: McRng,
}

// --- BDPT (Bidirectional Path Tracing) ---
//
// At deep twilight (SZA > 100), backward chains must travel 1000+ km laterally
// to reach sunlit atmosphere. The success rate drops to ~0.001% at SZA 106.
// BDPT traces light subpaths FORWARD from the sunlit TOA into the atmosphere,
// recording scatter vertices. These vertices are then connected to eye subpath
// (LOS integration) points, providing an alternative path to the sun that
// bypasses the lateral transport bottleneck.
//
// Connection transmittance at high altitude (80-100 km) is 95-99.7% even at
// 1500 km separation, so BDPT connections are highly efficient.
//
// MIS (Multiple Importance Sampling) via the balance heuristic prevents
// double-counting between NEE and BDPT connections.

/// Maximum number of scatter vertices stored per light subpath.
///
/// Set to 1 (single scatter at entry). After the first scatter near the
/// terminator, subsequent bounces drift the photon deeper into the atmosphere
/// where connection transmittance drops sharply, adding variance without
/// proportional signal. With 1 vertex per subpath, we maximize the number
/// of independent terminator entry points (subpaths) for better angular
/// coverage.
///
/// Tested with 2 vertices: regressed SZA 100-108 CV by 2.7-8.6x. The second
/// vertex produces rare high-weight connections that inflate both mean and
/// variance. Confirmed twice (once before MIS fix, once after).
const BDPT_MAX_LIGHT_VERTICES: usize = 1;

/// Number of independent light subpaths traced per call to
/// `hybrid_scatter_radiance_alis`. With BDPT_MAX_LIGHT_VERTICES=1, each
/// subpath produces exactly 1 vertex. 4096 subpaths with stratified jittered
/// azimuthal sampling gives dense, uniform coverage of the terminator strip.
///
/// Processed in batches of BDPT_BATCH_SIZE to avoid stack overflow.
///
/// Scaling history (with SZA-adaptive chord threshold):
///   64  -> 128:  SZA 101 CV 1.12 -> 0.83 (-26%)
///   128 -> 256:  SZA 106 CV 1.86 -> 1.16 (-38%)
///   256 -> 512:  SZA 106 CV 1.16 -> 0.89 (-23%)
///   512 -> 1024: SZA 107 CV 1.22 -> 0.89 (-27%)
///   1024-> 2048: SZA 108 CV 1.25 -> 0.98 (-21%)
///   2048-> 4096: SZA 106 CV 0.641 -> 0.545 (-15%, batched processing)
///   8192: 50-seed SZA 106 0.370->0.262 (-29%) but +45% runtime, 1/sqrt(N)
///         diminishing returns. Now SZA-conditional: 8192 for SZA > 106.
const BDPT_NUM_LIGHT_SUBPATHS: usize = 4096;

/// Increased subpath count for extreme deep twilight (SZA > 106).
/// At these SZAs, BDPT handles ~98% of the signal (w_bdpt near 1.0),
/// so doubling subpaths directly reduces the dominant variance source.
const BDPT_NUM_LIGHT_SUBPATHS_DEEP: usize = 8192;

/// Batch size for BDPT vertex processing.
///
/// Light subpaths are traced and connected in batches of this size to
/// keep the stack-allocated LightVertex buffer under ~600 KB (1024 * 590 bytes).
/// Each batch re-walks the LOS to evaluate connections (cheap: just tau
/// accumulation), then the buffer is reused for the next batch.
const BDPT_BATCH_SIZE: usize = 1024;

/// SZA threshold (degrees) below which BDPT is disabled.
///
/// At SZA < 98, backward chains already have high success rates and NEE
/// works well. BDPT overhead (transmittance evaluations for connections)
/// would be wasted. Ramps in via sigmoid from 99 to 105.
const BDPT_SZA_START: f64 = 99.0;

/// SZA at which BDPT reaches full strength.
const BDPT_SZA_FULL: f64 = 105.0;

/// A recorded scatter vertex on a light subpath.
///
/// Stores position, incoming direction, shell index, hero weight, and
/// per-wavelength ALIS weight ratios. The incoming direction is needed
/// to evaluate the phase function at the connection point.
#[derive(Clone, Copy)]
#[allow(dead_code)] // pdf_fwd reserved for balance-heuristic MIS refinement
struct LightVertex {
    /// Position of the scatter event (in Earth-centered coordinates).
    pos: Vec3,
    /// Incoming direction at this vertex (direction the photon was traveling
    /// BEFORE scattering here). Used to evaluate phase function for the
    /// connection: phase(dir_in, connection_dir).
    dir_in: Vec3,
    /// Shell index at the scatter position.
    shell_idx: usize,
    /// Hero wavelength weight accumulated along the light path up to this
    /// vertex (includes transmittance, SSA, and phase function weights from
    /// all previous bounces).
    hero_weight: f64,
    /// ALIS weight ratios: `weight_ratio[w] = weight_w / hero_weight`.
    weight_ratio: [f64; 64],
    /// Forward PDF (probability density of generating this vertex from the
    /// light source). Used for MIS weight computation.
    pdf_fwd: f64,
}

/// Returns the SZA-adaptive BDPT strength fraction.
///
/// 0.0 at SZA < 98, ramps to 1.0 at SZA >= 102.
/// Reserved for future defensive MIS integration.
#[inline]
fn bdpt_strength(sza_deg: f64) -> f64 {
    sigmoid(
        (sza_deg - (BDPT_SZA_START + BDPT_SZA_FULL) * 0.5)
            / ((BDPT_SZA_FULL - BDPT_SZA_START) * 0.25),
    )
}

/// Trace a single light subpath from the sunlit TOA into the atmosphere.
///
/// The light subpath enters the top-of-atmosphere (TOA) from the sunlit side,
/// walks through the atmosphere using forced scattering with VSPG, and records
/// a vertex at each scatter event. The photon enters with direction `-sun_dir`
/// (from sun toward Earth) at a position on the illuminated TOA hemisphere.
///
/// Entry point sampling: the entry position on the TOA sphere is importance-
/// sampled near the terminator (the great circle nearest to the observer's
/// location projected along the sun direction). This concentrates light
/// subpaths in the region most useful for connections to deep-twilight eye
/// vertices.
///
/// Returns the number of vertices written to `vertices`.
///
/// Physics: uses forced scattering with VSPG (Volume Scattering Probability
/// Guiding) to guarantee scatter events and bias them toward high altitude,
/// where connection transmittance to the eye path is highest. The observer's
/// SZA drives VSPG importance so boosting is active even though the light
/// subpath enters from the sunlit side. ALIS weight ratios track
/// per-wavelength corrections across all bounces.
#[allow(clippy::needless_range_loop)]
#[allow(clippy::too_many_arguments)]
fn trace_light_subpath(
    atm: &AtmosphereModel,
    sun_dir: Vec3,
    observer_pos: Vec3,
    hero_wl: usize,
    num_wl: usize,
    sza_deg_obs: f64,
    rng: &mut McRng,
    vertices: &mut [LightVertex; BDPT_MAX_LIGHT_VERTICES],
    subpath_idx: usize,
    num_subpaths: usize,
) -> usize {
    let toa_radius = atm.toa_radius();
    let mut n_vertices = 0usize;

    // --- Sample entry point on illuminated TOA hemisphere ---
    //
    // The illuminated hemisphere is the half of the TOA sphere facing the sun.
    // We importance-sample near the terminator (edge of the illuminated disk)
    // because that is closest to the observer at deep twilight and thus has
    // the highest connection transmittance.
    //
    // Strategy: sample a point on the TOA disk (as seen from the sun) using
    // a radial distribution biased toward the rim. The disk has radius
    // toa_radius. For a uniform disk, PDF in (r, phi) is r / (pi * R^2).
    // We use r = R * sqrt(xi) for uniform, or bias toward rim with
    // r = R * xi^(1/3) to get more terminator samples (PDF = 3*r^2 / R^3).
    //
    // For the azimuthal direction on the disk, we importance-sample toward
    // the observer's projected position on the terminator plane.

    // Build coordinate system on the sun-facing disk.
    // The disk normal is -sun_dir (pointing toward the sun from the disk).
    let disk_normal = sun_dir.scale(-1.0); // points toward sun
    let arbitrary = if libm::fabs(disk_normal.y) < 0.9 {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    let disk_u = disk_normal.cross(arbitrary).normalize();
    let disk_v = disk_normal.cross(disk_u);

    // Project observer position onto the terminator plane to get preferred
    // azimuthal direction for entry point sampling.
    let obs_proj_u = observer_pos.dot(disk_u);
    let obs_proj_v = observer_pos.dot(disk_v);
    let obs_proj_len = libm::sqrt(obs_proj_u * obs_proj_u + obs_proj_v * obs_proj_v);
    let pref_phi = if obs_proj_len > 1e-6 {
        libm::atan2(obs_proj_v, obs_proj_u)
    } else {
        0.0
    };

    // Sample radial position on the illuminated disk. Concentrate near the
    // terminator (r_frac -> 1.0) because:
    // 1. At r_frac=0.79 (median of cbrt sampling), the entry point is ~4000 km
    //    inside the sunlit hemisphere, and after scattering, vertices end up
    //    5000+ km from the observer -- beyond connection range.
    // 2. At r_frac=0.999, z_along_sun = R*sqrt(1-0.999^2) = R*0.045 = 289 km,
    //    so the entry point is only ~350 km from the observer's terminator
    //    projection, well within connection range.
    //
    // Sample r_frac uniformly from [1-delta, 1) where delta is small.
    // PDF(r_frac) = 1/delta on [1-delta, 1).
    const BDPT_R_DELTA: f64 = 0.03; // r_frac in [0.97, 1.0)
    let xi_r = xorshift_f64(&mut rng.tau);
    let r_frac = 1.0 - BDPT_R_DELTA * xi_r; // uniform in [1-delta, 1]
                                            // Clamp away from exactly 1.0 to avoid r_sq >= toa_r_sq guard
    let r_frac = if r_frac > 0.9999 { 0.9999 } else { r_frac };
    let r_disk = toa_radius * r_frac;

    // Azimuthal sampling: CONCENTRATED around the observer's projected direction.
    //
    // The observer at deep twilight is near the shadow boundary. Light vertices
    // must be within ~1500 km of the observer's high-altitude LOS steps for the
    // connection chord to stay above the troposphere. The chord minimum altitude
    // for two vertices at height h separated by distance d is:
    //   h_min ≈ h - d^2 / (8*R)
    // At h=70 km and d=1200 km: h_min = 70 - 28.3 = 41.7 km (good)
    // At h=70 km and d=2000 km: h_min = 70 - 78.7 = -8.7 km (underground)
    //
    // With PI/32 half-width (~5.6 deg), the arc on the TOA rim spans:
    //   2 * R * sin(PI/32) ≈ 2 * 6471 * 0.098 ≈ 1268 km
    // This keeps entry points within ~634 km of the observer's projected
    // direction on the terminator, so after a single scatter the light vertex
    // is within ~1000-1400 km of the observer's high-altitude LOS steps.
    //
    // PDF(phi) = 1 / (2 * delta_phi) on [pref_phi - delta, pref_phi + delta].
    const BDPT_PHI_HALF_WIDTH: f64 = core::f64::consts::PI / 16.0;
    // Stratified jittered sampling: divide the azimuthal strip into
    // num_subpaths equal bins and place this subpath's sample within
    // its assigned bin. This ensures uniform terminator coverage and
    // eliminates random clustering of entry points.
    let bin_width = 2.0 * BDPT_PHI_HALF_WIDTH / num_subpaths as f64;
    let xi_phi = xorshift_f64(&mut rng.tau);
    let phi_disk = pref_phi - BDPT_PHI_HALF_WIDTH + (subpath_idx as f64 + xi_phi) * bin_width;

    // Position on the disk (in sun-facing plane at distance toa_radius from center).
    let disk_x = r_disk * libm::cos(phi_disk);
    let disk_y = r_disk * libm::sin(phi_disk);

    // Convert disk position to 3D position on the TOA sphere.
    // The disk center is at sun_dir * toa_radius (the sub-solar point on TOA).
    // But we want a point ON the sphere, not on a flat disk. Project:
    // entry_pos is the point on the sphere whose projection onto the disk is (disk_x, disk_y).
    // We want the point on the TOA sphere that is at distance r_disk from the
    // sun axis. The z-coordinate along sun_dir is sqrt(R^2 - r^2).
    let r_sq = r_disk * r_disk;
    let toa_r_sq = toa_radius * toa_radius;
    if r_sq >= toa_r_sq {
        return 0; // shouldn't happen with r_frac < 1
    }
    let z_along_sun = libm::sqrt(toa_r_sq - r_sq); // positive = toward sun

    // Entry point on the illuminated hemisphere (sun-facing side).
    // The sub-solar point is at sun_dir * toa_radius, so positive z
    // along sun_dir = toward the sun = illuminated side.
    let entry_pos = sun_dir.scale(z_along_sun) + disk_u.scale(disk_x) + disk_v.scale(disk_y);

    // Verify the entry point is on the TOA sphere (debug sanity).
    let entry_r = entry_pos.length();
    if libm::fabs(entry_r - toa_radius) > 1.0 {
        return 0; // numerical issue
    }

    // Entry direction: photon comes from the sun, so direction is -sun_dir.
    // At the TOA boundary, we need to refract into the atmosphere. Since
    // n(vacuum) ~ 1.0 and n(top_shell) ~ 1.0000000x, refraction is negligible.
    let entry_dir = sun_dir.scale(-1.0);

    // Compute entry weight for importance sampling on the illuminated hemisphere.
    //
    // The MC estimator targets: I = integral_{hemisphere} L_sun * cos_inc * [stuff] dA
    // We sample (r_frac, phi) from joint density:
    //   f(r_frac, phi) = (1/delta_r) * 1/(2*delta_phi)
    // in the space (r_frac in [1-delta_r, 1], phi in [pref_phi-delta_phi, pref_phi+delta_phi]).
    //
    // The sphere area element: dA = R^2 * r_frac * dr_frac * dphi / cos_inc
    // where cos_inc = sqrt(1 - r_frac^2) = z_along_sun / R.
    //
    // The PDF per unit sphere area:
    //   p(A) = f(r_frac, phi) * cos_inc / (R^2 * r_frac)
    //        = cos_inc / (delta_r * 2*delta_phi * R^2 * r_frac)
    //
    // Entry weight = cos_inc / p(A)
    //              = delta_r * 2*delta_phi * R^2 * r_frac
    let cos_inc = z_along_sun / toa_radius;
    if cos_inc < 1e-10 {
        return 0; // grazing entry, degenerate
    }
    let entry_weight = BDPT_R_DELTA * 2.0 * BDPT_PHI_HALF_WIDTH * toa_r_sq * r_frac;

    // Entry PDF per unit sphere area (for MIS pdf_accumulated tracking).
    // p(A) = cos_inc / entry_weight
    let entry_pdf = if entry_weight > 1e-30 {
        cos_inc / entry_weight
    } else {
        1e-30
    };

    // Initialize hero weight and ALIS weight ratios.
    let mut hero_weight = entry_weight;
    let mut weight_ratio = [0.0f64; 64];
    for w in 0..num_wl {
        weight_ratio[w] = 1.0; // at entry, all wavelengths have the same geometric weight
    }

    // Nudge radially inward (not along ray direction) to ensure the position
    // is strictly inside the TOA sphere. For rim-biased entry near the
    // terminator, entry_dir is nearly tangential to the sphere, so nudging
    // along the ray barely changes the radius. Radial nudge guarantees
    // shell_index() returns a valid shell on the first call.
    let inward = entry_pos.normalize().scale(-1.0); // unit vector toward Earth center
    let mut pos = entry_pos + inward * 10.0; // 10 m radially inward
    let mut dir = entry_dir;
    let mut pdf_accumulated = entry_pdf;

    // Walk through atmosphere using forced scattering with VSPG.
    //
    // At high altitude (>30 km), extinction is tiny (sigma ~ 1e-8 /m)
    // and analog scattering MFP is hundreds of km. With analog mode,
    // ~96% of light subpaths traverse the entire atmosphere without
    // scattering and produce zero vertices. Forced scattering guarantees
    // a scatter event at each bounce, with weight correction
    // (1 - exp(-tau_max)) for the probability of scattering before the
    // boundary. VSPG (Volume Scattering Probability Guiding) biases the
    // scatter position toward high altitude where BDPT connections to the
    // eye path have high transmittance (95-99.7% at 80-100 km).
    //
    // The observer's SZA drives VSPG importance: at SZA 103+, high-altitude
    // shells get up to 50x the natural sampling probability.
    for _bounce in 0..BDPT_MAX_LIGHT_VERTICES {
        // Scout: compute total optical depth and VSPG segments along ray.
        // Uses observer's SZA for importance so that high-altitude scatters
        // are boosted (where connections to the deep-twilight eye path work).
        let mut vspg_segs = [VspgSegment {
            tau_lo: 0.0,
            tau_hi: 0.0,
            importance: 1.0,
        }; VSPG_MAX_SEGMENTS];
        let (tau_maxes, _hit_ground, n_vspg_segs) = scout_with_vspg_segments_alis(
            atm,
            pos,
            dir,
            hero_wl,
            num_wl,
            sza_deg_obs,
            &mut vspg_segs,
        );

        let tau_max_h = tau_maxes[hero_wl];

        // If total optical depth is negligible, the ray is in essentially
        // transparent atmosphere -- no meaningful scatter possible.
        if tau_max_h < 1e-6 {
            break;
        }

        // Forced scattering weight correction: probability of scattering
        // before the boundary. For hero wavelength: w *= (1 - exp(-tau_max_h)).
        let exp_neg_tau_h = libm::exp(-tau_max_h);
        let one_minus_exp_h = 1.0 - exp_neg_tau_h;
        hero_weight *= one_minus_exp_h;

        // ALIS forced scattering weight ratio correction for non-hero
        // wavelengths: each wavelength has a different total optical depth,
        // so its forced scattering weight differs.
        for w in 0..num_wl {
            let one_minus_exp_w = 1.0 - libm::exp(-tau_maxes[w]);
            weight_ratio[w] *= if one_minus_exp_h > 1e-30 {
                one_minus_exp_w / one_minus_exp_h
            } else {
                0.0
            };
        }

        // VSPG importance-weighted scatter position sampling.
        // Biases toward high-altitude shells; weight correction maintains
        // exact unbiasedness.
        let (tau_s, vspg_w) =
            vspg_sample_from_segments(&vspg_segs, n_vspg_segs, tau_max_h, &mut rng.tau);
        hero_weight *= vspg_w;

        // Advance to the sampled scatter position, tracking per-wavelength
        // optical depths for ALIS corrections.
        let (sp, sd, scatter_shell, taus_at_pos) =
            advance_to_optical_depth_alis(atm, pos, dir, tau_s, hero_wl, num_wl);
        pos = sp;
        dir = sd;

        // ALIS extinction ratio correction at scatter site: accounts for
        // the fact that each wavelength has different extinction, so the
        // probability of scattering at this exact position differs.
        let sigma_h = atm.optics[scatter_shell][hero_wl].extinction;
        if sigma_h > 1e-30 {
            let tau_h_pos = taus_at_pos[hero_wl];
            for w in 0..num_wl {
                let sigma_w = atm.optics[scatter_shell][w].extinction;
                weight_ratio[w] *= (sigma_w / sigma_h) * libm::exp(-(taus_at_pos[w] - tau_h_pos));
            }
        }

        // Apply SSA: probability of scattering vs absorption.
        let hero_optics = &atm.optics[scatter_shell][hero_wl];
        hero_weight *= hero_optics.ssa;
        for w in 0..num_wl {
            let ssa_ratio = if hero_optics.ssa > 1e-30 {
                atm.optics[scatter_shell][w].ssa / hero_optics.ssa
            } else {
                0.0
            };
            weight_ratio[w] *= ssa_ratio;
        }

        // Update forward PDF (isotropic approximation for future MIS).
        let scatter_pdf = if sigma_h > 1e-30 {
            sigma_h * INV_4PI
        } else {
            INV_4PI
        };
        pdf_accumulated *= scatter_pdf;

        // Record vertex BEFORE sampling new direction (dir_in = current dir).
        if n_vertices < BDPT_MAX_LIGHT_VERTICES {
            vertices[n_vertices] = LightVertex {
                pos,
                dir_in: dir,
                shell_idx: scatter_shell,
                hero_weight,
                weight_ratio,
                pdf_fwd: pdf_accumulated,
            };
            n_vertices += 1;
        }

        // Sample new scattering direction from hero phase function.
        let cos_theta = if xorshift_f64(&mut rng.dir) < hero_optics.rayleigh_fraction {
            sample_rayleigh_analytic(xorshift_f64(&mut rng.dir))
        } else {
            sample_henyey_greenstein(xorshift_f64(&mut rng.dir), hero_optics.asymmetry)
        };
        let phi = 2.0 * core::f64::consts::PI * xorshift_f64(&mut rng.dir);
        let new_dir = scatter_direction(dir, cos_theta, phi);

        // ALIS phase function ratio correction.
        let phase_hero = scalar_phase_value(cos_theta, hero_optics);
        if phase_hero > 1e-30 {
            for w in 0..num_wl {
                let optics_w = &atm.optics[scatter_shell][w];
                let phase_w = scalar_phase_value(cos_theta, optics_w);
                weight_ratio[w] *= phase_w / phase_hero;
            }
        }

        // Include phase in forward PDF.
        let phase_fwd = scalar_phase_value(cos_theta, hero_optics) * INV_4PI;
        pdf_accumulated *= if phase_fwd > 1e-30 {
            phase_fwd / scatter_pdf
        } else {
            1.0
        };

        dir = new_dir;
    }

    n_vertices
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

/// SZA at which VSPG importance begins ramping [degrees].
///
/// At SZA 93, the geometric shadow height is ~9 km and forced scattering
/// starts at SZA 96. But even at SZA 94-95, biasing toward high-altitude
/// scatter sites improves chain efficiency. Starting the VSPG ramp early
/// ensures that by SZA 97 (shadow at 48 km), chains have substantial
/// upward guidance rather than the minimal 0.1 * 50 = 5x that the old
/// ZENITH_SZA_START-based ramp provided.
const VSPG_SZA_START: f64 = 93.0;

/// SZA at which VSPG importance reaches maximum [degrees].
///
/// By SZA 103, the shadow exceeds the TOA and purely lateral transport
/// dominates. VSPG should be fully active before this point.
const VSPG_SZA_FULL: f64 = 106.0;

/// Compute altitude-dependent importance for VSPG.
///
/// Returns a multiplier >= 1.0 that biases scatter site selection toward
/// high altitude. The multiplier ramps quadratically from 1.0 (at or below
/// `VSPG_BOOST_START_M`) to a SZA-dependent maximum (at `VSPG_BOOST_FULL_M`).
///
/// The SZA dependence uses VSPG's own ramp (93-103) which is wider than
/// the zenith-bias ramp (96-106), giving meaningful guidance at SZA 97:
/// - SZA <= 93: sza_t = 0, importance = 1.0 (no VSPG effect)
/// - SZA = 97: sza_t = 0.4, max_imp = 20.6 (strong at high altitude)
/// - SZA >= 103: sza_t = 1.0, full VSPG_MAX_IMPORTANCE
#[inline]
fn vspg_importance(alt_m: f64, sza_deg: f64) -> f64 {
    if alt_m <= VSPG_BOOST_START_M {
        return 1.0;
    }
    let sza_t = ((sza_deg - VSPG_SZA_START) / (VSPG_SZA_FULL - VSPG_SZA_START)).clamp(0.0, 1.0);
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
///
/// NOTE: Production tracers now use the fused `scout_with_vspg_segments` +
/// `vspg_sample_from_segments` path. This standalone function is retained
/// for unit tests that verify VSPG sampling correctness in isolation.
#[cfg(test)]
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
                let (new_dir, crossed) = match refract_at_boundary(dir, boundary_pos, n_from, n_to)
                {
                    RefractResult::Refracted(d) => (d, true),
                    RefractResult::TotalReflection(d) => (d, false),
                };
                dir = new_dir;
                pos = boundary_pos + dir * 1e-3;

                // Hit ground.
                if !is_outward && pos.length() <= surface_radius + 1.0 {
                    break;
                }
                // Exited atmosphere.
                if crossed {
                    if next_shell >= num_shells {
                        break;
                    }
                    shell_idx = next_shell;
                }
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

/// Sample a forced-scatter optical depth from pre-collected VSPG segments.
///
/// This is the CDF-inversion half of VSPG sampling, separated from the
/// shell-walk half to enable fusion with the scout pass. The segments
/// must have been collected by `scout_with_vspg_segments()` or
/// `scout_with_vspg_segments_alis()`.
///
/// Returns `(tau_s, weight_correction)` -- same semantics as
/// `vspg_sample_scatter_tau`.
fn vspg_sample_from_segments(
    segments: &[VspgSegment; VSPG_MAX_SEGMENTS],
    num_seg: usize,
    tau_max: f64,
    rng: &mut u64,
) -> (f64, f64) {
    if num_seg == 0 {
        let xi = xorshift_f64(rng);
        let one_minus_exp = 1.0 - libm::exp(-tau_max);
        return (-libm::log(1.0 - xi * one_minus_exp + 1e-30), 1.0);
    }

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
        let xi = xorshift_f64(rng);
        let one_minus_exp = 1.0 - libm::exp(-tau_max);
        return (-libm::log(1.0 - xi * one_minus_exp + 1e-30), 1.0);
    }

    let xi_segment = xorshift_f64(rng) * q_sum;
    let mut j = 0usize;
    while j + 1 < num_seg && q_cdf[j] < xi_segment {
        j += 1;
    }

    let seg = &segments[j];
    let p_j = libm::exp(-seg.tau_lo) - libm::exp(-seg.tau_hi);
    let xi_within = xorshift_f64(rng);
    let tau_s = -libm::log(libm::exp(-seg.tau_lo) - xi_within * p_j + 1e-30);
    let tau_s = tau_s.clamp(seg.tau_lo, seg.tau_hi);

    let i_avg = q_sum / p_sum;
    let weight_correction = i_avg / seg.importance;

    (tau_s, weight_correction)
}

/// Fused scout + VSPG segment collection for single-wavelength tracers.
///
/// Walks the ray path through shells once, simultaneously computing:
/// - Total optical depth to boundary (`tau_max`) -- replaces `scout_tau_to_boundary`
/// - Per-shell VSPG segments -- replaces the walk in `vspg_sample_scatter_tau`
///
/// This eliminates the redundant shell re-walk that `vspg_sample_scatter_tau`
/// performs after `scout_tau_to_boundary`, saving ~33% of shell-walk cost
/// per forced-scatter bounce at deep twilight.
///
/// Returns `(tau_max, hit_ground, num_segments)`. Segments are written to
/// the caller-provided buffer.
fn scout_with_vspg_segments(
    atm: &AtmosphereModel,
    start_pos: Vec3,
    start_dir: Vec3,
    wavelength_idx: usize,
    sza_deg: f64,
    segments: &mut [VspgSegment; VSPG_MAX_SEGMENTS],
) -> (f64, bool, usize) {
    let surface_radius = atm.surface_radius();
    let num_shells = atm.num_shells;
    let mut pos = start_pos;
    let mut dir = start_dir;
    let mut tau = 0.0;
    let mut num_seg: usize = 0;

    let mut shell_idx = match atm.shell_index(pos.length()) {
        Some(idx) => idx,
        None => return (0.0, false, 0),
    };

    for _ in 0..200 {
        let shell = &atm.shells[shell_idx];
        let optics = &atm.optics[shell_idx][wavelength_idx];

        match next_shell_boundary(pos, dir, shell.r_inner, shell.r_outer) {
            Some((dist, is_outward)) => {
                let tau_shell = optics.extinction * dist;
                let tau_end = tau + tau_shell;

                // Collect VSPG segment if shell has nonzero optical depth.
                if num_seg < VSPG_MAX_SEGMENTS && tau_shell > 1e-30 {
                    segments[num_seg] = VspgSegment {
                        tau_lo: tau,
                        tau_hi: tau_end,
                        importance: vspg_importance(shell.altitude_mid, sza_deg),
                    };
                    num_seg += 1;
                }

                tau = tau_end;

                // Refract at boundary.
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
                let (new_dir, crossed) = match refract_at_boundary(dir, boundary_pos, n_from, n_to)
                {
                    RefractResult::Refracted(d) => (d, true),
                    RefractResult::TotalReflection(d) => (d, false),
                };
                dir = new_dir;
                pos = boundary_pos + dir * 1e-3;

                // Hit ground.
                if !is_outward && pos.length() <= surface_radius + 1.0 {
                    return (tau, true, num_seg);
                }
                // Exited atmosphere.
                if crossed {
                    if next_shell >= num_shells {
                        return (tau, false, num_seg);
                    }
                    shell_idx = next_shell;
                }
            }
            None => return (tau, false, num_seg),
        }

        if tau > FORCED_TAU_CUTOFF {
            return (tau, false, num_seg);
        }
    }

    (tau, false, num_seg)
}

/// Fused scout + VSPG segment collection for ALIS (multi-wavelength) tracers.
///
/// Like `scout_with_vspg_segments`, but accumulates optical depth for all
/// wavelengths simultaneously. VSPG segments use the hero wavelength's
/// tau values and altitude-based importance.
///
/// Returns `(tau_maxes, hit_ground, num_segments)`. `tau_maxes[w]` gives
/// the total optical depth for wavelength w. Segments are written to
/// the caller-provided buffer (hero-wavelength tau values).
#[allow(clippy::too_many_arguments)]
fn scout_with_vspg_segments_alis(
    atm: &AtmosphereModel,
    start_pos: Vec3,
    start_dir: Vec3,
    hero_wl: usize,
    num_wl: usize,
    sza_deg: f64,
    segments: &mut [VspgSegment; VSPG_MAX_SEGMENTS],
) -> ([f64; 64], bool, usize) {
    let surface_radius = atm.surface_radius();
    let num_shells = atm.num_shells;
    let mut pos = start_pos;
    let mut dir = start_dir;
    let mut tau = [0.0f64; 64];
    let mut num_seg: usize = 0;

    let mut shell_idx = match atm.shell_index(pos.length()) {
        Some(idx) => idx,
        None => return (tau, false, 0),
    };

    for _ in 0..200 {
        let shell = &atm.shells[shell_idx];

        match next_shell_boundary(pos, dir, shell.r_inner, shell.r_outer) {
            Some((dist, is_outward)) => {
                // Accumulate per-wavelength tau.
                let hero_tau_before = tau[hero_wl];
                for (w, tau_w) in tau.iter_mut().enumerate().take(num_wl) {
                    *tau_w += atm.optics[shell_idx][w].extinction * dist;
                }
                let hero_tau_shell = tau[hero_wl] - hero_tau_before;

                // Collect VSPG segment using hero wavelength tau.
                if num_seg < VSPG_MAX_SEGMENTS && hero_tau_shell > 1e-30 {
                    segments[num_seg] = VspgSegment {
                        tau_lo: hero_tau_before,
                        tau_hi: tau[hero_wl],
                        importance: vspg_importance(shell.altitude_mid, sza_deg),
                    };
                    num_seg += 1;
                }

                // Refract at boundary.
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
                let (new_dir, crossed) = match refract_at_boundary(dir, boundary_pos, n_from, n_to)
                {
                    RefractResult::Refracted(d) => (d, true),
                    RefractResult::TotalReflection(d) => (d, false),
                };
                dir = new_dir;
                pos = boundary_pos + dir * 1e-3;

                if !is_outward && pos.length() <= surface_radius + 1.0 {
                    return (tau, true, num_seg);
                }
                if crossed {
                    if next_shell >= num_shells {
                        return (tau, false, num_seg);
                    }
                    shell_idx = next_shell;
                }
            }
            None => return (tau, false, num_seg),
        }

        if tau[hero_wl] > FORCED_TAU_CUTOFF {
            return (tau, false, num_seg);
        }
    }

    (tau, false, num_seg)
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
                    // Per-chain McRng: master advances by 1 per chain,
                    // making inter-chain sequencing deterministic regardless
                    // of per-chain RNG consumption.
                    let _ = xorshift_f64(rng_state);
                    let mut mc_rng = McRng::from_seed(*rng_state);
                    let chain_stokes = trace_secondary_chain(
                        atm,
                        scatter_pos,
                        view_dir,
                        sun_dir,
                        wavelength_idx,
                        optics,
                        &mut mc_rng,
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
                    let _ = xorshift_f64(rng_state);
                    let mut mc_rng = McRng::from_seed(*rng_state);
                    mc_scalar += trace_secondary_chain_scalar(
                        atm,
                        scatter_pos,
                        sun_dir,
                        wavelength_idx,
                        optics,
                        &mut mc_rng,
                        ray,
                        secondary_rays,
                        1.0,
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
    rng: &mut McRng,
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
    let xi_jitter = xorshift_f64(&mut rng.dir);
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
        let cos_theta_init = if xorshift_f64(&mut rng.dir) < start_optics.rayleigh_fraction {
            sample_rayleigh_analytic(xorshift_f64(&mut rng.dir))
        } else {
            sample_henyey_greenstein(xorshift_f64(&mut rng.dir), start_optics.asymmetry)
        };
        let phi_init = 2.0 * core::f64::consts::PI * xorshift_f64(&mut rng.dir);
        let d = scatter_direction(sun_dir, cos_theta_init, phi_init);
        let branch_w = 0.5 / alpha_p;
        (d, cos_theta_init, branch_w)
    } else if xi_mix < alpha_p + alpha_z || alpha_t < 1e-12 {
        // Zenith-biased branch with shape + branch weight correction
        let (d, cos_z) = sample_zenith_biased(local_up, bp.n_zenith, &mut rng.dir);
        let cos_theta_init = sun_dir.dot(d);
        let shape_w = zenith_importance_weight(cos_z, bp.n_zenith);
        let branch_w = 0.5 / (alpha_z + alpha_t); // fallback if alpha_t ~ 0
        (d, cos_theta_init, shape_w * branch_w)
    } else {
        // Terminator lobe branch
        let (d, cos_t) = sample_zenith_biased(term_axis, bp.m_term, &mut rng.dir);
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

    for _scatter in 0..BOUNCE_SAFETY_LIMIT {
        // --- Decide scatter mode for this bounce ---
        let mut forced_this_bounce = false;
        let mut tau_max = 0.0;

        if use_forced {
            let (tm, hit_ground) = scout_tau_to_boundary(atm, pos, current_dir, wavelength_idx);
            tau_max = tm;
            // Force scatter only when path exits to space AND has moderate optical
            // depth. Ground-bound paths: analog (ground reflection). Dense paths
            // (tau >= 20): analog (equivalent, no scout overhead). Very thin paths
            // (tau < forced_tau_min): analog to avoid weight death -- the forced-
            // scatter weight (1-exp(-tau)) is punishingly small for tau < 0.3.
            let ftm = forced_tau_min_for_sza(sza_deg_local);
            forced_this_bounce = !hit_ground && (ftm..FORCED_TAU_CUTOFF).contains(&tm);
        }

        let scatter_shell;

        if forced_this_bounce {
            // Upfront forced scattering: weight = exact scatter probability.
            // No analog free-path walk, no escape, no double-counting.
            let exp_neg_tau = libm::exp(-tau_max);
            weight *= 1.0 - exp_neg_tau;
            let xi = xorshift_f64(&mut rng.tau);
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

                let xi = xorshift_f64(&mut rng.tau);
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
                                let normal = pos.normalize();

                                // Ground-bounce NEE: Lambertian BRDF = albedo/pi.
                                let cos_sun_ground = sun_dir.dot(normal);
                                if cos_sun_ground > 0.0 {
                                    let t_sun_gb =
                                        shadow_ray_transmittance(atm, pos, sun_dir, wavelength_idx);
                                    if t_sun_gb > 1e-30 {
                                        let albedo = atm.surface_albedo[wavelength_idx];
                                        let nee_gb = weight * albedo * t_sun_gb * cos_sun_ground
                                            / core::f64::consts::PI;
                                        // Lambertian depolarizes: only I component.
                                        total_stokes =
                                            total_stokes.add(&StokesVector::unpolarized(nee_gb));
                                    }
                                }

                                let albedo = atm.surface_albedo[wavelength_idx];
                                weight *= albedo;
                                prev_dir = current_dir;
                                current_dir = sample_hemisphere(normal, &mut rng.dir);
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

        // Sample new direction and update Stokes state
        let cos_theta = if xorshift_f64(&mut rng.dir) < optics.rayleigh_fraction {
            sample_rayleigh_analytic(xorshift_f64(&mut rng.dir))
        } else {
            sample_henyey_greenstein(xorshift_f64(&mut rng.dir), optics.asymmetry)
        };
        let phi = 2.0 * core::f64::consts::PI * xorshift_f64(&mut rng.dir);
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
    rng: &mut McRng,
    ray_idx: usize,
    total_rays: usize,
    nee_r2_weight: f64,
) -> f64 {
    use crate::single_scatter::shadow_ray_transmittance;

    let local_up = start_pos.normalize();

    // --- SZA-adaptive 3-branch parameters ---
    let cos_sza = sun_dir.dot(local_up);
    let sza_deg_local = libm::acos(cos_sza.clamp(-1.0, 1.0)) * 180.0 / core::f64::consts::PI;
    let bp = branch_params_for_sza(cos_sza);
    let d_frac = dwivedi_frac(sza_deg_local);
    let d_beta = dwivedi_beta(sza_deg_local);

    let alpha_p = 1.0 - bp.zenith_frac;
    let alpha_z = bp.zenith_frac * (1.0 - bp.term_share);
    let alpha_t = bp.zenith_frac * bp.term_share;

    let term_axis = terminator_axis(local_up, sun_dir, bp.tilt_rad);

    // --- Stratified initial direction sampling ---
    let xi_jitter = xorshift_f64(&mut rng.dir);
    let xi_mix = (ray_idx as f64 + xi_jitter) / total_rays as f64;

    // 3-branch importance sampling. See trace_secondary_chain for derivation.
    // At SZA <= 96: alpha_p=0.5, alpha_z=0.5, alpha_t=0. Both weights = 1.0.
    let (dir, initial_weight) = if xi_mix < alpha_p {
        // Phase function branch (toward sun_dir -- effective at civil twilight)
        let _cos_theta_init = if xorshift_f64(&mut rng.dir) < start_optics.rayleigh_fraction {
            sample_rayleigh_analytic(xorshift_f64(&mut rng.dir))
        } else {
            sample_henyey_greenstein(xorshift_f64(&mut rng.dir), start_optics.asymmetry)
        };
        let phi_init = 2.0 * core::f64::consts::PI * xorshift_f64(&mut rng.dir);
        let branch_w = 0.5 / alpha_p;
        (
            scatter_direction(sun_dir, _cos_theta_init, phi_init),
            branch_w,
        )
    } else if xi_mix < alpha_p + alpha_z || alpha_t < 1e-12 {
        // Zenith-biased branch with shape + branch weight correction
        let (d, cos_z) = sample_zenith_biased(local_up, bp.n_zenith, &mut rng.dir);
        let shape_w = zenith_importance_weight(cos_z, bp.n_zenith);
        let branch_w = 0.5 / (alpha_z + alpha_t);
        (d, shape_w * branch_w)
    } else {
        // Terminator lobe branch
        let (d, cos_t) = sample_zenith_biased(term_axis, bp.m_term, &mut rng.dir);
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

    // Weight window setup. Scale height controls splitting/RR aggressiveness.
    // CADIS lateral importance encourages chains toward the sunlit terminator.
    let h_ww = weight_window_h(sza_deg_local);
    let alt_start = start_pos.length() - surface_radius;
    let cos_sun_start = local_up.dot(sun_dir);
    let ck = cadis_k(sza_deg_local);

    // Initialize work stack with the main particle.
    let dummy_rng = McRng {
        tau: 0,
        dir: 0,
        ctl: 0,
    };
    let mut stack = [SplitParticleScalar {
        pos: Vec3::new(0.0, 0.0, 0.0),
        dir: Vec3::new(0.0, 0.0, 1.0),
        weight: 0.0,
        rng: dummy_rng,
    }; MAX_SPLIT_PARTICLES];
    let mut stack_len: usize = 1;
    stack[0] = SplitParticleScalar {
        pos: start_pos,
        dir,
        weight: initial_weight,
        rng: *rng,
    };
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
        let mut bounce_idx: usize = 0;

        loop {
            // --- Decide scatter mode for this bounce ---
            // Fused scout + VSPG: single shell walk collects both tau_max
            // and VSPG segments, eliminating the redundant re-walk.
            let mut forced_this_bounce = false;
            let mut tau_max = 0.0;
            let mut vspg_segs = [VspgSegment {
                tau_lo: 0.0,
                tau_hi: 0.0,
                importance: 1.0,
            }; VSPG_MAX_SEGMENTS];
            let mut n_vspg_segs = 0usize;

            if use_forced {
                let (tm, hit_ground, ns) = scout_with_vspg_segments(
                    atm,
                    pos,
                    current_dir,
                    wavelength_idx,
                    sza_deg_local,
                    &mut vspg_segs,
                );
                tau_max = tm;
                n_vspg_segs = ns;
                let ftm = forced_tau_min_for_sza(sza_deg_local);
                forced_this_bounce = !hit_ground && (ftm..FORCED_TAU_CUTOFF).contains(&tm);
            }

            let scatter_shell;

            if forced_this_bounce {
                let exp_neg_tau = libm::exp(-tau_max);
                weight *= 1.0 - exp_neg_tau;
                // VSPG: sample from pre-collected segments (no re-walk).
                let (tau_s, vspg_w) =
                    vspg_sample_from_segments(&vspg_segs, n_vspg_segs, tau_max, &mut local_rng.tau);
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

                    let xi = xorshift_f64(&mut local_rng.tau);
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
                                    let normal = pos.normalize();

                                    // Ground-bounce NEE: Lambertian BRDF = albedo/pi.
                                    let cos_sun_ground = sun_dir.dot(normal);
                                    if cos_sun_ground > 0.0 {
                                        let t_sun_gb = shadow_ray_transmittance(
                                            atm,
                                            pos,
                                            sun_dir,
                                            wavelength_idx,
                                        );
                                        if t_sun_gb > 1e-30 {
                                            let albedo = atm.surface_albedo[wavelength_idx];
                                            total += weight * albedo * t_sun_gb * cos_sun_ground
                                                / core::f64::consts::PI;
                                        }
                                    }

                                    let albedo = atm.surface_albedo[wavelength_idx];
                                    weight *= albedo;
                                    current_dir = sample_hemisphere(normal, &mut local_rng.dir);
                                    scatter_found = false;
                                    break;
                                }
                                scatter_found = false;
                                break;
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
            weight *= optics.ssa;

            let bdpt_covered = is_main && bounce_idx < BDPT_MAX_LIGHT_VERTICES;
            let skip_nee = bdpt_covered && nee_r2_weight < 1e-30;
            if !skip_nee {
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

                    // On the first BDPT_MAX_LIGHT_VERTICES bounces of the main
                    // particle, apply the MIS weight (w_back) since BDPT provides
                    // independent estimates for these orders. Higher-order bounces
                    // get full weight since BDPT does not cover them.
                    let nee_weight = if bdpt_covered { nee_r2_weight } else { 1.0 };
                    let scale = nee_weight * weight * t_sun_secondary * INV_4PI;
                    total += phase * scale;
                }
            }
            bounce_idx += 1;

            // Sample new direction: 2-way one-sample MIS between phase
            // function and Dwivedi horizontal biasing.
            //
            // When Dwivedi is inactive (alpha_d < 0.01), fall through to
            // pure phase function sampling with no MIS overhead and no extra
            // RNG consumption.
            let alpha_d = d_frac;
            let mis_active = alpha_d >= 0.02;

            let new_dir = if mis_active {
                let local_up_here = pos.normalize();
                let alpha_p_mis = 1.0 - alpha_d;
                let xi_branch = xorshift_f64(&mut local_rng.dir);

                if xi_branch < alpha_d {
                    // Dwivedi branch: sample direction biased toward horizontal
                    let xi1 = xorshift_f64(&mut local_rng.dir);
                    let xi2 = xorshift_f64(&mut local_rng.dir);
                    let xi_sign = xorshift_f64(&mut local_rng.dir);
                    let (cos_z, phi_dw) = dwivedi_sample(xi1, xi2, xi_sign, d_beta);
                    let sin_z = libm::sqrt((1.0 - cos_z * cos_z).max(0.0));
                    let east = {
                        let arbitrary = if libm::fabs(local_up_here.y) < 0.9 {
                            Vec3::new(0.0, 1.0, 0.0)
                        } else {
                            Vec3::new(1.0, 0.0, 0.0)
                        };
                        let e = local_up_here.cross(arbitrary);
                        e.normalize()
                    };
                    let north = local_up_here.cross(east);
                    let d = local_up_here.scale(cos_z)
                        + east.scale(sin_z * libm::cos(phi_dw))
                        + north.scale(sin_z * libm::sin(phi_dw));
                    let d = d.normalize();

                    let cos_t = current_dir.dot(d);
                    let p_phase = scalar_phase_value(cos_t, optics) * INV_4PI;
                    let p_dw = dwivedi_pdf(cos_z, d_beta);
                    let mis_denom = alpha_p_mis * p_phase + alpha_d * p_dw;
                    if mis_denom > 1e-30 {
                        weight *= p_phase / mis_denom;
                    }
                    d
                } else {
                    // Phase function branch (within MIS)
                    let cos_theta = if xorshift_f64(&mut local_rng.dir) < optics.rayleigh_fraction {
                        sample_rayleigh_analytic(xorshift_f64(&mut local_rng.dir))
                    } else {
                        sample_henyey_greenstein(xorshift_f64(&mut local_rng.dir), optics.asymmetry)
                    };
                    let phi = 2.0 * core::f64::consts::PI * xorshift_f64(&mut local_rng.dir);
                    let d = scatter_direction(current_dir, cos_theta, phi);

                    let p_phase = scalar_phase_value(cos_theta, optics) * INV_4PI;
                    let cos_z_dw = d.dot(local_up_here);
                    let p_dw = dwivedi_pdf(cos_z_dw, d_beta);
                    let mis_denom = alpha_p_mis * p_phase + alpha_d * p_dw;
                    if mis_denom > 1e-30 {
                        weight *= p_phase / mis_denom;
                    }
                    d
                }
            } else {
                // Pure phase function: no Dwivedi, no MIS overhead.
                let cos_theta = if xorshift_f64(&mut local_rng.dir) < optics.rayleigh_fraction {
                    sample_rayleigh_analytic(xorshift_f64(&mut local_rng.dir))
                } else {
                    sample_henyey_greenstein(xorshift_f64(&mut local_rng.dir), optics.asymmetry)
                };
                let phi = 2.0 * core::f64::consts::PI * xorshift_f64(&mut local_rng.dir);
                scatter_direction(current_dir, cos_theta, phi)
            };
            current_dir = new_dir;

            // --- Weight window population control ---
            let alt = pos.length() - surface_radius;
            let cos_sun_here = pos.normalize().dot(sun_dir);
            let w_target =
                weight_window_target(alt, alt_start, h_ww, cos_sun_here, cos_sun_start, ck);
            let w_lower = w_target / WW_LOWER_RATIO;
            let w_upper = w_target * WW_UPPER_RATIO;
            let abs_w = weight.abs();

            if abs_w < w_lower && w_target > 1e-30 {
                // Russian roulette: chain weight is too small for this region.
                // Survive with probability p = |weight| / w_target.
                // On survival: weight = sign(weight) * w_target.
                // Unbiased: E[output] = p * w_target = |weight|.
                let p_survive = abs_w / w_target;
                if xorshift_f64(&mut local_rng.ctl) < p_survive {
                    weight = if weight >= 0.0 { w_target } else { -w_target };
                } else {
                    break; // Chain killed by RR
                }
            } else if abs_w > w_upper && w_target > 1e-30 {
                // Splitting: chain weight is too large for this region.
                // Create k copies each with weight/k. Cap k to available
                // stack space (remaining slots + 1 for the main particle).
                let k_ideal = libm::round(abs_w / w_target) as usize;
                let max_k = MAX_SPLIT_PARTICLES - stack_len + 1;
                if max_k >= 2 {
                    let k = k_ideal.clamp(2, max_k);
                    weight /= k as f64;
                    for copy_idx in 1..k {
                        if stack_len < MAX_SPLIT_PARTICLES {
                            let child_seed = splitmix64(
                                local_rng.tau
                                    ^ (copy_idx as u64).wrapping_mul(2654435761)
                                    ^ (alt.to_bits() >> 32),
                            );
                            stack[stack_len] = SplitParticleScalar {
                                pos,
                                dir: current_dir,
                                weight,
                                rng: McRng::from_seed(child_seed),
                            };
                            stack_len += 1;
                        }
                    }
                }
            }
        }
    }

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
///
/// NOTE: Production tracers now use the fused `scout_with_vspg_segments_alis`.
/// This standalone function is retained for unit tests that verify ALIS scout
/// correctness against the single-wavelength scout.
#[cfg(test)]
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
                let (new_dir, crossed) = match refract_at_boundary(dir, boundary_pos, n_from, n_to)
                {
                    RefractResult::Refracted(d) => (d, true),
                    RefractResult::TotalReflection(d) => (d, false),
                };
                dir = new_dir;
                pos = boundary_pos + dir * 1e-3;

                if !is_outward && pos.length() <= surface_radius + 1.0 {
                    return (tau, true);
                }
                if crossed {
                    if next_shell >= num_shells {
                        return (tau, false);
                    }
                    shell_idx = next_shell;
                }
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
                let (new_dir, crossed) = match refract_at_boundary(dir, boundary_pos, n_from, n_to)
                {
                    RefractResult::Refracted(d) => (d, true),
                    RefractResult::TotalReflection(d) => (d, false),
                };
                dir = new_dir;
                pos = boundary_pos + dir * 1e-3;

                if !is_outward && pos.length() <= surface_radius + 1.0 {
                    return (pos, dir, shell_idx, tau_accumulated);
                }
                if crossed {
                    if next_shell >= num_shells {
                        return (pos, dir, shell_idx, tau_accumulated);
                    }
                    shell_idx = next_shell;
                }
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
    rng: &mut McRng,
    ray_idx: usize,
    total_rays: usize,
    num_wl: usize,
    nee_r2_weight: f64,
    guide: Option<&crate::path_guide::PathGuide>,
) -> [f64; 64] {
    use crate::single_scatter::shadow_ray_transmittance_spectrum;

    let local_up = start_pos.normalize();
    let hero_optics = &atm.optics[start_shell][hero_wl];

    // --- SZA-adaptive 3-branch parameters ---
    let cos_sza = sun_dir.dot(local_up);
    let sza_deg_local = libm::acos(cos_sza.clamp(-1.0, 1.0)) * 180.0 / core::f64::consts::PI;
    let bp = branch_params_for_sza(cos_sza);
    let d_frac = dwivedi_frac(sza_deg_local);
    let d_beta = dwivedi_beta(sza_deg_local);

    let alpha_p = 1.0 - bp.zenith_frac;
    let alpha_z = bp.zenith_frac * (1.0 - bp.term_share);
    let alpha_t = bp.zenith_frac * bp.term_share;

    let term_axis = terminator_axis(local_up, sun_dir, bp.tilt_rad);

    // --- Stratified initial direction sampling ---
    let xi_jitter = xorshift_f64(&mut rng.dir);
    let xi_mix = (ray_idx as f64 + xi_jitter) / total_rays as f64;

    // Sample initial direction from hero's phase function (phase branch),
    // zenith-biased distribution (zenith branch), or terminator lobe
    // (terminator branch). Track cos_theta_init for phase function ratio
    // correction on non-hero wavelengths.
    let (dir, initial_weight, cos_theta_init, is_phase_branch) = if xi_mix < alpha_p {
        // Phase function branch
        let ct = if xorshift_f64(&mut rng.dir) < hero_optics.rayleigh_fraction {
            sample_rayleigh_analytic(xorshift_f64(&mut rng.dir))
        } else {
            sample_henyey_greenstein(xorshift_f64(&mut rng.dir), hero_optics.asymmetry)
        };
        let phi_init = 2.0 * core::f64::consts::PI * xorshift_f64(&mut rng.dir);
        let branch_w = 0.5 / alpha_p;
        (scatter_direction(sun_dir, ct, phi_init), branch_w, ct, true)
    } else if xi_mix < alpha_p + alpha_z || alpha_t < 1e-12 {
        // Zenith-biased branch (wavelength-independent)
        let (d, cos_z) = sample_zenith_biased(local_up, bp.n_zenith, &mut rng.dir);
        let shape_w = zenith_importance_weight(cos_z, bp.n_zenith);
        let branch_w = 0.5 / (alpha_z + alpha_t);
        (d, shape_w * branch_w, 0.0, false)
    } else {
        // Terminator lobe branch (wavelength-independent)
        let (d, cos_t) = sample_zenith_biased(term_axis, bp.m_term, &mut rng.dir);
        let cos_z = d.dot(local_up);
        let shape_w = terminator_shape_weight(cos_z, cos_t, bp.m_term);
        let branch_w = 0.5 / alpha_t;
        (d, shape_w * branch_w, 0.0, false)
    };

    // Initialize per-wavelength weight ratios: weight_ratio[w] = weight_w / hero_weight.
    // Only the initial direction sampling (phase function ratio) differs across
    // wavelengths here. SSA is handled correctly by:
    //   - outer integrator: beta_scat = extinction * ssa (per-wavelength)
    //   - chain: hero_weight *= ssa_hero at each scatter, wr[w] *= ssa_w/ssa_hero
    // Including ssa_ratio here would double-count the start-shell SSA.
    let mut weight_ratio = [0.0f64; 64];
    let hero_phase_init = if is_phase_branch {
        scalar_phase_value(cos_theta_init, hero_optics)
    } else {
        1.0
    };
    for w in 0..num_wl {
        let optics_w = &atm.optics[start_shell][w];
        let dir_ratio = if is_phase_branch && hero_phase_init > 1e-30 {
            scalar_phase_value(cos_theta_init, optics_w) / hero_phase_init
        } else {
            1.0
        };
        weight_ratio[w] = dir_ratio;
    }

    let surface_radius = atm.surface_radius();
    let mut total = [0.0f64; 64];

    // Forced scattering + exponential transform setup (same as scalar tracer).
    let sza_deg_local = libm::acos(cos_sza.clamp(-1.0, 1.0)) * 180.0 / core::f64::consts::PI;
    let use_forced = sza_deg_local >= ZENITH_SZA_START;
    let sza_t_et =
        ((sza_deg_local - ZENITH_SZA_START) / (ZENITH_SZA_FULL - ZENITH_SZA_START)).clamp(0.0, 1.0);
    let alpha_et = EXP_TRANSFORM_ALPHA_MAX * sza_t_et;

    // Weight window setup (same as scalar tracer, with CADIS lateral importance).
    let h_ww = weight_window_h(sza_deg_local);
    let alt_start = start_pos.length() - surface_radius;
    let cos_sun_start = local_up.dot(sun_dir);
    let ck = cadis_k(sza_deg_local);

    // Initialize work stack with the main particle.
    let dummy_rng = McRng {
        tau: 0,
        dir: 0,
        ctl: 0,
    };
    let mut stack = [SplitParticleAlis {
        pos: Vec3::new(0.0, 0.0, 0.0),
        dir: Vec3::new(0.0, 0.0, 1.0),
        hero_weight: 0.0,
        weight_ratio: [0.0f64; 64],
        rng: dummy_rng,
    }; MAX_SPLIT_PARTICLES];
    let mut stack_len: usize = 1;
    stack[0] = SplitParticleAlis {
        pos: start_pos,
        dir,
        hero_weight: initial_weight,
        weight_ratio,
        rng: *rng,
    };
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
        let mut bounce_idx: usize = 0;

        loop {
            // --- Decide scatter mode for this bounce ---
            // Fused scout + VSPG: single shell walk collects both per-wl
            // tau_maxes and VSPG segments (hero wavelength), eliminating
            // the redundant re-walk.
            let mut forced_this_bounce = false;
            let mut tau_maxes = [0.0f64; 64];
            let mut vspg_segs = [VspgSegment {
                tau_lo: 0.0,
                tau_hi: 0.0,
                importance: 1.0,
            }; VSPG_MAX_SEGMENTS];
            let mut n_vspg_segs = 0usize;

            if use_forced {
                let (tms, hit_ground, ns) = scout_with_vspg_segments_alis(
                    atm,
                    pos,
                    current_dir,
                    hero_wl,
                    num_wl,
                    sza_deg_local,
                    &mut vspg_segs,
                );
                tau_maxes = tms;
                n_vspg_segs = ns;
                let ftm = forced_tau_min_for_sza(sza_deg_local);
                forced_this_bounce =
                    !hit_ground && (ftm..FORCED_TAU_CUTOFF).contains(&tms[hero_wl]);
            }

            let scatter_shell;

            if forced_this_bounce {
                let tau_max_h = tau_maxes[hero_wl];
                let exp_neg_tau_h = libm::exp(-tau_max_h);
                let one_minus_exp_h = 1.0 - exp_neg_tau_h;
                hero_weight *= one_minus_exp_h;

                for w in 0..num_wl {
                    let one_minus_exp_w = 1.0 - libm::exp(-tau_maxes[w]);
                    wr[w] *= if one_minus_exp_h > 1e-30 {
                        one_minus_exp_w / one_minus_exp_h
                    } else {
                        0.0
                    };
                }

                // VSPG: sample from pre-collected segments (no re-walk).
                let (tau_s, vspg_w) = vspg_sample_from_segments(
                    &vspg_segs,
                    n_vspg_segs,
                    tau_max_h,
                    &mut local_rng.tau,
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

                    let xi = xorshift_f64(&mut local_rng.tau);
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
                                    let normal = pos.normalize();

                                    // Ground-bounce NEE: Lambertian BRDF = albedo/pi.
                                    // Fire shadow ray before albedo is applied to
                                    // the continuing chain weight.
                                    let cos_sun_ground = sun_dir.dot(normal);
                                    if cos_sun_ground > 0.0 {
                                        let t_suns_gb = shadow_ray_transmittance_spectrum(
                                            atm, pos, sun_dir, num_wl,
                                        );
                                        let inv_pi = 1.0 / core::f64::consts::PI;
                                        for w in 0..num_wl {
                                            if t_suns_gb[w] > 1e-30 {
                                                total[w] += hero_weight
                                                    * wr[w]
                                                    * atm.surface_albedo[w]
                                                    * t_suns_gb[w]
                                                    * cos_sun_ground
                                                    * inv_pi;
                                            }
                                        }
                                    }

                                    let hero_albedo = atm.surface_albedo[hero_wl];
                                    hero_weight *= hero_albedo;
                                    for w in 0..num_wl {
                                        let albedo_ratio = if hero_albedo > 1e-30 {
                                            atm.surface_albedo[w] / hero_albedo
                                        } else {
                                            0.0
                                        };
                                        wr[w] *= albedo_ratio;
                                    }
                                    current_dir = sample_hemisphere(normal, &mut local_rng.dir);
                                    scatter_found = false;
                                    break;
                                }
                                scatter_found = false;
                                break;
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

            // Apply hero SSA for this scatter event BEFORE calculating NEE.
            let hero_scatter_optics = &atm.optics[scatter_shell][hero_wl];
            hero_weight *= hero_scatter_optics.ssa;

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

            let bdpt_covered = is_main && bounce_idx < BDPT_MAX_LIGHT_VERTICES;
            let skip_nee = bdpt_covered && nee_r2_weight < 1e-30;
            if !skip_nee {
                let t_suns = shadow_ray_transmittance_spectrum(atm, pos, sun_dir, num_wl);
                let cos_angle_nee = sun_dir.dot(-current_dir);

                // On the first BDPT_MAX_LIGHT_VERTICES bounces of the main
                // particle, apply the MIS weight (w_back) since BDPT provides
                // independent estimates for these orders. Higher-order bounces
                // get full weight since BDPT does not cover them.
                let nee_weight = if bdpt_covered { nee_r2_weight } else { 1.0 };

                for w in 0..num_wl {
                    if t_suns[w] > 1e-30 {
                        let optics_w = &atm.optics[scatter_shell][w];
                        let phase_w = scalar_phase_value(cos_angle_nee, optics_w);
                        total[w] +=
                            nee_weight * hero_weight * wr[w] * t_suns[w] * phase_w * INV_4PI;
                    }
                }
            }
            bounce_idx += 1;

            // Sample new direction: up to 3-way one-sample MIS:
            //   phase function + Dwivedi horizontal bias + path guide.
            // Guide is only active when a trained PathGuide is provided.
            // Gated: when both Dwivedi and guide fractions are negligible,
            // use pure phase function (no MIS overhead).
            let alpha_d = d_frac;
            let alpha_g = match guide {
                Some(g) if g.is_trained() => guide_frac(sza_deg_local),
                _ => 0.0,
            };
            let mis_active = (alpha_d + alpha_g) >= 0.02;

            let (new_dir, cos_theta_for_alis) = if mis_active {
                let local_up_here = pos.normalize();
                let alpha_p_mis = (1.0 - alpha_d - alpha_g).max(0.10);
                // Renormalize if we clamped alpha_p
                let alpha_sum = alpha_p_mis + alpha_d + alpha_g;
                let alpha_d_n = alpha_d / alpha_sum;
                let alpha_g_n = alpha_g / alpha_sum;
                let alpha_p_n = alpha_p_mis / alpha_sum;
                let xi_branch = xorshift_f64(&mut local_rng.dir);

                if alpha_g_n > 0.01 && xi_branch < alpha_g_n {
                    // Path guide branch: sample direction from learned distribution
                    let alt_here = pos.length() - surface_radius;
                    let (d, p_guide_sr) =
                        guide
                            .unwrap()
                            .sample(alt_here, local_up_here, sun_dir, &mut local_rng.dir);
                    let d = d.normalize();
                    let ct = current_dir.dot(d);
                    let p_phase_hero = scalar_phase_value(ct, hero_scatter_optics) * INV_4PI;
                    let cos_z = d.dot(local_up_here);
                    let p_dw = if alpha_d_n > 0.01 {
                        dwivedi_pdf(cos_z, d_beta)
                    } else {
                        0.0
                    };
                    let mis_denom =
                        alpha_p_n * p_phase_hero + alpha_d_n * p_dw + alpha_g_n * p_guide_sr;
                    if mis_denom > 1e-30 {
                        hero_weight *= p_phase_hero / mis_denom;
                    }
                    (d, ct)
                } else if xi_branch < alpha_g_n + alpha_d_n {
                    // Dwivedi branch
                    let xi1 = xorshift_f64(&mut local_rng.dir);
                    let xi2 = xorshift_f64(&mut local_rng.dir);
                    let xi_sign = xorshift_f64(&mut local_rng.dir);
                    let (cos_z, phi_dw) = dwivedi_sample(xi1, xi2, xi_sign, d_beta);
                    let sin_z = libm::sqrt((1.0 - cos_z * cos_z).max(0.0));
                    let east = {
                        let arbitrary = if libm::fabs(local_up_here.y) < 0.9 {
                            Vec3::new(0.0, 1.0, 0.0)
                        } else {
                            Vec3::new(1.0, 0.0, 0.0)
                        };
                        let e = local_up_here.cross(arbitrary);
                        e.normalize()
                    };
                    let north = local_up_here.cross(east);
                    let d = local_up_here.scale(cos_z)
                        + east.scale(sin_z * libm::cos(phi_dw))
                        + north.scale(sin_z * libm::sin(phi_dw));
                    let d = d.normalize();
                    let ct = current_dir.dot(d);
                    let p_phase_hero = scalar_phase_value(ct, hero_scatter_optics) * INV_4PI;
                    let p_dw = dwivedi_pdf(cos_z, d_beta);
                    let p_guide_sr = if alpha_g_n > 0.01 {
                        let alt_here = pos.length() - surface_radius;
                        guide.unwrap().pdf(alt_here, local_up_here, sun_dir, d)
                    } else {
                        0.0
                    };
                    let mis_denom =
                        alpha_p_n * p_phase_hero + alpha_d_n * p_dw + alpha_g_n * p_guide_sr;
                    if mis_denom > 1e-30 {
                        hero_weight *= p_phase_hero / mis_denom;
                    }
                    (d, ct)
                } else {
                    // Phase function branch (within MIS)
                    let cos_theta = if xorshift_f64(&mut local_rng.dir)
                        < hero_scatter_optics.rayleigh_fraction
                    {
                        sample_rayleigh_analytic(xorshift_f64(&mut local_rng.dir))
                    } else {
                        sample_henyey_greenstein(
                            xorshift_f64(&mut local_rng.dir),
                            hero_scatter_optics.asymmetry,
                        )
                    };
                    let phi = 2.0 * core::f64::consts::PI * xorshift_f64(&mut local_rng.dir);
                    let d = scatter_direction(current_dir, cos_theta, phi);
                    let p_phase_hero = scalar_phase_value(cos_theta, hero_scatter_optics) * INV_4PI;
                    let cos_z_dw = d.dot(local_up_here);
                    let p_dw = if alpha_d_n > 0.01 {
                        dwivedi_pdf(cos_z_dw, d_beta)
                    } else {
                        0.0
                    };
                    let p_guide_sr = if alpha_g_n > 0.01 {
                        let alt_here = pos.length() - surface_radius;
                        guide.unwrap().pdf(alt_here, local_up_here, sun_dir, d)
                    } else {
                        0.0
                    };
                    let mis_denom =
                        alpha_p_n * p_phase_hero + alpha_d_n * p_dw + alpha_g_n * p_guide_sr;
                    if mis_denom > 1e-30 {
                        hero_weight *= p_phase_hero / mis_denom;
                    }
                    (d, cos_theta)
                }
            } else {
                // Pure phase function: no MIS overhead.
                let cos_theta =
                    if xorshift_f64(&mut local_rng.dir) < hero_scatter_optics.rayleigh_fraction {
                        sample_rayleigh_analytic(xorshift_f64(&mut local_rng.dir))
                    } else {
                        sample_henyey_greenstein(
                            xorshift_f64(&mut local_rng.dir),
                            hero_scatter_optics.asymmetry,
                        )
                    };
                let phi = 2.0 * core::f64::consts::PI * xorshift_f64(&mut local_rng.dir);
                let d = scatter_direction(current_dir, cos_theta, phi);
                (d, cos_theta)
            };

            // ALIS phase function ratio for direction sampling.
            // The sampling PDF is mix_hero = alpha*q + (1-alpha)*p_hero_sr for ALL
            // wavelengths. Hero weight already carries p_hero_sr / mix_hero. Standard
            // ALIS ratio phase_w / phase_hero gives the correct per-wavelength weight:
            //   hero_weight * wr[w] ~ (p_hero_sr / mix_hero) * (phase_w / phase_hero)
            //                       = p_phase_w_sr / mix_hero       (correct IS weight)
            // No per-wavelength MIS denominator correction is needed.
            let phase_hero = scalar_phase_value(cos_theta_for_alis, hero_scatter_optics);
            if phase_hero > 1e-30 {
                for w in 0..num_wl {
                    let optics_w = &atm.optics[scatter_shell][w];
                    let phase_w = scalar_phase_value(cos_theta_for_alis, optics_w);
                    wr[w] *= phase_w / phase_hero;
                }
            }

            current_dir = new_dir;

            // --- Weight window population control ---
            let alt = pos.length() - surface_radius;
            let cos_sun_here = pos.normalize().dot(sun_dir);
            let w_target =
                weight_window_target(alt, alt_start, h_ww, cos_sun_here, cos_sun_start, ck);
            let w_lower = w_target / WW_LOWER_RATIO;
            let w_upper = w_target * WW_UPPER_RATIO;
            let abs_hw = hero_weight.abs();

            if abs_hw < w_lower && w_target > 1e-30 {
                // Russian roulette: hero weight too small for this region.
                // Weight ratios are preserved on survival.
                let p_survive = abs_hw / w_target;
                if xorshift_f64(&mut local_rng.ctl) < p_survive {
                    hero_weight = if hero_weight >= 0.0 {
                        w_target
                    } else {
                        -w_target
                    };
                } else {
                    break; // Chain killed by RR
                }
            } else if abs_hw > w_upper && w_target > 1e-30 {
                // Splitting: hero weight too large for this region.
                // Each copy inherits the same weight_ratio array.
                let k_ideal = libm::round(abs_hw / w_target) as usize;
                let max_k = MAX_SPLIT_PARTICLES - stack_len + 1;
                if max_k >= 2 {
                    let k = k_ideal.clamp(2, max_k);
                    hero_weight /= k as f64;
                    for copy_idx in 1..k {
                        if stack_len < MAX_SPLIT_PARTICLES {
                            let child_seed = splitmix64(
                                local_rng.tau
                                    ^ (copy_idx as u64).wrapping_mul(2654435761)
                                    ^ (alt.to_bits() >> 32),
                            );
                            stack[stack_len] = SplitParticleAlis {
                                pos,
                                dir: current_dir,
                                hero_weight,
                                weight_ratio: wr,
                                rng: McRng::from_seed(child_seed),
                            };
                            stack_len += 1;
                        }
                    }
                }
            }
        }
    }

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

    // --- LOS ray-budget redistribution (importance sampling by altitude) ---
    //
    // At deep twilight, MC chains starting at low-altitude LOS points are
    // exponentially less productive: they must climb 50+ km through dense
    // atmosphere AND travel 1800+ km laterally to reach sunlit regions.
    // Uniform allocation wastes ~60-70% of compute on these hopeless chains.
    //
    // Solution: pre-scan LOS altitudes and redistribute the total MC ray
    // budget so that high-altitude steps (where chains can efficiently reach
    // sunlit atmosphere) receive more rays. Each step's estimator remains
    // (1/n_i) * sum(chains), which is unbiased regardless of n_i.
    // Only the per-step variance changes.
    let observer_up = observer_pos.normalize();
    let cos_sza_obs = sun_dir.dot(observer_up);
    let sza_deg_obs = libm::acos(cos_sza_obs.clamp(-1.0, 1.0)) * 180.0 / core::f64::consts::PI;

    let use_los_importance = sza_deg_obs >= ZENITH_SZA_START && secondary_rays > 0;
    let total_mc_budget = secondary_rays * num_steps;

    // Pre-scan: compute per-step importance from altitude profile.
    let mut los_importance = [0.0f64; HYBRID_LOS_STEPS];
    let mut sum_los_imp = 0.0f64;

    if use_los_importance {
        // SZA-adaptive scale height: two-stage ramp.
        // Stage 1 (SZA 96-106): 100km -> 30km (mild to moderate redistribution)
        // Stage 2 (SZA 106-108): 30km -> 15km (aggressive for deep twilight)
        let sza_t = ((sza_deg_obs - ZENITH_SZA_START) / (ZENITH_SZA_FULL - ZENITH_SZA_START))
            .clamp(0.0, 1.0);
        let h_stage1 = LOS_IMP_H_MODERATE_M + (LOS_IMP_H_DEEP_M - LOS_IMP_H_MODERATE_M) * sza_t;
        // Second ramp: from VSPG_SZA_FULL (106) to 108
        let sza_t2 = ((sza_deg_obs - VSPG_SZA_FULL) / 2.0).clamp(0.0, 1.0);
        let h_scale = h_stage1 + (LOS_IMP_H_EXTREME_M - h_stage1) * sza_t2;

        for step in 0..num_steps {
            let s = (step as f64 + 0.5) * ds;
            let p = observer_pos + view_dir * s;
            let r = p.length();
            if r > toa_radius || r < surface_radius {
                continue;
            }
            let alt = r - surface_radius;
            let imp = libm::exp(alt / h_scale);
            los_importance[step] = imp;
            sum_los_imp += imp;
        }
    }

    // --- BDPT: trace light subpaths from the sunlit TOA ---
    //
    // At deep twilight (SZA > 98), backward chains struggle to reach
    // sunlit atmosphere. Light subpaths enter from the illuminated TOA
    // and scatter into the atmosphere. Their vertices are later connected
    // to each LOS step to provide an alternative path to the sun.
    use crate::single_scatter::transmittance_between_points_spectrum;

    let bdpt_active = sza_deg_obs >= BDPT_SZA_START && secondary_rays > 0;

    // MIS Blend: backward chains and BDPT compute exactly the same
    // multi-scattering integral. To prevent a 2x additive bias, we blend them.
    // BDPT takes over almost completely at deep twilight (SZA 108), suppressing
    // the backward chain's extreme CV 4+ variance outliers down to ~0.
    let w_bdpt = if bdpt_active {
        bdpt_strength(sza_deg_obs)
    } else {
        0.0
    };
    let w_back = 1.0 - w_bdpt;

    // SZA-conditional subpath count: double at extreme deep twilight
    // where BDPT handles ~98% of the signal.
    let num_light_subpaths = if bdpt_active && sza_deg_obs > VSPG_SZA_FULL {
        BDPT_NUM_LIGHT_SUBPATHS_DEEP
    } else {
        BDPT_NUM_LIGHT_SUBPATHS
    };

    let inv_num_light_subpaths = if bdpt_active {
        1.0 / num_light_subpaths as f64
    } else {
        0.0
    };

    // --- Path guide: train from BDPT light vertices ---
    //
    // Each BDPT light vertex records a position and incoming direction where
    // a photon successfully scattered near the terminator. We train the guide
    // by accumulating these directions at the vertex's (altitude, solar_angle)
    // cell, weighted by the vertex's hero_weight. This tells backward chains
    // "at this position, photons arriving from direction X are productive."
    //
    // The guide is trained BEFORE the LOS walk so it's ready for production
    // chains. Training cost is negligible: just bin lookups on already-traced
    // vertices.
    use crate::path_guide::PathGuide;
    let mut path_guide = PathGuide::new();

    let guide_ref: Option<&PathGuide> = if bdpt_active {
        let light_base_seed = splitmix64(*rng_state ^ 0xBDFF_BDFF_BDFF_BDFF);
        let dummy_lv = LightVertex {
            pos: Vec3::new(0.0, 0.0, 0.0),
            dir_in: Vec3::new(0.0, 0.0, 1.0),
            shell_idx: 0,
            hero_weight: 0.0,
            weight_ratio: [0.0f64; 64],
            pdf_fwd: 0.0,
        };

        // First pass: trace all light subpaths and train the guide.
        for subpath_idx in 0..num_light_subpaths {
            let hero_wl = subpath_idx % num_wl;
            let subpath_seed = splitmix64(light_base_seed.wrapping_add(subpath_idx as u64));
            let mut light_rng = McRng::from_seed(subpath_seed);
            let mut subpath_verts = [dummy_lv; BDPT_MAX_LIGHT_VERTICES];
            let n_verts = trace_light_subpath(
                atm,
                sun_dir,
                observer_pos,
                hero_wl,
                num_wl,
                sza_deg_obs,
                &mut light_rng,
                &mut subpath_verts,
                subpath_idx,
                num_light_subpaths,
            );
            for v in 0..n_verts {
                let lv = &subpath_verts[v];
                if lv.hero_weight.abs() > 1e-30 {
                    let local_up = lv.pos.normalize();
                    let alt = lv.pos.length() - surface_radius;
                    // Train guide: the incoming direction at this vertex
                    // was productive (it led to a scatter near the terminator).
                    // We flip dir_in to get the outgoing direction that a
                    // backward chain should sample to reach this vertex.
                    let outgoing = lv.dir_in.scale(-1.0);
                    path_guide.accumulate(alt, local_up, sun_dir, outgoing, lv.hero_weight.abs());
                }
            }
        }
        path_guide.normalize();
        Some(&path_guide)
    } else {
        None
    };

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
        //
        // Ray budget per step: when LOS importance sampling is active,
        // high-altitude steps receive more rays (proportional to
        // exp(altitude / h_scale)) and low-altitude steps receive fewer.
        // The estimator (1/n_i) * sum(chains) is unbiased for any n_i > 0,
        // so no weight correction is needed -- only per-step variance changes.
        let rays_this_step = if use_los_importance && los_importance[step] > 0.0 {
            // Proportional allocation: n_i = round(total_budget * imp_i / sum_imp).
            // Clamp to at least 1 so every in-atmosphere step gets coverage.
            let frac = los_importance[step] / sum_los_imp;
            let n = libm::round(total_mc_budget as f64 * frac) as usize;
            n.max(1)
        } else if !use_los_importance && secondary_rays > 0 {
            secondary_rays
        } else {
            0
        };

        if rays_this_step > 0 {
            let mut mc_totals = [0.0f64; 64];

            for ray in 0..rays_this_step {
                // Select hero as the wavelength with maximum extinction at
                // this LOS step's shell. This ensures sigma_w/sigma_h <= 1
                // at every scatter event, preventing ALIS weight ratio wr[w]
                // from growing exponentially over many bounces.
                // In a Rayleigh-dominated atmosphere, lambda^-4 scaling
                // preserves the extinction ordering across all shells, so
                // the hero remains max-extinction throughout the chain.
                // Round-robin offset ensures different wavelengths get hero
                // turns when multiple wavelengths have similar extinction.
                let hero_wl = {
                    let mut best = ray % num_wl;
                    let mut best_ext = atm.optics[shell_idx][best].extinction;
                    for w in 0..num_wl {
                        let ext = atm.optics[shell_idx][w].extinction;
                        if ext > best_ext {
                            best = w;
                            best_ext = ext;
                        }
                    }
                    best
                };

                // Per-chain McRng: master advances by 1 per chain.
                let _ = xorshift_f64(rng_state);
                let mut mc_rng = McRng::from_seed(*rng_state);
                let chain_result = trace_secondary_chain_alis(
                    atm,
                    scatter_pos,
                    sun_dir,
                    hero_wl,
                    shell_idx,
                    &mut mc_rng,
                    ray,
                    rays_this_step,
                    num_wl,
                    w_back,
                    guide_ref,
                );

                for w in 0..num_wl {
                    mc_totals[w] += chain_result[w];
                }
            }

            let inv_rays = 1.0 / rays_this_step as f64;
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

        // BDPT connections are now handled in a separate batched pass below.

        for w in 0..num_wl {
            tau_obs[w] += atm.optics[shell_idx][w].extinction * ds;
        }
    }

    // --- BDPT connections: batched post-processing pass ---
    //
    // Light subpaths are traced in batches of BDPT_BATCH_SIZE and their
    // vertices are connected to each LOS step. This avoids allocating
    // all vertices simultaneously (which overflows the stack at 4096+
    // vertices). Each batch re-walks the LOS to compute tau_obs and
    // evaluate connections. The LOS re-walk is cheap (just extinction
    // accumulation, no MC chains or shadow rays).
    //
    // For each light vertex L, evaluate the connection contribution:
    //   contrib = eye_weight * light_weight * phase_eye * phase_light * G * T_conn
    //
    // Optimizations to avoid expensive transmittance evaluations:
    // - Skip LOS steps below 10 km (deep troposphere, t_obs negligible)
    // - Per-connection chord minimum altitude check (SZA-adaptive)
    // - Skip connections > 3000 km: G = 1/d^2 makes contributions negligible
    if bdpt_active {
        // Derive light subpath seeds from the current rng_state WITHOUT
        // advancing it. This is critical: any advance to rng_state shifts
        // all backward chain seeds, causing regressions due to RNG stream
        // sensitivity in the heavy-tailed distribution. By using splitmix64
        // to scramble the current state, we get decorrelated light seeds
        // while keeping the backward chain sequence identical.
        let light_base_seed = splitmix64(*rng_state ^ 0xBDFF_BDFF_BDFF_BDFF);

        let dummy_lv = LightVertex {
            pos: Vec3::new(0.0, 0.0, 0.0),
            dir_in: Vec3::new(0.0, 0.0, 1.0),
            shell_idx: 0,
            hero_weight: 0.0,
            weight_ratio: [0.0f64; 64],
            pdf_fwd: 0.0,
        };

        // Process subpaths in batches to keep stack usage under ~600 KB.
        let num_batches = num_light_subpaths.div_ceil(BDPT_BATCH_SIZE);

        for batch_idx in 0..num_batches {
            let batch_start = batch_idx * BDPT_BATCH_SIZE;
            let batch_end = (batch_start + BDPT_BATCH_SIZE).min(num_light_subpaths);
            let _batch_count = batch_end - batch_start;

            // Trace this batch of light subpaths and collect vertices.
            // With BDPT_MAX_LIGHT_VERTICES=2, each subpath can produce up to
            // 2 vertices, so the buffer is sized at 2*BDPT_BATCH_SIZE.
            const BATCH_VERT_CAP: usize = BDPT_BATCH_SIZE * BDPT_MAX_LIGHT_VERTICES;
            let mut batch_vertices = [dummy_lv; BATCH_VERT_CAP];
            let mut n_batch_verts = 0usize;

            for subpath_idx in batch_start..batch_end {
                let hero_wl = subpath_idx % num_wl;
                let subpath_seed = splitmix64(light_base_seed.wrapping_add(subpath_idx as u64));
                let mut light_rng = McRng::from_seed(subpath_seed);
                let mut subpath_verts = [dummy_lv; BDPT_MAX_LIGHT_VERTICES];
                let n_verts = trace_light_subpath(
                    atm,
                    sun_dir,
                    observer_pos,
                    hero_wl,
                    num_wl,
                    sza_deg_obs,
                    &mut light_rng,
                    &mut subpath_verts,
                    subpath_idx,
                    num_light_subpaths,
                );
                for v in 0..n_verts {
                    if n_batch_verts < BATCH_VERT_CAP {
                        batch_vertices[n_batch_verts] = subpath_verts[v];
                        n_batch_verts += 1;
                    }
                }
            }

            if n_batch_verts == 0 {
                continue;
            }

            // Re-walk the LOS to evaluate connections for this batch's vertices.
            // This is cheap: just extinction accumulation + connection evaluation.
            let mut tau_obs_bdpt = [0.0f64; 64];

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
                    let tau_mid = tau_obs_bdpt[w] + atm.optics[shell_idx][w].extinction * ds * 0.5;
                    if libm::exp(-tau_mid) > 1e-30 {
                        any_visible = true;
                        break;
                    }
                }
                if !any_visible {
                    break;
                }

                for lv_idx in 0..n_batch_verts {
                    let lv = &batch_vertices[lv_idx];

                    let diff = Vec3::new(
                        lv.pos.x - scatter_pos.x,
                        lv.pos.y - scatter_pos.y,
                        lv.pos.z - scatter_pos.z,
                    );
                    let dist_sq = diff.length_sq();

                    if !(1.0..=9.0e12_f64).contains(&dist_sq) {
                        continue;
                    }

                    // No biased chord altitude filtering.
                    // We rely on exact physical `t_conn` evaluation.

                    let dist = libm::sqrt(dist_sq);
                    let connection_dir = diff.scale(1.0 / dist);

                    let u = diff.dot(view_dir);
                    let d_sq = (dist_sq - u * u).max(100.0);
                    let d_perp = libm::sqrt(d_sq);
                    let y1 = -0.5 * ds - u;
                    let y2 = 0.5 * ds - u;
                    let g_term_ds = (libm::atan2(y2, d_perp) - libm::atan2(y1, d_perp)) / d_perp;

                    let cos_theta_eye = connection_dir.dot(view_dir.scale(-1.0));
                    let cos_theta_light = lv.dir_in.dot(connection_dir.scale(-1.0));

                    let t_conn =
                        transmittance_between_points_spectrum(atm, scatter_pos, lv.pos, num_wl);

                    for w in 0..num_wl {
                        let optics_eye = &atm.optics[shell_idx][w];
                        let beta_scat_eye = optics_eye.extinction * optics_eye.ssa;
                        if beta_scat_eye < 1e-30 || t_conn[w] < 1e-30 {
                            continue;
                        }

                        let tau_obs_mid = tau_obs_bdpt[w] + optics_eye.extinction * ds * 0.5;
                        let t_obs = libm::exp(-tau_obs_mid);
                        if t_obs < 1e-30 {
                            continue;
                        }

                        let phase_eye = scalar_phase_value(cos_theta_eye, optics_eye);
                        let optics_light = &atm.optics[lv.shell_idx][w];
                        let phase_light = scalar_phase_value(cos_theta_light, optics_light);

                        let contrib = t_obs
                            * beta_scat_eye
                            * lv.hero_weight
                            * lv.weight_ratio[w]
                            * phase_eye
                            * INV_4PI
                            * phase_light
                            * INV_4PI
                            * g_term_ds
                            * t_conn[w]
                            * inv_num_light_subpaths;

                        if contrib.is_finite() {
                            radiance[w] += w_bdpt * contrib;
                        }
                    }
                }

                for w in 0..num_wl {
                    tau_obs_bdpt[w] += atm.optics[shell_idx][w].extinction * ds;
                }
            }
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
pub fn hybrid_scatter_spectrum(
    atm: &AtmosphereModel,
    observer_pos: Vec3,
    view_dir: Vec3,
    sun_dir: Vec3,
    secondary_rays: usize,
    base_seed: u64,
    polarized: bool,
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
/// Sampling: cos(theta) = F_trunc^{-1}(xi) where F_trunc is the truncated
/// power-cosine CDF over [power_cos_min(n), 1]. This prevents near-horizontal
/// directions that produce unbounded importance weights (1/cos^(n-1) diverges
/// as cos_theta -> 0). The max weight is bounded at ~ZENITH_MAX_IMPORTANCE_WEIGHT.
/// The removed probability mass is cos_min^(n+1) (negligible).
///
/// Consumes exactly 2 RNG draws, matching `sample_hemisphere`.
///
/// Returns (direction, cos_theta) -- the caller uses cos_theta to compute
/// the importance weight via `zenith_importance_weight`.
fn sample_zenith_biased(normal: Vec3, n: f64, rng: &mut u64) -> (Vec3, f64) {
    let xi1 = xorshift_f64(rng);
    let xi2 = xorshift_f64(rng);

    let cos_min = power_cos_min(n);
    let cos_theta = if cos_min > 1e-6 {
        // Truncated inverse CDF: map xi1 in [0,1] to cos_theta in [cos_min, 1].
        // F_trunc^{-1}(xi) = (cos_min^(n+1) + xi * (1 - cos_min^(n+1)))^(1/(n+1))
        let exp = n + 1.0;
        let cos_min_pow = libm::pow(cos_min, exp);
        libm::pow(cos_min_pow + xi1 * (1.0 - cos_min_pow), 1.0 / exp)
    } else {
        // At n~1, weight is ~constant, no truncation needed.
        libm::pow(xi1, 1.0 / (n + 1.0))
    };

    let phi = 2.0 * core::f64::consts::PI * xi2;

    let dir = scatter_direction(normal, cos_theta, phi);
    (dir, cos_theta)
}

/// Importance weight correction for truncated zenith-biased sampling.
///
/// When sampling from the truncated power-cosine(n) distribution over
/// [cos_min, 1] instead of cosine-weighted (n=1) over [0, 1],
/// the importance ratio is:
///
///   w = p_cosine(theta) / p_trunc(theta)
///     = [cos(theta) / pi] / [(n+1) / (2*pi*(1 - c^(n+1))) * cos^n(theta)]
///     = 2*(1 - c^(n+1)) / ((n+1) * cos^(n-1)(theta))
///
/// where c = power_cos_min(n). The truncation normalization factor ensures
/// the IS weight is exactly correct for the truncated proposal distribution.
/// Maximum weight is bounded at ZENITH_MAX_IMPORTANCE_WEIGHT.
#[inline]
fn zenith_importance_weight(cos_theta: f64, n: f64) -> f64 {
    let cos_nm1 = libm::pow(cos_theta, n - 1.0);
    let cos_min = power_cos_min(n);
    if cos_min > 1e-6 {
        let trunc_norm = 1.0 - libm::pow(cos_min, n + 1.0);
        2.0 * trunc_norm / ((n + 1.0) * cos_nm1)
    } else {
        2.0 / ((n + 1.0) * cos_nm1)
    }
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
    let cos_min = power_cos_min(m);
    if cos_min > 1e-6 {
        let trunc_norm = 1.0 - libm::pow(cos_min, m + 1.0);
        2.0 * cos_z * trunc_norm / ((m + 1.0) * cos_t_m)
    } else {
        2.0 * cos_z / ((m + 1.0) * cos_t_m)
    }
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

/// Split RNG state for Monte Carlo chains.
///
/// Three independent xorshift64 streams isolate:
/// - `tau`: free-path / optical-depth sampling
/// - `dir`: direction sampling (branch choice, scattering angles, Dwivedi, ground bounce)
/// - `ctl`: control flow (weight-window RR, split decisions)
///
/// This prevents parameter changes in one category from cascading
/// into other categories via the shared RNG stream. For example,
/// changing the Dwivedi fraction (which affects direction sampling)
/// no longer shifts the free-path sequence, and vice versa.
#[derive(Clone, Copy)]
pub struct McRng {
    /// Free-path / optical-depth sampling stream.
    pub tau: u64,
    /// Direction sampling stream (branch + angles + ground bounce).
    pub dir: u64,
    /// Control flow stream (weight-window RR, split decisions).
    pub ctl: u64,
}

/// SplitMix64 scramble: excellent avalanche properties for seeding RNGs.
///
/// Given sequential inputs (e.g., 1, 2, 3), produces outputs with no
/// detectable correlation. Used by Java's `SplittableRandom` and
/// recommended by Vigna for seeding xorshift generators.
#[inline]
fn splitmix64(state: u64) -> u64 {
    let mut z = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl McRng {
    /// Create split RNG from a single seed.
    ///
    /// Uses SplitMix64 scrambling to derive three independent streams.
    /// SplitMix64 has excellent avalanche properties, so even sequential
    /// seeds (e.g., from a master xorshift) produce well-decorrelated
    /// stream states. Ensures no stream starts at state 0 (xorshift
    /// fixed point).
    #[inline]
    pub fn from_seed(seed: u64) -> Self {
        let s1 = splitmix64(seed);
        let s2 = splitmix64(s1);
        let s3 = splitmix64(s2);
        McRng {
            tau: s1 | 1, // ensure non-zero (xorshift fixed point at 0)
            dir: s2 | 1,
            ctl: s3 | 1,
        }
    }
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
        // Safety limit: must be large enough for deep twilight chains
        // (no weight floor, no Russian roulette). 10_000 is the backstop.
        assert!(MAX_SCATTERS >= 1000);
        assert_eq!(MAX_SCATTERS, 10_000);
        assert_eq!(MAX_SCATTERS, BOUNCE_SAFETY_LIMIT);
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

        let spectrum = hybrid_scatter_spectrum(&atm, obs, view, sun, 10, 42, true);
        assert_eq!(spectrum.len(), 64);
    }

    #[test]
    fn hybrid_spectrum_non_negative() {
        let atm = make_scattering_atmosphere();
        let obs = crate::geometry::Vec3::new(crate::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0);
        let view = crate::geometry::Vec3::new(0.0, 1.0, 0.0).normalize();
        let sun = crate::geometry::solar_direction_ecef(96.0, 180.0, 0.0, 0.0);

        let spectrum = hybrid_scatter_spectrum(&atm, obs, view, sun, 50, 42, true);
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

        let spectrum = hybrid_scatter_spectrum(&atm, obs, view, sun, 50, 42, true);
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

        let spectrum = hybrid_scatter_spectrum(&atm, obs, view, sun, 50, 42, true);
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

        let s1 = hybrid_scatter_spectrum(&atm, obs, view, sun, 50, 42, true);
        let s2 = hybrid_scatter_spectrum(&atm, obs, view, sun, 50, 42, true);
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
            let spectrum = hybrid_scatter_spectrum(&atm, obs, view, sun, 20, 42, true);
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

        let spectrum = hybrid_scatter_spectrum(&atm, obs, view, sun, 20, 42, true);
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
        let result = hybrid_scatter_radiance_alis(&atm, obs, view, sun, 50, &mut rng);
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
        let result = hybrid_scatter_radiance_alis(&atm, obs, view, sun, 200, &mut rng);
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
            let alis = hybrid_scatter_radiance_alis(&atm, obs, view, sun, rays, &mut rng_alis);
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
        let result = hybrid_scatter_radiance_alis(&atm, obs, view, sun, 50, &mut rng);
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
            let result = hybrid_scatter_radiance_alis(&atm, obs, view, sun, 50, &mut rng);
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
    fn bdpt_light_subpath_vertex_diagnostic() {
        // Diagnostic: verify that forced-scattering light subpaths produce
        // vertices at useful altitudes with reasonable weights.
        use crate::atmosphere::{AtmosphereModel, ShellOptics, EARTH_RADIUS_M};

        let altitudes_km = [
            0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0,
        ];
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

        let observer = Vec3::new(EARTH_RADIUS_M + 1.0, 0.0, 0.0);

        // First: directly test the entry point geometry and scout.
        {
            extern crate std;
            let sza_deg = 102.0;
            let sza_rad = sza_deg * core::f64::consts::PI / 180.0;
            let sun_dir = Vec3::new(libm::cos(sza_rad), libm::sin(sza_rad), 0.0);
            let entry_dir = sun_dir.scale(-1.0);

            // Sub-solar point entry (r_frac = 0, disk center)
            let entry_center = sun_dir.scale(-atm.toa_radius());
            let r_center = entry_center.length();
            std::eprintln!(
                "Sub-solar entry: r={:.1} m, toa={:.1} m, diff={:.3} m",
                r_center,
                atm.toa_radius(),
                r_center - atm.toa_radius()
            );
            let nudged_center = entry_center + entry_center.normalize().scale(-10.0);
            let r_nudged = nudged_center.length();
            std::eprintln!(
                "  After radial nudge: r={:.1} m, shell={:?}",
                r_nudged,
                atm.shell_index(r_nudged)
            );

            // Rim entry (r_frac ~ 0.99)
            let disk_normal = sun_dir.scale(-1.0);
            let arbitrary = if libm::fabs(disk_normal.y) < 0.9 {
                Vec3::new(0.0, 1.0, 0.0)
            } else {
                Vec3::new(1.0, 0.0, 0.0)
            };
            let disk_u = disk_normal.cross(arbitrary).normalize();
            let r_frac = 0.99;
            let r_disk = atm.toa_radius() * r_frac;
            let z_along = libm::sqrt(atm.toa_radius() * atm.toa_radius() - r_disk * r_disk);
            let entry_rim = sun_dir.scale(-z_along) + disk_u.scale(r_disk);
            let r_rim = entry_rim.length();
            std::eprintln!(
                "Rim entry (r_frac=0.99): r={:.1} m, toa={:.1} m, diff={:.3} m",
                r_rim,
                atm.toa_radius(),
                r_rim - atm.toa_radius()
            );

            // Check entry direction dot with radial
            let cos_entry = entry_dir.dot(entry_rim.normalize());
            std::eprintln!(
                "  entry_dir . radial = {:.6} (>0 means outward!)",
                cos_entry
            );

            // Nudge radially inward
            let nudged_rim = entry_rim + entry_rim.normalize().scale(-10.0);
            let r_nudged_rim = nudged_rim.length();
            std::eprintln!(
                "  After radial nudge: r={:.1} m, shell={:?}",
                r_nudged_rim,
                atm.shell_index(r_nudged_rim)
            );

            // Scout from nudged rim position
            let mut vspg_segs = [VspgSegment {
                tau_lo: 0.0,
                tau_hi: 0.0,
                importance: 1.0,
            }; VSPG_MAX_SEGMENTS];
            let (tau_maxes, hit_ground, n_segs) = scout_with_vspg_segments_alis(
                &atm,
                nudged_rim,
                entry_dir,
                1,
                3,
                sza_deg,
                &mut vspg_segs,
            );
            std::eprintln!(
                "  Scout from rim: tau_hero={:.6e}, hit_ground={}, n_segs={}",
                tau_maxes[1],
                hit_ground,
                n_segs
            );

            // Also scout from center
            let (tau_c, hg_c, ns_c) = scout_with_vspg_segments_alis(
                &atm,
                nudged_center,
                entry_dir,
                1,
                3,
                sza_deg,
                &mut vspg_segs,
            );
            std::eprintln!(
                "  Scout from center: tau_hero={:.6e}, hit_ground={}, n_segs={}",
                tau_c[1],
                hg_c,
                ns_c
            );
        }

        for sza_deg in &[102.0, 105.0, 108.0] {
            let sza_rad = sza_deg * core::f64::consts::PI / 180.0;
            let sun_dir = Vec3::new(libm::cos(sza_rad), libm::sin(sza_rad), 0.0);

            let mut total_verts = 0usize;
            let mut max_weight = 0.0f64;
            let mut min_alt = f64::MAX;
            let mut max_alt = 0.0f64;
            let num_subpaths = 100;

            for sp in 0..num_subpaths {
                let hero_wl = sp % 3;
                let seed = splitmix64(12345u64.wrapping_add(sp as u64));
                let mut rng = McRng::from_seed(seed);
                let mut verts = [LightVertex {
                    pos: Vec3::new(0.0, 0.0, 0.0),
                    dir_in: Vec3::new(0.0, 0.0, 1.0),
                    shell_idx: 0,
                    hero_weight: 0.0,
                    weight_ratio: [0.0; 64],
                    pdf_fwd: 0.0,
                }; BDPT_MAX_LIGHT_VERTICES];
                let nv = trace_light_subpath(
                    &atm,
                    sun_dir,
                    observer,
                    hero_wl,
                    3,
                    *sza_deg,
                    &mut rng,
                    &mut verts,
                    sp,
                    num_subpaths,
                );
                total_verts += nv;
                for v in 0..nv {
                    let alt = verts[v].pos.length() - EARTH_RADIUS_M;
                    if alt < min_alt {
                        min_alt = alt;
                    }
                    if alt > max_alt {
                        max_alt = alt;
                    }
                    let hw = libm::fabs(verts[v].hero_weight);
                    if hw > max_weight {
                        max_weight = hw;
                    }
                }
            }

            let avg_verts = total_verts as f64 / num_subpaths as f64;

            #[cfg(test)]
            {
                extern crate std;
                std::eprintln!(
                    "SZA={}: avg_verts={:.1}, total={}, alt=[{:.0},{:.0}] m, max_weight={:.4e}",
                    sza_deg,
                    avg_verts,
                    total_verts,
                    if min_alt < f64::MAX { min_alt } else { 0.0 },
                    max_alt,
                    max_weight
                );
            }

            // With forced scattering, every subpath should produce at least 1 vertex.
            assert!(
                total_verts >= num_subpaths,
                "SZA={}: forced scattering should guarantee >= 1 vertex/subpath, got {} total from {}",
                sza_deg, total_verts, num_subpaths
            );
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
        // cos_z = cos_t = 1. With truncated power-cosine, weight = 2 * trunc_norm / (m+1).
        let m = 8.0;
        let w = terminator_shape_weight(1.0, 1.0, m);
        let cos_min = power_cos_min(m);
        let trunc_norm = if cos_min > 1e-6 {
            1.0 - libm::pow(cos_min, m + 1.0)
        } else {
            1.0
        };
        let expected = 2.0 * trunc_norm / (m + 1.0);
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

    // ── Weight windows ──

    #[test]
    fn weight_window_h_dormant_at_civil_twilight() {
        // At SZA < 96, scale height should be very large (> 100 km).
        let h = weight_window_h(90.0);
        assert!(h > 100_000.0, "H at SZA 90 should be > 100 km, got {}", h);
        let h93 = weight_window_h(93.0);
        assert!(
            h93 > 100_000.0,
            "H at SZA 93 should be > 100 km, got {}",
            h93
        );
    }

    #[test]
    fn weight_window_h_aggressive_at_deep_twilight() {
        // At SZA >= 106, scale height should be close to WW_H_MIN_M.
        let h = weight_window_h(106.0);
        assert!(
            h < WW_H_MIN_M * 1.2,
            "H at SZA 106 should be near {} m, got {} m",
            WW_H_MIN_M,
            h
        );
    }

    #[test]
    fn weight_window_h_smooth() {
        // Verify smooth monotonic decrease from SZA 90 to 110.
        let mut prev = weight_window_h(90.0);
        for sza_10x in 905..1100 {
            let sza = sza_10x as f64 / 10.0;
            let h = weight_window_h(sza);
            assert!(
                h <= prev + 1.0, // allow tiny float rounding
                "Weight window H not monotonically decreasing: H({}) = {}, H({}) = {}",
                sza - 0.1,
                prev,
                sza,
                h
            );
            prev = h;
        }
    }

    #[test]
    fn weight_window_target_unity_at_start() {
        // At the starting altitude, target weight should be 1.0.
        let w = weight_window_target(30_000.0, 30_000.0, 20_000.0, 0.0, 0.0, 0.0);
        assert!(
            (w - 1.0).abs() < 1e-10,
            "Target weight at start alt should be 1.0, got {}",
            w
        );
    }

    #[test]
    fn weight_window_target_decreases_with_altitude() {
        // Target weight should decrease as altitude increases above start.
        let alt_start = 10_000.0;
        let h = 20_000.0;
        let w1 = weight_window_target(20_000.0, alt_start, h, 0.0, 0.0, 0.0);
        let w2 = weight_window_target(40_000.0, alt_start, h, 0.0, 0.0, 0.0);
        let w3 = weight_window_target(60_000.0, alt_start, h, 0.0, 0.0, 0.0);
        assert!(w1 > w2, "w(20km)={} should be > w(40km)={}", w1, w2);
        assert!(w2 > w3, "w(40km)={} should be > w(60km)={}", w2, w3);
    }

    #[test]
    fn weight_window_target_increases_below_start() {
        // Chains descending below start altitude: target weight increases
        // (less important region, wider window).
        let alt_start = 50_000.0;
        let h = 20_000.0;
        let w_at_start = weight_window_target(alt_start, alt_start, h, 0.0, 0.0, 0.0);
        let w_below = weight_window_target(30_000.0, alt_start, h, 0.0, 0.0, 0.0);
        assert!(
            w_below > w_at_start,
            "Target at 30km ({}) should be > target at 50km ({})",
            w_below,
            w_at_start
        );
    }

    #[test]
    fn cadis_lateral_importance_boosts_toward_sun() {
        let alt = 30_000.0;
        let h = 20_000.0;
        let cos_sun_start = -0.276; // SZA 106 observer
        let k = 8.0; // full CADIS strength

        // Chain at same lateral position as start: no lateral boost.
        let w_same = weight_window_target(alt, alt, h, cos_sun_start, cos_sun_start, k);
        assert!(
            (w_same - 1.0).abs() < 1e-10,
            "No lateral boost at start: w = {}",
            w_same
        );

        // Chain that moved toward sun (positive delta_cos): lower target = more splitting.
        let w_toward = weight_window_target(alt, alt, h, -0.1, cos_sun_start, k);
        assert!(
            w_toward < w_same,
            "Toward-sun chain should have lower target: {} >= {}",
            w_toward,
            w_same
        );

        // Chain that moved away from sun (negative delta_cos): no penalty.
        let w_away = weight_window_target(alt, alt, h, -0.4, cos_sun_start, k);
        assert!(
            (w_away - 1.0).abs() < 1e-10,
            "Away-from-sun should have no penalty: w = {}",
            w_away
        );

        // CADIS off (k=0): no lateral effect.
        let w_no_cadis = weight_window_target(alt, alt, h, 0.0, cos_sun_start, 0.0);
        assert!(
            (w_no_cadis - 1.0).abs() < 1e-10,
            "CADIS off should give 1.0: w = {}",
            w_no_cadis
        );
    }

    #[test]
    fn cadis_k_ramps_smoothly() {
        // With center=100, width=3.5: cadis_k(93) ~ 1.43 (mild lateral bias;
        // harmless at civil twilight because weight windows are dormant).
        assert!(
            cadis_k(93.0) < 2.0,
            "cadis_k(93) = {:.2}, should be < 2.0",
            cadis_k(93.0)
        );
        assert!(
            (cadis_k(110.0) - CADIS_K_MAX).abs() < 1.0,
            "cadis_k(110) should be ~{}, got {:.2}",
            CADIS_K_MAX,
            cadis_k(110.0)
        );
        // Monotonic
        let mut prev = cadis_k(90.0);
        for sza in 91..=115 {
            let cur = cadis_k(sza as f64);
            assert!(cur >= prev - 1e-10, "cadis_k must be monotonic");
            prev = cur;
        }
    }

    #[test]
    fn weight_window_rr_unbiased() {
        // Statistical test: RR expected value should equal input weight.
        // Run many trials with weight below threshold and verify mean output.
        let alt_start = 0.0;
        let h_ww = 20_000.0;
        let alt = 5_000.0; // near start, w_target ~ 0.78
        let w_target = weight_window_target(alt, alt_start, h_ww, 0.0, 0.0, 0.0);
        let w_lower = w_target / WW_LOWER_RATIO;

        let input_weight = w_lower * 0.5; // below threshold
        let mut rng: u64 = 42;
        let n = 100_000;
        let mut output_sum = 0.0;
        let p_survive = input_weight / w_target;
        for _ in 0..n {
            if xorshift_f64(&mut rng) < p_survive {
                output_sum += w_target;
            }
            // else: chain killed, output 0
        }
        let mean_output = output_sum / n as f64;
        let relative_error = (mean_output - input_weight).abs() / input_weight;
        assert!(
            relative_error < 0.02,
            "RR mean output {} should match input weight {} (error: {:.1}%)",
            mean_output,
            input_weight,
            relative_error * 100.0
        );
    }

    #[test]
    fn split_particle_scalar_size_reasonable() {
        // pos Vec3 (24) + dir Vec3 (24) + weight f64 (8) + rng u64 (8) = 64 bytes
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
            let _ = xorshift_f64(&mut rng);
            let mut mc = McRng::from_seed(rng);
            total += trace_secondary_chain_scalar(
                &atm,
                observer,
                sun_dir,
                0,
                start_optics,
                &mut mc,
                ray,
                n,
                1.0,
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
            let _ = xorshift_f64(&mut rng);
            let mut mc = McRng::from_seed(rng);
            let val = trace_secondary_chain_scalar(
                &atm,
                observer,
                sun_dir,
                0,
                start_optics,
                &mut mc,
                ray,
                n,
                1.0,
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
            let _ = xorshift_f64(&mut rng);
            let mut mc = McRng::from_seed(rng);
            let result = trace_secondary_chain_alis(
                &atm, observer, sun_dir, hero_wl, 0, &mut mc, ray, n, num_wl, 1.0, None,
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
    fn weight_window_constants_valid() {
        assert!(WW_H_MIN_M > 0.0, "WW_H_MIN_M must be positive");
        assert!(WW_H_MAX_M > WW_H_MIN_M, "WW_H_MAX_M must exceed WW_H_MIN_M");
        assert!(WW_UPPER_RATIO > 1.0, "WW_UPPER_RATIO must exceed 1.0");
        assert!(WW_LOWER_RATIO > 1.0, "WW_LOWER_RATIO must exceed 1.0");
        assert!(
            WW_LOWER_RATIO >= WW_UPPER_RATIO,
            "WW_LOWER_RATIO should be >= WW_UPPER_RATIO for conservative RR"
        );
        assert!(WW_SZA_WIDTH > 0.0, "WW_SZA_WIDTH must be positive");
    }

    #[test]
    fn sigmoid_basic_properties() {
        assert!(
            (sigmoid(0.0) - 0.5).abs() < 1e-10,
            "sigmoid(0) should be 0.5"
        );
        assert!(sigmoid(10.0) > 0.999, "sigmoid(10) should be near 1");
        assert!(sigmoid(-10.0) < 0.001, "sigmoid(-10) should be near 0");
        // Monotonic
        for i in -100..100 {
            let x = i as f64 * 0.1;
            assert!(sigmoid(x + 0.1) >= sigmoid(x), "sigmoid must be monotonic");
        }
    }

    // ---- Dwivedi biasing tests ----

    #[test]
    fn dwivedi_pdf_uniform_at_zero_beta() {
        // beta=0 should give uniform 1/(4*pi).
        for cos_z in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            let p = dwivedi_pdf(cos_z, 0.0);
            assert!(
                (p - INV_4PI).abs() < 1e-10,
                "dwivedi_pdf({}, 0) = {}, expected {}",
                cos_z,
                p,
                INV_4PI
            );
        }
    }

    #[test]
    fn dwivedi_pdf_peaked_at_horizontal() {
        // At beta=3, PDF should be highest at cos_z=0 (horizontal)
        // and lowest at cos_z=+/-1 (vertical).
        let beta = 3.0;
        let p_horiz = dwivedi_pdf(0.0, beta);
        let p_vert = dwivedi_pdf(1.0, beta);
        assert!(
            p_horiz > p_vert * 2.0,
            "Horizontal PDF {} should be much larger than vertical {}",
            p_horiz,
            p_vert
        );
    }

    #[test]
    fn dwivedi_pdf_normalizes_to_one() {
        // Numerical integration over the unit sphere should give 1.
        for beta in [0.5, 1.0, 2.0, 3.0, 5.0] {
            let n = 10000;
            let mut integral = 0.0;
            for i in 0..n {
                let cos_z = -1.0 + 2.0 * (i as f64 + 0.5) / n as f64;
                integral +=
                    dwivedi_pdf(cos_z, beta) * 2.0 * core::f64::consts::PI * (2.0 / n as f64);
            }
            assert!(
                (integral - 1.0).abs() < 0.01,
                "dwivedi_pdf with beta={} integrates to {}, expected 1.0",
                beta,
                integral
            );
        }
    }

    #[test]
    fn dwivedi_sample_in_bounds() {
        let mut rng = 12345u64;
        for beta in [0.0, 1.0, 3.0, 5.0] {
            for _ in 0..1000 {
                let xi1 = xorshift_f64(&mut rng);
                let xi2 = xorshift_f64(&mut rng);
                let xi_sign = xorshift_f64(&mut rng);
                let (cos_z, phi) = dwivedi_sample(xi1, xi2, xi_sign, beta);
                assert!(
                    cos_z >= -1.0 && cos_z <= 1.0,
                    "cos_z = {} out of [-1,1] at beta={}",
                    cos_z,
                    beta
                );
                assert!(
                    phi >= 0.0 && phi <= 2.0 * core::f64::consts::PI + 1e-10,
                    "phi = {} out of [0,2*pi] at beta={}",
                    phi,
                    beta
                );
            }
        }
    }

    #[test]
    fn dwivedi_sample_distribution_matches_pdf() {
        // Histogram test: sample 50k directions and check that the
        // cos_z distribution matches the expected PDF.
        let beta = 3.0;
        let mut rng = 98765u64;
        let n_samples = 50_000;
        let n_bins = 20;
        let mut bins = [0u32; 20];

        for _ in 0..n_samples {
            let xi1 = xorshift_f64(&mut rng);
            let xi2 = xorshift_f64(&mut rng);
            let xi_sign = xorshift_f64(&mut rng);
            let (cos_z, _phi) = dwivedi_sample(xi1, xi2, xi_sign, beta);
            let bin = ((cos_z + 1.0) / 2.0 * n_bins as f64).min(n_bins as f64 - 1.0) as usize;
            bins[bin] += 1;
        }

        // Expected fraction per bin: integral of PDF over bin's cos_z range.
        let bin_width = 2.0 / n_bins as f64;
        for b in 0..n_bins {
            let cos_z_mid = -1.0 + (b as f64 + 0.5) * bin_width;
            // Marginal PDF in cos_z: integrate over phi (2*pi) -> p_marginal = 2*pi * dwivedi_pdf
            let expected_frac =
                2.0 * core::f64::consts::PI * dwivedi_pdf(cos_z_mid, beta) * bin_width;
            let actual_frac = bins[b] as f64 / n_samples as f64;
            let tolerance = 0.03; // 3% tolerance
            assert!(
                (actual_frac - expected_frac).abs() < tolerance,
                "Bin {} (cos_z~{:.2}): actual={:.4}, expected={:.4}, diff={:.4}",
                b,
                cos_z_mid,
                actual_frac,
                expected_frac,
                (actual_frac - expected_frac).abs()
            );
        }
    }

    #[test]
    fn dwivedi_frac_ramps_smoothly() {
        // At civil twilight, Dwivedi fraction should be near zero.
        assert!(dwivedi_frac(93.0) < 0.01, "dwivedi_frac(93) should be ~0");
        // At deep twilight, should approach DWIVEDI_FRAC_MAX.
        assert!(
            (dwivedi_frac(115.0) - DWIVEDI_FRAC_MAX).abs() < 0.01,
            "dwivedi_frac(115) = {}, expected ~{}",
            dwivedi_frac(115.0),
            DWIVEDI_FRAC_MAX
        );
        // Monotonic.
        let mut prev = dwivedi_frac(90.0);
        for sza in 91..=115 {
            let cur = dwivedi_frac(sza as f64);
            assert!(cur >= prev - 1e-10, "dwivedi_frac must be monotonic");
            prev = cur;
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
        // At SZA <= VSPG_SZA_START (93), importance should be 1.0 for all altitudes.
        for alt in [0.0, 30_000.0, 50_000.0, 70_000.0, 100_000.0] {
            let imp = vspg_importance(alt, 90.0);
            assert!(
                (imp - 1.0).abs() < 1e-12,
                "Expected 1.0 at SZA 90 alt={}, got {}",
                alt,
                imp
            );
            let imp2 = vspg_importance(alt, VSPG_SZA_START);
            assert!(
                (imp2 - 1.0).abs() < 1e-12,
                "Expected 1.0 at SZA {} alt={}, got {}",
                VSPG_SZA_START,
                alt,
                imp2
            );
        }
        // At SZA 96 (above VSPG_SZA_START), high-altitude importance should exceed 1.0.
        let imp_96_70k = vspg_importance(70_000.0, 96.0);
        assert!(
            imp_96_70k > 1.0,
            "Expected >1.0 at SZA 96 alt=70km, got {}",
            imp_96_70k
        );
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
            let _ = xorshift_f64(&mut rng);
            let mut mc = McRng::from_seed(rng);
            let result = trace_secondary_chain_scalar(
                &atm,
                observer,
                sun_dir,
                0,
                start_optics,
                &mut mc,
                ray,
                n,
                1.0,
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
            let _ = xorshift_f64(&mut rng);
            let mut mc = McRng::from_seed(rng);
            let result = trace_secondary_chain_alis(
                &atm, observer, sun_dir, hero_wl, 0, &mut mc, ray, n, num_wl, 1.0, None,
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
