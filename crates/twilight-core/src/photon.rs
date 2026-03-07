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
    weight * transmittance * phase / (4.0 * core::f64::consts::PI)
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
    let factor = weight * transmittance / (4.0 * core::f64::consts::PI);
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
            let scale_1 = beta_scat / (4.0 * core::f64::consts::PI) * t_sun * t_obs * ds;

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

    // --- SZA-adaptive zenith parameters ---
    let cos_sza = sun_dir.dot(local_up);
    let (zenith_frac, n) = zenith_params_for_sza(cos_sza);

    // --- Stratified initial direction sampling ---
    let xi_jitter = xorshift_f64(rng_state);
    let xi_mix = (ray_idx as f64 + xi_jitter) / total_rays as f64;

    // Two-branch importance sampling with correct branch probability weights.
    //
    // The baseline estimator (no bias) is:
    //   E = 0.5 * E_phase + 0.5 * E_hemi
    //
    // We sample phase branch with probability (1 - zenith_frac) and zenith
    // branch with probability zenith_frac. To keep E unchanged, each branch
    // carries a branch probability weight of (baseline_prob / actual_prob):
    //   phase:  0.5 / (1.0 - zenith_frac)
    //   zenith: 0.5 / zenith_frac
    //
    // The zenith branch also carries a shape weight correcting cos^n -> cos^1.
    //
    // At SZA <= 96: n=1, zenith_frac=0.5. Both weights = 1.0 exactly.
    let (dir, cos_theta_init, initial_weight) = if xi_mix < (1.0 - zenith_frac) {
        // Phase function branch
        let cos_theta_init = if xorshift_f64(rng_state) < start_optics.rayleigh_fraction {
            sample_rayleigh_analytic(xorshift_f64(rng_state))
        } else {
            sample_henyey_greenstein(xorshift_f64(rng_state), start_optics.asymmetry)
        };
        let phi_init = 2.0 * core::f64::consts::PI * xorshift_f64(rng_state);
        let d = scatter_direction(sun_dir, cos_theta_init, phi_init);
        let branch_w = 0.5 / (1.0 - zenith_frac);
        (d, cos_theta_init, branch_w)
    } else {
        // Zenith-biased branch with shape + branch weight correction
        let (d, cos_z) = sample_zenith_biased(local_up, n, rng_state);
        let cos_theta_init = sun_dir.dot(d);
        let shape_w = zenith_importance_weight(cos_z, n);
        let branch_w = 0.5 / zenith_frac;
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

    for _bounce in 0..HYBRID_MAX_BOUNCES {
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
                    let (new_pos, new_dir) =
                        cross_boundary(pos, current_dir, dist, is_outward, shell_idx, atm);
                    pos = new_pos;
                    current_dir = new_dir;
                    continue;
                }
                None => break,
            }
        }

        let xi = xorshift_f64(rng_state);
        let free_path = -libm::log(1.0 - xi + 1e-30) / optics.extinction;

        match next_shell_boundary(pos, current_dir, shell.r_inner, shell.r_outer) {
            Some((boundary_dist, is_outward)) => {
                if free_path >= boundary_dist {
                    let (new_pos, new_dir) =
                        cross_boundary(pos, current_dir, boundary_dist, is_outward, shell_idx, atm);
                    pos = new_pos;
                    current_dir = new_dir;

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

        // Scatter event
        pos = pos + current_dir * free_path;

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

            let scale = weight * t_sun_secondary / (4.0 * core::f64::consts::PI);
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
) -> f64 {
    use crate::single_scatter::shadow_ray_transmittance;

    let local_up = start_pos.normalize();

    // --- SZA-adaptive zenith parameters ---
    let cos_sza = sun_dir.dot(local_up);
    let (zenith_frac, n) = zenith_params_for_sza(cos_sza);

    // --- Stratified initial direction sampling ---
    // Stratify across rays: the first (1-zenith_frac) fraction of rays use
    // phase function, the rest use zenith-biased hemisphere.
    let xi_jitter = xorshift_f64(rng_state);
    let xi_mix = (ray_idx as f64 + xi_jitter) / total_rays as f64;

    // Two-branch importance sampling with correct branch probability weights.
    // See trace_secondary_chain for derivation. At SZA <= 96: both weights = 1.0.
    let (dir, initial_weight) = if xi_mix < (1.0 - zenith_frac) {
        // Phase function branch (toward sun_dir -- effective at civil twilight)
        let _cos_theta_init = if xorshift_f64(rng_state) < start_optics.rayleigh_fraction {
            sample_rayleigh_analytic(xorshift_f64(rng_state))
        } else {
            sample_henyey_greenstein(xorshift_f64(rng_state), start_optics.asymmetry)
        };
        let phi_init = 2.0 * core::f64::consts::PI * xorshift_f64(rng_state);
        let branch_w = 0.5 / (1.0 - zenith_frac);
        (
            scatter_direction(sun_dir, _cos_theta_init, phi_init),
            branch_w,
        )
    } else {
        // Zenith-biased branch with shape + branch weight correction
        let (d, cos_z) = sample_zenith_biased(local_up, n, rng_state);
        let shape_w = zenith_importance_weight(cos_z, n);
        let branch_w = 0.5 / zenith_frac;
        (d, shape_w * branch_w)
    };

    let mut pos = start_pos;
    let mut current_dir = dir;
    let mut weight = start_optics.ssa * initial_weight;
    let mut total = 0.0_f64;

    for _bounce in 0..HYBRID_MAX_BOUNCES {
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
                    let (new_pos, new_dir) =
                        cross_boundary(pos, current_dir, dist, is_outward, shell_idx, atm);
                    pos = new_pos;
                    current_dir = new_dir;
                    continue;
                }
                None => break,
            }
        }

        let xi = xorshift_f64(rng_state);
        let free_path = -libm::log(1.0 - xi + 1e-30) / optics.extinction;

        match next_shell_boundary(pos, current_dir, shell.r_inner, shell.r_outer) {
            Some((boundary_dist, is_outward)) => {
                if free_path >= boundary_dist {
                    let (new_pos, new_dir) =
                        cross_boundary(pos, current_dir, boundary_dist, is_outward, shell_idx, atm);
                    pos = new_pos;
                    current_dir = new_dir;

                    // Ground reflection
                    if !is_outward && pos.length() <= atm.surface_radius() + 1.0 {
                        let albedo = atm.surface_albedo[wavelength_idx];
                        weight *= albedo;
                        if weight < 1e-30 {
                            break;
                        }
                        let normal = pos.normalize();
                        current_dir = sample_hemisphere(normal, rng_state);
                        continue;
                    }
                    continue;
                }
            }
            None => break,
        }

        // Scatter event
        pos = pos + current_dir * free_path;

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

            let scale = weight * t_sun_secondary / (4.0 * core::f64::consts::PI);
            total += phase * scale;
        }

        // Apply SSA
        weight *= optics.ssa;
        if weight < 1e-30 {
            break;
        }

        // Sample new direction (RNG consumption matches Stokes version)
        let cos_theta = if xorshift_f64(rng_state) < optics.rayleigh_fraction {
            sample_rayleigh_analytic(xorshift_f64(rng_state))
        } else {
            sample_henyey_greenstein(xorshift_f64(rng_state), optics.asymmetry)
        };
        let phi = 2.0 * core::f64::consts::PI * xorshift_f64(rng_state);
        let new_dir = scatter_direction(current_dir, cos_theta, phi);

        current_dir = new_dir;
    }

    total
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

/// Compute the SZA-adaptive zenith bias parameters.
///
/// Returns `(zenith_frac, n)` where:
/// - `zenith_frac` is the fraction of rays using zenith-biased sampling
///   (in [0.5, ZENITH_MAX_FRACTION])
/// - `n` is the power-cosine exponent (in [1.0, ZENITH_BIAS_N])
///
/// Both parameters vary linearly with SZA:
///
///   SZA <= 96:  (0.5, 1.0) -- standard 50/50 mix, cosine-weighted (baseline)
///   SZA >= 106: (0.95, 5.0) -- aggressive zenith bias
///   In between: linear interpolation
///
/// At the baseline point (sza_t=0, n=1, frac=0.5), branch weights evaluate
/// to exactly 1.0 for both branches -- no if/else bypass needed.
#[inline]
fn zenith_params_for_sza(cos_sza: f64) -> (f64, f64) {
    let sza_deg = libm::acos(cos_sza.clamp(-1.0, 1.0)) * 180.0 / core::f64::consts::PI;

    let sza_t =
        ((sza_deg - ZENITH_SZA_START) / (ZENITH_SZA_FULL - ZENITH_SZA_START)).clamp(0.0, 1.0);

    let zenith_frac = 0.5 + (ZENITH_MAX_FRACTION - 0.5) * sza_t; // 0.5 -> 0.95
    let n = 1.0 + (ZENITH_BIAS_N - 1.0) * sza_t; // 1.0 -> 5.0

    (zenith_frac, n)
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
}
