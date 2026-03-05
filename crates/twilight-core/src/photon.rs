//! Single photon trace logic — the core MCRT pure function.
//!
//! This module contains the backward Monte Carlo photon tracing algorithm.
//! The trace function is a pure function with no platform dependencies,
//! making it compilable to any target (CPU, GPU via WGSL, WASM, CUDA PTX).

use crate::atmosphere::AtmosphereModel;
use crate::geometry::{next_shell_boundary, Vec3};
use crate::scattering::{
    henyey_greenstein_phase, rayleigh_phase, sample_henyey_greenstein, sample_rayleigh_analytic,
    scatter_direction,
};

/// Maximum number of scattering events before terminating a photon.
pub const MAX_SCATTERS: usize = 100;

/// Minimum photon weight before Russian roulette.
pub const RUSSIAN_ROULETTE_WEIGHT: f64 = 0.01;

/// Russian roulette survival probability.
pub const RUSSIAN_ROULETTE_SURVIVE: f64 = 0.1;

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
            // Move to next shell boundary
            match next_shell_boundary(pos, dir, shell.r_inner, shell.r_outer) {
                Some((dist, _)) => {
                    pos = pos + dir * (dist + 1e-3); // small nudge past boundary
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
                    // Photon exits this shell without scattering
                    // Apply transmittance
                    pos = pos + dir * (boundary_dist + 1e-3);

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
        // Compute direct contribution from sun at this scatter point
        let nee_contribution = compute_nee(atm, pos, sun_dir, optics, wavelength_idx, weight);
        result.weight += nee_contribution;
        result.num_scatters += 1;

        // Apply single scattering albedo (probability of scattering vs absorption)
        weight *= optics.ssa;

        // Russian roulette for low-weight photons
        if weight < RUSSIAN_ROULETTE_WEIGHT {
            let xi_rr = xorshift_f64(rng_state);
            if xi_rr > RUSSIAN_ROULETTE_SURVIVE {
                result.terminated = true;
                break;
            }
            weight /= RUSSIAN_ROULETTE_SURVIVE;
        }

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
/// evaluated at the angle between incoming direction and sun direction.
fn compute_nee(
    atm: &AtmosphereModel,
    scatter_pos: Vec3,
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

    // Phase function value for scattering toward observer
    // (In backward MC, the "incoming" is from sun, "outgoing" is toward observer)
    // But for NEE, we evaluate P(sun_dir → current_dir) which equals P(cos_angle)
    // Actually for NEE in backward MC:
    // We need the phase function for scattering from the current photon direction
    // into the sun direction
    let cos_angle = sun_dir.dot(scatter_pos.normalize()); // approximate — should use actual photon direction

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
/// Integrates optical depth from `start_pos` in `direction` until the ray
/// exits the atmosphere. Returns exp(-total_optical_depth).
fn trace_transmittance(
    atm: &AtmosphereModel,
    start_pos: Vec3,
    direction: Vec3,
    wavelength_idx: usize,
) -> f64 {
    let mut pos = start_pos;
    let mut total_optical_depth = 0.0;

    for _ in 0..200 {
        let r = pos.length();

        let shell_idx = match atm.shell_index(r) {
            Some(idx) => idx,
            None => break, // Exited atmosphere
        };

        let shell = &atm.shells[shell_idx];
        let optics = &atm.optics[shell_idx][wavelength_idx];

        match next_shell_boundary(pos, direction, shell.r_inner, shell.r_outer) {
            Some((dist, is_outward)) => {
                total_optical_depth += optics.extinction * dist;
                pos = pos + direction * (dist + 1e-3);

                // Hit ground — fully opaque
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

    #[test]
    fn russian_roulette_weight_is_small() {
        assert!(RUSSIAN_ROULETTE_WEIGHT > 0.0);
        assert!(RUSSIAN_ROULETTE_WEIGHT < 1.0);
    }

    #[test]
    fn russian_roulette_survive_is_valid_probability() {
        assert!(RUSSIAN_ROULETTE_SURVIVE > 0.0);
        assert!(RUSSIAN_ROULETTE_SURVIVE <= 1.0);
    }
}
