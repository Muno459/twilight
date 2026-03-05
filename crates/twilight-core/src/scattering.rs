//! Rayleigh and Mie/Henyey-Greenstein scattering phase functions.

use libm::cos;

/// Rayleigh scattering phase function.
///
/// P(cos_theta) = (3/4) * (1 + cos^2(theta))
///
/// This is the unpolarized form. Normalized so that integrating over
/// 4π steradians gives 4π (i.e., the single-scattering normalization).
#[inline]
pub fn rayleigh_phase(cos_theta: f64) -> f64 {
    0.75 * (1.0 + cos_theta * cos_theta)
}

/// Henyey-Greenstein phase function for aerosol/cloud scattering.
///
/// P(cos_theta; g) = (1 - g^2) / (1 + g^2 - 2g*cos_theta)^(3/2)
///
/// - `g`: asymmetry parameter (-1 to 1). g=0 is isotropic, g>0 is forward-peaked.
///
/// Normalized over 4π steradians.
#[inline]
pub fn henyey_greenstein_phase(cos_theta: f64, g: f64) -> f64 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    (1.0 - g2) / (denom * libm::sqrt(denom))
}

/// Sample a scattering angle from the Rayleigh phase function.
///
/// Uses the rejection method. `xi1` and `xi2` are uniform random numbers in [0, 1).
///
/// Returns cos(theta).
pub fn sample_rayleigh(xi1: f64, xi2: f64) -> f64 {
    // Analytical inversion method for Rayleigh:
    // Use the formula from Chandrasekhar:
    // cos(theta) = u where u is found from the CDF
    //
    // Simpler: rejection sampling on [-1, 1]
    // P(mu) = 0.75*(1 + mu^2), max = 1.5 at mu = ±1
    let mu = 2.0 * xi1 - 1.0; // uniform in [-1, 1]
    let p = 0.75 * (1.0 + mu * mu);
    if xi2 * 1.5 < p {
        mu
    } else {
        // Rare rejection — try deterministic fallback
        // In practice we'd loop, but in no_std we use the analytic CDF inverse
        sample_rayleigh_analytic(xi1)
    }
}

/// Analytically sample from Rayleigh phase function CDF.
///
/// `xi` is a uniform random number in [0, 1).
/// Returns cos(theta).
pub fn sample_rayleigh_analytic(xi: f64) -> f64 {
    use libm::cbrt;
    // CDF inversion for P(mu) ∝ (1 + mu^2):
    // The normalized CDF is: F(mu) = (1/8)(3*mu + mu^3 + 4)
    // Invert: 3*mu + mu^3 = 8*xi - 4
    // Let q = 8*xi - 4, solve mu^3 + 3*mu - q = 0
    let q = 8.0 * xi - 4.0;
    // Cardano's formula for depressed cubic t^3 + pt + q = 0 with p=3:
    // Only one real root since discriminant > 0
    let disc = q * q / 4.0 + 1.0; // p^3/27 + q^2/4 = 1 + q^2/4
    let sqrt_disc = libm::sqrt(disc);
    let u = cbrt(-q / 2.0 + sqrt_disc);
    let v = cbrt(-q / 2.0 - sqrt_disc);
    let mu = u + v;
    // Clamp to [-1, 1]
    if mu < -1.0 {
        -1.0
    } else if mu > 1.0 {
        1.0
    } else {
        mu
    }
}

/// Sample a scattering angle from the Henyey-Greenstein phase function.
///
/// Analytical CDF inversion. `xi` is a uniform random number in [0, 1).
/// `g` is the asymmetry parameter.
///
/// Returns cos(theta).
pub fn sample_henyey_greenstein(xi: f64, g: f64) -> f64 {
    if libm::fabs(g) < 1e-6 {
        // Isotropic: uniform on [-1, 1]
        return 2.0 * xi - 1.0;
    }

    let g2 = g * g;
    // CDF inversion formula:
    // cos(theta) = (1/(2g)) * [1 + g^2 - ((1-g^2)/(1 - g + 2*g*xi))^2]
    let s = (1.0 - g2) / (1.0 - g + 2.0 * g * xi);
    let mu = (1.0 + g2 - s * s) / (2.0 * g);

    // Clamp to [-1, 1]
    if mu < -1.0 {
        -1.0
    } else if mu > 1.0 {
        1.0
    } else {
        mu
    }
}

/// Generate a new direction vector after scattering.
///
/// Given the incoming direction and a sampled cos(theta) and azimuthal angle phi,
/// rotate the direction accordingly.
///
/// - `dir`: incoming direction (unit vector)
/// - `cos_theta`: cosine of polar scattering angle
/// - `phi`: azimuthal scattering angle in radians [0, 2π)
///
/// Returns the new direction (unit vector).
pub fn scatter_direction(
    dir: crate::geometry::Vec3,
    cos_theta: f64,
    phi: f64,
) -> crate::geometry::Vec3 {
    use crate::geometry::Vec3;
    use libm::{fabs, sin, sqrt};

    let sin_theta = sqrt((1.0 - cos_theta * cos_theta).max(0.0));
    let cos_phi = cos(phi);
    let sin_phi = sin(phi);

    // Build local coordinate system around incoming direction
    let w = dir;

    // Find a vector not parallel to w
    let up = if fabs(w.z) < 0.9 {
        Vec3::new(0.0, 0.0, 1.0)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };

    let u = w.cross(up).normalize();
    let v = w.cross(u);

    // New direction in local coordinates
    Vec3::new(
        sin_theta * cos_phi * u.x + sin_theta * sin_phi * v.x + cos_theta * w.x,
        sin_theta * cos_phi * u.y + sin_theta * sin_phi * v.y + cos_theta * w.y,
        sin_theta * cos_phi * u.z + sin_theta * sin_phi * v.z + cos_theta * w.z,
    )
    .normalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Vec3;

    const EPSILON: f64 = 1e-10;

    // ── Rayleigh phase function ──

    #[test]
    fn rayleigh_phase_forward() {
        // P(cos_theta=1) = 0.75*(1+1) = 1.5
        assert!((rayleigh_phase(1.0) - 1.5).abs() < EPSILON);
    }

    #[test]
    fn rayleigh_phase_backward() {
        // P(cos_theta=-1) = 0.75*(1+1) = 1.5 (symmetric)
        assert!((rayleigh_phase(-1.0) - 1.5).abs() < EPSILON);
    }

    #[test]
    fn rayleigh_phase_perpendicular() {
        // P(cos_theta=0) = 0.75*(1+0) = 0.75
        assert!((rayleigh_phase(0.0) - 0.75).abs() < EPSILON);
    }

    #[test]
    fn rayleigh_phase_symmetric() {
        // P(μ) = P(-μ) for all μ
        for mu in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            assert!(
                (rayleigh_phase(*mu) - rayleigh_phase(-*mu)).abs() < EPSILON,
                "Rayleigh phase not symmetric at mu={}",
                mu
            );
        }
    }

    #[test]
    fn rayleigh_phase_minimum_at_90deg() {
        // Minimum at cos_theta=0 (90°), P=0.75
        let p_min = rayleigh_phase(0.0);
        for mu in &[-1.0, -0.5, 0.5, 1.0] {
            assert!(
                rayleigh_phase(*mu) >= p_min - EPSILON,
                "Phase at mu={} should be >= minimum 0.75",
                mu
            );
        }
    }

    #[test]
    fn rayleigh_phase_normalization_integral() {
        // ∫₋₁¹ P(μ) dμ = 2 (normalized so ∫₄π P dΩ/(4π) = 1)
        // Numerical integration with Simpson's rule
        let n = 10000;
        let mut integral = 0.0;
        let dmu = 2.0 / n as f64;
        for i in 0..n {
            let mu = -1.0 + (i as f64 + 0.5) * dmu;
            integral += rayleigh_phase(mu) * dmu;
        }
        assert!(
            (integral - 2.0).abs() < 0.001,
            "∫₋₁¹ P(μ) dμ = {}, expected 2.0",
            integral
        );
    }

    // ── Henyey-Greenstein phase function ──

    #[test]
    fn hg_isotropic_when_g_zero() {
        // g=0 → P = (1-0)/(1+0-0)^(3/2) = 1.0 for all angles
        for mu in &[-1.0, -0.5, 0.0, 0.5, 1.0] {
            assert!(
                (henyey_greenstein_phase(*mu, 0.0) - 1.0).abs() < EPSILON,
                "HG(mu={}, g=0) should be 1.0",
                mu
            );
        }
    }

    #[test]
    fn hg_forward_peaked() {
        // For g=0.85 (cloud droplets), P(1) >> P(-1)
        let p_forward = henyey_greenstein_phase(1.0, 0.85);
        let p_backward = henyey_greenstein_phase(-1.0, 0.85);
        assert!(
            p_forward > p_backward * 10.0,
            "HG(g=0.85) should be strongly forward-peaked: forward={}, backward={}",
            p_forward,
            p_backward
        );
    }

    #[test]
    fn hg_backward_peaked() {
        // For g=-0.85, P(-1) >> P(1)
        let p_forward = henyey_greenstein_phase(1.0, -0.85);
        let p_backward = henyey_greenstein_phase(-1.0, -0.85);
        assert!(
            p_backward > p_forward * 10.0,
            "HG(g=-0.85) should be backward-peaked"
        );
    }

    #[test]
    fn hg_normalization_integral() {
        // ∫₋₁¹ P(μ;g) dμ = 2 for any g
        for g in &[0.0, 0.3, 0.7, 0.85, -0.5] {
            let n = 10000;
            let dmu = 2.0 / n as f64;
            let mut integral = 0.0;
            for i in 0..n {
                let mu = -1.0 + (i as f64 + 0.5) * dmu;
                integral += henyey_greenstein_phase(mu, *g) * dmu;
            }
            assert!(
                (integral - 2.0).abs() < 0.01,
                "∫₋₁¹ HG(μ;g={}) dμ = {}, expected 2.0",
                g,
                integral
            );
        }
    }

    #[test]
    fn hg_forward_exact_value() {
        // P(1; g) = (1-g²)/(1+g²-2g)^(3/2) = (1-g²)/(1-g)³ = (1+g)/(1-g)²
        let g = 0.5;
        let expected = (1.0 + g) / ((1.0 - g) * (1.0 - g));
        assert!(
            (henyey_greenstein_phase(1.0, g) - expected).abs() < EPSILON,
            "HG(1, 0.5) = {}, expected {}",
            henyey_greenstein_phase(1.0, g),
            expected
        );
    }

    // ── sample_rayleigh_analytic ──

    #[test]
    fn sample_rayleigh_analytic_bounds() {
        // Output should always be in [-1, 1]
        for i in 0..1000 {
            let xi = i as f64 / 999.0;
            let mu = sample_rayleigh_analytic(xi);
            assert!(
                (-1.0..=1.0).contains(&mu),
                "sample_rayleigh_analytic({}) = {} out of bounds",
                xi,
                mu
            );
        }
    }

    #[test]
    fn sample_rayleigh_analytic_endpoints() {
        // The CDF inversion maps xi=0 → mu=+1 and xi=1 → mu=-1
        // (inverted convention but valid — the Rayleigh phase function
        // is symmetric P(μ) = P(-μ), so this doesn't affect physics)
        let mu_low = sample_rayleigh_analytic(0.0);
        let mu_high = sample_rayleigh_analytic(1.0 - 1e-15);
        // One should be near +1 and the other near -1
        assert!(
            mu_low.abs() > 0.9,
            "xi=0 should give |mu| near 1, got {}",
            mu_low
        );
        assert!(
            mu_high.abs() > 0.9,
            "xi≈1 should give |mu| near 1, got {}",
            mu_high
        );
        // They should be opposite signs
        assert!(
            mu_low * mu_high < 0.0,
            "Endpoints should map to opposite signs: mu(0)={}, mu(1)={}",
            mu_low,
            mu_high
        );
    }

    #[test]
    fn sample_rayleigh_analytic_midpoint() {
        // xi=0.5 → mu should be near 0 (by symmetry of the CDF)
        let mu = sample_rayleigh_analytic(0.5);
        assert!(mu.abs() < 0.1, "xi=0.5 should give mu near 0, got {}", mu);
    }

    #[test]
    fn sample_rayleigh_analytic_distribution() {
        // Verify the CDF inverse: if we sample many xi uniformly, the resulting
        // distribution should match the Rayleigh phase function.
        // Check: mean of cos²(theta) = 2/5 analytically for Rayleigh.
        let n = 100_000;
        let mut sum_mu2 = 0.0;
        for i in 0..n {
            let xi = (i as f64 + 0.5) / n as f64;
            let mu = sample_rayleigh_analytic(xi);
            sum_mu2 += mu * mu;
        }
        let mean_mu2 = sum_mu2 / n as f64;
        // <μ²> = ∫₋₁¹ μ² × P(μ)/2 dμ where P(μ)=0.75(1+μ²)
        // = 0.75 × ∫₋₁¹ (μ² + μ⁴)/2 dμ = 0.75 × (2/3 + 2/5)/2 = 0.75 × 16/30 = 0.4
        assert!(
            (mean_mu2 - 0.4).abs() < 0.01,
            "<μ²> = {}, expected 0.4",
            mean_mu2
        );
    }

    // ── sample_henyey_greenstein ──

    #[test]
    fn sample_hg_isotropic_is_linear() {
        // g≈0: cos_theta = 2*xi - 1
        for xi in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let mu = sample_henyey_greenstein(*xi, 0.0);
            let expected = 2.0 * xi - 1.0;
            assert!(
                (mu - expected).abs() < 1e-6,
                "HG sample(xi={}, g=0) = {}, expected {}",
                xi,
                mu,
                expected
            );
        }
    }

    #[test]
    fn sample_hg_bounds() {
        for g in &[0.0, 0.3, 0.7, 0.85, 0.99, -0.5, -0.85] {
            for i in 0..100 {
                let xi = i as f64 / 99.0;
                let mu = sample_henyey_greenstein(xi, *g);
                assert!(
                    (-1.0..=1.0).contains(&mu),
                    "HG sample(xi={}, g={}) = {} out of bounds",
                    xi,
                    g,
                    mu
                );
            }
        }
    }

    #[test]
    fn sample_hg_forward_bias() {
        // For g=0.85, most samples should be > 0 (forward scattering)
        let n = 10000;
        let mut n_forward = 0;
        for i in 0..n {
            let xi = (i as f64 + 0.5) / n as f64;
            let mu = sample_henyey_greenstein(xi, 0.85);
            if mu > 0.0 {
                n_forward += 1;
            }
        }
        let frac = n_forward as f64 / n as f64;
        assert!(
            frac > 0.8,
            "HG(g=0.85): {}% forward, expected >80%",
            frac * 100.0
        );
    }

    // ── sample_rayleigh (rejection method) ──

    #[test]
    fn sample_rayleigh_bounds() {
        // Should always return values in [-1, 1]
        for i in 0..100 {
            for j in 0..100 {
                let xi1 = i as f64 / 99.0;
                let xi2 = j as f64 / 99.0;
                let mu = sample_rayleigh(xi1, xi2);
                assert!(
                    (-1.0..=1.0).contains(&mu),
                    "sample_rayleigh({}, {}) = {} out of bounds",
                    xi1,
                    xi2,
                    mu
                );
            }
        }
    }

    // ── scatter_direction ──

    #[test]
    fn scatter_direction_preserves_unit_length() {
        let dir = Vec3::new(0.0, 0.0, 1.0);
        for cos_theta in &[-0.9, -0.5, 0.0, 0.5, 0.9] {
            for phi in &[0.0, 1.0, 3.14159, 5.0] {
                let new_dir = scatter_direction(dir, *cos_theta, *phi);
                assert!(
                    (new_dir.length() - 1.0).abs() < 1e-6,
                    "scatter_direction not unit: cos_theta={}, phi={}, len={}",
                    cos_theta,
                    phi,
                    new_dir.length()
                );
            }
        }
    }

    #[test]
    fn scatter_direction_forward_preserves_direction() {
        // cos_theta=1 (forward scattering) → same direction
        let dir = Vec3::new(0.0, 0.0, 1.0);
        let new_dir = scatter_direction(dir, 1.0, 0.0);
        assert!(
            (new_dir.dot(dir) - 1.0).abs() < 1e-6,
            "Forward scatter should preserve direction: dot={}",
            new_dir.dot(dir)
        );
    }

    #[test]
    fn scatter_direction_backward_reverses_direction() {
        // cos_theta=-1 (backward scattering) → opposite direction
        let dir = Vec3::new(0.0, 0.0, 1.0);
        let new_dir = scatter_direction(dir, -1.0, 0.0);
        assert!(
            (new_dir.dot(dir) - (-1.0)).abs() < 1e-6,
            "Backward scatter should reverse direction: dot={}",
            new_dir.dot(dir)
        );
    }

    #[test]
    fn scatter_direction_perpendicular() {
        // cos_theta=0 → new direction should be perpendicular to original
        let dir = Vec3::new(0.0, 0.0, 1.0);
        let new_dir = scatter_direction(dir, 0.0, 0.0);
        assert!(
            new_dir.dot(dir).abs() < 1e-6,
            "90° scatter should be perpendicular: dot={}",
            new_dir.dot(dir)
        );
    }

    #[test]
    fn scatter_direction_works_for_all_cardinal_directions() {
        // Test that scatter_direction works when input dir is along each axis
        let dirs = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
        ];
        for dir in &dirs {
            let new_dir = scatter_direction(*dir, 0.5, 1.0);
            assert!(
                (new_dir.length() - 1.0).abs() < 1e-6,
                "Failed for dir={:?}",
                dir
            );
        }
    }
}
