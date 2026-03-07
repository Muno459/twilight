//! Rayleigh and Mie/Henyey-Greenstein scattering phase functions.
//!
//! This module provides both scalar (unpolarized) and full Stokes vector
//! (polarized) scattering. The scalar functions are used when the `--fast`
//! flag is set; the Stokes/Mueller framework is used by default for
//! physically rigorous polarized radiative transfer.
//!
//! Stokes vector convention: S = (I, Q, U, V) where I is total intensity,
//! Q and U describe linear polarization, V describes circular polarization.
//! The reference plane for Q/U is the scattering plane (containing the
//! incident and scattered ray directions).

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
    mu.clamp(-1.0, 1.0)
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
    mu.clamp(-1.0, 1.0)
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
    use libm::{fabs, sincos, sqrt};

    let sin_theta = sqrt((1.0 - cos_theta * cos_theta).max(0.0));
    let (sin_phi, cos_phi) = sincos(phi);

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

    // (u, v, w) is orthonormal by construction: u is normalized,
    // v = w x u with both unit and orthogonal, w is unit input.
    // The linear combination has magnitude sqrt(sin_theta^2 + cos_theta^2) = 1,
    // so the result is already unit length -- no final normalize needed.
    let sc = sin_theta * cos_phi;
    let ss = sin_theta * sin_phi;
    Vec3::new(
        sc * u.x + ss * v.x + cos_theta * w.x,
        sc * u.y + ss * v.y + cos_theta * w.y,
        sc * u.z + ss * v.z + cos_theta * w.z,
    )
}

// ── Stokes vector polarized radiative transfer ─────────────────────────

/// Stokes vector: (I, Q, U, V).
///
/// I = total intensity, Q = linear polarization (0/90 deg), U = linear
/// polarization (45/135 deg), V = circular polarization.
///
/// The reference plane is the local scattering plane. When a photon
/// changes scattering plane between bounces, a rotation matrix must be
/// applied (see [`rotation_mueller`]).
#[derive(Debug, Clone, Copy)]
pub struct StokesVector {
    pub s: [f64; 4],
}

impl StokesVector {
    /// Unpolarized light with given intensity.
    #[inline]
    pub const fn unpolarized(intensity: f64) -> Self {
        Self {
            s: [intensity, 0.0, 0.0, 0.0],
        }
    }

    /// Create from explicit components.
    #[inline]
    pub const fn new(i: f64, q: f64, u: f64, v: f64) -> Self {
        Self { s: [i, q, u, v] }
    }

    /// Total intensity (first Stokes parameter).
    #[inline]
    pub fn intensity(&self) -> f64 {
        self.s[0]
    }

    /// Degree of polarization: sqrt(Q^2 + U^2 + V^2) / I.
    #[inline]
    pub fn degree_of_polarization(&self) -> f64 {
        let i = self.s[0];
        if i < 1e-30 {
            return 0.0;
        }
        let q = self.s[1];
        let u = self.s[2];
        let v = self.s[3];
        libm::sqrt(q * q + u * u + v * v) / i
    }

    /// Degree of linear polarization: sqrt(Q^2 + U^2) / I.
    #[inline]
    pub fn degree_of_linear_polarization(&self) -> f64 {
        let i = self.s[0];
        if i < 1e-30 {
            return 0.0;
        }
        libm::sqrt(self.s[1] * self.s[1] + self.s[2] * self.s[2]) / i
    }

    /// Scale all components by a factor.
    #[inline]
    pub fn scale(&self, factor: f64) -> Self {
        Self {
            s: [
                self.s[0] * factor,
                self.s[1] * factor,
                self.s[2] * factor,
                self.s[3] * factor,
            ],
        }
    }

    /// Add two Stokes vectors (incoherent superposition).
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        Self {
            s: [
                self.s[0] + other.s[0],
                self.s[1] + other.s[1],
                self.s[2] + other.s[2],
                self.s[3] + other.s[3],
            ],
        }
    }
}

/// 4x4 Mueller matrix for polarized scattering.
///
/// Transforms a Stokes vector: S_out = M * S_in.
/// Row-major storage: m[row][col].
#[derive(Debug, Clone, Copy)]
pub struct MuellerMatrix {
    pub m: [[f64; 4]; 4],
}

impl MuellerMatrix {
    /// Identity Mueller matrix (no change to Stokes vector).
    #[inline]
    pub const fn identity() -> Self {
        Self {
            m: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    /// Zero Mueller matrix.
    #[inline]
    pub const fn zero() -> Self {
        Self { m: [[0.0; 4]; 4] }
    }

    /// Multiply this Mueller matrix by a Stokes vector: S_out = M * S_in.
    #[inline]
    pub fn apply(&self, s: &StokesVector) -> StokesVector {
        let mut out = [0.0f64; 4];
        for (i, out_i) in out.iter_mut().enumerate() {
            *out_i = self.m[i][0] * s.s[0]
                + self.m[i][1] * s.s[1]
                + self.m[i][2] * s.s[2]
                + self.m[i][3] * s.s[3];
        }
        StokesVector { s: out }
    }

    /// Multiply two Mueller matrices: C = A * B (self = A).
    #[inline]
    pub fn mul(&self, other: &Self) -> Self {
        let mut out = [[0.0f64; 4]; 4];
        for (i, out_row) in out.iter_mut().enumerate() {
            for (j, out_ij) in out_row.iter_mut().enumerate() {
                *out_ij = self.m[i][0] * other.m[0][j]
                    + self.m[i][1] * other.m[1][j]
                    + self.m[i][2] * other.m[2][j]
                    + self.m[i][3] * other.m[3][j];
            }
        }
        Self { m: out }
    }

    /// Scale all elements by a factor.
    #[inline]
    pub fn scale(&self, factor: f64) -> Self {
        let mut out = self.m;
        for row in &mut out {
            for val in row.iter_mut() {
                *val *= factor;
            }
        }
        Self { m: out }
    }
}

/// Rayleigh scattering Mueller matrix.
///
/// For scattering angle theta (given cos_theta), the exact Rayleigh
/// scattering matrix is:
///
/// ```text
///         3     [ cos^2(t)+1   cos^2(t)-1       0        0     ]
/// M = --------- [  cos^2(t)-1  cos^2(t)+1       0        0     ]
///       4       [      0           0         2cos(t)      0     ]
///               [      0           0            0      2cos(t)  ]
/// ```
///
/// This is in the scattering plane reference frame. The M(1,1) element
/// (top-left) equals the scalar Rayleigh phase function: 0.75*(1+cos^2(t)).
///
/// Reference: Chandrasekhar (1960), van de Hulst (1981).
pub fn rayleigh_mueller(cos_theta: f64) -> MuellerMatrix {
    let c2 = cos_theta * cos_theta;
    let a = c2 + 1.0;
    let b = c2 - 1.0;
    let d = 2.0 * cos_theta;

    MuellerMatrix {
        m: [
            [0.75 * a, 0.75 * b, 0.0, 0.0],
            [0.75 * b, 0.75 * a, 0.0, 0.0],
            [0.0, 0.0, 0.75 * d, 0.0],
            [0.0, 0.0, 0.0, 0.75 * d],
        ],
    }
}

/// Henyey-Greenstein Mueller matrix (approximate).
///
/// The HG phase function has no analytical polarization matrix because
/// it is a phenomenological fit, not derived from Mie theory. Following
/// standard practice (Hovenier & van der Mee, 1983), we use the scalar
/// HG phase function as the M(1,1) element with the off-diagonal
/// structure of a general spherical particle scattering matrix:
///
/// ```text
///     [ P11   P12   0    0  ]
/// M = [ P12   P22   0    0  ]
///     [  0     0   P33  P34 ]
///     [  0     0  -P34  P44 ]
/// ```
///
/// For the HG approximation, we set P22 = P11 (which is exact for
/// spherical droplets in the far field) and P12 = P33 = P34 = P44 = 0.
/// This makes HG scattering effectively unpolarizing -- the intensity
/// transforms according to the HG phase function, but no polarization
/// is induced. This is a common and physically reasonable approximation
/// for aerosol/cloud scattering when Mie-computed phase matrices are
/// not available.
pub fn hg_mueller(cos_theta: f64, g: f64) -> MuellerMatrix {
    let p11 = henyey_greenstein_phase(cos_theta, g);
    MuellerMatrix {
        m: [
            [p11, 0.0, 0.0, 0.0],
            [0.0, p11, 0.0, 0.0],
            [0.0, 0.0, p11, 0.0],
            [0.0, 0.0, 0.0, p11],
        ],
    }
}

/// Mueller matrix for rotation of the reference plane by angle phi.
///
/// When the scattering plane changes between successive scattering events,
/// the Stokes vector must be rotated from the old scattering plane to the
/// new one. This rotation only affects Q and U (the linear polarization
/// components); I and V are invariant.
///
/// ```text
///     [ 1     0         0       0 ]
/// R = [ 0   cos(2phi)  sin(2phi) 0 ]
///     [ 0  -sin(2phi)  cos(2phi) 0 ]
///     [ 0     0         0       1 ]
/// ```
///
/// `phi` is the angle between the old and new scattering planes,
/// measured in radians.
pub fn rotation_mueller(phi: f64) -> MuellerMatrix {
    let c = libm::cos(2.0 * phi);
    let s = libm::sin(2.0 * phi);
    MuellerMatrix {
        m: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, s, 0.0],
            [0.0, -s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
}

/// Compute the rotation angle between two scattering planes.
///
/// Given the previous ray direction (`dir_in`), the current ray direction
/// after scattering (`dir_out`), and the new scattering direction
/// (`dir_next`), compute the angle phi needed to rotate the Stokes
/// reference frame from the old scattering plane (dir_in, dir_out)
/// to the new scattering plane (dir_out, dir_next).
///
/// The scattering plane is defined by the cross product of the incident
/// and scattered directions. The rotation angle is the angle between
/// the two planes' normal vectors.
pub fn scattering_plane_rotation(
    dir_in: crate::geometry::Vec3,
    dir_out: crate::geometry::Vec3,
    dir_next: crate::geometry::Vec3,
) -> f64 {
    // Normal to old scattering plane
    let n_old = dir_in.cross(dir_out);
    let n_old_len = n_old.length();

    // Normal to new scattering plane
    let n_new = dir_out.cross(dir_next);
    let n_new_len = n_new.length();

    // If either plane is degenerate (forward/backward scattering), no rotation needed
    if n_old_len < 1e-12 || n_new_len < 1e-12 {
        return 0.0;
    }

    let n_old_unit = n_old * (1.0 / n_old_len);
    let n_new_unit = n_new * (1.0 / n_new_len);

    // cos(phi) = n_old . n_new
    let cos_phi = n_old_unit.dot(n_new_unit);
    // sin(phi) = (n_old x n_new) . dir_out (signed angle)
    let sin_phi = n_old_unit.cross(n_new_unit).dot(dir_out);

    libm::atan2(sin_phi, cos_phi)
}

/// Compute the combined Mueller matrix for a scattering event,
/// including reference frame rotation.
///
/// This is the full polarized scattering operation: rotate the Stokes
/// vector from the old scattering plane to the new one, then apply
/// the scattering Mueller matrix.
///
/// `M_total = M_scatter * R(phi)`
///
/// where R(phi) rotates from the old plane to the current scattering plane.
pub fn scatter_mueller(
    cos_theta: f64,
    rayleigh_fraction: f64,
    asymmetry: f64,
    rotation_angle: f64,
) -> MuellerMatrix {
    let rot = rotation_mueller(rotation_angle);

    let scatter = if rayleigh_fraction > 0.99 {
        rayleigh_mueller(cos_theta)
    } else if rayleigh_fraction < 0.01 {
        hg_mueller(cos_theta, asymmetry)
    } else {
        // Weighted combination of Rayleigh and HG Mueller matrices
        let mr = rayleigh_mueller(cos_theta).scale(rayleigh_fraction);
        let mh = hg_mueller(cos_theta, asymmetry).scale(1.0 - rayleigh_fraction);
        let mut m = MuellerMatrix::zero();
        for i in 0..4 {
            for j in 0..4 {
                m.m[i][j] = mr.m[i][j] + mh.m[i][j];
            }
        }
        m
    };

    scatter.mul(&rot)
}

// ── Trig-free fused scatter+rotate+apply for the hot path ──────────────
//
// The standard path builds two 4x4 matrices (scatter, rotation), multiplies
// them (64 muls + 48 adds), then applies to a 4-vector (16 muls + 12 adds).
// Total: ~140 FP ops + 3 libm trig calls (atan2, cos, sin).
//
// For our atmosphere (Rayleigh + spherical HG), Mueller symmetries give:
//   P22 = P11, P33 = P44, P34 = 0, P12_HG = 0
// which reduces M_scatter * R(phi) * S to 4 scalar equations with 3 params
// (A, B, C) and trig-free double-angle rotation. Total: ~20 FP ops, 0 trig.

/// Compute cos(phi) and sin(phi) for the scattering plane rotation.
///
/// Same geometry as [`scattering_plane_rotation`], but returns the
/// raw (cos, sin) pair instead of the angle. This avoids the atan2 call.
#[inline(always)]
pub fn scattering_plane_cos_sin(
    dir_in: crate::geometry::Vec3,
    dir_out: crate::geometry::Vec3,
    dir_next: crate::geometry::Vec3,
) -> (f64, f64) {
    let n_old = dir_in.cross(dir_out);
    let n_old_len = n_old.length();

    let n_new = dir_out.cross(dir_next);
    let n_new_len = n_new.length();

    if n_old_len < 1e-12 || n_new_len < 1e-12 {
        return (1.0, 0.0); // phi = 0 -> cos=1, sin=0
    }

    let inv_old = 1.0 / n_old_len;
    let inv_new = 1.0 / n_new_len;
    let n_old_unit = n_old * inv_old;
    let n_new_unit = n_new * inv_new;

    let cos_phi = n_old_unit.dot(n_new_unit);
    let sin_phi = n_old_unit.cross(n_new_unit).dot(dir_out);
    (cos_phi, sin_phi)
}

/// Apply scatter + rotate directly to a Stokes vector without building matrices.
///
/// Exploits Mueller symmetries of Rayleigh (P22=P11, P33=P44, P34=0) and
/// spherical HG (diagonal, P12=0) to reduce the combined operation
/// `M_scatter * R(phi) * S` to 4 scalar equations:
///
/// ```text
///   A = alpha*P11_R + (1-alpha)*P11_HG    // mixed scalar phase function
///   B = alpha*P12_R                        // polarization coupling
///   C = alpha*P33_R + (1-alpha)*P11_HG    // cross-pol + HG diagonal
///
///   c2 = 2*cos(phi)^2 - 1   // double-angle, no trig
///   s2 = 2*sin(phi)*cos(phi)
///
///   I' = A*I + B*(c2*Q + s2*U)
///   Q' = B*I + A*(c2*Q + s2*U)     // D=A for Rayleigh symmetry
///   U' = C*(c2*U - s2*Q)
///   V' = C*V                        // E=C for Rayleigh symmetry
/// ```
///
/// Cost: ~20 FP ops, 0 trig calls. Replaces scatter_mueller().apply() which
/// costs ~140 FP ops + 3 libm trig calls.
#[inline(always)]
pub fn scatter_stokes_fast(
    stokes: &StokesVector,
    cos_theta: f64,
    rayleigh_fraction: f64,
    asymmetry: f64,
    cos_phi: f64,
    sin_phi: f64,
) -> StokesVector {
    // Rayleigh components (3/4 normalization)
    let ct2 = cos_theta * cos_theta;
    let p11_r = 0.75 * (ct2 + 1.0);
    let p12_r = 0.75 * (ct2 - 1.0);
    let p33_r = 1.5 * cos_theta;

    // HG component
    let p11_h = henyey_greenstein_phase(cos_theta, asymmetry);

    // Mixed A, B, C
    let alpha = rayleigh_fraction;
    let one_minus_alpha = 1.0 - alpha;
    let a = alpha * p11_r + one_minus_alpha * p11_h;
    let b = alpha * p12_r;
    let c = alpha * p33_r + one_minus_alpha * p11_h;

    // Double-angle rotation (no trig)
    let c2 = 2.0 * cos_phi * cos_phi - 1.0;
    let s2 = 2.0 * sin_phi * cos_phi;

    // Rotated Q, U terms
    let q_rot = c2 * stokes.s[1] + s2 * stokes.s[2];
    let u_rot = c2 * stokes.s[2] - s2 * stokes.s[1];

    StokesVector {
        s: [
            a * stokes.s[0] + b * q_rot,
            b * stokes.s[0] + a * q_rot, // D = A (Rayleigh symmetry)
            c * u_rot,
            c * stokes.s[3], // E = C (Rayleigh symmetry)
        ],
    }
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
            for phi in &[0.0, 1.0, core::f64::consts::PI, 5.0] {
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

    // ── StokesVector ──

    #[test]
    fn stokes_unpolarized_has_zero_qvu() {
        let s = StokesVector::unpolarized(1.0);
        assert!((s.s[0] - 1.0).abs() < EPSILON);
        assert!(s.s[1].abs() < EPSILON);
        assert!(s.s[2].abs() < EPSILON);
        assert!(s.s[3].abs() < EPSILON);
    }

    #[test]
    fn stokes_degree_of_polarization_unpolarized() {
        let s = StokesVector::unpolarized(5.0);
        assert!(s.degree_of_polarization() < EPSILON);
    }

    #[test]
    fn stokes_degree_of_polarization_fully_polarized() {
        // Fully linearly polarized: Q = I, U = V = 0
        let s = StokesVector::new(1.0, 1.0, 0.0, 0.0);
        assert!((s.degree_of_polarization() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn stokes_degree_of_polarization_partially_polarized() {
        let s = StokesVector::new(1.0, 0.5, 0.0, 0.0);
        assert!((s.degree_of_polarization() - 0.5).abs() < EPSILON);
    }

    #[test]
    fn stokes_degree_of_linear_polarization() {
        let s = StokesVector::new(1.0, 0.3, 0.4, 0.0);
        let expected = libm::sqrt(0.3 * 0.3 + 0.4 * 0.4);
        assert!((s.degree_of_linear_polarization() - expected).abs() < EPSILON);
    }

    #[test]
    fn stokes_scale() {
        let s = StokesVector::new(1.0, 0.5, -0.3, 0.1);
        let s2 = s.scale(2.0);
        assert!((s2.s[0] - 2.0).abs() < EPSILON);
        assert!((s2.s[1] - 1.0).abs() < EPSILON);
        assert!((s2.s[2] - (-0.6)).abs() < EPSILON);
        assert!((s2.s[3] - 0.2).abs() < EPSILON);
    }

    #[test]
    fn stokes_add() {
        let a = StokesVector::new(1.0, 0.5, 0.0, 0.0);
        let b = StokesVector::new(1.0, -0.5, 0.3, 0.1);
        let c = a.add(&b);
        assert!((c.s[0] - 2.0).abs() < EPSILON);
        assert!((c.s[1] - 0.0).abs() < EPSILON);
        assert!((c.s[2] - 0.3).abs() < EPSILON);
        assert!((c.s[3] - 0.1).abs() < EPSILON);
    }

    #[test]
    fn stokes_intensity_method() {
        let s = StokesVector::new(2.78, 0.1, 0.2, 0.3);
        assert!((s.intensity() - 2.78).abs() < EPSILON);
    }

    #[test]
    fn stokes_zero_intensity_dop() {
        let s = StokesVector::new(0.0, 0.0, 0.0, 0.0);
        assert!(s.degree_of_polarization() < EPSILON);
    }

    // ── MuellerMatrix ──

    #[test]
    fn mueller_identity_preserves_stokes() {
        let m = MuellerMatrix::identity();
        let s = StokesVector::new(1.0, 0.5, -0.3, 0.1);
        let out = m.apply(&s);
        for i in 0..4 {
            assert!(
                (out.s[i] - s.s[i]).abs() < EPSILON,
                "Identity should preserve component {}: {} vs {}",
                i,
                out.s[i],
                s.s[i]
            );
        }
    }

    #[test]
    fn mueller_zero_zeroes_stokes() {
        let m = MuellerMatrix::zero();
        let s = StokesVector::new(1.0, 0.5, -0.3, 0.1);
        let out = m.apply(&s);
        for i in 0..4 {
            assert!(out.s[i].abs() < EPSILON);
        }
    }

    #[test]
    fn mueller_mul_identity_is_identity() {
        let id = MuellerMatrix::identity();
        let m = rayleigh_mueller(0.5);
        let result = m.mul(&id);
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (result.m[i][j] - m.m[i][j]).abs() < EPSILON,
                    "M*I should equal M at [{},{}]",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn mueller_scale() {
        let m = rayleigh_mueller(0.5);
        let ms = m.scale(2.0);
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (ms.m[i][j] - 2.0 * m.m[i][j]).abs() < EPSILON,
                    "Scale 2x failed at [{},{}]",
                    i,
                    j
                );
            }
        }
    }

    // ── Rayleigh Mueller matrix ──

    #[test]
    fn rayleigh_mueller_forward_scattering() {
        // cos_theta = 1 (forward): M should be 0.75 * [[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,2]]
        // = 1.5 * I
        let m = rayleigh_mueller(1.0);
        assert!((m.m[0][0] - 1.5).abs() < EPSILON);
        assert!((m.m[1][1] - 1.5).abs() < EPSILON);
        assert!((m.m[2][2] - 1.5).abs() < EPSILON);
        assert!((m.m[3][3] - 1.5).abs() < EPSILON);
        assert!(m.m[0][1].abs() < EPSILON); // off-diag zero at theta=0
        assert!(m.m[1][0].abs() < EPSILON);
    }

    #[test]
    fn rayleigh_mueller_backward_scattering() {
        // cos_theta = -1: same as forward (Rayleigh is symmetric)
        let m = rayleigh_mueller(-1.0);
        assert!((m.m[0][0] - 1.5).abs() < EPSILON);
        assert!((m.m[0][1]).abs() < EPSILON);
    }

    #[test]
    fn rayleigh_mueller_90_degree_scattering() {
        // cos_theta = 0 (90 deg): maximum polarization
        let m = rayleigh_mueller(0.0);
        // M(1,1) = 0.75*(0+1) = 0.75
        assert!((m.m[0][0] - 0.75).abs() < EPSILON);
        // M(1,2) = 0.75*(0-1) = -0.75
        assert!((m.m[0][1] - (-0.75)).abs() < EPSILON);
        // M(2,1) = -0.75
        assert!((m.m[1][0] - (-0.75)).abs() < EPSILON);
        // M(3,3) = 0.75 * 2*0 = 0
        assert!(m.m[2][2].abs() < EPSILON);
        assert!(m.m[3][3].abs() < EPSILON);
    }

    #[test]
    fn rayleigh_mueller_m11_equals_scalar_phase() {
        // The M(1,1) element should equal the scalar Rayleigh phase function
        for cos_theta in &[-1.0, -0.5, 0.0, 0.5, 1.0] {
            let m = rayleigh_mueller(*cos_theta);
            let p = rayleigh_phase(*cos_theta);
            assert!(
                (m.m[0][0] - p).abs() < EPSILON,
                "M[0][0]={} != P(cos_theta={})={}",
                m.m[0][0],
                cos_theta,
                p
            );
        }
    }

    #[test]
    fn rayleigh_mueller_unpolarized_gives_scalar_result() {
        // Applying Rayleigh Mueller to unpolarized light should give
        // intensity = scalar phase function * I_in.
        let cos_theta = 0.6;
        let m = rayleigh_mueller(cos_theta);
        let s_in = StokesVector::unpolarized(1.0);
        let s_out = m.apply(&s_in);
        let expected_i = rayleigh_phase(cos_theta);
        assert!(
            (s_out.s[0] - expected_i).abs() < EPSILON,
            "Unpolarized: I_out={}, expected {}",
            s_out.s[0],
            expected_i
        );
    }

    #[test]
    fn rayleigh_mueller_90deg_produces_max_polarization() {
        // At 90 degrees, unpolarized light becomes fully linearly polarized.
        let m = rayleigh_mueller(0.0);
        let s_in = StokesVector::unpolarized(1.0);
        let s_out = m.apply(&s_in);
        // DOP should be 1.0 (fully polarized)
        let dop = s_out.degree_of_polarization();
        assert!(
            (dop - 1.0).abs() < EPSILON,
            "90-deg Rayleigh should fully polarize: DOP={}",
            dop
        );
    }

    #[test]
    fn rayleigh_mueller_symmetric_in_cos_theta() {
        // P(mu) = P(-mu) for M(1,1), and M(1,2) should also be symmetric
        for mu in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            let mp = rayleigh_mueller(*mu);
            let mn = rayleigh_mueller(-*mu);
            assert!((mp.m[0][0] - mn.m[0][0]).abs() < EPSILON);
            assert!((mp.m[0][1] - mn.m[0][1]).abs() < EPSILON);
        }
    }

    // ── HG Mueller matrix ──

    #[test]
    fn hg_mueller_m11_equals_scalar_phase() {
        let g = 0.7;
        for cos_theta in &[-1.0, -0.5, 0.0, 0.5, 1.0] {
            let m = hg_mueller(*cos_theta, g);
            let p = henyey_greenstein_phase(*cos_theta, g);
            assert!(
                (m.m[0][0] - p).abs() < EPSILON,
                "HG M[0][0]={} != P={}",
                m.m[0][0],
                p
            );
        }
    }

    #[test]
    fn hg_mueller_is_diagonal() {
        // Our HG Mueller approximation is diagonal (no polarization induced)
        let m = hg_mueller(0.5, 0.85);
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    assert!(
                        m.m[i][j].abs() < EPSILON,
                        "HG Mueller off-diagonal [{},{}]={}",
                        i,
                        j,
                        m.m[i][j]
                    );
                }
            }
        }
    }

    #[test]
    fn hg_mueller_unpolarized_gives_scalar_result() {
        let cos_theta = 0.3;
        let g = 0.85;
        let m = hg_mueller(cos_theta, g);
        let s_in = StokesVector::unpolarized(1.0);
        let s_out = m.apply(&s_in);
        let expected = henyey_greenstein_phase(cos_theta, g);
        assert!(
            (s_out.s[0] - expected).abs() < EPSILON,
            "HG unpolarized: I_out={}, expected {}",
            s_out.s[0],
            expected
        );
    }

    // ── Rotation Mueller matrix ──

    #[test]
    fn rotation_mueller_zero_angle_is_identity() {
        let r = rotation_mueller(0.0);
        let id = MuellerMatrix::identity();
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (r.m[i][j] - id.m[i][j]).abs() < EPSILON,
                    "R(0) should be identity at [{},{}]",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn rotation_mueller_preserves_intensity() {
        // Rotation should not change I or V
        let r = rotation_mueller(0.7);
        let s = StokesVector::new(1.0, 0.5, 0.3, 0.1);
        let out = r.apply(&s);
        assert!((out.s[0] - 1.0).abs() < EPSILON, "I should be preserved");
        assert!((out.s[3] - 0.1).abs() < EPSILON, "V should be preserved");
    }

    #[test]
    fn rotation_mueller_preserves_dop() {
        // Rotation should not change the degree of polarization
        let s = StokesVector::new(1.0, 0.5, 0.3, 0.1);
        let dop_before = s.degree_of_polarization();
        let r = rotation_mueller(1.23);
        let out = r.apply(&s);
        let dop_after = out.degree_of_polarization();
        assert!(
            (dop_before - dop_after).abs() < 1e-8,
            "Rotation should preserve DOP: {} -> {}",
            dop_before,
            dop_after
        );
    }

    #[test]
    fn rotation_mueller_pi_flips_qu() {
        // Rotating by pi: cos(2pi) = 1, sin(2pi) = 0 → identity
        // Actually R(pi): cos(2*pi) = 1, sin(2*pi) ≈ 0 → identity!
        // Rotating by pi/2: cos(pi) = -1, sin(pi) = 0 → Q → -Q, U → -U
        let r = rotation_mueller(core::f64::consts::PI / 2.0);
        let s = StokesVector::new(1.0, 0.5, 0.3, 0.1);
        let out = r.apply(&s);
        assert!((out.s[0] - 1.0).abs() < EPSILON);
        assert!((out.s[1] - (-0.5)).abs() < EPSILON);
        assert!((out.s[2] - (-0.3)).abs() < EPSILON);
        assert!((out.s[3] - 0.1).abs() < EPSILON);
    }

    #[test]
    fn rotation_mueller_quarter_rotation() {
        // phi = pi/4: cos(pi/2) = 0, sin(pi/2) = 1
        // Q_out = 0*Q + 1*U = U
        // U_out = -1*Q + 0*U = -Q
        let r = rotation_mueller(core::f64::consts::PI / 4.0);
        let s = StokesVector::new(1.0, 0.6, 0.3, 0.0);
        let out = r.apply(&s);
        assert!((out.s[0] - 1.0).abs() < EPSILON);
        assert!((out.s[1] - 0.3).abs() < EPSILON, "Q_out should be U_in");
        assert!((out.s[2] - (-0.6)).abs() < EPSILON, "U_out should be -Q_in");
    }

    #[test]
    fn rotation_mueller_double_rotation_adds() {
        // R(a) * R(b) should equal R(a+b) for the Q,U block
        let a = 0.3;
        let b = 0.7;
        let ra = rotation_mueller(a);
        let rb = rotation_mueller(b);
        let rab = ra.mul(&rb);
        let r_sum = rotation_mueller(a + b);
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (rab.m[i][j] - r_sum.m[i][j]).abs() < 1e-8,
                    "R(a)*R(b) != R(a+b) at [{},{}]",
                    i,
                    j
                );
            }
        }
    }

    // ── scattering_plane_rotation ──

    #[test]
    fn scattering_plane_rotation_same_plane() {
        // If all three directions are coplanar, phi should be 0
        let dir_in = Vec3::new(0.0, 0.0, 1.0);
        let dir_out = Vec3::new(0.0, 0.5, 0.866).normalize();
        let dir_next = Vec3::new(0.0, 0.866, 0.5).normalize();
        let phi = scattering_plane_rotation(dir_in, dir_out, dir_next);
        assert!(
            phi.abs() < 1e-6,
            "Coplanar directions should give phi=0, got {}",
            phi
        );
    }

    #[test]
    fn scattering_plane_rotation_forward_scatter() {
        // Forward scattering (dir_in = dir_out) → degenerate plane → phi = 0
        let dir = Vec3::new(0.0, 0.0, 1.0);
        let dir_next = Vec3::new(0.5, 0.0, 0.866).normalize();
        let phi = scattering_plane_rotation(dir, dir, dir_next);
        assert!(
            phi.abs() < EPSILON,
            "Forward scatter should give phi=0, got {}",
            phi
        );
    }

    // ── scatter_mueller (combined) ──

    #[test]
    fn scatter_mueller_pure_rayleigh_at_zero_rotation() {
        let cos_theta = 0.5;
        let m = scatter_mueller(cos_theta, 1.0, 0.0, 0.0);
        let mr = rayleigh_mueller(cos_theta);
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (m.m[i][j] - mr.m[i][j]).abs() < EPSILON,
                    "Pure Rayleigh at rotation=0 should match at [{},{}]",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn scatter_mueller_pure_hg_at_zero_rotation() {
        let cos_theta = 0.5;
        let g = 0.85;
        let m = scatter_mueller(cos_theta, 0.0, g, 0.0);
        let mh = hg_mueller(cos_theta, g);
        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    (m.m[i][j] - mh.m[i][j]).abs() < EPSILON,
                    "Pure HG at rotation=0 should match at [{},{}]",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn scatter_mueller_mixed_interpolates() {
        // With 50% Rayleigh, 50% HG, M(1,1) should be average of both
        let cos_theta = 0.5;
        let g = 0.7;
        let m = scatter_mueller(cos_theta, 0.5, g, 0.0);
        let expected_m11 =
            0.5 * rayleigh_phase(cos_theta) + 0.5 * henyey_greenstein_phase(cos_theta, g);
        assert!(
            (m.m[0][0] - expected_m11).abs() < EPSILON,
            "Mixed M[0][0]={}, expected {}",
            m.m[0][0],
            expected_m11
        );
    }

    #[test]
    fn scatter_mueller_unpolarized_intensity_matches_scalar() {
        // For unpolarized input, the output intensity from the Mueller
        // framework should exactly match the scalar phase function
        let cos_theta = 0.3;
        let s_in = StokesVector::unpolarized(1.0);

        // Pure Rayleigh
        let m_r = scatter_mueller(cos_theta, 1.0, 0.0, 0.0);
        let s_r = m_r.apply(&s_in);
        assert!(
            (s_r.s[0] - rayleigh_phase(cos_theta)).abs() < EPSILON,
            "Rayleigh scalar mismatch"
        );

        // Pure HG
        let g = 0.85;
        let m_h = scatter_mueller(cos_theta, 0.0, g, 0.0);
        let s_h = m_h.apply(&s_in);
        assert!(
            (s_h.s[0] - henyey_greenstein_phase(cos_theta, g)).abs() < EPSILON,
            "HG scalar mismatch"
        );
    }
}
