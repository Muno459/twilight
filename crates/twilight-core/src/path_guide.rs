//! 2D Path Guiding for atmospheric MCRT.
//!
//! Learns directional importance at each (altitude, solar_angle) position
//! from pilot MC chains, then guides production chains toward high-contribution
//! directions via one-sample MIS with the phase function.
//!
//! The atmosphere's spherical symmetry reduces the full guiding problem to a
//! compact 2D spatial grid (altitude x solar_angle) with a coarse directional
//! distribution at each grid cell. Total storage: 16 KiB in fixed arrays,
//! trivially fits in `#![no_std]`.
//!
//! # Direction parameterization
//!
//! At each scatter vertex, the outgoing direction is parameterized by:
//! - cos_zenith: dot(dir, local_up) -- 4 bands
//! - sun_side: dot(dir, sun_horiz) >= 0 -- 2 half-planes
//!
//! Total: 8 direction bins per spatial cell.
//!
//! # Training
//!
//! Run pilot chains with standard sampling. At each scatter vertex where
//! NEE produces nonzero contribution, accumulate the contribution into the
//! bin corresponding to the chain's current direction. After each pilot
//! iteration, normalize to get probability distributions. Laplace smoothing
//! prevents zero bins.
//!
//! # Production
//!
//! At each bounce (after the first), use one-sample MIS between the guide
//! distribution and the phase function. The MIS weight keeps the estimator
//! exactly unbiased. At SZA <= 96 the guide has no effect (uniform bins).

use crate::geometry::Vec3;

/// Number of altitude bins (0 to 100 km).
pub const NUM_ALT_BINS: usize = 32;

/// Number of solar angle bins at the scatter vertex.
/// Solar angle = acos(dot(sun_dir, local_up)), range [0, pi].
pub const NUM_SOLAR_BINS: usize = 16;

/// Number of direction bins per spatial cell.
/// 4 cos_zenith bands x 2 sun-relative azimuthal half-planes.
pub const NUM_DIR_BINS: usize = 8;

/// Top of atmosphere altitude for binning [m].
const TOA_ALT_M: f64 = 100_000.0;

/// Altitude bin width [m].
const ALT_BIN_WIDTH: f64 = TOA_ALT_M / NUM_ALT_BINS as f64;

/// Boundaries for cos_zenith bands.
///   [1.0,  0.5): strongly upward   (bin 0,1)
///   [0.5,  0.0): moderately upward  (bin 2,3)
///   [0.0, -0.5): moderately downward(bin 4,5)
///   [-0.5,-1.0]: strongly downward  (bin 6,7)
/// Even index = toward sun, odd = away from sun.
const COS_Z_BOUNDS: [f64; 3] = [0.5, 0.0, -0.5];

/// Solid angle of each direction bin.
///
/// Each bin covers delta_cos_z = 0.5 and delta_phi = pi (half-plane).
/// Solid angle = delta_cos_z * delta_phi = 0.5 * pi = pi/2.
const BIN_SOLID_ANGLE: f64 = core::f64::consts::PI / 2.0;

/// Xorshift64 RNG -- local copy to avoid circular dependency with photon.rs.
#[inline]
fn xorshift_f64(state: &mut u64) -> f64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

/// 2D path guiding table for atmospheric MCRT.
///
/// Fixed-size, `no_std`-compatible. Total memory:
/// 32 * 16 * 8 * 4 bytes = 16,384 bytes = 16 KiB.
#[derive(Clone)]
pub struct PathGuide {
    /// Directional probability weights.
    /// After normalization, each `[alt][solar]` slice sums to 1.0.
    table: [[[f32; NUM_DIR_BINS]; NUM_SOLAR_BINS]; NUM_ALT_BINS],
    /// Whether the guide has been trained (at least one normalize pass).
    trained: bool,
}

impl Default for PathGuide {
    fn default() -> Self {
        Self::new()
    }
}

impl PathGuide {
    /// Create a new uniform (untrained) path guide.
    pub fn new() -> Self {
        let uniform = 1.0 / NUM_DIR_BINS as f32;
        Self {
            table: [[[uniform; NUM_DIR_BINS]; NUM_SOLAR_BINS]; NUM_ALT_BINS],
            trained: false,
        }
    }

    /// Whether the guide has been trained.
    #[inline]
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Compute altitude bin index (clamped to valid range).
    #[inline]
    fn alt_bin(alt_m: f64) -> usize {
        let bin = (alt_m / ALT_BIN_WIDTH) as usize;
        if bin >= NUM_ALT_BINS {
            NUM_ALT_BINS - 1
        } else {
            bin
        }
    }

    /// Compute solar angle bin index.
    ///
    /// cos_solar = dot(sun_dir, local_up), in [-1, 1].
    /// cos=1 (sun overhead) -> bin 0. cos=-1 (sun nadir) -> bin 15.
    #[inline]
    fn solar_bin(cos_solar: f64) -> usize {
        let t = (1.0 - cos_solar) * 0.5; // [0, 1]
        let bin = (t * NUM_SOLAR_BINS as f64) as usize;
        if bin >= NUM_SOLAR_BINS {
            NUM_SOLAR_BINS - 1
        } else {
            bin
        }
    }

    /// Compute direction bin index.
    ///
    /// 4 cos_zenith bands x 2 sun-relative azimuthal half-planes = 8 bins.
    /// Even index = toward sun, odd index = away from sun.
    #[inline]
    fn dir_bin(cos_zenith: f64, toward_sun: bool) -> usize {
        let z_band = if cos_zenith >= COS_Z_BOUNDS[0] {
            0
        } else if cos_zenith >= COS_Z_BOUNDS[1] {
            1
        } else if cos_zenith >= COS_Z_BOUNDS[2] {
            2
        } else {
            3
        };
        z_band * 2 + if toward_sun { 0 } else { 1 }
    }

    /// Compute the sun-horizontal unit vector.
    ///
    /// Projects sun_dir onto the plane perpendicular to local_up.
    /// If the sun is at zenith/nadir, falls back to an arbitrary horizontal.
    #[inline]
    fn sun_horiz(local_up: Vec3, sun_dir: Vec3) -> Vec3 {
        let proj = sun_dir.dot(local_up);
        let h = Vec3::new(
            sun_dir.x - proj * local_up.x,
            sun_dir.y - proj * local_up.y,
            sun_dir.z - proj * local_up.z,
        );
        let len = h.length();
        if len > 1e-10 {
            h.scale(1.0 / len)
        } else {
            // Sun at zenith/nadir: pick arbitrary horizontal.
            // Use cross product with a non-parallel vector.
            let arbitrary = if local_up.x.abs() < 0.9 {
                Vec3::new(1.0, 0.0, 0.0)
            } else {
                Vec3::new(0.0, 1.0, 0.0)
            };
            let c = local_up.cross(arbitrary);
            let cl = c.length();
            if cl > 1e-10 {
                c.scale(1.0 / cl)
            } else {
                Vec3::new(1.0, 0.0, 0.0)
            }
        }
    }

    /// Full bin lookup: position + direction -> (alt_bin, solar_bin, dir_bin).
    #[inline]
    pub fn lookup(alt_m: f64, local_up: Vec3, sun_dir: Vec3, dir: Vec3) -> (usize, usize, usize) {
        let a = Self::alt_bin(alt_m);
        let cos_solar = sun_dir.dot(local_up).clamp(-1.0, 1.0);
        let s = Self::solar_bin(cos_solar);
        let cos_z = dir.dot(local_up);
        let sun_h = Self::sun_horiz(local_up, sun_dir);
        let toward = dir.dot(sun_h) >= 0.0;
        let d = Self::dir_bin(cos_z, toward);
        (a, s, d)
    }

    /// Accumulate a training sample.
    ///
    /// Called during pilot iterations at each scatter vertex where NEE > 0.
    /// `contribution` is the absolute NEE weight (positive). The direction
    /// `dir` is the chain's current travel direction at this vertex.
    #[inline]
    pub fn accumulate(
        &mut self,
        alt_m: f64,
        local_up: Vec3,
        sun_dir: Vec3,
        dir: Vec3,
        contribution: f64,
    ) {
        let (a, s, d) = Self::lookup(alt_m, local_up, sun_dir, dir);
        self.table[a][s][d] += contribution as f32;
    }

    /// Normalize all spatial cells to probability distributions.
    ///
    /// Each `[alt][solar]` slice is normalized to sum to 1.0. Applies
    /// Laplace smoothing (adds a small uniform baseline before normalizing)
    /// to prevent zero-probability bins that would cause MIS singularities.
    pub fn normalize(&mut self) {
        let smoothing = 1.0 / NUM_DIR_BINS as f32;
        for a in 0..NUM_ALT_BINS {
            for s in 0..NUM_SOLAR_BINS {
                let cell = &mut self.table[a][s];
                let mut sum = 0.0f32;
                for item in cell.iter_mut() {
                    *item += smoothing;
                    sum += *item;
                }
                if sum > 0.0 {
                    let inv = 1.0 / sum;
                    for item in cell.iter_mut() {
                        *item *= inv;
                    }
                } else {
                    for item in cell.iter_mut() {
                        *item = 1.0 / NUM_DIR_BINS as f32;
                    }
                }
            }
        }
        self.trained = true;
    }

    /// Reset all bins to zero (before a new training iteration).
    pub fn reset(&mut self) {
        for a in 0..NUM_ALT_BINS {
            for s in 0..NUM_SOLAR_BINS {
                for d in 0..NUM_DIR_BINS {
                    self.table[a][s][d] = 0.0;
                }
            }
        }
        self.trained = false;
    }

    /// Get the directional PDF for a given direction [sr^-1].
    ///
    /// PDF = P(bin) / solid_angle(bin) = table[a][s][d] / (pi/2).
    #[inline]
    pub fn pdf(&self, alt_m: f64, local_up: Vec3, sun_dir: Vec3, dir: Vec3) -> f64 {
        let (a, s, d) = Self::lookup(alt_m, local_up, sun_dir, dir);
        self.table[a][s][d] as f64 / BIN_SOLID_ANGLE
    }

    /// Sample a direction from the guide distribution.
    ///
    /// Uses CDF inversion to select a direction bin, then samples uniformly
    /// within the bin (uniform in cos_zenith and azimuth).
    ///
    /// Returns `(direction, pdf)` where pdf is in sr^-1.
    pub fn sample(&self, alt_m: f64, local_up: Vec3, sun_dir: Vec3, rng: &mut u64) -> (Vec3, f64) {
        let a = Self::alt_bin(alt_m);
        let cos_solar = sun_dir.dot(local_up).clamp(-1.0, 1.0);
        let s = Self::solar_bin(cos_solar);
        let cell = &self.table[a][s];

        // CDF inversion to select direction bin.
        let xi = xorshift_f64(rng) as f32;
        let mut cumulative = 0.0f32;
        let mut selected = NUM_DIR_BINS - 1;
        for (d, &prob) in cell.iter().enumerate() {
            cumulative += prob;
            if xi < cumulative {
                selected = d;
                break;
            }
        }

        // Decode bin -> (z_band, toward_sun)
        let z_band = selected / 2;
        let toward_sun = selected.is_multiple_of(2);

        // Sample cos_zenith uniformly within the band.
        let (cos_z_lo, cos_z_hi) = match z_band {
            0 => (COS_Z_BOUNDS[0], 1.0),
            1 => (COS_Z_BOUNDS[1], COS_Z_BOUNDS[0]),
            2 => (COS_Z_BOUNDS[2], COS_Z_BOUNDS[1]),
            _ => (-1.0, COS_Z_BOUNDS[2]),
        };
        let xi_z = xorshift_f64(rng);
        let cos_z = cos_z_lo + (cos_z_hi - cos_z_lo) * xi_z;
        let sin_z = libm::sqrt((1.0 - cos_z * cos_z).max(0.0));

        // Sample azimuth uniformly within the half-plane.
        let xi_phi = xorshift_f64(rng);
        let phi_half = xi_phi * core::f64::consts::PI;
        let phi = if toward_sun {
            phi_half
        } else {
            phi_half + core::f64::consts::PI
        };

        // Build direction in local frame (up, sun_horiz, cross).
        let sun_h = Self::sun_horiz(local_up, sun_dir);
        let cross = local_up.cross(sun_h);

        let cos_phi = libm::cos(phi);
        let sin_phi = libm::sin(phi);

        let dir = Vec3::new(
            cos_z * local_up.x + sin_z * (cos_phi * sun_h.x + sin_phi * cross.x),
            cos_z * local_up.y + sin_z * (cos_phi * sun_h.y + sin_phi * cross.y),
            cos_z * local_up.z + sin_z * (cos_phi * sun_h.z + sin_phi * cross.z),
        );

        let pdf_val = cell[selected] as f64 / BIN_SOLID_ANGLE;
        (dir, pdf_val)
    }

    /// Get the raw bin probability (not divided by solid angle).
    #[inline]
    pub fn bin_probability(&self, alt_m: f64, local_up: Vec3, sun_dir: Vec3, dir: Vec3) -> f64 {
        let (a, s, d) = Self::lookup(alt_m, local_up, sun_dir, dir);
        self.table[a][s][d] as f64
    }

    /// Total memory footprint in bytes.
    pub const fn memory_bytes() -> usize {
        NUM_ALT_BINS * NUM_SOLAR_BINS * NUM_DIR_BINS * core::mem::size_of::<f32>()
    }

    /// Check if a spatial cell has non-uniform distribution (useful for testing).
    pub fn cell_entropy(&self, alt_bin: usize, solar_bin: usize) -> f64 {
        let cell = &self.table[alt_bin][solar_bin];
        let mut entropy = 0.0f64;
        for &val in cell.iter() {
            let p = val as f64;
            if p > 1e-10 {
                entropy -= p * libm::log(p);
            }
        }
        entropy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_uniform() {
        let guide = PathGuide::new();
        let expected = 1.0 / NUM_DIR_BINS as f32;
        for a in 0..NUM_ALT_BINS {
            for s in 0..NUM_SOLAR_BINS {
                for d in 0..NUM_DIR_BINS {
                    assert!(
                        (guide.table[a][s][d] - expected).abs() < 1e-7,
                        "Non-uniform at [{a}][{s}][{d}]: {}",
                        guide.table[a][s][d]
                    );
                }
            }
        }
    }

    #[test]
    fn new_is_not_trained() {
        let guide = PathGuide::new();
        assert!(!guide.is_trained());
    }

    #[test]
    fn normalize_sets_trained() {
        let mut guide = PathGuide::new();
        assert!(!guide.is_trained());
        guide.normalize();
        assert!(guide.is_trained());
    }

    #[test]
    fn reset_clears_trained() {
        let mut guide = PathGuide::new();
        guide.normalize();
        assert!(guide.is_trained());
        guide.reset();
        assert!(!guide.is_trained());
    }

    #[test]
    fn normalize_sums_to_one() {
        let mut guide = PathGuide::new();
        // Accumulate some non-uniform data.
        let up = Vec3::new(1.0, 0.0, 0.0);
        let sun = Vec3::new(0.0, 1.0, 0.0);
        guide.accumulate(30_000.0, up, sun, Vec3::new(1.0, 0.0, 0.0), 10.0);
        guide.accumulate(30_000.0, up, sun, Vec3::new(0.0, 1.0, 0.0), 5.0);
        guide.accumulate(30_000.0, up, sun, Vec3::new(-1.0, 0.0, 0.0), 1.0);
        guide.normalize();

        for a in 0..NUM_ALT_BINS {
            for s in 0..NUM_SOLAR_BINS {
                let sum: f32 = guide.table[a][s].iter().sum();
                assert!((sum - 1.0).abs() < 1e-5, "Cell [{a}][{s}] sums to {sum}");
            }
        }
    }

    #[test]
    fn accumulate_increases_correct_bin() {
        let mut guide = PathGuide::new();
        guide.reset(); // zero everything

        let up = Vec3::new(1.0, 0.0, 0.0);
        let sun = Vec3::new(0.0, 1.0, 0.0);
        let dir = Vec3::new(0.8, 0.6, 0.0); // upward, toward sun

        let (a, s, d) = PathGuide::lookup(50_000.0, up, sun, dir);
        guide.accumulate(50_000.0, up, sun, dir, 42.0);

        assert!(
            (guide.table[a][s][d] - 42.0).abs() < 1e-5,
            "Accumulated value should be 42.0, got {}",
            guide.table[a][s][d]
        );
        // Other bins should be zero.
        for d2 in 0..NUM_DIR_BINS {
            if d2 != d {
                assert!(
                    guide.table[a][s][d2].abs() < 1e-10,
                    "Bin {d2} should be 0, got {}",
                    guide.table[a][s][d2]
                );
            }
        }
    }

    #[test]
    fn alt_bin_clamped() {
        assert_eq!(PathGuide::alt_bin(0.0), 0);
        assert_eq!(PathGuide::alt_bin(50_000.0), 16);
        assert_eq!(PathGuide::alt_bin(200_000.0), NUM_ALT_BINS - 1);
        assert_eq!(PathGuide::alt_bin(-100.0), 0); // negative -> 0
    }

    #[test]
    fn solar_bin_range() {
        // Sun overhead (cos=1) -> bin 0
        assert_eq!(PathGuide::solar_bin(1.0), 0);
        // Sun at nadir (cos=-1) -> bin 15
        assert_eq!(PathGuide::solar_bin(-1.0), NUM_SOLAR_BINS - 1);
        // Horizon (cos=0) -> bin 8
        assert_eq!(PathGuide::solar_bin(0.0), NUM_SOLAR_BINS / 2);
    }

    #[test]
    fn dir_bin_8_bins() {
        // Strongly upward, toward sun
        assert_eq!(PathGuide::dir_bin(0.8, true), 0);
        // Strongly upward, away from sun
        assert_eq!(PathGuide::dir_bin(0.8, false), 1);
        // Moderately upward, toward sun
        assert_eq!(PathGuide::dir_bin(0.3, true), 2);
        // Moderately downward, toward sun
        assert_eq!(PathGuide::dir_bin(-0.3, true), 4);
        // Strongly downward, away from sun
        assert_eq!(PathGuide::dir_bin(-0.8, false), 7);
    }

    #[test]
    fn sample_returns_unit_vector() {
        let guide = PathGuide::new();
        let up = Vec3::new(0.0, 0.0, 1.0);
        let sun = Vec3::new(1.0, 0.0, 0.0);
        let mut rng: u64 = 42;

        for _ in 0..100 {
            let (dir, pdf) = guide.sample(30_000.0, up, sun, &mut rng);
            let len = dir.length();
            assert!(
                (len - 1.0).abs() < 1e-10,
                "Direction should be unit vector, length = {len}"
            );
            assert!(pdf > 0.0, "PDF should be positive, got {pdf}");
            assert!(pdf.is_finite(), "PDF should be finite, got {pdf}");
        }
    }

    #[test]
    fn pdf_positive_for_all_directions() {
        let mut guide = PathGuide::new();
        guide.normalize(); // Laplace smoothing ensures no zeros

        let up = Vec3::new(0.0, 0.0, 1.0);
        let sun = Vec3::new(1.0, 0.0, 0.0);

        // Test various directions
        let dirs = [
            Vec3::new(0.0, 0.0, 1.0),   // up
            Vec3::new(0.0, 0.0, -1.0),  // down
            Vec3::new(1.0, 0.0, 0.0),   // toward sun
            Vec3::new(-1.0, 0.0, 0.0),  // away from sun
            Vec3::new(0.5, 0.5, 0.707), // diagonal
        ];

        for dir in &dirs {
            let p = guide.pdf(30_000.0, up, sun, *dir);
            assert!(p > 0.0, "PDF should be > 0 for dir {:?}, got {}", dir, p);
        }
    }

    #[test]
    fn uniform_pdf_integrates_to_one() {
        let guide = PathGuide::new();
        let _up = Vec3::new(0.0, 0.0, 1.0);
        let _sun = Vec3::new(1.0, 0.0, 0.0);

        // Each bin has probability 1/8, solid angle pi/2.
        // PDF = (1/8) / (pi/2) = 1/(4*pi).
        // Integral = 8 * (1/8) / (pi/2) * (pi/2) = 1.0.
        // Or equivalently: sum of bin_probs = 1.0.
        let mut sum = 0.0;
        for d in 0..NUM_DIR_BINS {
            let prob = guide.table[0][0][d] as f64;
            sum += prob;
        }
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Bin probabilities should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn trained_guide_has_structure() {
        let mut guide = PathGuide::new();
        guide.reset();

        // Simulate training: strongly upward directions toward the sun
        // are most important at high altitude and deep twilight.
        let up = Vec3::new(0.0, 0.0, 1.0);
        let sun = Vec3::new(0.1, 0.0, -0.995); // sun below horizon
        let upward_sun = Vec3::new(0.3, 0.0, 0.954).normalize(); // up + toward sun

        guide.accumulate(60_000.0, up, sun, upward_sun, 100.0);
        guide.normalize();

        // The bin containing upward_sun should have higher probability
        // than other bins in the same cell.
        let (a, s, d_up) = PathGuide::lookup(60_000.0, up, sun, upward_sun);
        let prob_up = guide.table[a][s][d_up];

        let downward = Vec3::new(0.0, 0.0, -1.0);
        let (_, _, d_down) = PathGuide::lookup(60_000.0, up, sun, downward);
        let prob_down = guide.table[a][s][d_down];

        assert!(
            prob_up > prob_down,
            "Upward bin ({}) should be more probable than downward ({}) after training",
            prob_up,
            prob_down
        );
    }

    #[test]
    fn sample_respects_distribution() {
        let mut guide = PathGuide::new();
        guide.reset();

        // Train: heavily weight upward directions.
        let up = Vec3::new(0.0, 0.0, 1.0);
        let sun = Vec3::new(1.0, 0.0, 0.0);

        for _ in 0..1000 {
            guide.accumulate(50_000.0, up, sun, Vec3::new(0.0, 0.0, 1.0), 100.0);
        }
        guide.normalize();

        // Sample many directions and count upward vs downward.
        let mut rng: u64 = 123;
        let mut upward_count = 0;
        let n = 1000;
        for _ in 0..n {
            let (dir, _) = guide.sample(50_000.0, up, sun, &mut rng);
            if dir.z > 0.0 {
                upward_count += 1;
            }
        }

        // With heavy upward training, most samples should be upward.
        let frac = upward_count as f64 / n as f64;
        assert!(
            frac > 0.6,
            "Expected >60% upward samples after upward training, got {:.1}%",
            frac * 100.0
        );
    }

    #[test]
    fn memory_footprint() {
        let bytes = PathGuide::memory_bytes();
        assert_eq!(bytes, 16384, "Expected 16 KiB, got {} bytes", bytes);
    }

    #[test]
    fn laplace_smoothing_prevents_zero_bins() {
        let mut guide = PathGuide::new();
        guide.reset();
        // Only accumulate into one bin.
        let up = Vec3::new(0.0, 0.0, 1.0);
        let sun = Vec3::new(1.0, 0.0, 0.0);
        guide.accumulate(50_000.0, up, sun, Vec3::new(0.0, 0.0, 1.0), 1000.0);
        guide.normalize();

        // All bins should be > 0 due to Laplace smoothing.
        let (a, s, _) = PathGuide::lookup(50_000.0, up, sun, Vec3::new(0.0, 0.0, 1.0));
        for d in 0..NUM_DIR_BINS {
            assert!(
                guide.table[a][s][d] > 0.0,
                "Bin [{a}][{s}][{d}] should be > 0 after smoothing, got {}",
                guide.table[a][s][d]
            );
        }
    }

    #[test]
    fn sun_horiz_perpendicular_to_up() {
        let up = Vec3::new(0.0, 0.0, 1.0);
        let sun = Vec3::new(0.3, 0.4, -0.866);
        let sh = PathGuide::sun_horiz(up, sun);
        let dot = sh.dot(up);
        assert!(
            dot.abs() < 1e-10,
            "sun_horiz should be perpendicular to up, dot = {dot}"
        );
        let len = sh.length();
        assert!(
            (len - 1.0).abs() < 1e-10,
            "sun_horiz should be unit, length = {len}"
        );
    }

    #[test]
    fn sun_horiz_fallback_at_zenith() {
        let up = Vec3::new(0.0, 0.0, 1.0);
        let sun = Vec3::new(0.0, 0.0, 1.0); // sun at zenith
        let sh = PathGuide::sun_horiz(up, sun);
        let dot = sh.dot(up);
        assert!(
            dot.abs() < 1e-10,
            "Fallback sun_horiz should be perpendicular, dot = {dot}"
        );
    }

    #[test]
    fn cell_entropy_uniform_is_max() {
        let guide = PathGuide::new();
        let max_entropy = libm::log(NUM_DIR_BINS as f64);
        let entropy = guide.cell_entropy(0, 0);
        assert!(
            (entropy - max_entropy).abs() < 1e-6,
            "Uniform guide should have max entropy {max_entropy}, got {entropy}"
        );
    }

    #[test]
    fn cell_entropy_decreases_with_training() {
        let mut guide = PathGuide::new();
        let max_entropy = guide.cell_entropy(16, 8);

        guide.reset();
        let up = Vec3::new(1.0, 0.0, 0.0);
        let sun = Vec3::new(0.0, 1.0, 0.0);
        guide.accumulate(50_000.0, up, sun, Vec3::new(1.0, 0.0, 0.0), 100.0);
        guide.normalize();

        let (a, s, _) = PathGuide::lookup(50_000.0, up, sun, Vec3::new(1.0, 0.0, 0.0));
        let trained_entropy = guide.cell_entropy(a, s);
        assert!(
            trained_entropy < max_entropy,
            "Trained entropy ({trained_entropy}) should be less than uniform ({max_entropy})"
        );
    }
}
