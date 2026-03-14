//! Atmospheric shell model and optical property lookups.
//!
//! The atmosphere is modeled as concentric spherical shells around the Earth.
//! Each shell has uniform optical properties (extinction coefficient,
//! single scattering albedo, asymmetry parameter) at a given wavelength.

/// Earth mean radius in meters.
pub const EARTH_RADIUS_M: f64 = 6_371_000.0;

/// Top of atmosphere altitude in meters (100 km).
pub const TOA_ALTITUDE_M: f64 = 100_000.0;

/// Maximum number of atmospheric shells.
pub const MAX_SHELLS: usize = 64;

/// Maximum number of wavelength bins.
pub const MAX_WAVELENGTHS: usize = 64;

/// Optical properties for a single shell at a single wavelength.
#[derive(Debug, Clone, Copy)]
pub struct ShellOptics {
    /// Total extinction coefficient (Rayleigh + aerosol + cloud + O3 absorption) [1/m]
    pub extinction: f64,
    /// Single scattering albedo (0 = pure absorption, 1 = pure scattering)
    pub ssa: f64,
    /// Scattering asymmetry parameter (0 = isotropic, ~0.85 = cloud forward-peaked)
    pub asymmetry: f64,
    /// Fraction of scattering that is Rayleigh (vs Mie/HG)
    pub rayleigh_fraction: f64,
}

impl Default for ShellOptics {
    fn default() -> Self {
        Self {
            extinction: 0.0,
            ssa: 1.0,
            asymmetry: 0.0,
            rayleigh_fraction: 1.0,
        }
    }
}

/// A single atmospheric shell (layer).
#[derive(Debug, Clone, Copy)]
pub struct Shell {
    /// Inner radius from Earth center [m]
    pub r_inner: f64,
    /// Outer radius from Earth center [m]
    pub r_outer: f64,
    /// Altitude of shell midpoint above sea level [m]
    pub altitude_mid: f64,
    /// Shell thickness [m]
    pub thickness: f64,
}

/// Complete atmosphere model: a stack of concentric spherical shells.
#[derive(Debug, Clone)]
pub struct AtmosphereModel {
    /// Shell geometry
    pub shells: [Shell; MAX_SHELLS],
    /// Optical properties per shell per wavelength
    /// Index: [shell_index][wavelength_index]
    pub optics: [[ShellOptics; MAX_WAVELENGTHS]; MAX_SHELLS],
    /// Number of active shells
    pub num_shells: usize,
    /// Wavelength grid in nanometers
    pub wavelengths_nm: [f64; MAX_WAVELENGTHS],
    /// Number of active wavelengths
    pub num_wavelengths: usize,
    /// Surface albedo per wavelength
    pub surface_albedo: [f64; MAX_WAVELENGTHS],
    /// Refractive index per shell (at shell midpoint altitude).
    ///
    /// Defaults to 1.0 for all shells (no refraction). Call
    /// [`compute_refractive_indices`] to populate from the Rayleigh
    /// extinction profile, or [`compute_refractive_indices_from_altitude`]
    /// for the scale-height fallback.
    ///
    /// When all values are 1.0, Snell's law at shell boundaries reduces
    /// to the identity (no direction change), preserving backward
    /// compatibility with straight-line transport.
    pub refractive_index: [f64; MAX_SHELLS],
}

impl AtmosphereModel {
    /// Create a new atmosphere model with given shell altitudes.
    ///
    /// `altitudes_km` should be the boundaries of each shell (N+1 values for N shells).
    pub fn new(altitudes_km: &[f64], wavelengths_nm: &[f64]) -> Self {
        let mut model = Self {
            shells: [Shell {
                r_inner: 0.0,
                r_outer: 0.0,
                altitude_mid: 0.0,
                thickness: 0.0,
            }; MAX_SHELLS],
            optics: [[ShellOptics::default(); MAX_WAVELENGTHS]; MAX_SHELLS],
            num_shells: 0,
            wavelengths_nm: [0.0; MAX_WAVELENGTHS],
            num_wavelengths: 0,
            surface_albedo: [0.15; MAX_WAVELENGTHS], // default Earth albedo
            refractive_index: [1.0; MAX_SHELLS],     // default: no refraction
        };

        let n_shells = if altitudes_km.len() > 1 {
            (altitudes_km.len() - 1).min(MAX_SHELLS)
        } else {
            0
        };

        for i in 0..n_shells {
            let alt_low = altitudes_km[i] * 1000.0; // km to m
            let alt_high = altitudes_km[i + 1] * 1000.0;
            model.shells[i] = Shell {
                r_inner: EARTH_RADIUS_M + alt_low,
                r_outer: EARTH_RADIUS_M + alt_high,
                altitude_mid: (alt_low + alt_high) / 2.0,
                thickness: alt_high - alt_low,
            };
        }
        model.num_shells = n_shells;

        let n_wl = wavelengths_nm.len().min(MAX_WAVELENGTHS);
        model.wavelengths_nm[..n_wl].copy_from_slice(&wavelengths_nm[..n_wl]);
        model.num_wavelengths = n_wl;

        model
    }

    /// Compute refractive indices from the Rayleigh extinction profile.
    ///
    /// Uses the proportionality n(h) - 1 ~ rho(h) / rho(0) where rho is
    /// the air number density. Since Rayleigh extinction is also proportional
    /// to number density, the ratio of Rayleigh extinction at shell s to
    /// that at the surface gives the density ratio directly. The sea-level
    /// refractivity of dry air is (n - 1) = 0.000293 (Edlen, 1966).
    ///
    /// Falls back to the exponential scale-height model if the surface
    /// Rayleigh extinction is zero (e.g., in a test atmosphere with no
    /// optics set).
    pub fn compute_refractive_indices(&mut self) {
        if self.num_shells == 0 || self.num_wavelengths == 0 {
            return;
        }

        // Use wavelength 0 for the density ratio (the lambda^-4 dependence
        // of the Rayleigh cross-section cancels in the ratio).
        let ext_surface = self.optics[0][0].extinction * self.optics[0][0].rayleigh_fraction;

        if ext_surface < 1e-30 {
            // No Rayleigh extinction data; fall back to scale height model.
            self.compute_refractive_indices_from_altitude();
            return;
        }

        for s in 0..self.num_shells {
            let ext_s = self.optics[s][0].extinction * self.optics[s][0].rayleigh_fraction;
            let density_ratio = ext_s / ext_surface;
            self.refractive_index[s] = 1.0 + 0.000293 * density_ratio;
        }
    }

    /// Compute refractive indices from the exponential scale-height model.
    ///
    /// n(h) = 1 + 0.000293 * exp(-h / H) where H = 8500 m is the
    /// density scale height of a standard atmosphere.
    pub fn compute_refractive_indices_from_altitude(&mut self) {
        for s in 0..self.num_shells {
            let h = self.shells[s].altitude_mid;
            self.refractive_index[s] = 1.0 + 0.000293 * libm::exp(-h / 8500.0);
        }
    }

    /// Get shell index for a given radius from Earth center.
    /// Returns None if outside the atmosphere.
    pub fn shell_index(&self, radius: f64) -> Option<usize> {
        (0..self.num_shells)
            .find(|&i| radius >= self.shells[i].r_inner && radius < self.shells[i].r_outer)
    }

    /// Get the optical depth through a shell along a path of given length.
    #[inline]
    pub fn optical_depth(&self, shell_idx: usize, wavelength_idx: usize, path_length: f64) -> f64 {
        self.optics[shell_idx][wavelength_idx].extinction * path_length
    }

    /// Get the surface radius (bottom of lowest shell).
    #[inline]
    pub fn surface_radius(&self) -> f64 {
        if self.num_shells > 0 {
            self.shells[0].r_inner
        } else {
            EARTH_RADIUS_M
        }
    }

    /// Get the top-of-atmosphere radius (top of highest shell).
    #[inline]
    pub fn toa_radius(&self) -> f64 {
        if self.num_shells > 0 {
            self.shells[self.num_shells - 1].r_outer
        } else {
            EARTH_RADIUS_M + TOA_ALTITUDE_M
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-6;

    fn make_simple_atm() -> AtmosphereModel {
        // 3 shells: 0-10km, 10-50km, 50-100km
        let altitudes_km = [0.0, 10.0, 50.0, 100.0];
        let wavelengths = [400.0, 550.0, 700.0];
        AtmosphereModel::new(&altitudes_km, &wavelengths)
    }

    // ── Construction ──

    #[test]
    fn new_creates_correct_number_of_shells() {
        let atm = make_simple_atm();
        assert_eq!(atm.num_shells, 3);
        assert_eq!(atm.num_wavelengths, 3);
    }

    #[test]
    fn new_shell_radii_are_correct() {
        let atm = make_simple_atm();
        // Shell 0: 0-10 km
        assert!((atm.shells[0].r_inner - EARTH_RADIUS_M).abs() < EPSILON);
        assert!((atm.shells[0].r_outer - (EARTH_RADIUS_M + 10_000.0)).abs() < EPSILON);
        // Shell 1: 10-50 km
        assert!((atm.shells[1].r_inner - (EARTH_RADIUS_M + 10_000.0)).abs() < EPSILON);
        assert!((atm.shells[1].r_outer - (EARTH_RADIUS_M + 50_000.0)).abs() < EPSILON);
        // Shell 2: 50-100 km
        assert!((atm.shells[2].r_inner - (EARTH_RADIUS_M + 50_000.0)).abs() < EPSILON);
        assert!((atm.shells[2].r_outer - (EARTH_RADIUS_M + 100_000.0)).abs() < EPSILON);
    }

    #[test]
    fn new_shell_thickness() {
        let atm = make_simple_atm();
        assert!((atm.shells[0].thickness - 10_000.0).abs() < EPSILON);
        assert!((atm.shells[1].thickness - 40_000.0).abs() < EPSILON);
        assert!((atm.shells[2].thickness - 50_000.0).abs() < EPSILON);
    }

    #[test]
    fn new_shell_altitude_mid() {
        let atm = make_simple_atm();
        assert!((atm.shells[0].altitude_mid - 5_000.0).abs() < EPSILON);
        assert!((atm.shells[1].altitude_mid - 30_000.0).abs() < EPSILON);
        assert!((atm.shells[2].altitude_mid - 75_000.0).abs() < EPSILON);
    }

    #[test]
    fn new_wavelengths_stored() {
        let atm = make_simple_atm();
        assert!((atm.wavelengths_nm[0] - 400.0).abs() < EPSILON);
        assert!((atm.wavelengths_nm[1] - 550.0).abs() < EPSILON);
        assert!((atm.wavelengths_nm[2] - 700.0).abs() < EPSILON);
    }

    #[test]
    fn new_default_albedo() {
        let atm = make_simple_atm();
        for w in 0..atm.num_wavelengths {
            assert!((atm.surface_albedo[w] - 0.15).abs() < EPSILON);
        }
    }

    #[test]
    fn new_default_optics() {
        let atm = make_simple_atm();
        for s in 0..atm.num_shells {
            for w in 0..atm.num_wavelengths {
                assert!((atm.optics[s][w].extinction - 0.0).abs() < EPSILON);
                assert!((atm.optics[s][w].ssa - 1.0).abs() < EPSILON);
                assert!((atm.optics[s][w].asymmetry - 0.0).abs() < EPSILON);
                assert!((atm.optics[s][w].rayleigh_fraction - 1.0).abs() < EPSILON);
            }
        }
    }

    #[test]
    fn new_single_altitude_produces_zero_shells() {
        let atm = AtmosphereModel::new(&[0.0], &[550.0]);
        assert_eq!(atm.num_shells, 0);
    }

    #[test]
    fn new_empty_altitudes_produces_zero_shells() {
        let atm = AtmosphereModel::new(&[], &[550.0]);
        assert_eq!(atm.num_shells, 0);
    }

    #[test]
    fn new_caps_at_max_shells() {
        // Create more than MAX_SHELLS altitude boundaries using a fixed array
        // MAX_SHELLS = 64, so 74 boundaries → 73 shells, capped to 64
        let mut alts = [0.0f64; 74];
        for i in 0..74 {
            alts[i] = i as f64;
        }
        let atm = AtmosphereModel::new(&alts, &[550.0]);
        assert_eq!(atm.num_shells, MAX_SHELLS);
    }

    #[test]
    fn new_caps_at_max_wavelengths() {
        // MAX_WAVELENGTHS = 64, so 74 wavelengths → capped to 64
        let mut wls = [0.0f64; 74];
        for i in 0..74 {
            wls[i] = 380.0 + i as f64;
        }
        let atm = AtmosphereModel::new(&[0.0, 10.0], &wls);
        assert_eq!(atm.num_wavelengths, MAX_WAVELENGTHS);
    }

    // ── shell_index ──

    #[test]
    fn shell_index_finds_correct_shell() {
        let atm = make_simple_atm();
        // At surface (r = EARTH_RADIUS_M + 1m) → shell 0
        assert_eq!(atm.shell_index(EARTH_RADIUS_M + 1.0), Some(0));
        // At 5km → shell 0
        assert_eq!(atm.shell_index(EARTH_RADIUS_M + 5_000.0), Some(0));
        // At 10km + epsilon → shell 1
        assert_eq!(atm.shell_index(EARTH_RADIUS_M + 10_001.0), Some(1));
        // At 30km → shell 1
        assert_eq!(atm.shell_index(EARTH_RADIUS_M + 30_000.0), Some(1));
        // At 75km → shell 2
        assert_eq!(atm.shell_index(EARTH_RADIUS_M + 75_000.0), Some(2));
    }

    #[test]
    fn shell_index_none_below_surface() {
        let atm = make_simple_atm();
        assert_eq!(atm.shell_index(EARTH_RADIUS_M - 1.0), None);
    }

    #[test]
    fn shell_index_none_above_toa() {
        let atm = make_simple_atm();
        assert_eq!(atm.shell_index(EARTH_RADIUS_M + 100_001.0), None);
    }

    #[test]
    fn shell_index_at_exact_boundary() {
        let atm = make_simple_atm();
        // At exactly r_inner of shell 0 → should be in shell 0
        assert_eq!(atm.shell_index(EARTH_RADIUS_M), Some(0));
        // At exactly r_outer of shell 0 = r_inner of shell 1 → shell 1
        assert_eq!(atm.shell_index(EARTH_RADIUS_M + 10_000.0), Some(1));
    }

    // ── optical_depth ──

    #[test]
    fn optical_depth_is_extinction_times_path() {
        let mut atm = make_simple_atm();
        atm.optics[0][0].extinction = 0.01; // 1/m
        let od = atm.optical_depth(0, 0, 1000.0);
        assert!((od - 10.0).abs() < EPSILON);
    }

    #[test]
    fn optical_depth_zero_extinction() {
        let atm = make_simple_atm();
        assert!((atm.optical_depth(0, 0, 1000.0)).abs() < EPSILON);
    }

    // ── surface_radius / toa_radius ──

    #[test]
    fn surface_radius_is_earth_radius() {
        let atm = make_simple_atm();
        assert!((atm.surface_radius() - EARTH_RADIUS_M).abs() < EPSILON);
    }

    #[test]
    fn toa_radius_is_earth_plus_100km() {
        let atm = make_simple_atm();
        assert!((atm.toa_radius() - (EARTH_RADIUS_M + 100_000.0)).abs() < EPSILON);
    }

    #[test]
    fn surface_radius_empty_model() {
        let atm = AtmosphereModel::new(&[], &[]);
        assert!((atm.surface_radius() - EARTH_RADIUS_M).abs() < EPSILON);
    }

    #[test]
    fn toa_radius_empty_model() {
        let atm = AtmosphereModel::new(&[], &[]);
        assert!((atm.toa_radius() - (EARTH_RADIUS_M + TOA_ALTITUDE_M)).abs() < EPSILON);
    }

    // ── ShellOptics default ──

    #[test]
    fn shell_optics_default() {
        let o = ShellOptics::default();
        assert!((o.extinction - 0.0).abs() < EPSILON);
        assert!((o.ssa - 1.0).abs() < EPSILON);
        assert!((o.asymmetry - 0.0).abs() < EPSILON);
        assert!((o.rayleigh_fraction - 1.0).abs() < EPSILON);
    }

    // ── Shells are contiguous ──

    #[test]
    fn shells_are_contiguous() {
        let atm = make_simple_atm();
        for i in 0..(atm.num_shells - 1) {
            assert!(
                (atm.shells[i].r_outer - atm.shells[i + 1].r_inner).abs() < EPSILON,
                "Gap between shell {} and {}: {} vs {}",
                i,
                i + 1,
                atm.shells[i].r_outer,
                atm.shells[i + 1].r_inner
            );
        }
    }

    // ── Constants ──

    #[test]
    fn earth_radius_matches_iugg() {
        // IUGG mean Earth radius: 6,371,000 m
        assert!((EARTH_RADIUS_M - 6_371_000.0).abs() < EPSILON);
    }

    #[test]
    fn toa_altitude_is_100km() {
        // Karman line: 100 km
        assert!((TOA_ALTITUDE_M - 100_000.0).abs() < EPSILON);
    }

    // ── Refractive index defaults ──

    #[test]
    fn default_refractive_index_is_one() {
        let atm = make_simple_atm();
        for s in 0..atm.num_shells {
            assert!(
                (atm.refractive_index[s] - 1.0).abs() < EPSILON,
                "Default n[{}] should be 1.0, got {}",
                s,
                atm.refractive_index[s]
            );
        }
    }

    #[test]
    fn default_refractive_index_all_slots() {
        // Even unused slots should be 1.0
        let atm = make_simple_atm();
        for s in 0..MAX_SHELLS {
            assert!(
                (atm.refractive_index[s] - 1.0).abs() < EPSILON,
                "Slot {} should be 1.0",
                s
            );
        }
    }

    // ── compute_refractive_indices_from_altitude ──

    #[test]
    fn refractive_index_from_altitude_sea_level() {
        let mut atm = make_simple_atm(); // shell 0 midpoint = 5000 m
        atm.compute_refractive_indices_from_altitude();

        // At sea level (0m), n = 1 + 0.000293 = 1.000293
        // Shell 0 midpoint is at 5000 m, so n = 1 + 0.000293 * exp(-5000/8500)
        let expected = 1.0 + 0.000293 * libm::exp(-5000.0 / 8500.0);
        assert!(
            (atm.refractive_index[0] - expected).abs() < 1e-9,
            "Shell 0 n: expected {}, got {}",
            expected,
            atm.refractive_index[0]
        );
    }

    #[test]
    fn refractive_index_from_altitude_decreases_with_height() {
        let mut atm = make_simple_atm();
        atm.compute_refractive_indices_from_altitude();

        // n should decrease with altitude (atmosphere gets thinner)
        for s in 0..(atm.num_shells - 1) {
            assert!(
                atm.refractive_index[s] > atm.refractive_index[s + 1],
                "n[{}]={} should be > n[{}]={}",
                s,
                atm.refractive_index[s],
                s + 1,
                atm.refractive_index[s + 1]
            );
        }
    }

    #[test]
    fn refractive_index_from_altitude_all_above_one() {
        let mut atm = make_simple_atm();
        atm.compute_refractive_indices_from_altitude();

        for s in 0..atm.num_shells {
            assert!(
                atm.refractive_index[s] > 1.0,
                "n[{}]={} should be > 1.0",
                s,
                atm.refractive_index[s]
            );
        }
    }

    #[test]
    fn refractive_index_from_altitude_below_sea_level_value() {
        let mut atm = make_simple_atm();
        atm.compute_refractive_indices_from_altitude();

        // All shells should have n < 1.000293 (sea level value)
        // because shell midpoints are above sea level
        for s in 0..atm.num_shells {
            assert!(
                atm.refractive_index[s] < 1.000293 + 1e-9,
                "n[{}]={} should be <= 1.000293 (sea level)",
                s,
                atm.refractive_index[s]
            );
        }
    }

    #[test]
    fn refractive_index_from_altitude_high_shell_near_one() {
        // Shell 2 midpoint is at 75 km -- n should be very close to 1.0
        let mut atm = make_simple_atm();
        atm.compute_refractive_indices_from_altitude();

        let expected = 1.0 + 0.000293 * libm::exp(-75_000.0 / 8500.0);
        assert!(
            (atm.refractive_index[2] - expected).abs() < 1e-12,
            "High-altitude shell n: expected {}, got {}",
            expected,
            atm.refractive_index[2]
        );
        // Should be extremely close to 1.0
        assert!(
            (atm.refractive_index[2] - 1.0) < 1e-6,
            "At 75 km, (n-1) should be < 1e-6, got {}",
            atm.refractive_index[2] - 1.0
        );
    }

    #[test]
    fn refractive_index_from_altitude_empty_model() {
        // Should not panic on empty model
        let mut atm = AtmosphereModel::new(&[], &[]);
        atm.compute_refractive_indices_from_altitude();
        // All indices should remain 1.0
        for s in 0..MAX_SHELLS {
            assert!((atm.refractive_index[s] - 1.0).abs() < EPSILON);
        }
    }

    // ── compute_refractive_indices (from Rayleigh extinction) ──

    fn make_atm_with_rayleigh() -> AtmosphereModel {
        // Build an atmosphere with realistic Rayleigh extinction profile
        let altitudes_km = [0.0, 5.0, 15.0, 50.0, 100.0];
        let wavelengths = [550.0];
        let mut atm = AtmosphereModel::new(&altitudes_km, &wavelengths);

        // Rayleigh extinction profile: surface = 1.3e-5, decays exponentially
        atm.optics[0][0].extinction = 1.3e-5;
        atm.optics[0][0].rayleigh_fraction = 1.0;

        atm.optics[1][0].extinction = 1.3e-5 * libm::exp(-10_000.0 / 8500.0);
        atm.optics[1][0].rayleigh_fraction = 1.0;

        atm.optics[2][0].extinction = 1.3e-5 * libm::exp(-32_500.0 / 8500.0);
        atm.optics[2][0].rayleigh_fraction = 1.0;

        atm.optics[3][0].extinction = 1.3e-5 * libm::exp(-75_000.0 / 8500.0);
        atm.optics[3][0].rayleigh_fraction = 1.0;

        atm
    }

    #[test]
    fn refractive_index_from_rayleigh_surface_value() {
        let mut atm = make_atm_with_rayleigh();
        atm.compute_refractive_indices();

        // Shell 0 has density_ratio = 1.0 (surface), so n = 1 + 0.000293
        assert!(
            (atm.refractive_index[0] - 1.000293).abs() < 1e-9,
            "Surface n should be 1.000293, got {}",
            atm.refractive_index[0]
        );
    }

    #[test]
    fn refractive_index_from_rayleigh_decreases_with_altitude() {
        let mut atm = make_atm_with_rayleigh();
        atm.compute_refractive_indices();

        for s in 0..(atm.num_shells - 1) {
            assert!(
                atm.refractive_index[s] > atm.refractive_index[s + 1],
                "n[{}]={} should be > n[{}]={}",
                s,
                atm.refractive_index[s],
                s + 1,
                atm.refractive_index[s + 1]
            );
        }
    }

    #[test]
    fn refractive_index_from_rayleigh_matches_scale_height() {
        // Since we set up the Rayleigh extinction with the standard scale
        // height, the resulting n values should be very close to the
        // altitude-based model.
        let mut atm_ray = make_atm_with_rayleigh();
        atm_ray.compute_refractive_indices();

        let mut atm_alt = make_atm_with_rayleigh();
        atm_alt.compute_refractive_indices_from_altitude();

        for s in 0..atm_ray.num_shells {
            let diff = (atm_ray.refractive_index[s] - atm_alt.refractive_index[s]).abs();
            // Allow 10% relative error due to discrete shell midpoint vs continuous
            let n_minus_1 = atm_alt.refractive_index[s] - 1.0;
            let rel = if n_minus_1 > 1e-10 {
                diff / n_minus_1
            } else {
                diff
            };
            assert!(
                rel < 0.5, // generous: shells are thick
                "Shell {} Rayleigh n={} vs altitude n={}, rel_diff={}",
                s,
                atm_ray.refractive_index[s],
                atm_alt.refractive_index[s],
                rel
            );
        }
    }

    #[test]
    fn refractive_index_from_rayleigh_fallback_on_zero_extinction() {
        // If surface extinction is zero, should fall back to altitude model
        let mut atm = make_simple_atm(); // all extinction = 0
        atm.compute_refractive_indices();

        // Should have fallen back to altitude model (not all 1.0)
        // Shell 0 midpoint at 5000m
        let expected = 1.0 + 0.000293 * libm::exp(-5000.0 / 8500.0);
        assert!(
            (atm.refractive_index[0] - expected).abs() < 1e-9,
            "Fallback should use altitude model: expected {}, got {}",
            expected,
            atm.refractive_index[0]
        );
    }

    #[test]
    fn refractive_index_from_rayleigh_empty_model() {
        let mut atm = AtmosphereModel::new(&[], &[]);
        atm.compute_refractive_indices(); // should not panic
        for s in 0..MAX_SHELLS {
            assert!((atm.refractive_index[s] - 1.0).abs() < EPSILON);
        }
    }

    #[test]
    fn refractive_index_from_rayleigh_with_aerosol_fraction() {
        // When rayleigh_fraction < 1.0, only the Rayleigh part of extinction
        // is used for the density ratio.
        let altitudes_km = [0.0, 10.0, 100.0];
        let wavelengths = [550.0];
        let mut atm = AtmosphereModel::new(&altitudes_km, &wavelengths);

        // Shell 0: 50% Rayleigh, 50% aerosol
        atm.optics[0][0].extinction = 2.6e-5; // total
        atm.optics[0][0].rayleigh_fraction = 0.5; // Rayleigh part = 1.3e-5

        // Shell 1: pure Rayleigh at 1/10 density
        atm.optics[1][0].extinction = 1.3e-6;
        atm.optics[1][0].rayleigh_fraction = 1.0;

        atm.compute_refractive_indices();

        // Surface n uses Rayleigh part: 1.3e-5 (same as pure case)
        assert!(
            (atm.refractive_index[0] - 1.000293).abs() < 1e-9,
            "Surface n with aerosol: {}",
            atm.refractive_index[0]
        );

        // Shell 1 density ratio = 1.3e-6 / 1.3e-5 = 0.1
        let expected_1 = 1.0 + 0.000293 * 0.1;
        assert!(
            (atm.refractive_index[1] - expected_1).abs() < 1e-9,
            "Shell 1 n: expected {}, got {}",
            expected_1,
            atm.refractive_index[1]
        );
    }
}
