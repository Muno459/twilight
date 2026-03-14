//! CPU f64 oracle: generates reference test vectors for GPU validation.
//!
//! The CPU engine's f64 results are the source of truth. This module
//! generates deterministic test cases covering the full physics:
//!
//! - Ray-sphere intersection geometry
//! - Phase functions (Rayleigh, Henyey-Greenstein)
//! - Shadow ray transmittance
//! - Single-scatter radiance at various SZA
//! - Hybrid multi-scatter radiance
//! - xorshift RNG output sequences
//!
//! Each test case is a struct that GPU tests can compare against with
//! known tolerance bounds.

use twilight_core::atmosphere::{AtmosphereModel, ShellOptics, EARTH_RADIUS_M};
use twilight_core::geometry::{ray_sphere_intersect, solar_direction_ecef, Vec3};
use twilight_core::photon::xorshift_f64;
use twilight_core::scattering::{henyey_greenstein_phase, rayleigh_phase};
use twilight_core::single_scatter::{
    shadow_ray_transmittance, single_scatter_radiance, single_scatter_spectrum,
};

// ── Oracle atmosphere builder ───────────────────────────────────────────

/// Build a standardized test atmosphere used by all oracle test vectors.
///
/// This is the canonical atmosphere for GPU validation:
/// - 3 shells: 0-10km, 10-50km, 50-100km
/// - 3 wavelengths: 400nm, 550nm, 700nm
/// - Rayleigh-dominated with wavelength-dependent extinction
/// - Surface albedo 0.15
pub fn oracle_atmosphere() -> AtmosphereModel {
    let altitudes_km = [0.0, 10.0, 50.0, 100.0];
    let wavelengths = [400.0, 550.0, 700.0];
    let mut atm = AtmosphereModel::new(&altitudes_km, &wavelengths);

    // Shell 0 (0-10km): dense troposphere
    for w in 0..3 {
        let lambda_factor = match w {
            0 => 4.0, // 400nm: strong Rayleigh
            1 => 1.0, // 550nm: reference
            _ => 0.3, // 700nm: weak Rayleigh
        };
        atm.optics[0][w] = ShellOptics {
            extinction: 1e-5 * lambda_factor,
            ssa: 1.0,
            asymmetry: 0.0,
            rayleigh_fraction: 1.0,
        };
    }
    // Shell 1 (10-50km): stratosphere
    for w in 0..3 {
        let lambda_factor = match w {
            0 => 4.0,
            1 => 1.0,
            _ => 0.3,
        };
        atm.optics[1][w] = ShellOptics {
            extinction: 1e-6 * lambda_factor,
            ssa: 1.0,
            asymmetry: 0.0,
            rayleigh_fraction: 1.0,
        };
    }
    // Shell 2 (50-100km): mesosphere
    for w in 0..3 {
        let lambda_factor = match w {
            0 => 4.0,
            1 => 1.0,
            _ => 0.3,
        };
        atm.optics[2][w] = ShellOptics {
            extinction: 1e-8 * lambda_factor,
            ssa: 1.0,
            asymmetry: 0.0,
            rayleigh_fraction: 1.0,
        };
    }

    atm
}

/// Build a test atmosphere with aerosols (non-zero asymmetry, mixed phase).
pub fn oracle_atmosphere_aerosol() -> AtmosphereModel {
    let mut atm = oracle_atmosphere();

    // Add aerosol to the troposphere (shell 0)
    for w in 0..3 {
        atm.optics[0][w].extinction += 5e-6; // add aerosol extinction
        atm.optics[0][w].ssa = 0.92;
        atm.optics[0][w].asymmetry = 0.65;
        atm.optics[0][w].rayleigh_fraction = 0.6;
    }

    atm
}

// ── Oracle test vector types ────────────────────────────────────────────

/// A single ray-sphere intersection test case.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct RaySphereCase {
    pub origin: [f64; 3],
    pub direction: [f64; 3],
    pub radius: f64,
    pub hits: bool,
    pub t_near: f64,
    pub t_far: f64,
}

/// A phase function test case.
#[derive(Debug, Clone)]
pub struct PhaseCase {
    pub cos_theta: f64,
    pub g: f64, // 0 for Rayleigh
    pub rayleigh_value: f64,
    pub hg_value: f64,
}

/// A shadow ray transmittance test case.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ShadowRayCase {
    pub scatter_pos: [f64; 3],
    pub sun_dir: [f64; 3],
    pub wavelength_idx: usize,
    pub transmittance: f64,
    pub label: &'static str,
}

/// A single-scatter radiance test case.
#[derive(Debug, Clone)]
pub struct SingleScatterCase {
    pub sza_deg: f64,
    pub wavelength_idx: usize,
    pub radiance: f64,
    pub label: &'static str,
}

/// A full spectral test case (3 wavelengths).
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SpectralCase {
    pub sza_deg: f64,
    pub radiance: [f64; 3],
    pub label: &'static str,
}

/// An RNG sequence test case.
#[derive(Debug, Clone)]
pub struct RngCase {
    pub seed: u64,
    pub num_values: usize,
    pub values: Vec<f64>,
}

// ── Oracle generators ───────────────────────────────────────────────────

/// Generate ray-sphere intersection oracle cases.
pub fn ray_sphere_cases() -> Vec<RaySphereCase> {
    let r_earth = EARTH_RADIUS_M;
    let r_toa = r_earth + 100_000.0;

    let mut cases = Vec::new();

    // Case 1: miss
    cases.push({
        let o = Vec3::new(0.0, r_toa + 1000.0, 0.0);
        let d = Vec3::new(1.0, 0.0, 0.0);
        let result = ray_sphere_intersect(o, d, r_earth);
        RaySphereCase {
            origin: [o.x, o.y, o.z],
            direction: [d.x, d.y, d.z],
            radius: r_earth,
            hits: result.is_some(),
            t_near: result.map_or(0.0, |h| h.t_near),
            t_far: result.map_or(0.0, |h| h.t_far),
        }
    });

    // Case 2: hit along x-axis
    cases.push({
        let o = Vec3::new(-r_toa - 1000.0, 0.0, 0.0);
        let d = Vec3::new(1.0, 0.0, 0.0);
        let result = ray_sphere_intersect(o, d, r_earth);
        RaySphereCase {
            origin: [o.x, o.y, o.z],
            direction: [d.x, d.y, d.z],
            radius: r_earth,
            hits: result.is_some(),
            t_near: result.map_or(0.0, |h| h.t_near),
            t_far: result.map_or(0.0, |h| h.t_far),
        }
    });

    // Case 3: from inside sphere
    cases.push({
        let o = Vec3::new(r_earth * 0.5, 0.0, 0.0);
        let d = Vec3::new(1.0, 0.0, 0.0);
        let result = ray_sphere_intersect(o, d, r_earth);
        RaySphereCase {
            origin: [o.x, o.y, o.z],
            direction: [d.x, d.y, d.z],
            radius: r_earth,
            hits: result.is_some(),
            t_near: result.map_or(0.0, |h| h.t_near),
            t_far: result.map_or(0.0, |h| h.t_far),
        }
    });

    // Case 4: from surface looking up at TOA
    cases.push({
        let o = Vec3::new(r_earth, 0.0, 0.0);
        let d = Vec3::new(1.0, 0.0, 0.0);
        let result = ray_sphere_intersect(o, d, r_toa);
        RaySphereCase {
            origin: [o.x, o.y, o.z],
            direction: [d.x, d.y, d.z],
            radius: r_toa,
            hits: result.is_some(),
            t_near: result.map_or(0.0, |h| h.t_near),
            t_far: result.map_or(0.0, |h| h.t_far),
        }
    });

    // Case 5: grazing ray near Earth limb
    cases.push({
        let o = Vec3::new(0.0, r_earth + 50_000.0, 0.0);
        let d = Vec3::new(1.0, 0.0, 0.0); // horizontal at 50km altitude
        let result = ray_sphere_intersect(o, d, r_earth);
        RaySphereCase {
            origin: [o.x, o.y, o.z],
            direction: [d.x, d.y, d.z],
            radius: r_earth,
            hits: result.is_some(),
            t_near: result.map_or(0.0, |h| h.t_near),
            t_far: result.map_or(0.0, |h| h.t_far),
        }
    });

    cases
}

/// Generate phase function oracle cases.
pub fn phase_function_cases() -> Vec<PhaseCase> {
    let mut cases = Vec::new();

    // Sweep cos_theta from -1 to 1
    let cos_values = [
        -1.0, -0.9, -0.7, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0,
    ];
    let g_values = [0.0, 0.3, 0.65, 0.85, -0.5];

    for &cos_theta in &cos_values {
        for &g in &g_values {
            cases.push(PhaseCase {
                cos_theta,
                g,
                rayleigh_value: rayleigh_phase(cos_theta),
                hg_value: henyey_greenstein_phase(cos_theta, g),
            });
        }
    }

    cases
}

/// Generate shadow ray transmittance oracle cases.
pub fn shadow_ray_cases() -> Vec<ShadowRayCase> {
    let atm = oracle_atmosphere();
    let obs_r = EARTH_RADIUS_M + 1.0;

    let mut cases = Vec::new();

    // Various SZA from 80 to 120 degrees
    let szas = [80.0, 90.0, 96.0, 100.0, 104.0, 108.0, 112.0, 120.0];
    let labels = [
        "sza_80_sunlit",
        "sza_90_horizon",
        "sza_96_civil",
        "sza_100_nautical",
        "sza_104_deep",
        "sza_108_astronomical",
        "sza_112_deep_shadow",
        "sza_120_night",
    ];

    for (i, &sza) in szas.iter().enumerate() {
        let sun_dir = solar_direction_ecef(sza, 180.0, 0.0, 0.0);
        let scatter_pos = Vec3::new(obs_r, 0.0, 0.0);

        for w in 0..3 {
            let t = shadow_ray_transmittance(&atm, scatter_pos, sun_dir, w);
            cases.push(ShadowRayCase {
                scatter_pos: [scatter_pos.x, scatter_pos.y, scatter_pos.z],
                sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z],
                wavelength_idx: w,
                transmittance: t,
                label: labels[i],
            });
        }
    }

    // High-altitude scatter point (40km, should be sunlit even at deep twilight)
    let high_pos = Vec3::new(EARTH_RADIUS_M + 40_000.0, 0.0, 0.0);
    for &sza in &[96.0, 108.0] {
        let sun_dir = solar_direction_ecef(sza, 180.0, 0.0, 0.0);
        for w in 0..3 {
            let t = shadow_ray_transmittance(&atm, high_pos, sun_dir, w);
            cases.push(ShadowRayCase {
                scatter_pos: [high_pos.x, high_pos.y, high_pos.z],
                sun_dir: [sun_dir.x, sun_dir.y, sun_dir.z],
                wavelength_idx: w,
                transmittance: t,
                label: if sza < 100.0 {
                    "high_alt_civil"
                } else {
                    "high_alt_astronomical"
                },
            });
        }
    }

    cases
}

/// Generate single-scatter radiance oracle cases.
pub fn single_scatter_cases() -> Vec<SingleScatterCase> {
    let atm = oracle_atmosphere();
    let obs = Vec3::new(EARTH_RADIUS_M + 1.0, 0.0, 0.0);
    let view = Vec3::new(0.0, 1.0, 0.0).normalize(); // horizontal

    let mut cases = Vec::new();

    let szas = [80.0, 90.0, 92.0, 96.0, 100.0, 104.0, 108.0, 120.0];
    let labels = [
        "sza_80_day",
        "sza_90_sunset",
        "sza_92_civil",
        "sza_96_mid_civil",
        "sza_100_nautical",
        "sza_104_deep",
        "sza_108_astronomical",
        "sza_120_night",
    ];

    for (i, &sza) in szas.iter().enumerate() {
        let sun = solar_direction_ecef(sza, 180.0, 0.0, 0.0);
        for w in 0..3 {
            let rad = single_scatter_radiance(&atm, obs, view, sun, w);
            cases.push(SingleScatterCase {
                sza_deg: sza,
                wavelength_idx: w,
                radiance: rad,
                label: labels[i],
            });
        }
    }

    cases
}

/// Generate full spectral oracle cases.
pub fn spectral_cases() -> Vec<SpectralCase> {
    let atm = oracle_atmosphere();
    let obs = Vec3::new(EARTH_RADIUS_M + 1.0, 0.0, 0.0);
    let view = Vec3::new(0.0, 1.0, 0.0).normalize();

    let mut cases = Vec::new();

    let szas = [80.0, 92.0, 96.0, 100.0, 108.0];
    let labels = [
        "spectrum_day",
        "spectrum_civil",
        "spectrum_mid_civil",
        "spectrum_nautical",
        "spectrum_astronomical",
    ];

    for (i, &sza) in szas.iter().enumerate() {
        let sun = solar_direction_ecef(sza, 180.0, 0.0, 0.0);
        let spec = single_scatter_spectrum(&atm, obs, view, sun);
        cases.push(SpectralCase {
            sza_deg: sza,
            radiance: [spec[0], spec[1], spec[2]],
            label: labels[i],
        });
    }

    cases
}

/// Generate xorshift64 RNG oracle sequences.
pub fn rng_cases() -> Vec<RngCase> {
    let mut cases = Vec::new();

    for &seed in &[1u64, 42, 123456789, 0xDEADBEEF] {
        let mut state = seed;
        let mut values = Vec::new();
        for _ in 0..20 {
            values.push(xorshift_f64(&mut state));
        }
        cases.push(RngCase {
            seed,
            num_values: 20,
            values,
        });
    }

    cases
}

/// Total number of oracle test cases across all categories.
pub fn total_oracle_cases() -> usize {
    ray_sphere_cases().len()
        + phase_function_cases().len()
        + shadow_ray_cases().len()
        + single_scatter_cases().len()
        + spectral_cases().len()
        + rng_cases().len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oracle_atmosphere_has_expected_structure() {
        let atm = oracle_atmosphere();
        assert_eq!(atm.num_shells, 3);
        assert_eq!(atm.num_wavelengths, 3);
        assert!((atm.wavelengths_nm[0] - 400.0).abs() < 0.01);
        assert!((atm.wavelengths_nm[1] - 550.0).abs() < 0.01);
        assert!((atm.wavelengths_nm[2] - 700.0).abs() < 0.01);
    }

    #[test]
    fn oracle_atmosphere_aerosol_has_mixed_phase() {
        let atm = oracle_atmosphere_aerosol();
        assert!(atm.optics[0][0].rayleigh_fraction < 1.0);
        assert!(atm.optics[0][0].asymmetry > 0.0);
        assert!(atm.optics[0][0].ssa < 1.0);
    }

    #[test]
    fn ray_sphere_oracle_generates_cases() {
        let cases = ray_sphere_cases();
        assert!(cases.len() >= 5, "expected >= 5 cases, got {}", cases.len());
    }

    #[test]
    fn ray_sphere_oracle_miss_case() {
        let cases = ray_sphere_cases();
        let miss = &cases[0]; // first case is a miss
        assert!(!miss.hits, "first case should be a miss");
    }

    #[test]
    fn ray_sphere_oracle_hit_case() {
        let cases = ray_sphere_cases();
        let hit = &cases[1]; // second case is a hit
        assert!(hit.hits, "second case should be a hit");
        assert!(hit.t_near < hit.t_far, "t_near should be < t_far");
    }

    #[test]
    fn phase_function_oracle_count() {
        let cases = phase_function_cases();
        // 13 cos_theta values * 5 g values = 65
        assert_eq!(cases.len(), 65);
    }

    #[test]
    fn phase_function_oracle_rayleigh_values() {
        let cases = phase_function_cases();
        for c in &cases {
            assert!(c.rayleigh_value >= 0.75, "Rayleigh phase should be >= 0.75");
            assert!(c.rayleigh_value <= 1.5, "Rayleigh phase should be <= 1.5");
        }
    }

    #[test]
    fn phase_function_oracle_hg_positive() {
        let cases = phase_function_cases();
        for c in &cases {
            assert!(
                c.hg_value > 0.0,
                "HG phase should be positive: cos={}, g={}, val={}",
                c.cos_theta,
                c.g,
                c.hg_value,
            );
        }
    }

    #[test]
    fn shadow_ray_oracle_count() {
        let cases = shadow_ray_cases();
        // 8 SZA * 3 wavelengths + 2 high-alt SZA * 3 wavelengths = 30
        assert_eq!(cases.len(), 30);
    }

    #[test]
    fn shadow_ray_oracle_sunlit_positive() {
        let cases = shadow_ray_cases();
        // SZA=80 at surface should have positive transmittance
        let sunlit: Vec<_> = cases
            .iter()
            .filter(|c| c.label == "sza_80_sunlit")
            .collect();
        for c in &sunlit {
            assert!(
                c.transmittance > 0.0,
                "SZA=80 should have positive transmittance: T={}",
                c.transmittance,
            );
        }
    }

    #[test]
    fn shadow_ray_oracle_night_zero() {
        let cases = shadow_ray_cases();
        let night: Vec<_> = cases
            .iter()
            .filter(|c| c.label == "sza_120_night")
            .collect();
        for c in &night {
            assert!(
                c.transmittance < 1e-10,
                "SZA=120 should have near-zero transmittance: T={}",
                c.transmittance,
            );
        }
    }

    #[test]
    fn shadow_ray_oracle_transmittance_in_range() {
        let cases = shadow_ray_cases();
        for c in &cases {
            assert!(
                c.transmittance >= 0.0 && c.transmittance <= 1.0,
                "Transmittance out of [0,1]: {} at {}",
                c.transmittance,
                c.label,
            );
        }
    }

    #[test]
    fn single_scatter_oracle_count() {
        let cases = single_scatter_cases();
        // 8 SZA * 3 wavelengths = 24
        assert_eq!(cases.len(), 24);
    }

    #[test]
    fn single_scatter_oracle_non_negative() {
        let cases = single_scatter_cases();
        for c in &cases {
            assert!(
                c.radiance >= 0.0,
                "Radiance should be non-negative: {} at SZA={}, wl={}",
                c.radiance,
                c.sza_deg,
                c.wavelength_idx,
            );
        }
    }

    #[test]
    fn single_scatter_oracle_decreases_with_sza() {
        let cases = single_scatter_cases();
        // For wavelength 1 (550nm), radiance should generally decrease with SZA
        let wl1: Vec<_> = cases.iter().filter(|c| c.wavelength_idx == 1).collect();
        for pair in wl1.windows(2) {
            if pair[0].sza_deg < 108.0 && pair[1].sza_deg > pair[0].sza_deg {
                assert!(
                    pair[1].radiance <= pair[0].radiance + 1e-20,
                    "Radiance should decrease: SZA {} -> {}: {:.4e} -> {:.4e}",
                    pair[0].sza_deg,
                    pair[1].sza_deg,
                    pair[0].radiance,
                    pair[1].radiance,
                );
            }
        }
    }

    #[test]
    fn spectral_oracle_count() {
        let cases = spectral_cases();
        assert_eq!(cases.len(), 5);
    }

    #[test]
    fn spectral_oracle_red_dominates_at_twilight() {
        // At SZA=92 (civil twilight), the long slant path attenuates blue
        // much more than red (Rayleigh lambda^-4 works AGAINST blue at long paths).
        // This is the correct physics of why the twilight sky turns red/orange.
        let cases = spectral_cases();
        let civil = cases.iter().find(|c| c.sza_deg == 92.0).unwrap();
        assert!(
            civil.radiance[2] > civil.radiance[0],
            "At SZA=92, red ({:.4e}) should > blue ({:.4e}) due to path attenuation",
            civil.radiance[2],
            civil.radiance[0],
        );
    }

    #[test]
    fn rng_oracle_count() {
        let cases = rng_cases();
        assert_eq!(cases.len(), 4);
    }

    #[test]
    fn rng_oracle_values_in_unit_interval() {
        let cases = rng_cases();
        for c in &cases {
            assert_eq!(c.values.len(), c.num_values);
            for &v in &c.values {
                assert!(
                    (0.0..1.0).contains(&v),
                    "RNG value {} out of [0,1) for seed {}",
                    v,
                    c.seed,
                );
            }
        }
    }

    #[test]
    fn rng_oracle_is_deterministic() {
        let cases1 = rng_cases();
        let cases2 = rng_cases();
        for (c1, c2) in cases1.iter().zip(cases2.iter()) {
            assert_eq!(c1.seed, c2.seed);
            for (v1, v2) in c1.values.iter().zip(c2.values.iter()) {
                assert_eq!(*v1, *v2, "RNG should be deterministic");
            }
        }
    }

    #[test]
    fn total_oracle_cases_is_reasonable() {
        let total = total_oracle_cases();
        // 5 + 65 + 30 + 24 + 5 + 4 = 133
        assert!(
            total >= 100 && total <= 200,
            "Expected ~133 oracle cases, got {}",
            total,
        );
    }
}
