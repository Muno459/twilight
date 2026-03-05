//! Cross-backend test infrastructure for GPU validation.
//!
//! Tests are organized in layers:
//!
//! 1. **Buffer packing** -- verify f64->f32 roundtrip (always runs, no GPU)
//! 2. **Oracle generation** -- verify CPU reference produces consistent values
//! 3. **Physics invariants** -- non-negative radiance, monotonic decrease, etc.
//! 4. **Backend integration** -- GPU result vs CPU oracle (feature-gated)
//! 5. **Cross-backend parity** -- all available backends agree (Phase 11g)
//!
//! Layers 1-3 run without any GPU backend. Layers 4-5 are feature-gated.

use crate::buffers::*;
use crate::oracle;

// ── Tolerance constants ─────────────────────────────────────────────────

/// Relative tolerance for f32 vs f64 comparison on scalar values.
/// Based on precision analysis: f32 rounding ~1e-7 relative.
#[allow(dead_code)]
const F32_RTOL: f64 = 1e-5;

/// Absolute tolerance for values near zero.
#[allow(dead_code)]
const F32_ATOL: f64 = 1e-20;

/// Relative tolerance for single-scatter GPU vs CPU oracle.
/// Allows for f32 accumulation error over 200 LOS steps.
#[allow(dead_code)]
const SINGLE_SCATTER_RTOL: f64 = 1e-3;

/// Relative tolerance for MC results (dominated by stochastic noise).
/// MC noise is ~1% with 10k photons, so 5% tolerance is conservative.
#[allow(dead_code)]
const MC_RTOL: f64 = 0.05;

// ── Helper: approximate equality ────────────────────────────────────────

#[allow(dead_code)]
fn approx_eq(a: f64, b: f64, rtol: f64, atol: f64) -> bool {
    let diff = (a - b).abs();
    diff < atol + rtol * a.abs().max(b.abs())
}

// ── Layer 1: Buffer packing tests (always run) ──────────────────────────

#[test]
fn buffer_magic_encodes_correctly_in_f32() {
    let as_f32 = f32::from_bits(BUFFER_MAGIC);
    let roundtrip = as_f32.to_bits();
    assert_eq!(roundtrip, BUFFER_MAGIC);
}

#[test]
fn buffer_version_encodes_correctly_in_f32() {
    let as_f32 = f32::from_bits(BUFFER_VERSION);
    let roundtrip = as_f32.to_bits();
    assert_eq!(roundtrip, BUFFER_VERSION);
}

#[test]
fn packed_atmosphere_size_fits_gpu_memory() {
    // Even the smallest mobile GPU (Mali, ~1GB) can fit this
    let atm = oracle::oracle_atmosphere();
    let packed = PackedAtmosphere::pack(&atm);
    assert!(
        packed.size_bytes() < 1_000_000,
        "Atmosphere buffer {} bytes should be < 1MB",
        packed.size_bytes(),
    );
}

#[test]
fn packed_atmosphere_all_buffers_total_size() {
    // Atmosphere + solar + vision < 100KB total
    let atm = oracle::oracle_atmosphere();
    let packed_atm = PackedAtmosphere::pack(&atm);
    let packed_solar = PackedSolarSpectrum::pack();
    let packed_vision = PackedVisionLuts::pack();

    let total =
        packed_atm.size_bytes() + packed_solar.data.len() * 4 + packed_vision.data.len() * 4;

    assert!(
        total < 100_000,
        "All GPU buffers = {} bytes, should be < 100KB",
        total,
    );
}

// ── Layer 2: Oracle consistency ─────────────────────────────────────────

#[test]
fn oracle_generates_expected_case_count() {
    assert_eq!(oracle::ray_sphere_cases().len(), 5);
    assert_eq!(oracle::phase_function_cases().len(), 65);
    assert_eq!(oracle::shadow_ray_cases().len(), 30);
    assert_eq!(oracle::single_scatter_cases().len(), 24);
    assert_eq!(oracle::spectral_cases().len(), 5);
    assert_eq!(oracle::rng_cases().len(), 4);
}

#[test]
fn oracle_is_deterministic() {
    // Generate oracle twice, results must be identical
    let ss1 = oracle::single_scatter_cases();
    let ss2 = oracle::single_scatter_cases();
    for (a, b) in ss1.iter().zip(ss2.iter()) {
        assert_eq!(
            a.radiance, b.radiance,
            "Oracle not deterministic at SZA={}, wl={}",
            a.sza_deg, a.wavelength_idx,
        );
    }
}

#[test]
fn oracle_phase_rayleigh_symmetry() {
    let cases = oracle::phase_function_cases();
    for c in &cases {
        // Rayleigh is symmetric: P(mu) = P(-mu)
        let p_neg = twilight_core::scattering::rayleigh_phase(-c.cos_theta);
        assert!(
            (c.rayleigh_value - p_neg).abs() < 1e-12,
            "Rayleigh not symmetric: P({}) = {}, P({}) = {}",
            c.cos_theta,
            c.rayleigh_value,
            -c.cos_theta,
            p_neg,
        );
    }
}

#[test]
fn oracle_phase_hg_normalization() {
    // For each g value, numerical integral of HG should be 2
    let g_values = [0.0, 0.3, 0.65, 0.85, -0.5];
    for &g in &g_values {
        let n = 10_000;
        let dmu = 2.0 / n as f64;
        let mut integral = 0.0;
        for i in 0..n {
            let mu = -1.0 + (i as f64 + 0.5) * dmu;
            integral += twilight_core::scattering::henyey_greenstein_phase(mu, g) * dmu;
        }
        assert!(
            (integral - 2.0).abs() < 0.01,
            "HG(g={}) integral = {}, expected 2.0",
            g,
            integral,
        );
    }
}

// ── Layer 3: Physics invariant tests ────────────────────────────────────

#[test]
fn physics_radiance_non_negative() {
    let cases = oracle::single_scatter_cases();
    for c in &cases {
        assert!(
            c.radiance >= 0.0,
            "Negative radiance: {} at SZA={}, wl={} ({})",
            c.radiance,
            c.sza_deg,
            c.wavelength_idx,
            c.label,
        );
    }
}

#[test]
fn physics_radiance_monotonic_decrease_with_sza() {
    // For each wavelength, radiance should decrease as SZA increases
    // (sun goes deeper below horizon = less light)
    let cases = oracle::single_scatter_cases();
    for w in 0..3 {
        let wl_cases: Vec<_> = cases.iter().filter(|c| c.wavelength_idx == w).collect();
        for pair in wl_cases.windows(2) {
            if pair[0].sza_deg < pair[1].sza_deg {
                assert!(
                    pair[1].radiance <= pair[0].radiance + 1e-20,
                    "Radiance should decrease: wl={}, SZA {} ({:.4e}) -> {} ({:.4e})",
                    w,
                    pair[0].sza_deg,
                    pair[0].radiance,
                    pair[1].sza_deg,
                    pair[1].radiance,
                );
            }
        }
    }
}

#[test]
fn physics_deep_night_radiance_negligible() {
    let cases = oracle::single_scatter_cases();
    let night: Vec<_> = cases.iter().filter(|c| c.sza_deg >= 120.0).collect();
    for c in &night {
        assert!(
            c.radiance < 1e-20,
            "SZA={} should have negligible radiance: {:.4e}",
            c.sza_deg,
            c.radiance,
        );
    }
}

#[test]
fn physics_transmittance_decreases_with_sza() {
    // At surface level, transmittance should generally decrease as the sun
    // goes below the horizon (longer slant path through atmosphere)
    let cases = oracle::shadow_ray_cases();
    for w in 0..3 {
        let surface_cases: Vec<_> = cases
            .iter()
            .filter(|c| c.wavelength_idx == w && c.scatter_pos[0] < 6_371_010.0)
            .collect();

        for pair in surface_cases.windows(2) {
            // Only compare if transmittances are both meaningfully positive
            if pair[0].transmittance > 1e-10 && pair[1].transmittance > 1e-10 {
                // Allow small violation from discretization
                assert!(
                    pair[1].transmittance <= pair[0].transmittance * 1.01,
                    "Transmittance should decrease: wl={}, {} ({}) -> {} ({})",
                    w,
                    pair[0].label,
                    pair[0].transmittance,
                    pair[1].label,
                    pair[1].transmittance,
                );
            }
        }
    }
}

#[test]
fn physics_red_dominates_blue_at_twilight() {
    // At civil twilight (SZA=92), long slant paths attenuate blue far more
    // than red. Even though Rayleigh scattering coefficient is ~13x larger
    // for blue, the exponential extinction along 100+ km paths overwhelms
    // the scattering advantage. This is why the twilight sky is red/orange.
    let cases = oracle::spectral_cases();
    let civil = cases.iter().find(|c| c.sza_deg == 92.0).unwrap();
    assert!(
        civil.radiance[2] > civil.radiance[0],
        "At SZA=92, red ({:.4e}) should dominate blue ({:.4e}) due to path attenuation",
        civil.radiance[2],
        civil.radiance[0],
    );
}

#[test]
fn physics_rng_uniform_distribution() {
    let cases = oracle::rng_cases();
    for c in &cases {
        let mean: f64 = c.values.iter().sum::<f64>() / c.values.len() as f64;
        // With only 20 values, mean can deviate significantly, so use loose bound
        assert!(
            mean > 0.1 && mean < 0.9,
            "RNG mean = {} for seed {} (expected ~0.5)",
            mean,
            c.seed,
        );
    }
}

// ── Layer 3b: f32 packing preserves physics ─────────────────────────────

#[test]
fn f32_packing_preserves_extinction_order() {
    // After f32 packing, the relative ordering of extinction values must
    // be preserved (blue > green > red for Rayleigh)
    let atm = oracle::oracle_atmosphere();
    let packed = PackedAtmosphere::pack(&atm);
    let unpacked = packed.unpack();

    for s in 0..atm.num_shells {
        let ext_blue = unpacked.optics[s][0].extinction;
        let ext_green = unpacked.optics[s][1].extinction;
        let ext_red = unpacked.optics[s][2].extinction;

        if ext_blue > 0.0 {
            assert!(
                ext_blue > ext_green,
                "shell[{}]: blue ext ({}) should > green ext ({})",
                s,
                ext_blue,
                ext_green,
            );
            assert!(
                ext_green > ext_red,
                "shell[{}]: green ext ({}) should > red ext ({})",
                s,
                ext_green,
                ext_red,
            );
        }
    }
}

#[test]
fn f32_packing_earth_radius_precision() {
    // EARTH_RADIUS_M = 6,371,000.0. In f32, this is representable exactly
    // (it's < 2^23 * 1, actually 6.371e6 which needs ~23 bits of mantissa).
    // f32 has 23 bits of mantissa, so the ULP at 6.371e6 is about 0.5m.
    let r = twilight_core::atmosphere::EARTH_RADIUS_M;
    let r_f32 = r as f32;
    let roundtrip = r_f32 as f64;
    assert!(
        (roundtrip - r).abs() < 1.0,
        "Earth radius f32 roundtrip: {} -> {} -> {} (err={}m)",
        r,
        r_f32,
        roundtrip,
        (roundtrip - r).abs(),
    );
}

#[test]
fn f32_packing_small_extinction_preserved() {
    // Extinction values ~1e-8 should survive f32 conversion
    let val = 1e-8_f64;
    let f32_val = val as f32;
    let roundtrip = f32_val as f64;
    let rel_err = (roundtrip - val).abs() / val;
    assert!(
        rel_err < 1e-6,
        "Small extinction f32 roundtrip: {:.4e} -> {:.4e} (rel_err={:.4e})",
        val,
        roundtrip,
        rel_err,
    );
}

// ── Dispatch parameter tests ────────────────────────────────────────────

#[test]
fn dispatch_params_earth_radius_scale() {
    // Observer at Earth surface: coordinates are ~6.371e6
    // In f32, ULP is about 0.5m -- fine for our purposes
    let obs = [twilight_core::atmosphere::EARTH_RADIUS_M + 1.0, 0.0, 0.0];
    let p = PackedDispatchParams::new(obs, [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], 10000, 100, 42);

    let packed_x = p.data[0] as f64;
    assert!(
        (packed_x - obs[0]).abs() < 1.0,
        "Observer x: packed={}, original={}, err={}",
        packed_x,
        obs[0],
        (packed_x - obs[0]).abs(),
    );
}
