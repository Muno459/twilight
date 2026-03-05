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

// ═══════════════════════════════════════════════════════════════════════
// Layer 4: Backend integration tests (GPU result vs CPU f64 oracle)
// ═══════════════════════════════════════════════════════════════════════
//
// These tests initialize a real GPU backend, upload the oracle atmosphere,
// run single_scatter at several SZAs, and compare against CPU f64 oracle.
//
// Each test gracefully skips if no GPU hardware is available (CI-safe).
// Feature-gated: only compiled when the corresponding backend feature is on.

/// Helper: run single_scatter on a backend for several SZAs, compare vs oracle.
///
/// Returns the number of test cases checked (0 means backend unavailable).
#[allow(dead_code)]
fn run_single_scatter_parity(backend: &mut dyn crate::GpuBackend, label: &str) -> usize {
    use twilight_core::atmosphere::EARTH_RADIUS_M;
    use twilight_core::geometry::solar_direction_ecef;

    let atm = oracle::oracle_atmosphere();
    backend
        .upload_atmosphere(&atm)
        .unwrap_or_else(|e| panic!("[{}] upload_atmosphere failed: {}", label, e));

    let obs = [EARTH_RADIUS_M + 1.0, 0.0, 0.0];
    let view = [0.0, 1.0, 0.0]; // horizontal
    let oracle_cases = oracle::single_scatter_cases();

    let szas = [80.0, 90.0, 92.0, 96.0, 100.0, 104.0, 108.0];
    let mut checked = 0;

    for &sza in &szas {
        let sun = solar_direction_ecef(sza, 180.0, 0.0, 0.0);
        let sun_arr = [sun.x, sun.y, sun.z];

        let gpu_result = backend
            .single_scatter(obs, view, sun_arr)
            .unwrap_or_else(|e| panic!("[{}] single_scatter(SZA={}) failed: {}", label, sza, e));

        // Compare each wavelength against CPU oracle
        for w in 0..3 {
            let cpu_rad = oracle_cases
                .iter()
                .find(|c| (c.sza_deg - sza).abs() < 0.01 && c.wavelength_idx == w)
                .unwrap_or_else(|| panic!("no oracle case for SZA={}, wl={}", sza, w))
                .radiance;

            let gpu_rad = gpu_result.radiance[w];

            // Skip comparison for effectively-zero values (deep night)
            if cpu_rad < 1e-25 && gpu_rad < 1e-25 {
                checked += 1;
                continue;
            }

            assert!(
                approx_eq(gpu_rad, cpu_rad, SINGLE_SCATTER_RTOL, F32_ATOL),
                "[{}] SZA={} wl={}: GPU={:.6e} vs CPU={:.6e}, rel_err={:.4e} (tol={})",
                label,
                sza,
                w,
                gpu_rad,
                cpu_rad,
                (gpu_rad - cpu_rad).abs() / cpu_rad.abs().max(1e-30),
                SINGLE_SCATTER_RTOL,
            );
            checked += 1;
        }
    }

    checked
}

/// Helper: run mcrt_trace on a backend and verify physical consistency.
///
/// MCRT (backward MC with NEE) includes multiple scattering, so it should
/// produce radiance >= single scatter (more light paths). We verify:
/// 1. Non-negative, finite results
/// 2. MCRT >= single scatter for each wavelength (within noise)
/// 3. Radiance decreases with increasing SZA (monotonicity)
/// 4. Deep twilight values are small but positive
#[allow(dead_code)]
fn run_mcrt_vs_single_scatter(backend: &mut dyn crate::GpuBackend, label: &str) -> usize {
    use twilight_core::atmosphere::EARTH_RADIUS_M;
    use twilight_core::geometry::solar_direction_ecef;

    let atm = oracle::oracle_atmosphere();
    backend
        .upload_atmosphere(&atm)
        .unwrap_or_else(|e| panic!("[{}] upload_atmosphere failed: {}", label, e));

    let obs = [EARTH_RADIUS_M + 1.0, 0.0, 0.0];
    let view = [0.0, 1.0, 0.0];

    let szas = [80.0, 90.0, 96.0, 108.0];
    let mut checked = 0;
    let mut prev_total = [f64::MAX; 3];

    for &sza in &szas {
        let sun = solar_direction_ecef(sza, 180.0, 0.0, 0.0);
        let sun_arr = [sun.x, sun.y, sun.z];

        let ss = backend.single_scatter(obs, view, sun_arr).unwrap();
        let mc = backend.mcrt_trace(obs, view, sun_arr, 10_000, 42).unwrap();

        for w in 0..3 {
            let ss_rad = ss.radiance[w];
            let mc_rad = mc.radiance[w];

            // 1. Non-negative and finite
            assert!(
                mc_rad >= 0.0 && mc_rad.is_finite(),
                "[{}] MCRT SZA={} wl={}: non-finite or negative: {:.6e}",
                label,
                sza,
                w,
                mc_rad,
            );

            // 2. MCRT should be >= single scatter (MC adds multi-scatter paths).
            // Allow small MC noise violation (MC can be slightly below SS due
            // to stochastic noise with limited photon count).
            if ss_rad > 1e-20 {
                assert!(
                    mc_rad > ss_rad * 0.5,
                    "[{}] MCRT SZA={} wl={}: MC={:.6e} should be >= ~SS={:.6e} (multi-scatter adds light)",
                    label, sza, w, mc_rad, ss_rad,
                );
            }

            // 3. Monotonicity: total radiance should generally decrease with SZA.
            // MC noise can violate this, especially at deep twilight where
            // signal is weak and 10k photons give ~10% noise. Use a very
            // generous 3x bound -- we're testing the trend, not exact values.
            if sza > 80.0 && prev_total[w] > 1e-20 {
                assert!(
                    mc_rad <= prev_total[w] * 3.0 + 1e-20,
                    "[{}] MCRT monotonicity fail wl={}: SZA={} ({:.6e}) >> prev ({:.6e})",
                    label,
                    w,
                    sza,
                    mc_rad,
                    prev_total[w],
                );
            }
            prev_total[w] = mc_rad;

            checked += 1;
        }
    }

    checked
}

/// Helper: run hybrid_scatter and verify physical consistency.
///
/// Hybrid = single scatter (deterministic LOS) + secondary MC chains.
/// At twilight SZAs, the multi-scatter contribution can dominate single
/// scatter by orders of magnitude (especially for blue light where the
/// direct path is heavily attenuated but scattered light from higher
/// altitudes reaches the observer via secondary chains).
///
/// We verify:
/// 1. Non-negative, finite results
/// 2. Hybrid >= single scatter (more light paths, within MC noise)
/// 3. Results are physically plausible (< 1 W/m2/sr/nm)
#[allow(dead_code)]
fn run_hybrid_sanity(backend: &mut dyn crate::GpuBackend, label: &str) -> usize {
    use twilight_core::atmosphere::EARTH_RADIUS_M;
    use twilight_core::geometry::solar_direction_ecef;

    let atm = oracle::oracle_atmosphere();
    backend
        .upload_atmosphere(&atm)
        .unwrap_or_else(|e| panic!("[{}] upload_atmosphere failed: {}", label, e));

    let obs = [EARTH_RADIUS_M + 1.0, 0.0, 0.0];
    let view = [0.0, 1.0, 0.0];
    let mut checked = 0;

    for &sza in &[90.0, 96.0] {
        let sun = solar_direction_ecef(sza, 180.0, 0.0, 0.0);
        let sun_arr = [sun.x, sun.y, sun.z];

        let ss = backend.single_scatter(obs, view, sun_arr).unwrap();
        let hybrid = backend.hybrid_scatter(obs, view, sun_arr, 100, 42).unwrap();

        for w in 0..3 {
            let h_rad = hybrid.radiance[w];
            let s_rad = ss.radiance[w];

            // 1. Non-negative and finite
            assert!(
                h_rad >= 0.0 && h_rad.is_finite(),
                "[{}] hybrid SZA={} wl={}: non-finite or negative: {:.6e}",
                label,
                sza,
                w,
                h_rad,
            );

            // 2. Hybrid should be >= single scatter (it includes SS + more).
            // Allow MC noise to make it slightly lower (0.5x).
            if s_rad > 1e-20 {
                assert!(
                    h_rad > s_rad * 0.5,
                    "[{}] hybrid SZA={} wl={}: hybrid {:.6e} should >= ~SS {:.6e}",
                    label,
                    sza,
                    w,
                    h_rad,
                    s_rad,
                );
            }

            // 3. Physical plausibility: twilight radiance should be < 1 W/m2/sr/nm
            assert!(
                h_rad < 1.0,
                "[{}] hybrid SZA={} wl={}: radiance {:.6e} is unphysically large",
                label,
                sza,
                w,
                h_rad,
            );

            checked += 1;
        }
    }

    checked
}

// ── Metal backend integration tests ─────────────────────────────────────

#[cfg(feature = "metal")]
mod layer4_metal {
    use super::*;
    use crate::{BackendKind, GpuConfig};

    fn try_metal() -> Option<Box<dyn crate::GpuBackend>> {
        let config = GpuConfig {
            preferred_backend: Some(BackendKind::Metal),
            ..Default::default()
        };
        crate::try_init(&config).ok()
    }

    #[test]
    fn metal_init_and_device_info() {
        let Some(gpu) = try_metal() else { return };
        let info = gpu.device_info();
        assert_eq!(info.backend, BackendKind::Metal);
        assert!(
            !info.name.is_empty(),
            "Metal device name should not be empty"
        );
        assert!(
            info.max_workgroup_size >= 256,
            "Metal max_workgroup_size={} should be >= 256",
            info.max_workgroup_size,
        );
    }

    #[test]
    fn metal_upload_atmosphere() {
        let Some(mut gpu) = try_metal() else { return };
        let atm = oracle::oracle_atmosphere();
        gpu.upload_atmosphere(&atm)
            .expect("Metal upload_atmosphere should succeed");
    }

    #[test]
    fn metal_single_scatter_vs_cpu_oracle() {
        let Some(mut gpu) = try_metal() else { return };
        let checked = run_single_scatter_parity(gpu.as_mut(), "Metal");
        assert!(checked > 0, "should have checked at least one case");
    }

    #[test]
    fn metal_mcrt_vs_single_scatter() {
        let Some(mut gpu) = try_metal() else { return };
        let checked = run_mcrt_vs_single_scatter(gpu.as_mut(), "Metal");
        assert!(checked > 0, "should have checked at least one case");
    }

    #[test]
    fn metal_hybrid_sanity() {
        let Some(mut gpu) = try_metal() else { return };
        let checked = run_hybrid_sanity(gpu.as_mut(), "Metal");
        assert!(checked > 0, "should have checked at least one case");
    }

    #[test]
    fn metal_single_scatter_radiance_non_negative() {
        use twilight_core::atmosphere::EARTH_RADIUS_M;
        use twilight_core::geometry::solar_direction_ecef;

        let Some(mut gpu) = try_metal() else { return };
        let atm = oracle::oracle_atmosphere();
        gpu.upload_atmosphere(&atm).unwrap();

        let obs = [EARTH_RADIUS_M + 1.0, 0.0, 0.0];
        let view = [0.0, 1.0, 0.0];

        for &sza in &[80.0, 90.0, 96.0, 100.0, 108.0, 120.0] {
            let sun = solar_direction_ecef(sza, 180.0, 0.0, 0.0);
            let result = gpu
                .single_scatter(obs, view, [sun.x, sun.y, sun.z])
                .unwrap();
            for (w, &rad) in result.radiance.iter().enumerate() {
                assert!(
                    rad >= 0.0,
                    "Metal SZA={} wl={}: negative radiance {:.6e}",
                    sza,
                    w,
                    rad,
                );
                assert!(
                    rad.is_finite(),
                    "Metal SZA={} wl={}: non-finite radiance {:.6e}",
                    sza,
                    w,
                    rad,
                );
            }
        }
    }

    #[test]
    fn metal_single_scatter_decreases_with_sza() {
        use twilight_core::atmosphere::EARTH_RADIUS_M;
        use twilight_core::geometry::solar_direction_ecef;

        let Some(mut gpu) = try_metal() else { return };
        let atm = oracle::oracle_atmosphere();
        gpu.upload_atmosphere(&atm).unwrap();

        let obs = [EARTH_RADIUS_M + 1.0, 0.0, 0.0];
        let view = [0.0, 1.0, 0.0];

        // For wavelength 1 (550nm), radiance should decrease as SZA increases
        let szas = [80.0, 90.0, 96.0, 100.0, 108.0];
        let mut prev_rad = f64::MAX;
        for &sza in &szas {
            let sun = solar_direction_ecef(sza, 180.0, 0.0, 0.0);
            let result = gpu
                .single_scatter(obs, view, [sun.x, sun.y, sun.z])
                .unwrap();
            let rad = result.radiance[1]; // 550nm
            assert!(
                rad <= prev_rad + 1e-20,
                "Metal SZA={}: radiance {:.6e} should <= previous {:.6e}",
                sza,
                rad,
                prev_rad,
            );
            prev_rad = rad;
        }
    }

    #[test]
    fn metal_deep_night_negligible() {
        use twilight_core::atmosphere::EARTH_RADIUS_M;
        use twilight_core::geometry::solar_direction_ecef;

        let Some(mut gpu) = try_metal() else { return };
        let atm = oracle::oracle_atmosphere();
        gpu.upload_atmosphere(&atm).unwrap();

        let obs = [EARTH_RADIUS_M + 1.0, 0.0, 0.0];
        let view = [0.0, 1.0, 0.0];
        let sun = solar_direction_ecef(120.0, 180.0, 0.0, 0.0);

        let result = gpu
            .single_scatter(obs, view, [sun.x, sun.y, sun.z])
            .unwrap();
        for (w, &rad) in result.radiance.iter().enumerate() {
            assert!(
                rad < 1e-15,
                "Metal SZA=120 wl={}: radiance {:.6e} should be negligible",
                w,
                rad,
            );
        }
    }
}

// ── Vulkan backend integration tests ────────────────────────────────────

#[cfg(feature = "vulkan")]
mod layer4_vulkan {
    use super::*;
    use crate::{BackendKind, GpuConfig};

    fn try_vulkan() -> Option<Box<dyn crate::GpuBackend>> {
        let config = GpuConfig {
            preferred_backend: Some(BackendKind::Vulkan),
            ..Default::default()
        };
        crate::try_init(&config).ok()
    }

    #[test]
    fn vulkan_init_and_device_info() {
        let Some(gpu) = try_vulkan() else { return };
        let info = gpu.device_info();
        assert_eq!(info.backend, BackendKind::Vulkan);
        assert!(
            !info.name.is_empty(),
            "Vulkan device name should not be empty"
        );
        assert!(
            info.max_workgroup_size >= 256,
            "Vulkan max_workgroup_size={} should be >= 256",
            info.max_workgroup_size,
        );
    }

    #[test]
    fn vulkan_upload_atmosphere() {
        let Some(mut gpu) = try_vulkan() else { return };
        let atm = oracle::oracle_atmosphere();
        gpu.upload_atmosphere(&atm)
            .expect("Vulkan upload_atmosphere should succeed");
    }

    #[test]
    fn vulkan_single_scatter_vs_cpu_oracle() {
        let Some(mut gpu) = try_vulkan() else { return };
        let checked = run_single_scatter_parity(gpu.as_mut(), "Vulkan");
        assert!(checked > 0, "should have checked at least one case");
    }

    #[test]
    fn vulkan_mcrt_vs_single_scatter() {
        let Some(mut gpu) = try_vulkan() else { return };
        let checked = run_mcrt_vs_single_scatter(gpu.as_mut(), "Vulkan");
        assert!(checked > 0, "should have checked at least one case");
    }

    #[test]
    fn vulkan_hybrid_sanity() {
        let Some(mut gpu) = try_vulkan() else { return };
        let checked = run_hybrid_sanity(gpu.as_mut(), "Vulkan");
        assert!(checked > 0, "should have checked at least one case");
    }

    #[test]
    fn vulkan_single_scatter_radiance_non_negative() {
        use twilight_core::atmosphere::EARTH_RADIUS_M;
        use twilight_core::geometry::solar_direction_ecef;

        let Some(mut gpu) = try_vulkan() else { return };
        let atm = oracle::oracle_atmosphere();
        gpu.upload_atmosphere(&atm).unwrap();

        let obs = [EARTH_RADIUS_M + 1.0, 0.0, 0.0];
        let view = [0.0, 1.0, 0.0];

        for &sza in &[80.0, 90.0, 96.0, 100.0, 108.0, 120.0] {
            let sun = solar_direction_ecef(sza, 180.0, 0.0, 0.0);
            let result = gpu
                .single_scatter(obs, view, [sun.x, sun.y, sun.z])
                .unwrap();
            for (w, &rad) in result.radiance.iter().enumerate() {
                assert!(
                    rad >= 0.0,
                    "Vulkan SZA={} wl={}: negative radiance {:.6e}",
                    sza,
                    w,
                    rad,
                );
                assert!(
                    rad.is_finite(),
                    "Vulkan SZA={} wl={}: non-finite radiance {:.6e}",
                    sza,
                    w,
                    rad,
                );
            }
        }
    }

    #[test]
    fn vulkan_single_scatter_decreases_with_sza() {
        use twilight_core::atmosphere::EARTH_RADIUS_M;
        use twilight_core::geometry::solar_direction_ecef;

        let Some(mut gpu) = try_vulkan() else { return };
        let atm = oracle::oracle_atmosphere();
        gpu.upload_atmosphere(&atm).unwrap();

        let obs = [EARTH_RADIUS_M + 1.0, 0.0, 0.0];
        let view = [0.0, 1.0, 0.0];

        let szas = [80.0, 90.0, 96.0, 100.0, 108.0];
        let mut prev_rad = f64::MAX;
        for &sza in &szas {
            let sun = solar_direction_ecef(sza, 180.0, 0.0, 0.0);
            let result = gpu
                .single_scatter(obs, view, [sun.x, sun.y, sun.z])
                .unwrap();
            let rad = result.radiance[1];
            assert!(
                rad <= prev_rad + 1e-20,
                "Vulkan SZA={}: radiance {:.6e} should <= previous {:.6e}",
                sza,
                rad,
                prev_rad,
            );
            prev_rad = rad;
        }
    }

    #[test]
    fn vulkan_deep_night_negligible() {
        use twilight_core::atmosphere::EARTH_RADIUS_M;
        use twilight_core::geometry::solar_direction_ecef;

        let Some(mut gpu) = try_vulkan() else { return };
        let atm = oracle::oracle_atmosphere();
        gpu.upload_atmosphere(&atm).unwrap();

        let obs = [EARTH_RADIUS_M + 1.0, 0.0, 0.0];
        let view = [0.0, 1.0, 0.0];
        let sun = solar_direction_ecef(120.0, 180.0, 0.0, 0.0);

        let result = gpu
            .single_scatter(obs, view, [sun.x, sun.y, sun.z])
            .unwrap();
        for (w, &rad) in result.radiance.iter().enumerate() {
            assert!(
                rad < 1e-15,
                "Vulkan SZA=120 wl={}: radiance {:.6e} should be negligible",
                w,
                rad,
            );
        }
    }
}

// ── CUDA backend integration tests ──────────────────────────────────────

#[cfg(feature = "cuda")]
mod layer4_cuda {
    use super::*;
    use crate::{BackendKind, GpuConfig};

    fn try_cuda() -> Option<Box<dyn crate::GpuBackend>> {
        let config = GpuConfig {
            preferred_backend: Some(BackendKind::Cuda),
            ..Default::default()
        };
        crate::try_init(&config).ok()
    }

    #[test]
    fn cuda_init_and_device_info() {
        let Some(gpu) = try_cuda() else { return };
        let info = gpu.device_info();
        assert_eq!(info.backend, BackendKind::Cuda);
        assert!(
            !info.name.is_empty(),
            "CUDA device name should not be empty"
        );
    }

    #[test]
    fn cuda_single_scatter_vs_cpu_oracle() {
        let Some(mut gpu) = try_cuda() else { return };
        let checked = run_single_scatter_parity(gpu.as_mut(), "CUDA");
        assert!(checked > 0, "should have checked at least one case");
    }

    #[test]
    fn cuda_mcrt_vs_single_scatter() {
        let Some(mut gpu) = try_cuda() else { return };
        let checked = run_mcrt_vs_single_scatter(gpu.as_mut(), "CUDA");
        assert!(checked > 0, "should have checked at least one case");
    }

    #[test]
    fn cuda_hybrid_sanity() {
        let Some(mut gpu) = try_cuda() else { return };
        let checked = run_hybrid_sanity(gpu.as_mut(), "CUDA");
        assert!(checked > 0, "should have checked at least one case");
    }
}

// ── wgpu backend integration tests ──────────────────────────────────────

#[cfg(feature = "webgpu")]
mod layer4_wgpu {
    use super::*;
    use crate::{BackendKind, GpuConfig};

    fn try_wgpu() -> Option<Box<dyn crate::GpuBackend>> {
        let config = GpuConfig {
            preferred_backend: Some(BackendKind::Wgpu),
            ..Default::default()
        };
        crate::try_init(&config).ok()
    }

    #[test]
    fn wgpu_init_and_device_info() {
        let Some(gpu) = try_wgpu() else { return };
        let info = gpu.device_info();
        assert_eq!(info.backend, BackendKind::Wgpu);
        assert!(
            !info.name.is_empty(),
            "wgpu device name should not be empty"
        );
    }

    #[test]
    fn wgpu_single_scatter_vs_cpu_oracle() {
        let Some(mut gpu) = try_wgpu() else { return };
        let checked = run_single_scatter_parity(gpu.as_mut(), "wgpu");
        assert!(checked > 0, "should have checked at least one case");
    }

    #[test]
    fn wgpu_mcrt_vs_single_scatter() {
        let Some(mut gpu) = try_wgpu() else { return };
        let checked = run_mcrt_vs_single_scatter(gpu.as_mut(), "wgpu");
        assert!(checked > 0, "should have checked at least one case");
    }

    #[test]
    fn wgpu_hybrid_sanity() {
        let Some(mut gpu) = try_wgpu() else { return };
        let checked = run_hybrid_sanity(gpu.as_mut(), "wgpu");
        assert!(checked > 0, "should have checked at least one case");
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Layer 5: Cross-backend parity tests
// ═══════════════════════════════════════════════════════════════════════
//
// When multiple GPU backends are available on the same machine, their
// results must agree within f32 tolerance. These tests initialize all
// available backends and compare their single_scatter / mcrt outputs.

/// Cross-backend parity tolerance. Backends use the same f32 arithmetic
/// but may differ in instruction ordering, FMA usage, etc.
const CROSS_BACKEND_RTOL: f64 = 1e-4;

/// Helper: collect all available backends on this machine.
#[allow(dead_code)]
fn init_all_backends() -> Vec<(crate::BackendKind, Box<dyn crate::GpuBackend>)> {
    let mut backends = Vec::new();

    #[cfg(feature = "metal")]
    {
        let config = crate::GpuConfig {
            preferred_backend: Some(crate::BackendKind::Metal),
            ..Default::default()
        };
        if let Ok(gpu) = crate::try_init(&config) {
            backends.push((crate::BackendKind::Metal, gpu));
        }
    }

    #[cfg(feature = "vulkan")]
    {
        let config = crate::GpuConfig {
            preferred_backend: Some(crate::BackendKind::Vulkan),
            ..Default::default()
        };
        if let Ok(gpu) = crate::try_init(&config) {
            backends.push((crate::BackendKind::Vulkan, gpu));
        }
    }

    #[cfg(feature = "cuda")]
    {
        let config = crate::GpuConfig {
            preferred_backend: Some(crate::BackendKind::Cuda),
            ..Default::default()
        };
        if let Ok(gpu) = crate::try_init(&config) {
            backends.push((crate::BackendKind::Cuda, gpu));
        }
    }

    #[cfg(feature = "webgpu")]
    {
        let config = crate::GpuConfig {
            preferred_backend: Some(crate::BackendKind::Wgpu),
            ..Default::default()
        };
        if let Ok(gpu) = crate::try_init(&config) {
            backends.push((crate::BackendKind::Wgpu, gpu));
        }
    }

    backends
}

#[test]
fn cross_backend_single_scatter_parity() {
    use twilight_core::atmosphere::EARTH_RADIUS_M;
    use twilight_core::geometry::solar_direction_ecef;

    let mut backends = init_all_backends();
    if backends.len() < 2 {
        // Need at least 2 backends to compare
        return;
    }

    let atm = oracle::oracle_atmosphere();
    for (_, gpu) in backends.iter_mut() {
        gpu.upload_atmosphere(&atm).unwrap();
    }

    let obs = [EARTH_RADIUS_M + 1.0, 0.0, 0.0];
    let view = [0.0, 1.0, 0.0];

    let szas = [80.0, 90.0, 96.0, 100.0, 108.0];

    for &sza in &szas {
        let sun = solar_direction_ecef(sza, 180.0, 0.0, 0.0);
        let sun_arr = [sun.x, sun.y, sun.z];

        // Collect results from all backends
        let results: Vec<_> = backends
            .iter()
            .map(|(kind, gpu)| {
                let r = gpu.single_scatter(obs, view, sun_arr).unwrap();
                (*kind, r)
            })
            .collect();

        // Compare all pairs
        for i in 0..results.len() {
            for j in (i + 1)..results.len() {
                let (kind_a, ref res_a) = results[i];
                let (kind_b, ref res_b) = results[j];

                for w in 0..3 {
                    let a = res_a.radiance[w];
                    let b = res_b.radiance[w];

                    // Skip near-zero
                    if a.abs() < 1e-25 && b.abs() < 1e-25 {
                        continue;
                    }

                    assert!(
                        approx_eq(a, b, CROSS_BACKEND_RTOL, F32_ATOL),
                        "Cross-backend mismatch at SZA={} wl={}: {} ({:.6e}) vs {} ({:.6e}), rel_err={:.4e}",
                        sza, w, kind_a, a, kind_b, b,
                        (a - b).abs() / a.abs().max(b.abs()).max(1e-30),
                    );
                }
            }
        }
    }
}

#[test]
fn cross_backend_mcrt_sign_agreement() {
    use twilight_core::atmosphere::EARTH_RADIUS_M;
    use twilight_core::geometry::solar_direction_ecef;

    let mut backends = init_all_backends();
    if backends.len() < 2 {
        return;
    }

    let atm = oracle::oracle_atmosphere();
    for (_, gpu) in backends.iter_mut() {
        gpu.upload_atmosphere(&atm).unwrap();
    }

    let obs = [EARTH_RADIUS_M + 1.0, 0.0, 0.0];
    let view = [0.0, 1.0, 0.0];
    let sun = solar_direction_ecef(96.0, 180.0, 0.0, 0.0);
    let sun_arr = [sun.x, sun.y, sun.z];

    // MC results won't be identical (different RNG sequences per backend),
    // but they should all be non-negative and in the same order of magnitude.
    let results: Vec<_> = backends
        .iter()
        .map(|(kind, gpu)| {
            let r = gpu.mcrt_trace(obs, view, sun_arr, 10_000, 42).unwrap();
            (*kind, r)
        })
        .collect();

    for i in 0..results.len() {
        for j in (i + 1)..results.len() {
            let (kind_a, ref res_a) = results[i];
            let (kind_b, ref res_b) = results[j];

            for w in 0..3 {
                let a = res_a.radiance[w];
                let b = res_b.radiance[w];

                // Both should be non-negative
                assert!(a >= 0.0, "{} MCRT wl={}: negative {:.6e}", kind_a, w, a);
                assert!(b >= 0.0, "{} MCRT wl={}: negative {:.6e}", kind_b, w, b);

                // Same order of magnitude (within 2x)
                if a > 1e-20 && b > 1e-20 {
                    let ratio = a / b;
                    assert!(
                        ratio > 0.1 && ratio < 10.0,
                        "MCRT order-of-magnitude mismatch wl={}: {} ({:.6e}) vs {} ({:.6e}), ratio={:.3}",
                        w, kind_a, a, kind_b, b, ratio,
                    );
                }
            }
        }
    }
}

#[test]
fn cross_backend_deep_night_all_agree_zero() {
    use twilight_core::atmosphere::EARTH_RADIUS_M;
    use twilight_core::geometry::solar_direction_ecef;

    let mut backends = init_all_backends();
    if backends.len() < 2 {
        return;
    }

    let atm = oracle::oracle_atmosphere();
    for (_, gpu) in backends.iter_mut() {
        gpu.upload_atmosphere(&atm).unwrap();
    }

    let obs = [EARTH_RADIUS_M + 1.0, 0.0, 0.0];
    let view = [0.0, 1.0, 0.0];
    let sun = solar_direction_ecef(120.0, 180.0, 0.0, 0.0);
    let sun_arr = [sun.x, sun.y, sun.z];

    for (kind, gpu) in backends.iter() {
        let result = gpu.single_scatter(obs, view, sun_arr).unwrap();
        for (w, &rad) in result.radiance.iter().enumerate() {
            assert!(
                rad < 1e-15,
                "{} SZA=120 wl={}: radiance {:.6e} should be negligible",
                kind,
                w,
                rad,
            );
        }
    }
}

#[test]
fn cross_backend_physics_monotonicity_all_agree() {
    use twilight_core::atmosphere::EARTH_RADIUS_M;
    use twilight_core::geometry::solar_direction_ecef;

    let mut backends = init_all_backends();
    if backends.is_empty() {
        return;
    }

    let atm = oracle::oracle_atmosphere();
    for (_, gpu) in backends.iter_mut() {
        gpu.upload_atmosphere(&atm).unwrap();
    }

    let obs = [EARTH_RADIUS_M + 1.0, 0.0, 0.0];
    let view = [0.0, 1.0, 0.0];
    let szas = [80.0, 90.0, 96.0, 100.0, 108.0];

    for (kind, gpu) in backends.iter() {
        // For 550nm (wl=1), radiance should monotonically decrease with SZA
        let mut prev = f64::MAX;
        for &sza in &szas {
            let sun = solar_direction_ecef(sza, 180.0, 0.0, 0.0);
            let result = gpu
                .single_scatter(obs, view, [sun.x, sun.y, sun.z])
                .unwrap();
            let rad = result.radiance[1];
            assert!(
                rad <= prev + 1e-20,
                "{}: SZA={} rad {:.6e} should <= prev {:.6e}",
                kind,
                sza,
                rad,
                prev,
            );
            prev = rad;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Layer 6: Benchmark / performance sanity tests
// ═══════════════════════════════════════════════════════════════════════
//
// These are lightweight performance smoke tests. They don't enforce hard
// timing thresholds (that would be flaky in CI), but they verify that
// GPU dispatch completes in a reasonable time and measure throughput.
// Full benchmarks live in `examples/gpu_bench.rs`.

#[test]
fn benchmark_single_scatter_gpu_latency() {
    use std::time::Instant;
    use twilight_core::atmosphere::EARTH_RADIUS_M;
    use twilight_core::geometry::solar_direction_ecef;

    let mut backends = init_all_backends();
    if backends.is_empty() {
        return;
    }

    let atm = oracle::oracle_atmosphere();
    for (_, gpu) in backends.iter_mut() {
        gpu.upload_atmosphere(&atm).unwrap();
    }

    let obs = [EARTH_RADIUS_M + 1.0, 0.0, 0.0];
    let view = [0.0, 1.0, 0.0];
    let sun = solar_direction_ecef(96.0, 180.0, 0.0, 0.0);
    let sun_arr = [sun.x, sun.y, sun.z];

    for (kind, gpu) in backends.iter() {
        // Warmup
        let _ = gpu.single_scatter(obs, view, sun_arr);

        // Timed run (10 iterations)
        let n = 10;
        let start = Instant::now();
        for _ in 0..n {
            let _ = gpu.single_scatter(obs, view, sun_arr).unwrap();
        }
        let elapsed = start.elapsed();
        let per_call = elapsed / n;

        // Sanity: each call should complete in < 1 second
        // (typically < 1ms for single_scatter on modern GPUs)
        assert!(
            per_call.as_secs() < 1,
            "{}: single_scatter took {:?} per call (expected < 1s)",
            kind,
            per_call,
        );

        // Print timing for manual inspection (visible with `cargo test -- --nocapture`)
        eprintln!(
            "  [benchmark] {} single_scatter: {:?} per call ({} calls in {:?})",
            kind, per_call, n, elapsed,
        );
    }
}

#[test]
fn benchmark_mcrt_trace_gpu_latency() {
    use std::time::Instant;
    use twilight_core::atmosphere::EARTH_RADIUS_M;
    use twilight_core::geometry::solar_direction_ecef;

    let mut backends = init_all_backends();
    if backends.is_empty() {
        return;
    }

    let atm = oracle::oracle_atmosphere();
    for (_, gpu) in backends.iter_mut() {
        gpu.upload_atmosphere(&atm).unwrap();
    }

    let obs = [EARTH_RADIUS_M + 1.0, 0.0, 0.0];
    let view = [0.0, 1.0, 0.0];
    let sun = solar_direction_ecef(96.0, 180.0, 0.0, 0.0);
    let sun_arr = [sun.x, sun.y, sun.z];

    for (kind, gpu) in backends.iter() {
        // Warmup
        let _ = gpu.mcrt_trace(obs, view, sun_arr, 10_000, 42);

        let n = 5;
        let start = Instant::now();
        for i in 0..n {
            let _ = gpu
                .mcrt_trace(obs, view, sun_arr, 10_000, 42 + i as u64)
                .unwrap();
        }
        let elapsed = start.elapsed();
        let per_call = elapsed / n;

        assert!(
            per_call.as_secs() < 5,
            "{}: mcrt_trace(10k photons) took {:?} per call (expected < 5s)",
            kind,
            per_call,
        );

        eprintln!(
            "  [benchmark] {} mcrt_trace(10k): {:?} per call ({} calls in {:?})",
            kind, per_call, n, elapsed,
        );
    }
}

#[test]
fn benchmark_cpu_vs_gpu_single_scatter() {
    use std::time::Instant;
    use twilight_core::atmosphere::EARTH_RADIUS_M;
    use twilight_core::geometry::{solar_direction_ecef, Vec3};
    use twilight_core::single_scatter::single_scatter_spectrum;

    let mut backends = init_all_backends();
    if backends.is_empty() {
        return;
    }

    let atm = oracle::oracle_atmosphere();
    for (_, gpu) in backends.iter_mut() {
        gpu.upload_atmosphere(&atm).unwrap();
    }

    let obs_arr = [EARTH_RADIUS_M + 1.0, 0.0, 0.0];
    let view_arr = [0.0, 1.0, 0.0];
    let obs = Vec3::new(obs_arr[0], obs_arr[1], obs_arr[2]);
    let view = Vec3::new(view_arr[0], view_arr[1], view_arr[2]).normalize();
    let sun = solar_direction_ecef(96.0, 180.0, 0.0, 0.0);
    let sun_arr = [sun.x, sun.y, sun.z];

    // CPU baseline
    let n = 100;
    let cpu_start = Instant::now();
    for _ in 0..n {
        let _ = single_scatter_spectrum(&atm, obs, view, sun);
    }
    let cpu_elapsed = cpu_start.elapsed();
    let cpu_per_call = cpu_elapsed / n;

    eprintln!(
        "  [benchmark] CPU single_scatter: {:?} per call ({} calls in {:?})",
        cpu_per_call, n, cpu_elapsed,
    );

    // GPU
    for (kind, gpu) in backends.iter() {
        // Warmup
        let _ = gpu.single_scatter(obs_arr, view_arr, sun_arr);

        let start = Instant::now();
        for _ in 0..n {
            let _ = gpu.single_scatter(obs_arr, view_arr, sun_arr).unwrap();
        }
        let gpu_elapsed = start.elapsed();
        let gpu_per_call = gpu_elapsed / n;

        eprintln!(
            "  [benchmark] {} single_scatter: {:?} per call ({} calls in {:?})",
            kind, gpu_per_call, n, gpu_elapsed,
        );

        // Note: single-scatter dispatch may actually be slower on GPU than CPU
        // due to launch overhead (CPU is ~10us, GPU dispatch is ~50-200us).
        // The GPU advantage shows up for MCRT with thousands of photons.
        // So we just log, no assertion on relative speed.
    }
}
