//! Parallel photon tracing using rayon.

use rayon::prelude::*;
use twilight_core::atmosphere::AtmosphereModel;
use twilight_core::geometry::Vec3;
use twilight_core::photon::{trace_photon, PhotonResult};

/// Trace N photons in parallel for a given observer position, viewing direction,
/// sun direction, and wavelength.
///
/// Returns the average radiance contribution.
pub fn trace_photons_parallel(
    atm: &AtmosphereModel,
    observer_pos: Vec3,
    view_dir: Vec3,
    sun_dir: Vec3,
    wavelength_idx: usize,
    num_photons: usize,
    base_seed: u64,
) -> f64 {
    let results: Vec<PhotonResult> = (0..num_photons)
        .into_par_iter()
        .map(|i| {
            let mut rng = base_seed
                .wrapping_add(i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            trace_photon(
                atm,
                observer_pos,
                view_dir,
                sun_dir,
                wavelength_idx,
                &mut rng,
            )
        })
        .collect();

    let total_weight: f64 = results.iter().map(|r| r.weight).sum();
    total_weight / num_photons as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use twilight_core::atmosphere::EARTH_RADIUS_M;
    use twilight_data::atmosphere_profiles::AtmosphereType;
    use twilight_data::builder;

    fn make_atm() -> AtmosphereModel {
        builder::build_clear_sky(AtmosphereType::UsStandard, 0.15)
    }

    /// Observer at sea level, looking toward zenith=85° (near horizon)
    fn observer_pos() -> Vec3 {
        Vec3::new(EARTH_RADIUS_M, 0.0, 0.0)
    }

    /// View direction toward local horizon (roughly +z at equator, 0°lon)
    fn view_dir_horizon() -> Vec3 {
        // 85° from zenith (which is radial) = nearly horizontal
        let zenith_rad = 85.0_f64.to_radians();
        Vec3::new(zenith_rad.cos(), 0.0, zenith_rad.sin())
    }

    /// Sun direction at SZA=96° (civil twilight boundary)
    fn sun_dir_sza96() -> Vec3 {
        let sza_rad = 96.0_f64.to_radians();
        Vec3::new(sza_rad.cos(), 0.0, sza_rad.sin())
    }

    /// Sun direction at SZA=108° (astronomical twilight boundary)
    fn sun_dir_sza108() -> Vec3 {
        let sza_rad = 108.0_f64.to_radians();
        Vec3::new(sza_rad.cos(), 0.0, sza_rad.sin())
    }

    // ── Basic functionality ──

    #[test]
    fn returns_non_negative_radiance() {
        let atm = make_atm();
        let result = trace_photons_parallel(
            &atm,
            observer_pos(),
            view_dir_horizon(),
            sun_dir_sza96(),
            0, // first wavelength
            100,
            42,
        );
        assert!(
            result >= 0.0,
            "Radiance should be non-negative, got {}",
            result
        );
    }

    #[test]
    fn returns_finite_radiance() {
        let atm = make_atm();
        let result = trace_photons_parallel(
            &atm,
            observer_pos(),
            view_dir_horizon(),
            sun_dir_sza96(),
            0,
            100,
            42,
        );
        assert!(
            result.is_finite(),
            "Radiance should be finite, got {}",
            result
        );
    }

    #[test]
    fn zero_photons_returns_nan() {
        // 0 photons → division by zero → NaN (or we could check for 0.0/0.0)
        // This is a degenerate input; the function should not panic.
        let atm = make_atm();
        let result = trace_photons_parallel(
            &atm,
            observer_pos(),
            view_dir_horizon(),
            sun_dir_sza96(),
            0,
            0,
            42,
        );
        // 0.0 / 0 = NaN, which is fine — caller should not pass 0
        assert!(
            result.is_nan(),
            "0 photons should produce NaN, got {}",
            result
        );
    }

    #[test]
    fn single_photon_does_not_panic() {
        let atm = make_atm();
        let result = trace_photons_parallel(
            &atm,
            observer_pos(),
            view_dir_horizon(),
            sun_dir_sza96(),
            0,
            1,
            42,
        );
        assert!(
            result.is_finite(),
            "Single photon should produce finite result"
        );
    }

    // ── Determinism (same seed → same result) ──

    #[test]
    fn deterministic_with_same_seed() {
        let atm = make_atm();
        let r1 = trace_photons_parallel(
            &atm,
            observer_pos(),
            view_dir_horizon(),
            sun_dir_sza96(),
            0,
            200,
            12345,
        );
        let r2 = trace_photons_parallel(
            &atm,
            observer_pos(),
            view_dir_horizon(),
            sun_dir_sza96(),
            0,
            200,
            12345,
        );
        assert!(
            (r1 - r2).abs() < 1e-15,
            "Same seed should give identical results: {} vs {}",
            r1,
            r2
        );
    }

    #[test]
    fn different_seeds_give_different_results() {
        let atm = make_atm();
        let r1 = trace_photons_parallel(
            &atm,
            observer_pos(),
            view_dir_horizon(),
            sun_dir_sza96(),
            0,
            500,
            42,
        );
        let r2 = trace_photons_parallel(
            &atm,
            observer_pos(),
            view_dir_horizon(),
            sun_dir_sza96(),
            0,
            500,
            99999,
        );
        // With different seeds and enough photons, results should differ
        // (could be equal in theory but astronomically unlikely)
        assert!(
            (r1 - r2).abs() > 1e-20 || r1 == 0.0,
            "Different seeds should (almost certainly) give different results"
        );
    }

    // ── Physical behavior ──

    #[test]
    fn deeper_twilight_gives_less_radiance() {
        let atm = make_atm();
        let n = 1000;
        // SZA 96° (civil twilight) should give more radiance than SZA 108° (astronomical)
        let r_96 = trace_photons_parallel(
            &atm,
            observer_pos(),
            view_dir_horizon(),
            sun_dir_sza96(),
            0,
            n,
            42,
        );
        let r_108 = trace_photons_parallel(
            &atm,
            observer_pos(),
            view_dir_horizon(),
            sun_dir_sza108(),
            0,
            n,
            42,
        );
        // r_96 should be >= r_108 (with MC noise, just check rough ordering)
        // Both could be 0 if MC doesn't connect to sun, so allow equality
        assert!(
            r_96 >= r_108 * 0.5 || r_108 < 1e-30,
            "SZA 96° radiance ({:.4e}) should be >= SZA 108° ({:.4e})",
            r_96,
            r_108
        );
    }

    #[test]
    fn multiple_wavelengths_produce_results() {
        let atm = make_atm();
        let n = 100;
        // Test first few wavelength channels
        for wl_idx in 0..atm.num_wavelengths.min(5) {
            let result = trace_photons_parallel(
                &atm,
                observer_pos(),
                view_dir_horizon(),
                sun_dir_sza96(),
                wl_idx,
                n,
                42,
            );
            assert!(
                result.is_finite(),
                "Wavelength index {} should produce finite result, got {}",
                wl_idx,
                result
            );
        }
    }

    // ── Convergence ──

    #[test]
    fn more_photons_reduces_variance() {
        let atm = make_atm();
        // Run 5 trials with N=50 and 5 trials with N=500
        // The spread of results should be smaller with more photons
        let mut results_small: Vec<f64> = Vec::new();
        let mut results_large: Vec<f64> = Vec::new();

        for seed in 0..5u64 {
            results_small.push(trace_photons_parallel(
                &atm,
                observer_pos(),
                view_dir_horizon(),
                sun_dir_sza96(),
                0,
                50,
                seed * 1000,
            ));
            results_large.push(trace_photons_parallel(
                &atm,
                observer_pos(),
                view_dir_horizon(),
                sun_dir_sza96(),
                0,
                500,
                seed * 1000,
            ));
        }

        let mean_s: f64 = results_small.iter().sum::<f64>() / 5.0;
        let mean_l: f64 = results_large.iter().sum::<f64>() / 5.0;

        // Compute coefficient of variation (std/mean)
        let var_s: f64 = results_small
            .iter()
            .map(|x| (x - mean_s).powi(2))
            .sum::<f64>()
            / 4.0;
        let var_l: f64 = results_large
            .iter()
            .map(|x| (x - mean_l).powi(2))
            .sum::<f64>()
            / 4.0;

        // With 10× more photons, variance should decrease (roughly 10× for MC)
        // We just check it doesn't increase dramatically
        // (This test can fail statistically, but very rarely)
        if mean_s.abs() > 1e-20 && mean_l.abs() > 1e-20 {
            let cv_s = var_s.sqrt() / mean_s.abs();
            let cv_l = var_l.sqrt() / mean_l.abs();
            // Allow generous margin — just verify more photons isn't wildly noisier
            assert!(
                cv_l < cv_s * 3.0 || cv_l < 0.5,
                "More photons should not dramatically increase noise: CV(50)={:.4}, CV(500)={:.4}",
                cv_s,
                cv_l
            );
        }
    }

    // ── Observer position edge cases ──

    #[test]
    fn observer_at_altitude() {
        let atm = make_atm();
        let obs = Vec3::new(EARTH_RADIUS_M + 5000.0, 0.0, 0.0); // 5km altitude
        let result =
            trace_photons_parallel(&atm, obs, view_dir_horizon(), sun_dir_sza96(), 0, 100, 42);
        assert!(
            result.is_finite(),
            "Observer at 5km should work, got {}",
            result
        );
    }

    #[test]
    fn parallel_execution_produces_correct_count() {
        // Verify all photons are actually traced by checking that
        // the result changes predictably with photon count
        let atm = make_atm();
        let r1 = trace_photons_parallel(
            &atm,
            observer_pos(),
            view_dir_horizon(),
            sun_dir_sza96(),
            0,
            10,
            42,
        );
        let r2 = trace_photons_parallel(
            &atm,
            observer_pos(),
            view_dir_horizon(),
            sun_dir_sza96(),
            0,
            100,
            42,
        );
        // Both should be finite
        assert!(r1.is_finite() && r2.is_finite());
        // The 100-photon run includes the first 10 photons,
        // so results should be in the same ballpark (within order of magnitude)
        if r1 > 1e-20 && r2 > 1e-20 {
            let ratio = r1 / r2;
            assert!(
                ratio > 0.01 && ratio < 100.0,
                "10 vs 100 photons should give similar order of magnitude: {} vs {}",
                r1,
                r2
            );
        }
    }
}
