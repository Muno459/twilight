// G3-LEDGER: explicit-MS chain (Stage 2) under uniform 1D decks, summed
// radiance at SZA 95/100, OD-3 and OD-10 stratus, Hybrid mode. Measures
// the delta vs the Stage-1 T_diff closure (recorded separately). Also
// probes convergence (photon count) to separate bias from variance.

use twilight_cpu::simulation::{self, ScatteringMode, SimulationConfig};
use twilight_data::atmosphere_profiles::AtmosphereType;
use twilight_data::cloud::CloudProperties;

fn run(od: f64, sza: f64, photons: usize, seeds: u64) -> f64 {
    let props = CloudProperties {
        base_km: 0.5,
        top_km: 1.5,
        optical_depth: od,
        ssa: 0.999,
        asymmetry: 0.85,
    };
    let cloudy = twilight_data::builder::build_with_cloud_properties(
        AtmosphereType::UsStandard,
        0.15,
        &props,
    );
    let mut acc = 0.0;
    for s in 0..seeds {
        let config = SimulationConfig {
            view_zenith: 85.0,
            scattering_mode: ScatteringMode::Hybrid,
            photons_per_wavelength: photons,
            polarized: false,
            seed_salt: s.wrapping_mul(0x9E3779B97F4A7C15),
            ..SimulationConfig::default()
        };
        let r: f64 = simulation::simulate_at_sza(&cloudy, &config, sza, None)
            .radiance
            .iter()
            .sum();
        acc += r;
    }
    acc / seeds as f64
}

fn main() {
    println!("G3-LEDGER explicit-MS (1D fallback, Hybrid, scalar/ALIS)");
    for &od in &[3.0, 10.0] {
        for &sza in &[95.0, 100.0] {
            let lo = run(od, sza, 50, 8);
            let hi = run(od, sza, 400, 8);
            println!(
                "OD {od:4.1} SZA {sza:5.1}: P=50 {lo:.4e}   P=400 {hi:.4e}   ratio {:.2}",
                if lo > 0.0 { hi / lo } else { 0.0 }
            );
        }
    }
}
