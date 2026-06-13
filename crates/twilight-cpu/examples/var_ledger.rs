// G-VAR ledger: coefficient of variation (CV = std/mean) of summed
// radiance over K independent seeds at a fixed photon count, on the real
// Padborg field. Run on the Stage-1 commit to record the baseline, then
// again after Stage 2 to bound the regression factor.
//
// Usage: cargo run --release -p twilight-cpu --example var_ledger -- /tmp/padborg_field.bin

use twilight_cpu::simulation::{self, ScatteringMode, SimulationConfig};
use twilight_data::atmosphere_profiles::AtmosphereType;

fn cv(samples: &[f64]) -> (f64, f64, f64) {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let var = samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();
    let cv = if mean.abs() > 1e-300 { std / mean } else { 0.0 };
    (mean, std, cv)
}

fn main() {
    let p = std::env::args().nth(1).expect("field path");
    let owned = twilight_weather::cloud3d::load_field(std::path::Path::new(&p)).expect("load");
    let view = owned.view();
    let mut atm = twilight_data::builder::build_clear_sky(AtmosphereType::UsStandard, 0.15);
    atm.cloud_g_scaled = owned.g_default;
    // Field owns all cloud: zero the shell channel.
    atm.cloud_extinction = [0.0; twilight_core::atmosphere::MAX_SHELLS];

    let k_seeds = 8u64;
    let photons = 64usize;

    for &polarized in &[false, true] {
        for &sza in &[96.0, 100.0] {
            let mut samples = Vec::new();
            for s in 0..k_seeds {
                let config = SimulationConfig {
                    latitude: 54.83,
                    longitude: 9.36,
                    scattering_mode: ScatteringMode::Hybrid,
                    photons_per_wavelength: photons,
                    polarized,
                    seed_salt: s.wrapping_mul(0x9E3779B97F4A7C15),
                    ..SimulationConfig::default()
                };
                let r: f64 = simulation::simulate_at_sza(&atm, &config, sza, Some(&view))
                    .radiance
                    .iter()
                    .sum();
                samples.push(r);
            }
            let (mean, std, cv) = cv(&samples);
            let mode = if polarized { "Stokes" } else { "ALIS  " };
            println!(
                "{mode} SZA {sza:5.1}: mean {mean:.6e}  std {std:.3e}  CV {cv:.4}  (K={k_seeds}, P={photons})"
            );
        }
    }
}
