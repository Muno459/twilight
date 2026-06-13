use twilight_cpu::simulation::{self, ScatteringMode, SimulationConfig};
use twilight_data::atmosphere_profiles::AtmosphereType;

fn main() {
    let p = std::env::args().nth(1).expect("field path");
    let owned =
        twilight_weather::cloud3d::load_field(std::path::Path::new(&p)).expect("load");
    let mut atm = twilight_data::builder::build_clear_sky(AtmosphereType::UsStandard, 0.15);
    atm.cloud_g_scaled = owned.g_default;
    let config = SimulationConfig {
        latitude: 54.83,
        longitude: 9.36,
        scattering_mode: ScatteringMode::Single,
        ..SimulationConfig::default()
    };
    for sza in [94.0, 96.0, 99.5] {
        let a: f64 = simulation::simulate_at_sza(&atm, &config, sza, None)
            .radiance
            .iter()
            .sum();
        let b: f64 = simulation::simulate_at_sza(&atm, &config, sza, Some(&owned.view()))
            .radiance
            .iter()
            .sum();
        println!(
            "SZA {sza}: clear {a:.6e}  field {b:.6e}  ratio {:.6}",
            b / a
        );
    }
}
