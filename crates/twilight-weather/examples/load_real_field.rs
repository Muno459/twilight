fn main() {
    let p = std::env::args().nth(1).expect("path");
    let f = twilight_weather::cloud3d::load_field(std::path::Path::new(&p)).expect("load");
    let nonzero = f.sigma.iter().filter(|v| **v > 0.0).count();
    let max = f.sigma.iter().cloned().fold(0.0f32, f32::max);
    println!(
        "dims {}x{}x{}  z0 {} dz {}  lat0 {:.3} dlat {:.5}  nonzero {} / {}  max sigma {:.4e}  src '{}' @ {}",
        f.nz, f.nlat, f.nlon, f.z0_m, f.dz_m, f.lat0_deg, f.dlat_deg,
        nonzero, f.sigma.len(), max, f.source, f.timestamp
    );
    assert!(nonzero > 0, "field is empty");
    let bg_max = f.background_column.iter().cloned().fold(0.0f32, f32::max);
    let mc_max = f.macrocell_max.iter().cloned().fold(0.0f32, f32::max);
    println!("background column max {bg_max:.4e}, macrocell max {mc_max:.4e}");
    assert!((mc_max - max).abs() / max < 1e-6, "macrocell majorant must cover max");
}
