fn main() {
    let cache = std::path::Path::new("data/skyglow");
    for (name, lat, lon) in [("Brondby DK", 55.653, 12.412), ("Mecca", 21.4225, 39.8262), ("mid-Atlantic", 45.0, -35.0)] {
        match twilight_skyglow::atlas::artificial_zenith(cache, lat, lon) {
            Some(a) => println!("{name}: {:.3} mcd/m^2 (ratio {:.3}, atlas {})", a.zenith_mcd, a.brightness_ratio, a.year),
            None => println!("{name}: no data"),
        }
    }
}
