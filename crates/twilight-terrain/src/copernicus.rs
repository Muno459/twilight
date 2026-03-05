//! Copernicus DEM GLO-30 backend.
//!
//! Downloads 1x1 degree tiles from the public AWS S3 bucket at 30m resolution.
//! Tiles are Cloud Optimized GeoTIFFs (COG) with float32 elevation values.
//!
//! Bucket: s3://copernicus-dem-30m (public, no auth required)
//! URL pattern: https://copernicus-dem-30m.s3.amazonaws.com/
//!   Copernicus_DSM_COG_10_{NS}{lat:02}_00_{EW}{lon:03}_00_DEM/
//!   Copernicus_DSM_COG_10_{NS}{lat:02}_00_{EW}{lon:03}_00_DEM.tif

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::cache;
use crate::geotiff::{ElevationTile, GeoTransform};
use crate::ElevationSource;

/// Copernicus DEM GLO-30 elevation source.
pub struct CopernicusDem30 {
    /// Cache directory for downloaded tiles.
    cache_dir: PathBuf,
    /// Loaded tiles, keyed by (lat_floor, lon_floor).
    tiles: HashMap<(i32, i32), Option<ElevationTile>>,
}

impl CopernicusDem30 {
    pub fn new(dem_dir: &Path) -> Self {
        CopernicusDem30 {
            cache_dir: cache::source_dir(dem_dir, "copernicus"),
            tiles: HashMap::new(),
        }
    }

    /// Get the tile key (southwest corner) for a coordinate.
    fn tile_key(lat: f64, lon: f64) -> (i32, i32) {
        (lat.floor() as i32, lon.floor() as i32)
    }

    /// Construct the S3 URL for a tile.
    fn tile_url(lat_floor: i32, lon_floor: i32) -> String {
        let ns = if lat_floor >= 0 { 'N' } else { 'S' };
        let ew = if lon_floor >= 0 { 'E' } else { 'W' };
        let lat_abs = lat_floor.unsigned_abs();
        let lon_abs = lon_floor.unsigned_abs();

        let dir_name = format!(
            "Copernicus_DSM_COG_10_{}{:02}_00_{}{:03}_00_DEM",
            ns, lat_abs, ew, lon_abs
        );
        format!(
            "https://copernicus-dem-30m.s3.amazonaws.com/{}/{}.tif",
            dir_name, dir_name
        )
    }

    /// Cache filename for a tile.
    fn tile_filename(lat_floor: i32, lon_floor: i32) -> String {
        let ns = if lat_floor >= 0 { 'N' } else { 'S' };
        let ew = if lon_floor >= 0 { 'E' } else { 'W' };
        format!(
            "cop30_{}{:02}_{}{:03}.tif",
            ns,
            lat_floor.unsigned_abs(),
            ew,
            lon_floor.unsigned_abs()
        )
    }

    /// Build the GeoTransform for a tile based on its position and actual dimensions.
    fn geo_transform_for_tile(
        lat_floor: i32,
        lon_floor: i32,
        width: u32,
        height: u32,
    ) -> GeoTransform {
        // Copernicus tiles cover exactly 1 degree in lat and lon.
        // Pixel scale = 1.0 / width for lon, 1.0 / height for lat.
        // Origin is top-left = (lon_floor, lat_floor + 1).
        let pixel_size_x = 1.0 / width as f64;
        let pixel_size_y = 1.0 / height as f64;
        GeoTransform {
            origin_x: lon_floor as f64,
            origin_y: (lat_floor + 1) as f64,
            pixel_size_x,
            pixel_size_y,
        }
    }

    /// Load a tile, downloading if necessary. Returns None for ocean tiles.
    fn load_tile(&mut self, lat_floor: i32, lon_floor: i32) -> Option<&ElevationTile> {
        let key = (lat_floor, lon_floor);
        if !self.tiles.contains_key(&key) {
            let tile = self.download_and_parse(lat_floor, lon_floor);
            self.tiles.insert(key, tile);
        }
        self.tiles.get(&key).and_then(|t| t.as_ref())
    }

    fn download_and_parse(&self, lat_floor: i32, lon_floor: i32) -> Option<ElevationTile> {
        let filename = Self::tile_filename(lat_floor, lon_floor);
        let url = Self::tile_url(lat_floor, lon_floor);

        let path = match cache::download_to_cache(&self.cache_dir, &filename, &url) {
            Ok(p) => p,
            Err(e) => {
                if e.contains("404") {
                    // Ocean tile, no data
                    return None;
                }
                eprintln!("Warning: failed to download DEM tile: {}", e);
                return None;
            }
        };

        // We need to figure out the tile dimensions to build the GeoTransform.
        // Read the file to get dimensions, then build the transform.
        match load_tile_from_path(&path, lat_floor, lon_floor) {
            Ok(tile) => Some(tile),
            Err(e) => {
                eprintln!("Warning: failed to parse DEM tile {}: {}", filename, e);
                None
            }
        }
    }
}

/// Load a tile from a file path with the correct GeoTransform.
fn load_tile_from_path(
    path: &Path,
    lat_floor: i32,
    lon_floor: i32,
) -> Result<ElevationTile, String> {
    use std::fs::File;
    use tiff::decoder::Decoder;

    // First pass: get dimensions
    let file = File::open(path).map_err(|e| format!("Failed to open: {}", e))?;
    let mut decoder = Decoder::new(file).map_err(|e| format!("Failed to decode: {}", e))?;
    let (width, height) = decoder
        .dimensions()
        .map_err(|e| format!("Failed to read dimensions: {}", e))?;

    let geo = CopernicusDem30::geo_transform_for_tile(lat_floor, lon_floor, width, height);
    ElevationTile::from_file_with_geo(path, geo)
}

impl ElevationSource for CopernicusDem30 {
    fn elevation_at(&self, lat: f64, lon: f64) -> Option<f32> {
        let (lat_floor, lon_floor) = Self::tile_key(lat, lon);
        // We need &mut self for lazy loading, but the trait takes &self.
        // This is a design tension. For now, all tiles must be loaded via prepare().
        let key = (lat_floor, lon_floor);
        self.tiles
            .get(&key)
            .and_then(|t| t.as_ref())
            .and_then(|tile| tile.elevation_at_geo(lon, lat))
    }

    fn resolution_m(&self) -> f64 {
        30.0
    }

    fn name(&self) -> &str {
        "Copernicus DEM GLO-30 (30m)"
    }

    fn prepare(
        &mut self,
        min_lat: f64,
        min_lon: f64,
        max_lat: f64,
        max_lon: f64,
    ) -> Result<(), String> {
        let lat_lo = min_lat.floor() as i32;
        let lat_hi = max_lat.floor() as i32;
        let lon_lo = min_lon.floor() as i32;
        let lon_hi = max_lon.floor() as i32;

        let total = ((lat_hi - lat_lo + 1) * (lon_hi - lon_lo + 1)) as usize;
        let mut loaded = 0;
        let errors: Vec<String> = Vec::new();

        for lat_f in lat_lo..=lat_hi {
            for lon_f in lon_lo..=lon_hi {
                self.load_tile(lat_f, lon_f);
                loaded += 1;
                if loaded < total {
                    eprint!("\r  Loading DEM tiles: {}/{}", loaded, total);
                }
            }
        }
        if total > 1 {
            eprintln!("\r  Loaded {} DEM tiles.          ", total);
        }

        if !errors.is_empty() {
            return Err(errors.join("; "));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_key_positive() {
        assert_eq!(CopernicusDem30::tile_key(21.4225, 39.8262), (21, 39));
    }

    #[test]
    fn tile_key_negative() {
        assert_eq!(CopernicusDem30::tile_key(-33.87, -73.95), (-34, -74));
    }

    #[test]
    fn tile_key_boundary() {
        // Exactly on a degree boundary
        assert_eq!(CopernicusDem30::tile_key(55.0, 9.0), (55, 9));
    }

    #[test]
    fn tile_url_mecca() {
        let url = CopernicusDem30::tile_url(21, 39);
        assert_eq!(
            url,
            "https://copernicus-dem-30m.s3.amazonaws.com/\
             Copernicus_DSM_COG_10_N21_00_E039_00_DEM/\
             Copernicus_DSM_COG_10_N21_00_E039_00_DEM.tif"
        );
    }

    #[test]
    fn tile_url_southern_western() {
        let url = CopernicusDem30::tile_url(-34, -74);
        assert!(url.contains("S34"));
        assert!(url.contains("W074"));
    }

    #[test]
    fn tile_filename_format() {
        let f = CopernicusDem30::tile_filename(55, 9);
        assert_eq!(f, "cop30_N55_E009.tif");

        let f = CopernicusDem30::tile_filename(-34, -74);
        assert_eq!(f, "cop30_S34_W074.tif");
    }

    #[test]
    fn geo_transform_equatorial() {
        let geo = CopernicusDem30::geo_transform_for_tile(21, 39, 3600, 3600);
        assert!((geo.origin_x - 39.0).abs() < 1e-10);
        assert!((geo.origin_y - 22.0).abs() < 1e-10);
        assert!((geo.pixel_size_x - 1.0 / 3600.0).abs() < 1e-15);
    }

    #[test]
    fn geo_transform_high_latitude() {
        // At 70°N, tiles are 1200 pixels wide (fewer lon pixels)
        let geo = CopernicusDem30::geo_transform_for_tile(70, 25, 1200, 3600);
        assert!((geo.origin_x - 25.0).abs() < 1e-10);
        assert!((geo.origin_y - 71.0).abs() < 1e-10);
        assert!((geo.pixel_size_x - 1.0 / 1200.0).abs() < 1e-15);
        assert!((geo.pixel_size_y - 1.0 / 3600.0).abs() < 1e-15);
    }

    // ── Integration tests with real DEM tiles ──
    // These are ignored by default (require downloaded tiles at /tmp/).
    // Run with: cargo test -p twilight-terrain -- --ignored

    #[test]
    #[ignore]
    fn real_tile_mecca_loads() {
        // Requires: /tmp/test_dem_mecca.tif (Copernicus N21_E039)
        let path = std::path::Path::new("/tmp/test_dem_mecca.tif");
        if !path.exists() {
            eprintln!("Skipping: /tmp/test_dem_mecca.tif not found");
            return;
        }
        let tile = load_tile_from_path(path, 21, 39).unwrap();
        assert_eq!(tile.width, 3600);
        assert_eq!(tile.height, 3600);
        assert!(tile.data.len() == 3600 * 3600);
    }

    #[test]
    #[ignore]
    fn real_tile_mecca_elevation_at_kaaba() {
        // The Kaaba is at approximately 21.4225°N, 39.8262°E, elevation ~277m
        let path = std::path::Path::new("/tmp/test_dem_mecca.tif");
        if !path.exists() {
            return;
        }
        let tile = load_tile_from_path(path, 21, 39).unwrap();
        let elev = tile.elevation_at_geo(39.8262, 21.4225);
        assert!(elev.is_some(), "Should have elevation at Kaaba");
        let e = elev.unwrap();
        // Kaaba area is ~270-290m
        assert!(
            e > 250.0 && e < 320.0,
            "Kaaba elevation = {}m, expected 250-320m",
            e
        );
    }

    #[test]
    #[ignore]
    fn real_tile_mecca_horizon_profile() {
        let path = std::path::Path::new("/tmp/test_dem_mecca.tif");
        if !path.exists() {
            return;
        }

        // Load the tile manually so we can use it as an ElevationSource
        let tile = load_tile_from_path(path, 21, 39).unwrap();

        // Build a CopernicusDem30 with the tile pre-loaded
        let mut source = CopernicusDem30::new(std::path::Path::new("/tmp/copernicus_test"));
        source.tiles.insert((21, 39), Some(tile));

        // Compute horizon at the Kaaba
        let profile = crate::horizon::compute_horizon(&source, 21.4225, 39.8262, 15.0);

        // Mecca is in a valley surrounded by mountains.
        // The horizon should have some positive angles.
        let max_hz = profile.max_angle();
        assert!(
            max_hz > 0.5,
            "Mecca should have terrain masking (max horizon = {:.3}°)",
            max_hz
        );

        // Observer elevation should be around 270-290m
        assert!(
            profile.observer_elev_m > 250.0 && profile.observer_elev_m < 320.0,
            "Observer elevation = {:.1}m, expected ~277m",
            profile.observer_elev_m
        );

        eprintln!("Mecca horizon profile:");
        eprintln!(
            "  Observer: {:.1}m, max horizon: {:.3}°, min: {:.3}°",
            profile.observer_elev_m,
            max_hz,
            profile.min_angle()
        );
        eprintln!("  Approx sunrise delay: {:.1} minutes", max_hz * 4.0);

        // Print a few cardinal directions
        for &(name, az) in &[
            ("N", 0),
            ("NE", 45),
            ("E", 90),
            ("SE", 135),
            ("S", 180),
            ("SW", 225),
            ("W", 270),
            ("NW", 315),
        ] {
            eprintln!("  {:>3}: {:.3}°", name, profile.angles_deg[az]);
        }
    }
}
