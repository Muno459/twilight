//! Danish national LiDAR elevation model (DHM/Terrain).
//!
//! Denmark has complete 0.4m resolution LiDAR coverage through Dataforsyningen
//! (the Danish Agency for Data Supply and Infrastructure). The data is free
//! but requires an API key from https://datafordeler.dk.
//!
//! CRS: EPSG:25832 (UTM zone 32N on ETRS89/GRS80 ellipsoid)
//! Tile structure: 10x10 km tiles in UTM coordinates
//! Format: GeoTIFF
//!
//! Denmark bounding box (approximate):
//!   Lat: 54.5 - 57.8 N
//!   Lon:  7.5 - 15.3 E

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::cache;
use crate::geotiff::{ElevationTile, GeoTransform};
use crate::projection::wgs84_to_utm_zone;
use crate::ElevationSource;

/// Denmark bounding box (generous, includes Bornholm).
const DK_LAT_MIN: f64 = 54.4;
const DK_LAT_MAX: f64 = 57.9;
const DK_LON_MIN: f64 = 7.4;
const DK_LON_MAX: f64 = 15.4;

/// UTM zone for Denmark (EPSG:25832).
const DK_UTM_ZONE: u8 = 32;

/// Tile size in meters (10km x 10km).
const TILE_SIZE_M: f64 = 10_000.0;

/// Check whether a WGS84 coordinate falls within Denmark's bounding box.
pub fn covers(lat: f64, lon: f64) -> bool {
    lat >= DK_LAT_MIN && lat <= DK_LAT_MAX && lon >= DK_LON_MIN && lon <= DK_LON_MAX
}

/// Danish DHM (Digital Height Model) elevation source.
pub struct DanishDhm {
    /// Cache directory for downloaded tiles.
    cache_dir: PathBuf,
    /// Dataforsyningen API key.
    api_key: String,
    /// Loaded tiles, keyed by (easting_10km, northing_10km).
    tiles: HashMap<(i32, i32), Option<ElevationTile>>,
}

impl DanishDhm {
    pub fn new(dem_dir: &Path, api_key: &str) -> Self {
        DanishDhm {
            cache_dir: cache::source_dir(dem_dir, "denmark_dhm"),
            api_key: api_key.to_string(),
            tiles: HashMap::new(),
        }
    }

    /// Get the tile key for a UTM coordinate (easting, northing).
    /// Tiles are 10x10km, so we floor to nearest 10km.
    fn tile_key(easting: f64, northing: f64) -> (i32, i32) {
        let e = (easting / TILE_SIZE_M).floor() as i32;
        let n = (northing / TILE_SIZE_M).floor() as i32;
        (e, n)
    }

    /// Cache filename for a tile.
    fn tile_filename(e_key: i32, n_key: i32) -> String {
        format!("dhm_{}_{}.tif", e_key, n_key)
    }

    /// Build the GeoTransform for a Danish DHM tile.
    /// Origin is the SW corner in UTM coordinates.
    fn geo_transform_for_tile(e_key: i32, n_key: i32, width: u32, _height: u32) -> GeoTransform {
        let origin_easting = e_key as f64 * TILE_SIZE_M;
        let origin_northing = (n_key + 1) as f64 * TILE_SIZE_M; // top edge

        let pixel_size = TILE_SIZE_M / width as f64; // should be ~0.4m

        GeoTransform {
            origin_x: origin_easting,
            origin_y: origin_northing,
            pixel_size_x: pixel_size,
            pixel_size_y: pixel_size,
        }
    }

    /// Download and load a tile. Returns None if no data (ocean/outside coverage).
    fn load_tile(&mut self, e_key: i32, n_key: i32) -> Option<&ElevationTile> {
        let key = (e_key, n_key);
        if !self.tiles.contains_key(&key) {
            let tile = self.download_and_parse(e_key, n_key);
            self.tiles.insert(key, tile);
        }
        self.tiles.get(&key).and_then(|t| t.as_ref())
    }

    fn download_and_parse(&self, e_key: i32, n_key: i32) -> Option<ElevationTile> {
        let filename = Self::tile_filename(e_key, n_key);

        // Construct the Dataforsyningen WCS URL.
        // The exact API endpoint and parameters depend on the service version.
        // This will be finalized once the user provides their API key and we can
        // test against the real service.
        //
        // Expected format (WCS GetCoverage):
        // https://api.dataforsyningen.dk/dhm?service=WCS&version=2.0.1
        //   &request=GetCoverage&CoverageId=dhm_terraen
        //   &subset=x({emin},{emax})&subset=y({nmin},{nmax})
        //   &format=image/tiff
        //   &token={api_key}
        let emin = e_key as f64 * TILE_SIZE_M;
        let emax = emin + TILE_SIZE_M;
        let nmin = n_key as f64 * TILE_SIZE_M;
        let nmax = nmin + TILE_SIZE_M;

        let url = format!(
            "https://api.dataforsyningen.dk/dhm?service=WCS&version=2.0.1\
             &request=GetCoverage&CoverageId=dhm_terraen\
             &subset=x({},{})\
             &subset=y({},{})\
             &format=image/tiff\
             &token={}",
            emin, emax, nmin, nmax, self.api_key
        );

        let path = match cache::download_to_cache(&self.cache_dir, &filename, &url) {
            Ok(p) => p,
            Err(e) => {
                if e.contains("404") {
                    return None;
                }
                eprintln!("Warning: failed to download Danish DHM tile: {}", e);
                return None;
            }
        };

        // Parse the tile. We need dimensions first for the GeoTransform.
        match load_dk_tile(&path, e_key, n_key) {
            Ok(tile) => Some(tile),
            Err(e) => {
                eprintln!(
                    "Warning: failed to parse Danish DHM tile {}: {}",
                    filename, e
                );
                None
            }
        }
    }
}

/// Load a Danish DHM tile from a cached file.
fn load_dk_tile(path: &Path, e_key: i32, n_key: i32) -> Result<ElevationTile, String> {
    use std::fs::File;
    use tiff::decoder::Decoder;

    let file = File::open(path).map_err(|e| format!("Failed to open: {}", e))?;
    let mut decoder = Decoder::new(file).map_err(|e| format!("Failed to decode: {}", e))?;
    let (width, height) = decoder
        .dimensions()
        .map_err(|e| format!("Failed to read dimensions: {}", e))?;

    let geo = DanishDhm::geo_transform_for_tile(e_key, n_key, width, height);
    ElevationTile::from_file_with_geo(path, geo)
}

impl ElevationSource for DanishDhm {
    fn elevation_at(&self, lat: f64, lon: f64) -> Option<f32> {
        if !covers(lat, lon) {
            return None;
        }

        // Convert to UTM zone 32N
        let (easting, northing) = wgs84_to_utm_zone(lat, lon, DK_UTM_ZONE);
        let (e_key, n_key) = Self::tile_key(easting, northing);
        let key = (e_key, n_key);

        self.tiles
            .get(&key)
            .and_then(|t| t.as_ref())
            .and_then(|tile| tile.elevation_at_geo(easting, northing))
    }

    fn resolution_m(&self) -> f64 {
        0.4
    }

    fn name(&self) -> &str {
        "Danish DHM LiDAR (0.4m)"
    }

    fn prepare(
        &mut self,
        min_lat: f64,
        min_lon: f64,
        max_lat: f64,
        max_lon: f64,
    ) -> Result<(), String> {
        // Convert bounding box corners to UTM
        let (e1, n1) = wgs84_to_utm_zone(min_lat, min_lon, DK_UTM_ZONE);
        let (e2, n2) = wgs84_to_utm_zone(max_lat, max_lon, DK_UTM_ZONE);

        let emin = e1.min(e2);
        let emax = e1.max(e2);
        let nmin = n1.min(n2);
        let nmax = n1.max(n2);

        let e_lo = (emin / TILE_SIZE_M).floor() as i32;
        let e_hi = (emax / TILE_SIZE_M).floor() as i32;
        let n_lo = (nmin / TILE_SIZE_M).floor() as i32;
        let n_hi = (nmax / TILE_SIZE_M).floor() as i32;

        let total = ((e_hi - e_lo + 1) * (n_hi - n_lo + 1)) as usize;
        let mut loaded = 0;

        for e_key in e_lo..=e_hi {
            for n_key in n_lo..=n_hi {
                self.load_tile(e_key, n_key);
                loaded += 1;
                if total > 1 {
                    eprint!("\r  Loading Danish DHM tiles: {}/{}", loaded, total);
                }
            }
        }
        if total > 1 {
            eprintln!("\r  Loaded {} Danish DHM tiles.       ", total);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn covers_denmark() {
        // Copenhagen
        assert!(covers(55.6761, 12.5683));
        // Padborg (near German border)
        assert!(covers(54.83, 9.35));
        // Skagen (northernmost)
        assert!(covers(57.72, 10.59));
        // Bornholm
        assert!(covers(55.13, 14.92));
    }

    #[test]
    fn does_not_cover_outside() {
        // Mecca
        assert!(!covers(21.4225, 39.8262));
        // Stockholm
        assert!(!covers(59.33, 18.07));
        // Hamburg
        assert!(!covers(53.55, 9.99));
    }

    #[test]
    fn tile_key_calculation() {
        // A point in central Denmark in UTM zone 32N
        // Easting ~500km, Northing ~6200km
        let (e, n) = DanishDhm::tile_key(523_456.0, 6_178_901.0);
        assert_eq!(e, 52); // 523456 / 10000 = 52
        assert_eq!(n, 617); // 6178901 / 10000 = 617
    }

    #[test]
    fn tile_filename_format() {
        let f = DanishDhm::tile_filename(52, 617);
        assert_eq!(f, "dhm_52_617.tif");
    }

    #[test]
    fn geo_transform_dk_tile() {
        // 10km tile at key (52, 617), 25000x25000 pixels (0.4m resolution)
        let geo = DanishDhm::geo_transform_for_tile(52, 617, 25000, 25000);
        assert!((geo.origin_x - 520_000.0).abs() < 1e-6);
        assert!((geo.origin_y - 6_180_000.0).abs() < 1e-6);
        assert!((geo.pixel_size_x - 0.4).abs() < 1e-6);
    }
}
