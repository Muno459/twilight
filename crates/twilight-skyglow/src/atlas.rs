//! Satellite-derived light-pollution atlas (automatic skyglow).
//!
//! Fetches and decodes the David Lorenz world light-pollution atlas
//! (VIIRS-based, propagated artificial ZENITH sky brightness; the 2024
//! edition; <https://djlorenz.github.io/astronomy/lp/>), which models
//! upward radiance through atmospheric scattering - i.e. it is already
//! the observer-sky quantity the engine needs, unlike raw VIIRS DNB.
//!
//! Tile format (documented in the atlas viewer source):
//! - 5x5 degree tiles, 600x600 points (1/120 deg ~ 0.5 nmi resolution)
//! - URL: binary_tiles/{year}/binary_tile_{x}_{y}.dat.gz
//!   with x = floor(mod(lon+180,360)/5)+1, y = floor((lat+65)/5)+1 (1..28)
//! - gzip-compressed i8 stream, delta-coded: the first point is a 2-byte
//!   value (128*b0 + b1), every other byte is a delta to the previous
//!   point (first column = latitude steps, then along the row)
//! - decoded value c -> artificial/natural brightness ratio
//!   R = (5/195) * (exp(0.0195*c) - 1)
//! - artificial zenith luminance = R * L_natural where the atlas's
//!   natural reference is 22.0 mag/arcsec^2 = 0.1714 mcd/m^2.

use std::io::Read;
use std::path::{Path, PathBuf};

const ATLAS_BASE: &str = "https://djlorenz.github.io/astronomy/binary_tiles";
/// Newest published atlas edition.
const ATLAS_YEAR: u32 = 2024;
const REQUEST_TIMEOUT_MS: u64 = 15_000;
/// 22.0 mag/arcsec^2 in mcd/m^2: 10.8e4 * 10^(-0.4*22) * 1000.
const NATURAL_REF_MCD: f64 = 0.171_4;

/// Result of an atlas lookup.
#[derive(Debug, Clone, Copy)]
pub struct AtlasSkyglow {
    /// Artificial zenith luminance [mcd/m^2].
    pub zenith_mcd: f64,
    /// Artificial/natural brightness ratio (Lorenz's native quantity).
    pub brightness_ratio: f64,
    /// Atlas edition year.
    pub year: u32,
}

fn tile_indices(lat: f64, lon: f64) -> Option<(u32, u32, usize, usize)> {
    let lon_fdl = (lon + 180.0).rem_euclid(360.0);
    let lat_fs = lat + 65.0;
    let tx = (lon_fdl / 5.0).floor() as i64 + 1;
    let ty = (lat_fs / 5.0).floor() as i64 + 1;
    if !(1..=28).contains(&ty) || !(1..=72).contains(&tx) {
        return None; // outside atlas coverage (|lat| > 65-ish)
    }
    // nearest grid point inside the tile (mirrors the atlas viewer JS)
    let ix = (120.0 * (lon_fdl - 5.0 * (tx - 1) as f64 + 1.0 / 240.0)).round() as i64;
    let iy = (120.0 * (lat_fs - 5.0 * (ty - 1) as f64 + 1.0 / 240.0)).round() as i64;
    let ix = ix.clamp(1, 600) as usize;
    let iy = iy.clamp(1, 600) as usize;
    Some((tx as u32, ty as u32, ix, iy))
}

/// Decode the delta-coded tile at grid point (ix, iy), both 1-based.
fn decode_at(data: &[i8], ix: usize, iy: usize) -> Option<i64> {
    if data.len() < 600 * 600 + 1 {
        return None;
    }
    // First point: 2 bytes (i8 arithmetic exactly as the reference decoder).
    let first = 128i64 * data[0] as i64 + data[1] as i64;
    let mut change = 0i64;
    // Latitude steps along the first column.
    for i in 1..iy {
        change += data[600 * i + 1] as i64;
    }
    // Longitude steps along row iy.
    for i in 1..ix {
        change += data[600 * (iy - 1) + 1 + i] as i64;
    }
    Some(first + change)
}

fn compressed_to_ratio(c: i64) -> f64 {
    (5.0 / 195.0) * ((0.0195 * c as f64).exp() - 1.0)
}

fn fetch_tile(cache_dir: &Path, tx: u32, ty: u32) -> Result<Vec<i8>, String> {
    let cache: PathBuf = cache_dir.join(format!("lorenz{ATLAS_YEAR}_tile_{tx}_{ty}.dat"));
    let gz_bytes: Vec<u8> = if let Ok(b) = std::fs::read(&cache) {
        b
    } else {
        let url = format!("{ATLAS_BASE}/{ATLAS_YEAR}/binary_tile_{tx}_{ty}.dat.gz");
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(std::time::Duration::from_millis(REQUEST_TIMEOUT_MS)))
            .build()
            .new_agent();
        let mut resp = agent
            .get(&url)
            .call()
            .map_err(|e| format!("atlas fetch failed: {e}"))?;
        let mut bytes = Vec::new();
        resp.body_mut()
            .as_reader()
            .read_to_end(&mut bytes)
            .map_err(|e| format!("atlas read failed: {e}"))?;
        let _ = std::fs::create_dir_all(cache_dir);
        let _ = std::fs::write(&cache, &bytes);
        bytes
    };
    let mut decoder = flate2::read::GzDecoder::new(gz_bytes.as_slice());
    let mut raw = Vec::new();
    decoder
        .read_to_end(&mut raw)
        .map_err(|e| format!("atlas gunzip failed: {e}"))?;
    Ok(raw.into_iter().map(|b| b as i8).collect())
}

/// Look up the artificial zenith sky brightness at a location.
///
/// Returns None outside atlas coverage (|lat| beyond ~65-75) or on
/// network failure (callers fall back to Bortle/manual input).
pub fn artificial_zenith(cache_dir: &Path, lat: f64, lon: f64) -> Option<AtlasSkyglow> {
    let (tx, ty, ix, iy) = tile_indices(lat, lon)?;
    let data = fetch_tile(cache_dir, tx, ty).ok()?;
    let compressed = decode_at(&data, ix, iy)?;
    let ratio = compressed_to_ratio(compressed).max(0.0);
    Some(AtlasSkyglow {
        zenith_mcd: ratio * NATURAL_REF_MCD,
        brightness_ratio: ratio,
        year: ATLAS_YEAR,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_indices_known_locations() {
        // Brondby 55.653N 12.412E: lon_fdl=192.412 -> tx=39; lat_fs=120.653 -> ty=25
        let (tx, ty, ix, iy) = tile_indices(55.653, 12.412).unwrap();
        assert_eq!((tx, ty), (39, 25));
        assert!((1..=600).contains(&ix) && (1..=600).contains(&iy));
        // Mecca 21.42N 39.83E: tx = floor(219.83/5)+1 = 44; ty = floor(86.42/5)+1 = 18
        let (tx, ty, _, _) = tile_indices(21.4225, 39.8262).unwrap();
        assert_eq!((tx, ty), (44, 18));
        // Out of coverage
        assert!(tile_indices(80.0, 0.0).is_none());
    }

    #[test]
    fn delta_decode_synthetic_tile() {
        // Build a synthetic tile: first point = 200 (2 bytes: 1, 72),
        // column deltas +2 per row, row deltas +1 per column.
        let mut data = vec![0i8; 600 * 600 + 1];
        data[0] = 1; // 128
        data[1] = 72; // +72 => 200
        for i in 1..600 {
            data[600 * i + 1] = 2; // latitude steps
        }
        for iy in 1..=600usize {
            for i in 1..600 {
                let idx = 600 * (iy - 1) + 1 + i;
                if idx > 1 && data[idx] == 0 && i >= 1 {
                    data[idx] = 1; // longitude steps
                }
            }
        }
        // point (1,1) = 200
        assert_eq!(decode_at(&data, 1, 1).unwrap(), 200);
        // point (1,3): two latitude steps: 200 + 2 + 2 = 204
        assert_eq!(decode_at(&data, 1, 3).unwrap(), 204);
        // point (4,1): three longitude steps: 200 + 3 = 203
        assert_eq!(decode_at(&data, 4, 1).unwrap(), 203);
    }

    #[test]
    fn ratio_conversion_anchors() {
        // c=0 -> ratio 0 (pristine)
        assert!(compressed_to_ratio(0).abs() < 1e-12);
        // monotone increasing
        assert!(compressed_to_ratio(100) > compressed_to_ratio(50));
        // a heavily polluted city: ratio of order 10-100
        let r = compressed_to_ratio(300);
        assert!(r > 5.0 && r < 100.0, "ratio(300) = {r}");
    }
}
