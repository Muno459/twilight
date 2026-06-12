//! Live VIIRS Day/Night Band nighttime-lights feed (NASA Black Marble).
//!
//! Second, CURRENT satellite skyglow source complementing the static
//! Lorenz 2024 atlas: the `VIIRS_*_GapFilled_BRDF_Corrected_DayNightBand_
//! Radiance` GIBS layers serve the daily Black Marble VNP46A2 product —
//! upward nighttime-lights radiance with moonlight REMOVED, BRDF/seasonal
//! effects corrected and cloud gaps filled. That makes it directly usable
//! without lunar gating (the raw At_Sensor product is moon- and
//! cloud-contaminated).
//!
//! Tiles are paletted PNGs whose palette INDEX is the bin reference into
//! the official radiance colormap (embedded as [`DNB_RADIANCE_LUT`],
//! regenerable via tools/gen_gibs_colormaps.py). Units: nW/(cm^2 sr) —
//! the same unit the Garstang/Bortle conversions in [`crate::bortle`]
//! consume, so a measured current-month radiance plugs straight into the
//! existing skyglow model.
//!
//! How the two feeds combine (in the CLI): the atlas provides the
//! PROPAGATED artificial zenith brightness (the right observable, but
//! frozen at its 2024 epoch); the DNB provides the CURRENT upward
//! radiance. We fetch DNB at both epochs (the GIBS time dimension goes
//! back years) and scale the atlas by the same-sensor temporal ratio —
//! lighting growth/decline since 2024 — rather than re-deriving
//! propagation from scratch.

use std::path::Path;

use crate::dnb_colormap::DNB_RADIANCE_LUT;

/// Preferred layer (NOAA-20: current through ~yesterday) and fallback.
const LAYERS: [&str; 2] = [
    "VIIRS_NOAA20_GapFilled_BRDF_Corrected_DayNightBand_Radiance",
    "VIIRS_SNPP_GapFilled_BRDF_Corrected_DayNightBand_Radiance",
];
/// epsg4326 "500m" TileMatrixSet, level 7: 160x80 tiles of 512 px,
/// 2.25 deg/tile (same family as the 2km/level-5 grid in
/// twilight-weather::satellite, scaled by 2^2).
const TMS_LEVEL: u32 = 7;
const DEG_PER_TILE: f64 = 2.25;
const TILE_PX: u32 = 512;

/// A measured nighttime-lights sample.
#[derive(Debug, Clone)]
pub struct DnbSample {
    /// Median upward radiance over the sampled nights [nW/(cm^2 sr)].
    pub radiance_nw: f64,
    /// Dates (ISO) that contributed valid data.
    pub dates_used: Vec<String>,
    /// GIBS layer that served the data.
    pub layer: &'static str,
}

fn tile_for(lat: f64, lon: f64) -> (u32, u32, u32, u32) {
    let col = ((lon + 180.0) / DEG_PER_TILE).floor() as u32;
    let row = ((90.0 - lat) / DEG_PER_TILE).floor() as u32;
    let px = (((lon + 180.0) % DEG_PER_TILE) / DEG_PER_TILE * TILE_PX as f64) as u32;
    let py = (((90.0 - lat) % DEG_PER_TILE) / DEG_PER_TILE * TILE_PX as f64) as u32;
    (row, col, py.min(TILE_PX - 1), px.min(TILE_PX - 1))
}

/// Fetch one tile (disk-cached) and return the palette index at the pixel.
fn index_at(
    cache_dir: &Path,
    layer: &str,
    date: &str,
    lat: f64,
    lon: f64,
) -> Option<u8> {
    let (row, col, py, px) = tile_for(lat, lon);
    let path = cache_dir.join(format!("dnb_{layer}_{date}_{row}_{col}.png"));
    let bytes = if path.exists() {
        std::fs::read(&path).ok()?
    } else {
        let url = format!(
            "https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/{layer}/default/{date}/500m/{TMS_LEVEL}/{row}/{col}.png"
        );
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(std::time::Duration::from_secs(30)))
            .build()
            .new_agent();
        let mut resp = agent.get(&url).call().ok()?;
        if resp.status() != 200 {
            return None;
        }
        let body = resp.body_mut().read_to_vec().ok()?;
        let _ = std::fs::create_dir_all(cache_dir);
        let tmp = path.with_extension("part");
        if std::fs::write(&tmp, &body).is_ok() {
            let _ = std::fs::rename(&tmp, &path);
        }
        body
    };

    // Paletted PNG: we need the raw INDEX, not expanded RGB.
    let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    let mut reader = decoder.read_info().ok()?;
    if reader.info().color_type != png::ColorType::Indexed {
        return None;
    }
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    let w = info.width as usize;
    buf.get((py as usize) * w + px as usize).copied()
}

/// Radiance [nW/(cm^2 sr)] at a point on a given date, or None for
/// nodata/missing tile.
fn radiance_on(
    cache_dir: &Path,
    layer: &str,
    date: &str,
    lat: f64,
    lon: f64,
) -> Option<f64> {
    let idx = index_at(cache_dir, layer, date, lat, lon)? as usize;
    let v = *DNB_RADIANCE_LUT.get(idx)?;
    if v < 0.0 {
        None // nodata
    } else {
        Some(v)
    }
}

fn date_minus(date: (i32, u32, u32), days: u32) -> (i32, u32, u32) {
    // Simple Julian-day arithmetic (Gregorian).
    let (y, m, d) = date;
    let a = (14 - m as i32) / 12;
    let yy = y + 4800 - a;
    let mm = m as i32 + 12 * a - 3;
    let jdn = d as i64
        + ((153 * mm + 2) / 5) as i64
        + 365 * yy as i64
        + (yy / 4) as i64
        - (yy / 100) as i64
        + (yy / 400) as i64
        - 32045
        - days as i64;
    let a2 = jdn + 32044;
    let b = (4 * a2 + 3) / 146097;
    let c = a2 - 146097 * b / 4;
    let d2 = (4 * c + 3) / 1461;
    let e = c - 1461 * d2 / 4;
    let m2 = (5 * e + 2) / 153;
    (
        (100 * b + d2 - 4800 + m2 / 10) as i32,
        (m2 + 3 - 12 * (m2 / 10)) as u32,
        (e - (153 * m2 + 2) / 5 + 1) as u32,
    )
}

fn iso(d: (i32, u32, u32)) -> String {
    format!("{:04}-{:02}-{:02}", d.0, d.1, d.2)
}

/// Median nighttime-lights radiance around a reference date: walks back
/// from `start_days_back` collecting up to `want` valid daily values
/// within a 21-day window. The gap-filled product needs no moon/cloud
/// gating; the median guards against residual artifacts.
pub fn measure(
    cache_dir: &Path,
    lat: f64,
    lon: f64,
    today: (i32, u32, u32),
    start_days_back: u32,
) -> Option<DnbSample> {
    for layer in LAYERS {
        let mut vals: Vec<(f64, String)> = Vec::new();
        for back in start_days_back..(start_days_back + 21) {
            let date = iso(date_minus(today, back));
            if let Some(v) = radiance_on(cache_dir, layer, &date, lat, lon) {
                vals.push((v, date));
                if vals.len() >= 5 {
                    break;
                }
            }
        }
        if vals.len() >= 2 {
            let mut sorted: Vec<f64> = vals.iter().map(|v| v.0).collect();
            sorted.sort_by(|a, b| a.total_cmp(b));
            return Some(DnbSample {
                radiance_nw: sorted[sorted.len() / 2],
                dates_used: vals.into_iter().map(|v| v.1).collect(),
                layer,
            });
        }
    }
    None
}

/// Same-sensor temporal brightening ratio between now and the Lorenz
/// atlas epoch (2024), sampled in the SAME calendar month to control
/// seasonal albedo (snow, vegetation). Clamped to [0.5, 2.0] against
/// retrieval noise. Returns (ratio, now_sample, epoch_radiance).
pub fn epoch_ratio(
    cache_dir: &Path,
    lat: f64,
    lon: f64,
    today: (i32, u32, u32),
    atlas_year: i32,
) -> Option<(f64, DnbSample, f64)> {
    let now = measure(cache_dir, lat, lon, today, 1)?;
    let epoch = measure(cache_dir, lat, lon, (atlas_year, today.1, today.2), 1)?;
    if epoch.radiance_nw < 0.05 {
        // Dark at epoch: no meaningful ratio (avoid 0/0 explosions);
        // caller should use the absolute DNB value instead.
        return None;
    }
    let ratio = (now.radiance_nw / epoch.radiance_nw).clamp(0.5, 2.0);
    Some((ratio, now, epoch.radiance_nw))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_has_physical_values() {
        assert_eq!(DNB_RADIANCE_LUT[0], -1.0, "index 0 is nodata");
        // ref 1 = [0,0.1) -> 0.05; ref 180 = [38.2,inf) -> 38.2 floor
        assert!((DNB_RADIANCE_LUT[1] - 0.05).abs() < 1e-9);
        assert!((DNB_RADIANCE_LUT[180] - 38.2).abs() < 1e-9);
        // strictly increasing over the valid range
        for w in DNB_RADIANCE_LUT[1..].windows(2) {
            assert!(w[1] > w[0], "LUT must be monotone: {} -> {}", w[0], w[1]);
        }
    }

    #[test]
    fn tile_math_copenhagen() {
        // 55.676N 12.568E -> level-7 tile (15, 85), verified against a
        // live fetch during development.
        let (row, col, py, px) = tile_for(55.676, 12.568);
        assert_eq!((row, col), (15, 85));
        assert!(py < 512 && px < 512);
    }

    #[test]
    fn date_arithmetic_rolls_over() {
        assert_eq!(date_minus((2026, 1, 1), 1), (2025, 12, 31));
        assert_eq!(date_minus((2026, 3, 1), 1), (2026, 2, 28));
        assert_eq!(date_minus((2024, 3, 1), 1), (2024, 2, 29)); // leap
    }

    // Live network test: cargo test -p twilight-skyglow -- --ignored
    #[test]
    #[ignore]
    fn live_copenhagen_is_bright() {
        let dir = std::env::temp_dir().join("twilight_dnb_test");
        let s = measure(&dir, 55.676, 12.568, (2026, 6, 12), 1).expect("DNB fetch");
        eprintln!("Copenhagen DNB: {:?}", s);
        assert!(s.radiance_nw > 5.0, "city core should be bright: {s:?}");
    }
}
