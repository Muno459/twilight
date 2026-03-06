#![allow(clippy::needless_range_loop, clippy::if_same_then_else)]
//! Terrain masking for twilight prayer times.
//!
//! Provides elevation data from multiple sources (Copernicus GLO-30 globally,
//! national LiDAR where available) and computes horizon profiles that modify
//! sunrise/sunset and twilight times.

pub mod cache;
pub mod copernicus;
pub mod geotiff;
pub mod horizon;
pub mod lidar;
pub mod projection;

use std::path::Path;

/// Elevation data source. Implemented by each DEM backend.
pub trait ElevationSource {
    /// Elevation in meters above the WGS84 ellipsoid at a geographic coordinate.
    /// Returns None if the point is over ocean, outside coverage, or NODATA.
    fn elevation_at(&self, lat: f64, lon: f64) -> Option<f32>;

    /// Approximate ground resolution in meters.
    fn resolution_m(&self) -> f64;

    /// Human-readable name for CLI output.
    fn name(&self) -> &str;

    /// Ensure data is available for the given bounding box.
    /// Downloads and caches tiles as needed. Call before elevation_at().
    fn prepare(
        &mut self,
        min_lat: f64,
        min_lon: f64,
        max_lat: f64,
        max_lon: f64,
    ) -> Result<(), String>;
}

/// A 360-degree horizon profile around an observer.
#[derive(Debug, Clone)]
pub struct HorizonProfile {
    /// Elevation angle (degrees) of the horizon at each azimuth.
    /// Index 0 = North (0 deg), index 90 = East, index 180 = South, index 270 = West.
    /// Positive = terrain above geometric horizon, negative = below (depression).
    pub angles_deg: [f64; 360],
    /// Observer latitude (degrees).
    pub observer_lat: f64,
    /// Observer longitude (degrees).
    pub observer_lon: f64,
    /// Observer elevation above ellipsoid (meters), from DEM.
    pub observer_elev_m: f64,
    /// Search radius used (km).
    pub radius_km: f64,
    /// Name of the elevation source used.
    pub source_name: String,
}

impl HorizonProfile {
    /// Get the horizon elevation angle at a given azimuth (degrees, 0=N, clockwise).
    /// Interpolates linearly between the two nearest integer-degree bins.
    pub fn angle_at(&self, azimuth_deg: f64) -> f64 {
        let az = azimuth_deg.rem_euclid(360.0);
        let lo = az.floor() as usize % 360;
        let hi = (lo + 1) % 360;
        let frac = az - az.floor();
        self.angles_deg[lo] * (1.0 - frac) + self.angles_deg[hi] * frac
    }

    /// Maximum horizon angle across all azimuths.
    pub fn max_angle(&self) -> f64 {
        self.angles_deg
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Minimum horizon angle across all azimuths.
    pub fn min_angle(&self) -> f64 {
        self.angles_deg
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
    }
}

/// Resolve the best available elevation source for a location.
///
/// Checks for LiDAR availability first, falls back to Copernicus GLO-30.
pub fn resolve_source(
    lat: f64,
    lon: f64,
    dem_dir: &Path,
    dk_api_key: Option<&str>,
) -> Box<dyn ElevationSource> {
    // Check Denmark LiDAR
    if lidar::denmark::covers(lat, lon) {
        if let Some(key) = dk_api_key {
            return Box::new(lidar::denmark::DanishDhm::new(dem_dir, key));
        }
        eprintln!("Note: Location is in Denmark. Set --dk-api-key or TWILIGHT_DK_API_KEY");
        eprintln!("      for 0.4m LiDAR. Falling back to Copernicus GLO-30 (30m).");
        eprintln!("      Register free at https://datafordeler.dk");
    }

    // Default: Copernicus GLO-30
    Box::new(copernicus::CopernicusDem30::new(dem_dir))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn horizon_profile_angle_at_integer() {
        let mut profile = HorizonProfile {
            angles_deg: [0.0; 360],
            observer_lat: 0.0,
            observer_lon: 0.0,
            observer_elev_m: 0.0,
            radius_km: 30.0,
            source_name: String::new(),
        };
        profile.angles_deg[90] = 2.5;
        assert!((profile.angle_at(90.0) - 2.5).abs() < 1e-10);
    }

    #[test]
    fn horizon_profile_angle_at_interpolation() {
        let mut profile = HorizonProfile {
            angles_deg: [0.0; 360],
            observer_lat: 0.0,
            observer_lon: 0.0,
            observer_elev_m: 0.0,
            radius_km: 30.0,
            source_name: String::new(),
        };
        profile.angles_deg[90] = 2.0;
        profile.angles_deg[91] = 4.0;
        assert!((profile.angle_at(90.5) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn horizon_profile_angle_at_wraparound() {
        let mut profile = HorizonProfile {
            angles_deg: [0.0; 360],
            observer_lat: 0.0,
            observer_lon: 0.0,
            observer_elev_m: 0.0,
            radius_km: 30.0,
            source_name: String::new(),
        };
        profile.angles_deg[359] = 1.0;
        profile.angles_deg[0] = 3.0;
        assert!((profile.angle_at(359.5) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn horizon_profile_max_min() {
        let mut profile = HorizonProfile {
            angles_deg: [0.0; 360],
            observer_lat: 0.0,
            observer_lon: 0.0,
            observer_elev_m: 0.0,
            radius_km: 30.0,
            source_name: String::new(),
        };
        profile.angles_deg[45] = 5.0;
        profile.angles_deg[200] = -1.0;
        assert!((profile.max_angle() - 5.0).abs() < 1e-10);
        assert!((profile.min_angle() - (-1.0)).abs() < 1e-10);
    }
}
