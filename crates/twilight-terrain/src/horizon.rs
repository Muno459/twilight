//! Horizon profile computation via ray-marching.
//!
//! For each of 360 azimuth directions, march outward from the observer at
//! DEM-resolution steps, query elevation, apply Earth curvature correction,
//! and record the maximum elevation angle seen. The result is a 360-element
//! array of horizon elevation angles used to shift sunrise/sunset times.

use crate::projection::vincenty_forward;
use crate::ElevationSource;
use crate::HorizonProfile;

/// Mean Earth radius (m) for curvature correction.
/// Using the IUGG mean radius (arithmetic mean of semi-axes).
const EARTH_RADIUS_M: f64 = 6_371_008.8;

/// Default horizon scan radius in km.
pub const DEFAULT_RADIUS_KM: f64 = 30.0;

/// Compute a 360-degree horizon profile around an observer.
///
/// # Arguments
/// * `source` - Elevation data source (must have been `prepare()`d for the area)
/// * `lat` - Observer latitude (degrees)
/// * `lon` - Observer longitude (degrees)
/// * `radius_km` - Search radius in km (default 30)
///
/// # Algorithm
/// For each integer azimuth 0..360:
///   1. Step outward from observer at intervals of `source.resolution_m()`
///   2. At each step, use Vincenty forward to get the (lat, lon) of the sample point
///   3. Query the DEM for terrain elevation at that point
///   4. Compute the Earth curvature drop: `d^2 / (2 * R)` where d is distance
///   5. Compute the elevation angle: `atan2(h_terrain - h_observer - curvature_drop, distance)`
///   6. Track the maximum elevation angle for this azimuth
///
/// Atmospheric refraction is NOT applied here (it's handled in the solar engine).
pub fn compute_horizon(
    source: &dyn ElevationSource,
    lat: f64,
    lon: f64,
    radius_km: f64,
) -> HorizonProfile {
    let observer_elev = source.elevation_at(lat, lon).unwrap_or(0.0) as f64;

    let step_m = source.resolution_m();
    let radius_m = radius_km * 1000.0;
    let mut angles = [0.0_f64; 360];

    for az in 0..360 {
        let azimuth_deg = az as f64;
        let mut max_angle = f64::NEG_INFINITY;
        let mut dist = step_m;

        while dist <= radius_m {
            let (sample_lat, sample_lon) = vincenty_forward(lat, lon, azimuth_deg, dist);

            if let Some(terrain_elev) = source.elevation_at(sample_lat, sample_lon) {
                let h_terrain = terrain_elev as f64;

                // Earth curvature drop: the terrain "falls away" from the observer's
                // tangent plane by d^2 / (2R). This makes distant terrain appear lower.
                let curvature_drop = (dist * dist) / (2.0 * EARTH_RADIUS_M);

                // Height of terrain above observer's tangent plane
                let apparent_height = h_terrain - observer_elev - curvature_drop;

                // Elevation angle from observer to terrain point
                let angle_rad = apparent_height.atan2(dist);
                let angle_deg = angle_rad.to_degrees();

                if angle_deg > max_angle {
                    max_angle = angle_deg;
                }
            }

            dist += step_m;
        }

        // If no terrain was found at all (ocean everywhere), horizon is at 0 degrees
        angles[az] = if max_angle == f64::NEG_INFINITY {
            0.0
        } else {
            max_angle
        };
    }

    HorizonProfile {
        angles_deg: angles,
        observer_lat: lat,
        observer_lon: lon,
        observer_elev_m: observer_elev,
        radius_km,
        source_name: source.name().to_string(),
    }
}

/// Compute the time shift (in minutes) caused by a horizon obstruction.
///
/// When the horizon angle at the sun's azimuth is positive, the sun rises later
/// and sets earlier. The shift is approximately:
///   shift_minutes = horizon_angle_deg / 15.0 * 60.0
///   = horizon_angle_deg * 4.0
///
/// (The sun moves ~15 degrees/hour = 0.25 degrees/minute, so 1 degree of
/// horizon obstruction delays sunrise by ~4 minutes.)
///
/// This is a first-order approximation. The actual shift depends on the sun's
/// declination and the observer's latitude (the sun's path is not always
/// perpendicular to the horizon). A more precise formula uses the hour angle.
///
/// # Arguments
/// * `horizon_angle_deg` - Horizon elevation angle at the sun's azimuth (degrees)
/// * `lat_deg` - Observer latitude (degrees)
/// * `declination_deg` - Solar declination (degrees)
/// * `sunrise` - true for sunrise (delay), false for sunset (advance)
///
/// # Returns
/// Time shift in minutes. Positive = sunrise is later / sunset is earlier.
pub fn horizon_time_shift(
    horizon_angle_deg: f64,
    lat_deg: f64,
    declination_deg: f64,
    sunrise: bool,
) -> f64 {
    if horizon_angle_deg <= 0.0 {
        return 0.0; // No obstruction
    }

    // The rate at which the sun crosses the horizon depends on the angle
    // between the sun's path and the horizon. At the equator looking east,
    // it's cos(declination). At higher latitudes, the path is more oblique.
    //
    // Exact formula: the hour angle change for a given altitude change is
    //   dH = da / (cos(lat) * cos(dec) * sin(H))
    // where H is the hour angle at sunrise/sunset.
    //
    // Simplified: the sun's apparent speed across the horizon is
    //   v = 15 * cos(dec) / sqrt(1 - (sin(lat)*sin(dec) + cos(lat)*cos(dec))^2)
    // degrees/hour at the moment of rise/set, but this simplifies near zero altitude.
    //
    // A practical and accurate formula:
    //   shift = horizon_angle / (15 * cos_path_angle) * 60 minutes
    // where cos_path_angle = sqrt(cos^2(dec) - sin^2(lat) + ... )
    //
    // For simplicity and to avoid singularities, use the standard approximation:
    let lat_rad = lat_deg.to_radians();
    let dec_rad = declination_deg.to_radians();

    // cos of the angle between sun's path and the horizon at rise/set
    let cos_lat = lat_rad.cos();
    let cos_dec = dec_rad.cos();

    // The obliquity factor: how quickly the sun crosses a given altitude
    // At tropical latitudes this is ~1, at polar latitudes it can be very large
    let cos_product = cos_lat * cos_dec;
    if cos_product < 0.01 {
        // Near polar: sun barely moves vertically, shift can be hours
        // Return a large value clamped to something reasonable
        return horizon_angle_deg * 60.0; // 1 hour per degree
    }

    // Standard formula: dT = dh / (15 * cos_path)
    // where cos_path = cos(dec) * sin(hour_angle_at_rise)
    // At sunrise/sunset, cos(altitude) ≈ 1, and:
    //   sin(H0) = sqrt(1 - (tan(lat)*tan(dec))^2) / (cos(lat)*cos(dec))
    // but sin(H0) = sqrt(cos^2(H0) ... this gets circular.
    //
    // Use the simpler geometric derivation:
    // Time for sun to move through angle 'a' in altitude near the horizon:
    //   dt = a / (15 * cos(lat) * cos(dec) * sin(H0)) [hours]
    // where sin(H0) for standard sunrise (h=0):
    //   cos(H0) = -tan(lat)*tan(dec)
    //   sin(H0) = sqrt(1 - tan^2(lat)*tan^2(dec))
    let tan_product = lat_rad.tan() * dec_rad.tan();
    if tan_product.abs() >= 1.0 {
        // Polar day/night: no standard sunrise/sunset
        return horizon_angle_deg * 60.0;
    }

    let sin_h0 = (1.0 - tan_product * tan_product).sqrt();
    let rate = 15.0 * cos_lat * cos_dec * sin_h0; // degrees per hour

    if rate < 0.01 {
        return horizon_angle_deg * 60.0;
    }

    let shift_hours = horizon_angle_deg / rate;
    let shift_minutes = shift_hours * 60.0;

    // For sunrise: positive shift means later sunrise
    // For sunset: positive shift means earlier sunset
    if sunrise {
        shift_minutes
    } else {
        shift_minutes
    }
}

/// Compute the effective solar zenith angle for sunrise/sunset accounting for terrain.
///
/// Standard sunrise/sunset uses SZA = 90.8333 degrees (accounting for refraction
/// and solar semi-diameter). If terrain blocks the horizon at the sun's azimuth,
/// we need a smaller SZA (sun must be higher to clear the terrain).
///
/// Returns the adjusted SZA in degrees.
pub fn effective_sunrise_sza(horizon_angle_deg: f64) -> f64 {
    // Standard SZA at sunrise/sunset = 90.8333 (0.8333 = 50' refraction + semidiameter)
    // If horizon angle is h degrees above geometric horizon:
    //   effective_sza = 90.8333 - h
    // This means the sun's center must be at altitude h + 0.8333 degrees
    // when it clears the terrain.
    90.8333 - horizon_angle_deg
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat-plane elevation source for testing.
    struct FlatSource {
        elevation: f32,
    }

    impl ElevationSource for FlatSource {
        fn elevation_at(&self, _lat: f64, _lon: f64) -> Option<f32> {
            Some(self.elevation)
        }
        fn resolution_m(&self) -> f64 {
            100.0 // 100m steps for faster tests
        }
        fn name(&self) -> &str {
            "Flat test"
        }
        fn prepare(&mut self, _: f64, _: f64, _: f64, _: f64) -> Result<(), String> {
            Ok(())
        }
    }

    /// A source with a mountain at a specific bearing/distance from origin.
    struct MountainSource {
        /// Observer elevation
        base_elev: f32,
        /// Mountain peak elevation
        peak_elev: f32,
        /// Mountain center lat/lon
        peak_lat: f64,
        peak_lon: f64,
        /// Mountain radius in degrees (approximate)
        peak_radius_deg: f64,
    }

    impl ElevationSource for MountainSource {
        fn elevation_at(&self, lat: f64, lon: f64) -> Option<f32> {
            let dlat = lat - self.peak_lat;
            let dlon = lon - self.peak_lon;
            let dist = (dlat * dlat + dlon * dlon).sqrt();
            if dist < self.peak_radius_deg {
                // Simple cone shape
                let frac = 1.0 - dist / self.peak_radius_deg;
                let elev = self.base_elev as f64 + (self.peak_elev - self.base_elev) as f64 * frac;
                Some(elev as f32)
            } else {
                Some(self.base_elev)
            }
        }
        fn resolution_m(&self) -> f64 {
            100.0
        }
        fn name(&self) -> &str {
            "Mountain test"
        }
        fn prepare(&mut self, _: f64, _: f64, _: f64, _: f64) -> Result<(), String> {
            Ok(())
        }
    }

    #[test]
    fn flat_terrain_zero_horizon() {
        let source = FlatSource { elevation: 100.0 };
        let profile = compute_horizon(&source, 21.4225, 39.8262, 5.0);

        // On flat terrain, Earth curvature makes the horizon slightly negative
        for az in 0..360 {
            // Should be slightly negative (curvature dip) or zero
            assert!(
                profile.angles_deg[az] <= 0.01,
                "azimuth {} angle {} should be <= 0",
                az,
                profile.angles_deg[az]
            );
        }
    }

    #[test]
    fn flat_terrain_curvature_dip() {
        // On perfectly flat ground at elevation 0, the algorithm tracks the
        // MAXIMUM elevation angle at any step. The closest step (100m) has
        // near-zero curvature drop, so angle ≈ 0. Farther steps have progressively
        // more negative angles due to curvature.
        //
        // The max is therefore near zero (from the first step), but slightly
        // negative because even at 100m there's a tiny curvature drop:
        //   drop = 100^2 / (2 * 6371009) ≈ 0.000785m
        //   angle = atan2(-0.000785, 100) ≈ -0.00045 degrees
        let source = FlatSource { elevation: 0.0 };
        let profile = compute_horizon(&source, 0.0, 0.0, 30.0);

        for az in 0..360 {
            // Should be very slightly negative (curvature dip from nearest step)
            assert!(
                profile.angles_deg[az] < 0.0,
                "azimuth {} should be negative, got {}",
                az,
                profile.angles_deg[az]
            );
            // But very close to zero (the first step barely dips)
            assert!(
                profile.angles_deg[az].abs() < 0.001,
                "azimuth {} angle {} should be near zero",
                az,
                profile.angles_deg[az]
            );
        }
    }

    #[test]
    fn mountain_creates_positive_horizon() {
        // Place a 1000m mountain 10km north of the observer at equator
        // Observer at 0,0 elevation 0; mountain at ~0.09 N, 0 E
        let source = MountainSource {
            base_elev: 0.0,
            peak_elev: 1000.0,
            peak_lat: 0.09, // ~10km north at equator
            peak_lon: 0.0,
            peak_radius_deg: 0.05, // ~5.5km radius
        };

        let profile = compute_horizon(&source, 0.0, 0.0, 15.0);

        // North (azimuth 0) should have a significant positive angle
        // At 10km with 1000m peak: atan2(1000 - 0 - (10000^2/(2*6371009)), 10000)
        // curvature drop at 10km = 7.85m
        // angle = atan2(1000 - 7.85, 10000) = atan2(992.15, 10000) ≈ 5.67 degrees
        assert!(
            profile.angles_deg[0] > 3.0,
            "North horizon should be positive with mountain, got {}",
            profile.angles_deg[0]
        );

        // South (azimuth 180) should be near zero or negative (no mountain)
        assert!(
            profile.angles_deg[180] < 0.5,
            "South horizon should be near zero, got {}",
            profile.angles_deg[180]
        );
    }

    #[test]
    fn effective_sza_adjustment() {
        // No terrain: standard SZA
        assert!((effective_sunrise_sza(0.0) - 90.8333).abs() < 1e-4);

        // 2-degree horizon: sun must be higher to clear terrain
        assert!((effective_sunrise_sza(2.0) - 88.8333).abs() < 1e-4);

        // 5-degree horizon (big mountain):
        assert!((effective_sunrise_sza(5.0) - 85.8333).abs() < 1e-4);
    }

    #[test]
    fn horizon_time_shift_basic() {
        // At equator, equinox: cos_lat=1, cos_dec=1, tan_product=0
        // sin(H0) = 1, rate = 15 deg/hour
        // 1 degree of horizon = 1/15 hour = 4 minutes
        let shift = horizon_time_shift(1.0, 0.0, 0.0, true);
        assert!(
            (shift - 4.0).abs() < 0.1,
            "1 degree at equator equinox should be ~4 min, got {}",
            shift
        );
    }

    #[test]
    fn horizon_time_shift_zero_angle() {
        let shift = horizon_time_shift(0.0, 45.0, 23.44, true);
        assert!((shift).abs() < 1e-10);

        let shift = horizon_time_shift(-1.0, 45.0, 23.44, true);
        assert!((shift).abs() < 1e-10);
    }

    #[test]
    fn horizon_time_shift_high_latitude() {
        // At 60N, summer solstice (dec=23.44): sun path is very oblique
        // Should take longer than 4 min per degree
        let shift = horizon_time_shift(1.0, 60.0, 23.44, true);
        assert!(
            shift > 4.0,
            "High latitude should have larger shift, got {}",
            shift
        );
    }

    #[test]
    fn observer_elevation_from_source() {
        let source = FlatSource { elevation: 350.0 };
        let profile = compute_horizon(&source, 21.4225, 39.8262, 5.0);
        assert!((profile.observer_elev_m - 350.0).abs() < 1e-6);
    }

    #[test]
    fn source_name_recorded() {
        let source = FlatSource { elevation: 0.0 };
        let profile = compute_horizon(&source, 0.0, 0.0, 5.0);
        assert_eq!(profile.source_name, "Flat test");
    }
}
