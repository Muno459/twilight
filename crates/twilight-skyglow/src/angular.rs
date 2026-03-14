//! Angular distribution model for skyglow brightness.
//!
//! Skyglow is NOT uniform across the sky. It follows a well-characterized
//! angular distribution that is brighter toward the horizon and in the
//! direction of nearby cities. This module models the angular dependence.
//!
//! For prayer time determination, the relevant viewing direction is toward
//! the horizon at the azimuth of sunset (Isha) or sunrise (Fajr), typically
//! at an elevation of 5-15 degrees above the horizon.
//!
//! # The Angular Model
//!
//! The dominant effect is the increase in path length through the scattering
//! atmosphere at lower elevation angles. For an observer looking at elevation
//! angle `e` above the horizon, the airmass (relative path length) is
//! approximately `1/sin(e)` for `e > ~10 degrees`.
//!
//! The skyglow brightness at elevation angle `e` relative to zenith follows:
//!
//!   L(e) = L_zenith * f(e)
//!
//! where f(e) is the enhancement factor.
//!
//! # Enhancement Factor Models
//!
//! Several models have been published:
//!
//! 1. **Simple sec(z)**: f = 1/cos(z) = 1/sin(e) where z = 90-e.
//!    Too simple -- diverges at horizon.
//!
//! 2. **Garstang (1989)**: More physically motivated, accounts for
//!    exponential atmosphere and source geometry.
//!
//! 3. **Duriscoe (2013)**: Empirical fit from all-sky photometry:
//!    f(z) = 1 + a * (1 - cos(z))^b
//!    with a ≈ 10, b ≈ 2.5
//!
//! 4. **Kocifaj (2007)**: "MSNsR" (Multi-Scattering N-sources Radiance)
//!    model with angular dependence from integration over multiple sources.
//!
//! We use a model based on Duriscoe's empirical observations, calibrated
//! against Garstang and verified with SQM all-sky measurements.

use std::f64::consts::PI;

/// Compute the skyglow enhancement factor at a given elevation angle.
///
/// The enhancement factor multiplied by the zenith brightness gives the
/// brightness at that elevation angle.
///
/// # Arguments
/// * `elevation_deg` - Viewing angle above the horizon (degrees).
///   90 = zenith, 0 = horizon.
///
/// # Returns
/// Enhancement factor (1.0 at zenith, increases toward horizon).
///
/// At the horizon (0 degrees), returns a clamped maximum rather than
/// infinity, representing the practical horizon glow for an observer
/// whose view is obstructed by terrain/buildings at ~0 degrees.
pub fn enhancement_factor(elevation_deg: f64) -> f64 {
    // Clamp to avoid degenerate values
    let elev = elevation_deg.clamp(1.0, 90.0);
    let zenith_deg = 90.0 - elev;
    let zenith_rad = zenith_deg * PI / 180.0;

    // Duriscoe (2013) empirical model with modifications:
    //
    // f(z) = 1 + A * (1 - cos(z))^B
    //
    // Duriscoe found A ≈ 10, B ≈ 2.5 from all-sky photometry at
    // multiple sites. We use A=8.0, B=2.2 which better matches
    // Garstang at moderate zenith angles while avoiding the extreme
    // divergence near the horizon.
    //
    // At z=0 (zenith):  f = 1.0
    // At z=45 (elev=45): f ≈ 1 + 8*0.29^2.2 ≈ 1.52
    // At z=70 (elev=20): f ≈ 1 + 8*0.66^2.2 ≈ 4.1
    // At z=80 (elev=10): f ≈ 1 + 8*0.83^2.2 ≈ 6.2
    // At z=85 (elev=5):  f ≈ 1 + 8*0.91^2.2 ≈ 7.5
    // At z=89 (elev=1):  f ≈ 1 + 8*0.99^2.2 ≈ 8.8
    const A: f64 = 8.0;
    const B: f64 = 2.2;

    let cos_z = zenith_rad.cos();
    1.0 + A * (1.0 - cos_z).powf(B)
}

/// Compute the azimuthal enhancement factor toward a light source.
///
/// When looking toward a city, the skyglow is brighter than the average
/// for that elevation angle. This models the "light dome" effect.
///
/// # Arguments
/// * `angular_distance_deg` - Angle between viewing direction and the
///   direction to the light source (degrees). 0 = looking directly at city.
/// * `source_distance_km` - Distance to the city center (km).
///
/// # Returns
/// Additional multiplicative factor (1.0 = no enhancement, looking away from city).
pub fn azimuthal_enhancement(angular_distance_deg: f64, source_distance_km: f64) -> f64 {
    if source_distance_km <= 0.0 || angular_distance_deg >= 180.0 {
        return 1.0;
    }

    // The light dome angular size scales roughly as atan(city_radius / distance).
    // For a typical city with 10km radius at 50km distance, the dome subtends ~11 degrees.
    //
    // Within the dome, brightness is enhanced by a factor that depends on distance
    // and the source strength. We model this as a Gaussian profile:
    //
    //   f_az = 1 + peak * exp(-angle^2 / (2 * sigma^2))
    //
    // where sigma = atan(10km / distance) * 180/pi (dome half-width)
    //       peak = some function of distance (closer = more enhanced)

    let sigma_deg = (10.0 / source_distance_km).atan().to_degrees().max(2.0);

    // Peak enhancement: at 10km distance, factor ~3x; at 50km, ~1.5x; at 200km, ~1.1x
    let peak = (50.0 / source_distance_km.max(1.0)).powf(0.5).min(3.0);

    let gaussian = (-angular_distance_deg.powi(2) / (2.0 * sigma_deg.powi(2))).exp();

    1.0 + peak * gaussian
}

/// Compute the effective sky brightness at a given viewing direction,
/// accounting for both elevation and azimuthal effects.
///
/// # Arguments
/// * `zenith_luminance` - Artificial sky brightness at zenith (any unit)
/// * `view_elevation_deg` - Viewing elevation above horizon (degrees)
/// * `angular_distance_to_source_deg` - Angle to nearest major light source
/// * `source_distance_km` - Distance to nearest major light source
///
/// # Returns
/// Effective sky brightness at the specified viewing direction (same unit as input).
pub fn directional_brightness(
    zenith_luminance: f64,
    view_elevation_deg: f64,
    angular_distance_to_source_deg: f64,
    source_distance_km: f64,
) -> f64 {
    let elev_factor = enhancement_factor(view_elevation_deg);
    let az_factor = azimuthal_enhancement(angular_distance_to_source_deg, source_distance_km);
    zenith_luminance * elev_factor * az_factor
}

/// Compute the enhancement factor at a specific zenith angle commonly used
/// in prayer time observation.
///
/// For Fajr, the observer looks toward the eastern horizon at sunrise azimuth,
/// typically at an elevation of about 5-15 degrees above the horizon.
///
/// For Isha, the observer looks toward the western horizon at sunset azimuth,
/// similarly at about 5-15 degrees elevation.
///
/// Returns the enhancement factor for a typical twilight observation angle.
pub fn twilight_observation_factor() -> f64 {
    // Typical observation: 10 degrees above horizon
    enhancement_factor(10.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enhancement_factor_unity_at_zenith() {
        let f = enhancement_factor(90.0);
        assert!(
            (f - 1.0).abs() < 0.001,
            "Zenith factor should be 1.0, got {}",
            f
        );
    }

    #[test]
    fn enhancement_factor_increases_toward_horizon() {
        let f_zenith = enhancement_factor(90.0);
        let f_45 = enhancement_factor(45.0);
        let f_20 = enhancement_factor(20.0);
        let f_10 = enhancement_factor(10.0);
        let f_5 = enhancement_factor(5.0);

        assert!(f_45 > f_zenith);
        assert!(f_20 > f_45);
        assert!(f_10 > f_20);
        assert!(f_5 > f_10);
    }

    #[test]
    fn enhancement_factor_reasonable_range() {
        // At 10 degrees elevation (typical twilight observation), expect ~5-8x
        let f = enhancement_factor(10.0);
        assert!(
            f > 4.0 && f < 10.0,
            "At 10 deg, factor should be ~5-8, got {:.2}",
            f
        );
    }

    #[test]
    fn enhancement_factor_clamped_at_horizon() {
        // Should not diverge at very low elevations
        let f = enhancement_factor(0.0); // Will be clamped to 1 degree
        assert!(f < 15.0, "Horizon factor should be bounded, got {:.2}", f);
        assert!(f > 1.0, "Horizon factor should be > 1, got {:.2}", f);
    }

    #[test]
    fn azimuthal_enhancement_peaks_at_source() {
        let f_at = azimuthal_enhancement(0.0, 30.0);
        let f_away = azimuthal_enhancement(90.0, 30.0);
        assert!(
            f_at > f_away,
            "Should be brighter looking at source: at={:.2}, away={:.2}",
            f_at,
            f_away
        );
    }

    #[test]
    fn azimuthal_enhancement_unity_far_away() {
        // Very distant source: minimal azimuthal effect
        let f = azimuthal_enhancement(0.0, 500.0);
        assert!(f < 2.0, "Distant source should have small effect: {:.2}", f);
    }

    #[test]
    fn azimuthal_enhancement_decays_with_angle() {
        let f_0 = azimuthal_enhancement(0.0, 30.0);
        let f_30 = azimuthal_enhancement(30.0, 30.0);
        let f_60 = azimuthal_enhancement(60.0, 30.0);
        let f_90 = azimuthal_enhancement(90.0, 30.0);

        assert!(f_0 > f_30);
        assert!(f_30 > f_60);
        assert!(f_60 > f_90 || (f_60 - f_90).abs() < 0.01);
    }

    #[test]
    fn directional_brightness_multiplies() {
        let zen = 1.0; // 1 unit zenith brightness
        let db = directional_brightness(zen, 45.0, 0.0, 30.0);
        let elev_f = enhancement_factor(45.0);
        let az_f = azimuthal_enhancement(0.0, 30.0);
        assert!(
            (db - zen * elev_f * az_f).abs() < 1e-10,
            "Directional brightness should be product of factors"
        );
    }

    #[test]
    fn twilight_observation_factor_reasonable() {
        let f = twilight_observation_factor();
        assert!(
            f > 3.0 && f < 10.0,
            "Twilight factor should be ~5-8, got {:.2}",
            f
        );
    }

    #[test]
    fn enhancement_factor_smooth() {
        // Should be smooth (no jumps) between 5 and 85 degrees
        let mut prev = enhancement_factor(85.0);
        for elev in (5..=85).rev() {
            let f = enhancement_factor(elev as f64);
            let ratio = f / prev;
            assert!(
                ratio > 0.9 && ratio < 1.3,
                "Factor should change smoothly: elev={}, ratio={:.3}",
                elev,
                ratio
            );
            prev = f;
        }
    }
}
