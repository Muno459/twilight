//! Standard atmosphere profiles (Anderson/AFGL 1986).
//!
//! Provides temperature, pressure, and number density at standard altitude levels
//! for 6 reference atmospheres.

/// Standard altitude grid in km (50 levels from 0 to 100 km).
pub const ALTITUDE_GRID_KM: [f64; 51] = [
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0,
    65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, // Padding to fill 51 entries
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
];

/// Number of actual altitude levels in the grid.
pub const NUM_LEVELS: usize = 41;

/// US Standard Atmosphere 1976 — temperature profile [K]
pub const US_STD_TEMPERATURE_K: [f64; 41] = [
    288.15, 281.65, 275.15, 268.65, 262.15, 255.65, 249.15, 242.65, 236.15, 229.65, 223.15, 216.65,
    216.65, 216.65, 216.65, 216.65, 216.65, 216.65, 216.65, 216.65, 216.65, 217.65, 218.65, 219.65,
    220.65, 221.65, 226.65, 237.05, 251.05, 264.15, 270.65, 260.65, 247.05, 233.05, 219.15, 208.40,
    198.64, 188.89, 186.87, 188.42, 195.08,
];

/// US Standard Atmosphere 1976 — pressure profile [hPa]
pub const US_STD_PRESSURE_HPA: [f64; 41] = [
    1013.25, 898.76, 795.01, 701.21, 616.60, 540.48, 472.17, 411.05, 356.51, 308.00, 264.99,
    226.99, 194.02, 165.79, 141.70, 121.11, 103.52, 88.497, 75.652, 64.674, 55.293, 47.289, 40.475,
    34.668, 29.717, 25.492, 11.970, 5.746, 2.871, 1.491, 0.7978, 0.4253, 0.2196, 0.1093, 0.05221,
    0.02388, 0.01052, 0.00446, 0.00184, 0.000760, 0.000320,
];

/// US Standard Atmosphere 1976 — number density profile [molecules/cm³]
pub const US_STD_NUMBER_DENSITY: [f64; 41] = [
    2.547e19, 2.311e19, 2.093e19, 1.891e19, 1.703e19, 1.532e19, 1.373e19, 1.227e19, 1.093e19,
    9.711e18, 8.598e18, 7.585e18, 6.486e18, 5.543e18, 4.738e18, 4.049e18, 3.462e18, 2.960e18,
    2.529e18, 2.162e18, 1.849e18, 1.573e18, 1.341e18, 1.143e18, 9.759e17, 8.334e17, 3.828e17,
    1.757e17, 8.283e16, 4.084e16, 2.135e16, 1.181e16, 6.439e15, 3.393e15, 1.722e15, 8.300e14,
    3.838e14, 1.714e14, 7.116e13, 2.920e13, 1.189e13,
];

/// US Standard Atmosphere 1976 — ozone number density [molecules/cm³]
pub const US_STD_OZONE_DENSITY: [f64; 41] = [
    5.40e11, 5.40e11, 5.40e11, 5.00e11, 4.60e11, 4.20e11, 3.90e11, 3.60e11, 3.20e11, 2.90e11,
    2.64e11, 2.40e11, 2.34e11, 2.67e11, 3.24e11, 4.07e11, 5.00e11, 5.95e11, 6.87e11, 7.37e11,
    7.45e11, 7.09e11, 6.35e11, 5.33e11, 4.34e11, 3.45e11, 1.38e11, 4.19e10, 1.30e10, 4.30e9,
    1.50e9, 5.00e8, 2.00e8, 5.00e7, 5.00e6, 5.00e5, 5.00e4, 0.0, 0.0, 0.0, 0.0,
];

/// Atmosphere profile type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtmosphereType {
    /// US Standard 1976
    UsStandard,
    /// Tropical
    Tropical,
    /// Mid-latitude summer
    MidLatSummer,
    /// Mid-latitude winter
    MidLatWinter,
    /// Subarctic summer
    SubarcticSummer,
    /// Subarctic winter
    SubarcticWinter,
}

/// Get temperature at a given altitude by linear interpolation.
///
/// `altitude_km`: altitude in kilometers
/// Returns temperature in Kelvin.
pub fn temperature_at(altitude_km: f64, profile: AtmosphereType) -> f64 {
    let temps = match profile {
        AtmosphereType::UsStandard => &US_STD_TEMPERATURE_K,
        // TODO: add other profiles
        _ => &US_STD_TEMPERATURE_K,
    };
    interpolate_profile(&ALTITUDE_GRID_KM[..NUM_LEVELS], temps, altitude_km)
}

/// Get pressure at a given altitude by log-linear interpolation.
pub fn pressure_at(altitude_km: f64, profile: AtmosphereType) -> f64 {
    let pressures = match profile {
        AtmosphereType::UsStandard => &US_STD_PRESSURE_HPA,
        _ => &US_STD_PRESSURE_HPA,
    };
    // Log-linear interpolation for pressure
    let log_pressures: [f64; 41] = {
        let mut lp = [0.0; 41];
        let mut i = 0;
        while i < 41 {
            lp[i] = libm::log(pressures[i]);
            i += 1;
        }
        lp
    };
    let log_p = interpolate_profile(&ALTITUDE_GRID_KM[..NUM_LEVELS], &log_pressures, altitude_km);
    libm::exp(log_p)
}

/// Get number density at a given altitude by log-linear interpolation.
pub fn number_density_at(altitude_km: f64, profile: AtmosphereType) -> f64 {
    let densities = match profile {
        AtmosphereType::UsStandard => &US_STD_NUMBER_DENSITY,
        _ => &US_STD_NUMBER_DENSITY,
    };
    let log_densities: [f64; 41] = {
        let mut ld = [0.0; 41];
        let mut i = 0;
        while i < 41 {
            ld[i] = if densities[i] > 0.0 {
                libm::log(densities[i])
            } else {
                -100.0
            };
            i += 1;
        }
        ld
    };
    let log_n = interpolate_profile(&ALTITUDE_GRID_KM[..NUM_LEVELS], &log_densities, altitude_km);
    libm::exp(log_n)
}

/// Get ozone number density at a given altitude by log-linear interpolation.
///
/// Returns ozone number density in molecules/cm³.
pub fn ozone_density_at(altitude_km: f64, profile: AtmosphereType) -> f64 {
    let densities = match profile {
        AtmosphereType::UsStandard => &US_STD_OZONE_DENSITY,
        _ => &US_STD_OZONE_DENSITY,
    };
    let log_densities: [f64; 41] = {
        let mut ld = [0.0; 41];
        let mut i = 0;
        while i < 41 {
            ld[i] = if densities[i] > 0.0 {
                libm::log(densities[i])
            } else {
                -100.0
            };
            i += 1;
        }
        ld
    };
    let log_n = interpolate_profile(&ALTITUDE_GRID_KM[..NUM_LEVELS], &log_densities, altitude_km);
    let result = libm::exp(log_n);
    // Clamp very small values to zero
    if result < 1.0 {
        0.0
    } else {
        result
    }
}

/// Linear interpolation on a profile.
fn interpolate_profile(altitudes: &[f64], values: &[f64], alt_km: f64) -> f64 {
    if alt_km <= altitudes[0] {
        return values[0];
    }
    let n = altitudes.len();
    if alt_km >= altitudes[n - 1] {
        return values[n - 1];
    }
    for i in 0..(n - 1) {
        if alt_km >= altitudes[i] && alt_km < altitudes[i + 1] {
            let frac = (alt_km - altitudes[i]) / (altitudes[i + 1] - altitudes[i]);
            return values[i] + frac * (values[i + 1] - values[i]);
        }
    }
    values[n - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── US Standard 1976 reference values ──
    // These are well-known standard atmosphere values from the 1976 publication.

    #[test]
    fn temperature_at_sea_level() {
        // US Standard 1976: T(0 km) = 288.15 K (15°C)
        let t = temperature_at(0.0, AtmosphereType::UsStandard);
        assert!(
            (t - 288.15).abs() < 0.01,
            "T(0km) = {}, expected 288.15 K",
            t
        );
    }

    #[test]
    fn temperature_at_tropopause() {
        // US Standard 1976: T(11 km) = 216.65 K (-56.5°C)
        let t = temperature_at(11.0, AtmosphereType::UsStandard);
        assert!(
            (t - 216.65).abs() < 0.01,
            "T(11km) = {}, expected 216.65 K",
            t
        );
    }

    #[test]
    fn temperature_isothermal_in_stratosphere() {
        // US Standard 1976: T is constant at 216.65 K from 11 to 20 km
        for alt in &[12.0, 15.0, 18.0, 20.0] {
            let t = temperature_at(*alt, AtmosphereType::UsStandard);
            assert!(
                (t - 216.65).abs() < 0.1,
                "T({}km) = {}, expected ~216.65 K (isothermal layer)",
                alt,
                t
            );
        }
    }

    #[test]
    fn temperature_lapse_rate_troposphere() {
        // Tropospheric lapse rate: -6.5 K/km from 0-11 km
        let t0 = temperature_at(0.0, AtmosphereType::UsStandard);
        let t5 = temperature_at(5.0, AtmosphereType::UsStandard);
        let lapse = (t0 - t5) / 5.0; // K/km
        assert!(
            (lapse - 6.5).abs() < 0.1,
            "Tropospheric lapse rate: {} K/km, expected 6.5",
            lapse
        );
    }

    #[test]
    fn temperature_decreases_in_troposphere() {
        let mut prev = temperature_at(0.0, AtmosphereType::UsStandard);
        for alt_km in 1..=10 {
            let t = temperature_at(alt_km as f64, AtmosphereType::UsStandard);
            assert!(
                t < prev,
                "T should decrease: T({}km)={} >= T({}km)={}",
                alt_km,
                t,
                alt_km - 1,
                prev
            );
            prev = t;
        }
    }

    #[test]
    fn pressure_at_sea_level() {
        // US Standard 1976: P(0 km) = 1013.25 hPa
        let p = pressure_at(0.0, AtmosphereType::UsStandard);
        assert!(
            (p - 1013.25).abs() < 1.0,
            "P(0km) = {}, expected 1013.25 hPa",
            p
        );
    }

    #[test]
    fn pressure_decreases_with_altitude() {
        let mut prev = pressure_at(0.0, AtmosphereType::UsStandard);
        for alt in &[1.0, 5.0, 10.0, 20.0, 50.0, 80.0, 100.0] {
            let p = pressure_at(*alt, AtmosphereType::UsStandard);
            assert!(
                p < prev,
                "Pressure should decrease: P({}km)={} >= prev={}",
                alt,
                p,
                prev
            );
            prev = p;
        }
    }

    #[test]
    fn pressure_at_10km_approximately_264hpa() {
        // US Std 1976: P(10km) ≈ 264.99 hPa
        let p = pressure_at(10.0, AtmosphereType::UsStandard);
        assert!(
            (p - 264.99).abs() < 5.0,
            "P(10km) = {}, expected ~264.99 hPa",
            p
        );
    }

    #[test]
    fn pressure_at_50km_is_very_low() {
        // P(50km) ≈ 0.80 hPa
        let p = pressure_at(50.0, AtmosphereType::UsStandard);
        assert!(p < 2.0, "P(50km) should be < 2 hPa, got {}", p);
        assert!(p > 0.1, "P(50km) should be > 0.1 hPa, got {}", p);
    }

    #[test]
    fn number_density_at_sea_level() {
        // US Std 1976: n(0km) = 2.547e19 molecules/cm³
        let n = number_density_at(0.0, AtmosphereType::UsStandard);
        let ratio = n / 2.547e19;
        assert!(
            (ratio - 1.0).abs() < 0.01,
            "n(0km) = {:.3e}, expected 2.547e19",
            n
        );
    }

    #[test]
    fn number_density_decreases_with_altitude() {
        let mut prev = number_density_at(0.0, AtmosphereType::UsStandard);
        for alt in &[5.0, 10.0, 20.0, 50.0, 80.0] {
            let n = number_density_at(*alt, AtmosphereType::UsStandard);
            assert!(
                n < prev,
                "Number density should decrease: n({}km)={:.3e} >= prev={:.3e}",
                alt,
                n,
                prev
            );
            prev = n;
        }
    }

    #[test]
    fn ozone_density_peaks_near_20km() {
        // Ozone layer peaks at ~20 km altitude
        let o3_surface = ozone_density_at(0.0, AtmosphereType::UsStandard);
        let o3_peak = ozone_density_at(20.0, AtmosphereType::UsStandard);
        let o3_high = ozone_density_at(50.0, AtmosphereType::UsStandard);

        assert!(
            o3_peak > o3_surface,
            "O3 at 20km ({:.3e}) should exceed surface ({:.3e})",
            o3_peak,
            o3_surface
        );
        assert!(
            o3_peak > o3_high,
            "O3 at 20km ({:.3e}) should exceed 50km ({:.3e})",
            o3_peak,
            o3_high
        );
    }

    #[test]
    fn ozone_density_at_peak() {
        // US Std 1976: O3(20km) = 7.45e11 molecules/cm³
        let o3 = ozone_density_at(20.0, AtmosphereType::UsStandard);
        let ratio = o3 / 7.45e11;
        assert!(
            (ratio - 1.0).abs() < 0.05,
            "O3(20km) = {:.3e}, expected ~7.45e11",
            o3
        );
    }

    #[test]
    fn ozone_density_zero_above_90km() {
        // O3 density is zero above ~87 km in the table
        let o3 = ozone_density_at(95.0, AtmosphereType::UsStandard);
        assert!(o3 < 1.0, "O3(95km) should be ~0, got {:.3e}", o3);
    }

    #[test]
    fn interpolation_at_grid_points() {
        // At exact grid points, interpolation should return the table value exactly
        let t0 = temperature_at(0.0, AtmosphereType::UsStandard);
        assert!((t0 - US_STD_TEMPERATURE_K[0]).abs() < 0.001);

        let t10 = temperature_at(10.0, AtmosphereType::UsStandard);
        assert!((t10 - US_STD_TEMPERATURE_K[10]).abs() < 0.001);
    }

    #[test]
    fn interpolation_midpoint() {
        // At 0.5 km, should be midpoint of T(0) and T(1)
        let t = temperature_at(0.5, AtmosphereType::UsStandard);
        let expected = (US_STD_TEMPERATURE_K[0] + US_STD_TEMPERATURE_K[1]) / 2.0;
        assert!(
            (t - expected).abs() < 0.1,
            "T(0.5km) = {}, expected {}",
            t,
            expected
        );
    }

    #[test]
    fn interpolation_clamps_below_zero() {
        // Below 0 km should return surface value
        let t = temperature_at(-5.0, AtmosphereType::UsStandard);
        assert!((t - 288.15).abs() < 0.01);
    }

    #[test]
    fn interpolation_clamps_above_max() {
        // Above 100 km should return top value
        let t = temperature_at(200.0, AtmosphereType::UsStandard);
        let t_top = US_STD_TEMPERATURE_K[NUM_LEVELS - 1];
        assert!((t - t_top).abs() < 0.01);
    }

    #[test]
    fn all_profiles_fall_back_to_us_standard() {
        // Until other profiles are implemented, all should return US Std values
        let profiles = [
            AtmosphereType::Tropical,
            AtmosphereType::MidLatSummer,
            AtmosphereType::MidLatWinter,
            AtmosphereType::SubarcticSummer,
            AtmosphereType::SubarcticWinter,
        ];
        for profile in &profiles {
            let t = temperature_at(0.0, *profile);
            assert!(
                (t - 288.15).abs() < 0.01,
                "Profile {:?} should fallback to US Std",
                profile
            );
        }
    }

    #[test]
    fn altitude_grid_is_monotonic() {
        for i in 0..(NUM_LEVELS - 1) {
            assert!(
                ALTITUDE_GRID_KM[i + 1] > ALTITUDE_GRID_KM[i],
                "Altitude grid not monotonic at index {}: {} >= {}",
                i,
                ALTITUDE_GRID_KM[i],
                ALTITUDE_GRID_KM[i + 1]
            );
        }
    }

    #[test]
    fn altitude_grid_starts_at_zero() {
        assert!((ALTITUDE_GRID_KM[0]).abs() < 1e-10);
    }

    #[test]
    fn altitude_grid_ends_at_100km() {
        assert!((ALTITUDE_GRID_KM[NUM_LEVELS - 1] - 100.0).abs() < 1e-10);
    }
}
