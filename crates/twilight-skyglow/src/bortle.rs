//! Bortle Dark-Sky Scale conversions and VIIRS radiance mapping.
//!
//! The Bortle scale (1-9) classifies the night sky darkness at a site.
//! We provide conversions between:
//! - Bortle class <-> zenith luminance (mcd/m^2 or cd/m^2)
//! - Bortle class <-> VIIRS DNB radiance (nW/cm^2/sr)
//! - Bortle class <-> naked-eye limiting magnitude (NELM)
//!
//! Reference values are from Bortle (2001), Falchi et al. (2016),
//! Cinzano (2001), and cross-calibrated with SQM measurements.

/// Bortle class zenith luminance values in mcd/m^2 (millicandelas per square meter).
///
/// These represent the ARTIFICIAL component of zenith luminance.
/// The natural sky background (airglow + zodiacal + integrated starlight)
/// contributes approximately 0.17-0.22 mcd/m^2, which is the "floor"
/// for a pristine Bortle 1 site.
///
/// Sources:
/// - Bortle (2001) "Introducing the Bortle Dark-Sky Scale", S&T
/// - Falchi et al. (2016) "The new world atlas of artificial night sky brightness"
/// - Cinzano et al. (2001) "The first World Atlas of the artificial night sky brightness"
/// - SQM cross-calibration data from Globe at Night
///
/// The artificial component is total luminance minus natural background.
pub const BORTLE_LUMINANCE_MCD: [f64; 10] = [
    0.0,  // index 0: unused
    0.01, // Bortle 1: Excellent dark site. Essentially zero artificial light.
    0.04, // Bortle 2: Typical truly dark site. Slight brightening at horizon.
    0.08, // Bortle 3: Rural sky. Zodiacal light visible but dome glow on horizon.
    0.15, // Bortle 4: Rural/suburban transition. Light domes visible in several directions.
    0.4,  // Bortle 5: Suburban. Milky Way washed out at zenith.
    0.8,  // Bortle 6: Bright suburban. MW only visible at zenith if at all.
    2.5,  // Bortle 7: Suburban/urban transition. Entire sky grayish-white.
    7.0,  // Bortle 8: City sky. Only bright constellations visible.
    20.0, // Bortle 9: Inner city. Only the Moon, planets, and a few bright stars visible.
];

/// Natural sky background luminance (mcd/m^2) under pristine conditions.
///
/// This includes:
/// - Airglow: ~0.08-0.12 mcd/m^2 (variable, solar cycle dependent)
/// - Zodiacal light: ~0.05-0.08 mcd/m^2 (at zenith, varies with ecliptic angle)
/// - Integrated starlight: ~0.01-0.02 mcd/m^2
///
/// Total natural: approximately 0.17 mcd/m^2 (Bortle 1 total = 0.17-0.22).
pub const NATURAL_SKY_LUMINANCE_MCD: f64 = 0.17;

/// VIIRS DNB radiance values corresponding to each Bortle class (nW/cm^2/sr).
///
/// These are approximate mappings based on Falchi et al. (2016) and
/// empirical SQM-VIIRS cross-calibration. The relationship is roughly:
///   L_zenith_artificial ≈ 0.092 * R_viirs^0.72  (mcd/m^2, for R in nW/cm^2/sr)
///
/// where 0.092 and 0.72 are empirical fit parameters that account for
/// the Garstang scattering model converting ground-level upward radiance
/// to zenith sky brightness.
pub const BORTLE_VIIRS_RADIANCE: [f64; 10] = [
    0.0,   // index 0: unused
    0.1,   // Bortle 1: Essentially no artificial radiance
    0.3,   // Bortle 2: Trace artificial radiance
    0.8,   // Bortle 3: Rural, minor nearby sources
    2.0,   // Bortle 4: Small town visible
    6.0,   // Bortle 5: Suburban
    15.0,  // Bortle 6: Bright suburban
    40.0,  // Bortle 7: Urban transition
    100.0, // Bortle 8: City
    300.0, // Bortle 9: Inner city core
];

/// Naked-eye limiting magnitude (NELM) for each Bortle class.
///
/// NELM is the faintest star magnitude visible to the naked eye.
/// Brighter limiting magnitude = worse sky. Pristine sky reaches 7.6-8.0.
///
/// Conversion: NELM ≈ 21.58 - 5*log10(L_total / 108e-6) approximately,
/// but the Bortle scale is defined observationally, not strictly from luminance.
pub const BORTLE_NELM: [f64; 10] = [
    0.0, // index 0: unused
    7.6, // Bortle 1
    7.1, // Bortle 2
    6.6, // Bortle 3
    6.2, // Bortle 4
    5.6, // Bortle 5
    5.1, // Bortle 6
    4.6, // Bortle 7
    4.0, // Bortle 8
    3.5, // Bortle 9
];

/// SQM (Sky Quality Meter) readings in mag/arcsec^2 for each Bortle class.
///
/// SQM measures sky surface brightness. Higher = darker.
/// Conversion: SQM = 12.58 - 2.5*log10(L_total in cd/m^2 * 1e6)
/// where L_total is in cd/m^2.
pub const BORTLE_SQM: [f64; 10] = [
    0.0,   // index 0: unused
    21.99, // Bortle 1: Pristine
    21.89, // Bortle 2: Excellent dark site
    21.69, // Bortle 3: Rural
    21.25, // Bortle 4: Rural/suburban transition
    20.49, // Bortle 5: Suburban
    19.50, // Bortle 6: Bright suburban
    18.94, // Bortle 7: Urban transition
    18.38, // Bortle 8: City
    17.50, // Bortle 9: Inner city
];

/// Convert VIIRS DNB radiance (nW/cm^2/sr) to artificial zenith luminance (mcd/m^2).
///
/// Uses the empirical relationship from Falchi et al. (2016):
///   L_artificial = a * R^b
///
/// where a=0.092 and b=0.72 are fit parameters derived from SQM-calibrated
/// Garstang model inversions. This accounts for the atmospheric scattering
/// that converts ground-level upward radiance into zenith sky brightness.
///
/// The relationship is non-linear because higher radiance sources are typically
/// more concentrated (cities) and the scattering geometry changes.
pub fn radiance_to_zenith_luminance(radiance_nw: f64) -> f64 {
    if radiance_nw <= 0.0 {
        return 0.0;
    }
    // Falchi (2016) empirical fit: L = 0.092 * R^0.72
    // Units: radiance_nw in nW/cm^2/sr -> L in mcd/m^2
    0.092 * radiance_nw.powf(0.72)
}

/// Convert artificial zenith luminance (mcd/m^2) back to approximate VIIRS radiance.
///
/// Inverse of `radiance_to_zenith_luminance`.
pub fn zenith_luminance_to_radiance(luminance_mcd: f64) -> f64 {
    if luminance_mcd <= 0.0 {
        return 0.0;
    }
    // L = 0.092 * R^0.72  =>  R = (L / 0.092)^(1/0.72)
    (luminance_mcd / 0.092).powf(1.0 / 0.72)
}

/// Convert a Bortle class (1-9) to artificial zenith luminance (mcd/m^2).
pub fn bortle_to_luminance(bortle: u8) -> f64 {
    let b = bortle.clamp(1, 9) as usize;
    BORTLE_LUMINANCE_MCD[b]
}

/// Convert a Bortle class (1-9) to approximate VIIRS radiance (nW/cm^2/sr).
pub fn bortle_to_radiance(bortle: u8) -> f64 {
    let b = bortle.clamp(1, 9) as usize;
    BORTLE_VIIRS_RADIANCE[b]
}

/// Convert artificial zenith luminance (mcd/m^2) to Bortle class (1-9).
///
/// Returns the nearest Bortle class by finding the closest luminance match.
pub fn luminance_to_bortle(luminance_mcd: f64) -> u8 {
    if luminance_mcd <= BORTLE_LUMINANCE_MCD[1] {
        return 1;
    }
    for b in 1..9 {
        let mid = (BORTLE_LUMINANCE_MCD[b] + BORTLE_LUMINANCE_MCD[b + 1]) / 2.0;
        if luminance_mcd < mid {
            return b as u8;
        }
    }
    9
}

/// Convert VIIRS radiance to Bortle class.
pub fn radiance_to_bortle(radiance_nw: f64) -> u8 {
    let lum = radiance_to_zenith_luminance(radiance_nw);
    luminance_to_bortle(lum)
}

/// Convert zenith luminance (mcd/m^2) to SQM (mag/arcsec^2).
///
/// SQM = -2.5 * log10(L_total * pi / (12.566 * 108e3))
///
/// More precisely, for total luminance L_total in mcd/m^2:
///   SQM = 12.58 - 2.5 * log10(L_total * 1e-3)
///
/// (where the 12.58 constant absorbs unit conversions)
pub fn luminance_to_sqm(total_luminance_mcd: f64) -> f64 {
    if total_luminance_mcd <= 0.0 {
        return 25.0; // Darker than physically possible, return max
    }
    // total_luminance_mcd is in mcd/m^2 = 1e-3 cd/m^2
    // SQM = -2.5 * log10(L / L_ref) where L_ref corresponds to 0 mag/arcsec^2
    // Standard formula: SQM = 12.58 - 2.5 * log10(L_cd_per_m2)
    // where L_cd_per_m2 = total_luminance_mcd * 1e-3
    let l_cd = total_luminance_mcd * 1e-3;
    12.58 - 2.5 * l_cd.log10()
}

/// Convert SQM reading (mag/arcsec^2) to total luminance (mcd/m^2).
pub fn sqm_to_luminance(sqm: f64) -> f64 {
    // sqm = 12.58 - 2.5 * log10(L_cd)
    // log10(L_cd) = (12.58 - sqm) / 2.5
    // L_cd = 10^((12.58 - sqm) / 2.5)
    let l_cd = 10.0_f64.powf((12.58 - sqm) / 2.5);
    l_cd * 1e3 // convert cd/m^2 to mcd/m^2
}

/// Estimate the prayer time shift (minutes) caused by light pollution.
///
/// Light pollution adds an artificial background that:
/// - Delays the perceived disappearance of twilight glow (later Isha)
/// - Delays the perceived appearance of dawn glow (later Fajr)
///
/// This is a rough empirical estimate. The actual shift depends on the full
/// spectral computation and threshold model. Use the full pipeline for
/// accurate results.
///
/// # Arguments
/// * `artificial_mcd` - Artificial zenith luminance in mcd/m^2
///
/// # Returns
/// Approximate shift in minutes (positive = later)
pub fn estimated_prayer_shift_minutes(artificial_mcd: f64) -> f64 {
    if artificial_mcd <= 0.01 {
        return 0.0;
    }
    // Empirical: each doubling of sky brightness shifts twilight threshold
    // by approximately 2-3 minutes (Garstang 1989, cross-checked with
    // observations in Cinzano 2000).
    //
    // The natural twilight luminance at typical Isha depression (~15-18 deg)
    // is roughly 0.01-0.1 mcd/m^2. When artificial light matches or exceeds
    // this, the shift becomes significant.
    //
    // Simple log model: shift ≈ 2.5 * log2(1 + artificial / natural_at_threshold)
    // where natural_at_threshold ≈ 0.05 mcd/m^2 (typical Isha threshold luminance)
    let natural_threshold = 0.05;
    let ratio = artificial_mcd / natural_threshold;
    if ratio < 0.01 {
        return 0.0;
    }
    2.5 * (1.0 + ratio).log2()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radiance_luminance_roundtrip() {
        for &r in &[0.5, 2.0, 10.0, 50.0, 200.0] {
            let l = radiance_to_zenith_luminance(r);
            let r2 = zenith_luminance_to_radiance(l);
            assert!(
                (r - r2).abs() / r < 0.01,
                "Roundtrip failed for R={}: got {}",
                r,
                r2
            );
        }
    }

    #[test]
    fn radiance_to_luminance_monotonic() {
        let mut prev = 0.0;
        for r in [0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0] {
            let l = radiance_to_zenith_luminance(r);
            assert!(
                l > prev,
                "Should be monotonic: R={}, L={}, prev={}",
                r,
                l,
                prev
            );
            prev = l;
        }
    }

    #[test]
    fn radiance_zero_gives_zero() {
        assert_eq!(radiance_to_zenith_luminance(0.0), 0.0);
        assert_eq!(radiance_to_zenith_luminance(-5.0), 0.0);
    }

    #[test]
    fn bortle_luminance_monotonic() {
        for b in 1..9 {
            assert!(
                BORTLE_LUMINANCE_MCD[b] < BORTLE_LUMINANCE_MCD[b + 1],
                "Bortle luminance not monotonic at class {}",
                b
            );
        }
    }

    #[test]
    fn bortle_viirs_monotonic() {
        for b in 1..9 {
            assert!(
                BORTLE_VIIRS_RADIANCE[b] < BORTLE_VIIRS_RADIANCE[b + 1],
                "Bortle VIIRS radiance not monotonic at class {}",
                b
            );
        }
    }

    #[test]
    fn bortle_nelm_decreasing() {
        for b in 1..9 {
            assert!(
                BORTLE_NELM[b] > BORTLE_NELM[b + 1],
                "Bortle NELM not decreasing at class {}",
                b
            );
        }
    }

    #[test]
    fn bortle_sqm_decreasing() {
        for b in 1..9 {
            assert!(
                BORTLE_SQM[b] > BORTLE_SQM[b + 1],
                "Bortle SQM not decreasing at class {}",
                b
            );
        }
    }

    #[test]
    fn luminance_to_bortle_boundaries() {
        assert_eq!(luminance_to_bortle(0.0), 1);
        assert_eq!(luminance_to_bortle(0.005), 1);
        assert_eq!(luminance_to_bortle(0.01), 1);
        assert_eq!(luminance_to_bortle(25.0), 9);
        assert_eq!(luminance_to_bortle(100.0), 9);
    }

    #[test]
    fn luminance_to_bortle_covers_all_classes() {
        // Each Bortle luminance should map back to its own class
        for b in 1..=9u8 {
            let lum = bortle_to_luminance(b);
            let b2 = luminance_to_bortle(lum);
            assert_eq!(
                b, b2,
                "Bortle {} with lum {:.3} maps back to {}",
                b, lum, b2
            );
        }
    }

    #[test]
    fn radiance_to_bortle_city() {
        // 100 nW/cm^2/sr is roughly Bortle 8
        let b = radiance_to_bortle(100.0);
        assert!(b >= 7 && b <= 9, "100 nW should be Bortle 7-9, got {}", b);
    }

    #[test]
    fn radiance_to_bortle_dark_site() {
        let b = radiance_to_bortle(0.2);
        assert!(b <= 2, "0.2 nW should be Bortle 1-2, got {}", b);
    }

    #[test]
    fn sqm_luminance_roundtrip() {
        for &l in &[0.17, 0.5, 2.0, 10.0] {
            let sqm = luminance_to_sqm(l);
            let l2 = sqm_to_luminance(sqm);
            assert!(
                (l - l2).abs() / l < 0.01,
                "SQM roundtrip failed for L={}: got {}, sqm={}",
                l,
                l2,
                sqm
            );
        }
    }

    #[test]
    fn sqm_dark_site_high() {
        // A pristine site (Bortle 1) should have SQM > 21.5
        let sqm = luminance_to_sqm(NATURAL_SKY_LUMINANCE_MCD);
        assert!(sqm > 21.0, "Pristine SQM should be >21, got {:.2}", sqm);
    }

    #[test]
    fn sqm_city_low() {
        // A city (Bortle 8: 7 mcd/m^2 artificial + 0.17 natural)
        let sqm = luminance_to_sqm(7.0 + NATURAL_SKY_LUMINANCE_MCD);
        assert!(sqm < 19.5, "City SQM should be <19.5, got {:.2}", sqm);
    }

    #[test]
    fn prayer_shift_dark_site_negligible() {
        let shift = estimated_prayer_shift_minutes(0.01);
        assert!(
            shift.abs() < 0.1,
            "Dark site shift should be ~0, got {:.2}",
            shift
        );
    }

    #[test]
    fn prayer_shift_city_significant() {
        // Bortle 8 city: 7 mcd/m^2 artificial
        let shift = estimated_prayer_shift_minutes(7.0);
        assert!(
            shift > 10.0,
            "City shift should be >10 min, got {:.1}",
            shift
        );
    }

    #[test]
    fn prayer_shift_monotonic() {
        let mut prev = 0.0;
        for &l in &[0.01, 0.1, 0.5, 1.0, 5.0, 20.0] {
            let shift = estimated_prayer_shift_minutes(l);
            assert!(shift >= prev, "Shift not monotonic at L={}", l);
            prev = shift;
        }
    }

    #[test]
    fn bortle_clamp_low() {
        assert_eq!(bortle_to_luminance(0), BORTLE_LUMINANCE_MCD[1]);
    }

    #[test]
    fn bortle_clamp_high() {
        assert_eq!(bortle_to_luminance(10), BORTLE_LUMINANCE_MCD[9]);
    }
}
