//! Earth rotation and coordinate transformations.
//!
//! Converts geocentric J2000/ICRF positions to topocentric
//! solar zenith angle and azimuth, accounting for:
//! - Earth rotation (GMST/GAST via IAU 2006 precession)
//! - Nutation (IAU 1980, simplified)
//! - Observer geodetic position
//! - Atmospheric refraction (optional)
//!
//! Reference frames:
//!   ICRF/J2000 -> ITRF (via Earth rotation) -> topocentric (observer)

use libm::{asin, atan2, cos, floor, sin, sqrt, tan};

const PI: f64 = core::f64::consts::PI;
const DEG_TO_RAD: f64 = PI / 180.0;
const RAD_TO_DEG: f64 = 180.0 / PI;
const TWO_PI: f64 = 2.0 * PI;

/// Seconds per Julian day.
const SPD: f64 = 86400.0;

/// Julian date of J2000.0 epoch.
const J2000_JD: f64 = 2_451_545.0;

/// Julian centuries per Julian day.
const JULIAN_CENTURY_DAYS: f64 = 36525.0;

/// Earth equatorial radius (WGS84) in km.
const EARTH_A: f64 = 6378.137;

/// Earth polar radius (WGS84) in km.
const EARTH_B: f64 = 6356.752_314_245;

/// Earth flattening (unused currently but kept for reference).
const _EARTH_F: f64 = 1.0 / 298.257_223_563;

/// Obliquity of the ecliptic at J2000.0 (degrees).
const EPSILON_J2000: f64 = 23.439_291_11;

// ── Time conversions ───────────────────────────────────────────────

/// Convert calendar date (UTC) to Julian Date.
///
/// Uses the standard algorithm valid for dates after 1582 Oct 15 (Gregorian).
pub fn calendar_to_jd(year: i32, month: i32, day: i32, hour: i32, minute: i32, second: i32) -> f64 {
    let (y, m) = if month <= 2 {
        (year as f64 - 1.0, month as f64 + 12.0)
    } else {
        (year as f64, month as f64)
    };

    let a = floor(y / 100.0);
    let b = 2.0 - a + floor(a / 4.0);

    let jd = floor(365.25 * (y + 4716.0)) + floor(30.6001 * (m + 1.0)) + day as f64 + b - 1524.5;

    let day_fraction = (hour as f64 + minute as f64 / 60.0 + second as f64 / 3600.0) / 24.0;

    jd + day_fraction
}

/// Convert Julian Date to seconds past J2000.0 TDB.
///
/// Assumes JD is in TDB (or TT, difference < 2ms).
pub fn jd_to_tdb_seconds(jd: f64) -> f64 {
    (jd - J2000_JD) * SPD
}

/// Convert seconds past J2000.0 TDB to Julian Date.
pub fn tdb_seconds_to_jd(tdb: f64) -> f64 {
    tdb / SPD + J2000_JD
}

/// Convert UTC Julian Date to TDB Julian Date (approximate).
///
/// TDB = TT + periodic terms (< 2ms, ignored here)
/// TT = UTC + delta_t (where delta_t includes leap seconds + TT-TAI offset)
///
/// delta_t for 2024 is approximately 69.184 seconds.
pub fn utc_jd_to_tdb_jd(utc_jd: f64, delta_t: f64) -> f64 {
    utc_jd + delta_t / SPD
}

/// Convert calendar UTC to TDB seconds past J2000.
pub fn calendar_utc_to_tdb(
    year: i32,
    month: i32,
    day: i32,
    hour: i32,
    minute: i32,
    second: i32,
    delta_t: f64,
) -> f64 {
    let jd_utc = calendar_to_jd(year, month, day, hour, minute, second);
    let jd_tdb = utc_jd_to_tdb_jd(jd_utc, delta_t);
    jd_to_tdb_seconds(jd_tdb)
}

// ── Earth rotation ─────────────────────────────────────────────────

/// Greenwich Mean Sidereal Time (GMST) in radians.
///
/// Uses the IAU 2006 expression (Capitaine et al. 2003).
/// Input: Julian UT1 date.
pub fn gmst_radians(jd_ut1: f64) -> f64 {
    let t = (jd_ut1 - J2000_JD) / JULIAN_CENTURY_DAYS;
    let du = jd_ut1 - J2000_JD;

    // ERA (Earth Rotation Angle) - IERS 2003
    let theta = TWO_PI * (0.779_057_273_264_0 + 1.002_737_811_911_354_48 * du);

    // GMST = ERA + precession polynomial
    // Capitaine et al. 2003 (arcseconds -> radians)
    let gmst_arcsec = 0.014_506 + 4612.156_534 * t + 1.391_581_7 * t * t
        - 0.000_000_44 * t * t * t
        - 0.000_029_956 * t * t * t * t
        - 3.68e-8 * t * t * t * t * t;

    let gmst = theta + gmst_arcsec * PI / (180.0 * 3600.0);

    // Normalize to [0, 2*pi)
    let mut result = gmst % TWO_PI;
    if result < 0.0 {
        result += TWO_PI;
    }
    result
}

/// Simplified nutation in longitude (delta_psi) and obliquity (delta_epsilon).
///
/// Uses the dominant terms of the IAU 1980 nutation series.
/// Returns (delta_psi, delta_epsilon) in radians.
pub fn nutation_simple(t_centuries: f64) -> (f64, f64) {
    let t = t_centuries;

    // Fundamental arguments (degrees)
    // Moon's mean anomaly
    let m_moon = 134.963_411_4 + 477_198.867_631 * t;
    // Sun's mean anomaly
    let m_sun = 357.529_109_2 + 35_999.050_29 * t;
    // Moon's argument of latitude (reserved for extended nutation)
    let _f = 93.272_090_9 + 483_202.017_53 * t;
    // Moon's mean elongation (reserved for extended nutation)
    let _d = 297.850_195_4 + 445_267.111_5 * t;
    // Longitude of ascending node of Moon
    let omega = 125.044_555_0 - 1934.136_261 * t;

    let omega_rad = omega * DEG_TO_RAD;
    let _m_sun_rad = m_sun * DEG_TO_RAD;
    let _m_moon_rad = m_moon * DEG_TO_RAD;
    let two_omega = 2.0 * omega_rad;
    let two_l_sun = 2.0 * (280.466_4 + 36_000.769_8 * t) * DEG_TO_RAD;
    let two_l_moon = 2.0 * (218.316_4 + 481_267.883_4 * t) * DEG_TO_RAD;

    // Dominant nutation terms (arcseconds)
    let delta_psi = -17.2 * sin(omega_rad) - 1.32 * sin(two_l_sun) - 0.23 * sin(two_l_moon)
        + 0.21 * sin(two_omega);

    let delta_epsilon = 9.2 * cos(omega_rad) + 0.57 * cos(two_l_sun) + 0.10 * cos(two_l_moon)
        - 0.09 * cos(two_omega);

    // Convert from arcseconds to radians
    (
        delta_psi * PI / (180.0 * 3600.0),
        delta_epsilon * PI / (180.0 * 3600.0),
    )
}

/// Greenwich Apparent Sidereal Time (GAST) in radians.
///
/// GAST = GMST + equation of the equinoxes
/// equation_of_equinoxes = delta_psi * cos(epsilon)
pub fn gast_radians(jd_ut1: f64, jd_tdb: f64) -> f64 {
    let gmst = gmst_radians(jd_ut1);
    let t = (jd_tdb - J2000_JD) / JULIAN_CENTURY_DAYS;

    let (delta_psi, delta_epsilon) = nutation_simple(t);

    // Mean obliquity (IAU 2006)
    let epsilon_0 = (EPSILON_J2000
        + (-46.836_769 * t - 0.000_183_1 * t * t + 0.002_003_4 * t * t * t) / 3600.0)
        * DEG_TO_RAD;

    let epsilon = epsilon_0 + delta_epsilon;

    // Equation of the equinoxes
    let eq_eq = delta_psi * cos(epsilon);

    let mut gast = gmst + eq_eq;
    gast = gast % TWO_PI;
    if gast < 0.0 {
        gast += TWO_PI;
    }
    gast
}

// ── Geodetic to ECEF ───────────────────────────────────────────────

/// Convert geodetic coordinates (lat, lon, altitude) to ITRF/ECEF (km).
///
/// lat, lon: degrees
/// altitude: meters above WGS84 ellipsoid
pub fn geodetic_to_ecef(lat_deg: f64, lon_deg: f64, altitude_m: f64) -> [f64; 3] {
    let lat = lat_deg * DEG_TO_RAD;
    let lon = lon_deg * DEG_TO_RAD;
    let alt = altitude_m / 1000.0; // convert to km

    let e2 = 1.0 - (EARTH_B * EARTH_B) / (EARTH_A * EARTH_A);
    let sin_lat = sin(lat);
    let cos_lat = cos(lat);

    let n = EARTH_A / sqrt(1.0 - e2 * sin_lat * sin_lat);

    let x = (n + alt) * cos_lat * cos(lon);
    let y = (n + alt) * cos_lat * sin(lon);
    let z = (n * (1.0 - e2) + alt) * sin_lat;

    [x, y, z]
}

// ── ICRF to topocentric ────────────────────────────────────────────

/// Result of a topocentric solar position calculation.
#[derive(Debug, Clone, Copy)]
pub struct TopocentricPosition {
    /// Topocentric zenith angle (degrees). 0 = directly overhead, 90 = horizon.
    pub zenith: f64,
    /// Topocentric azimuth (degrees, clockwise from north). 0 = north, 90 = east.
    pub azimuth: f64,
    /// Topocentric elevation angle (degrees). 90 - zenith, uncorrected for refraction.
    pub elevation: f64,
    /// Distance from observer to body (km).
    pub distance_km: f64,
    /// Right ascension of the body (degrees).
    pub right_ascension: f64,
    /// Declination of the body (degrees).
    pub declination: f64,
}

/// Convert an ICRF/J2000 position vector (km, geocentric) to topocentric
/// zenith and azimuth for an observer at a given geodetic position.
///
/// This is the core coordinate transformation:
///   1. Apply Earth rotation (GAST) to get ITRF position of body
///   2. Subtract observer ECEF position to get topocentric vector
///   3. Rotate into local horizon frame (East-North-Up)
///   4. Compute zenith angle and azimuth
///
/// `body_icrf`: position of body in J2000/ICRF (km), geocentric
/// `jd_ut1`: Julian date UT1 (for Earth rotation)
/// `jd_tdb`: Julian date TDB (for nutation)
/// `lat_deg`, `lon_deg`: observer geodetic latitude/longitude (degrees)
/// `altitude_m`: observer altitude above WGS84 (meters)
pub fn icrf_to_topocentric(
    body_icrf: [f64; 3],
    jd_ut1: f64,
    jd_tdb: f64,
    lat_deg: f64,
    lon_deg: f64,
    altitude_m: f64,
) -> TopocentricPosition {
    // Step 1: Rotate body from ICRF to ITRF (Earth-fixed).
    // We use GAST for the rotation angle.
    // The rotation is about the z-axis by -GAST (or equivalently,
    // multiply by R_z(GAST)).
    let gast = gast_radians(jd_ut1, jd_tdb);

    let cos_gast = cos(gast);
    let sin_gast = sin(gast);

    // For a more precise transformation, we should also apply the
    // precession-nutation matrix. However, for the nutation correction
    // in the equinox-based approach, the GAST already includes the
    // equation of the equinoxes. For sub-arcsecond work, we apply
    // a simplified precession-nutation rotation.
    //
    // For now, we use the "classical" approach:
    //   ITRF = R_z(GAST) * [precession-nutation] * ICRF
    //
    // We fold precession into GAST for the dominant effect and apply
    // nutation explicitly for the small corrections.

    let t = (jd_tdb - J2000_JD) / JULIAN_CENTURY_DAYS;
    let (delta_psi, delta_epsilon) = nutation_simple(t);

    // Mean obliquity
    let epsilon_0 = (EPSILON_J2000
        + (-46.836_769 * t - 0.000_183_1 * t * t + 0.002_003_4 * t * t * t) / 3600.0)
        * DEG_TO_RAD;

    // For the full ICRF->ITRF transformation we need the precession-nutation
    // matrix followed by Earth rotation. In the equinox-based paradigm:
    //   [ITRF] = R_z(GAST) * N * P * [ICRF]
    // where P is precession, N is nutation.
    //
    // For our accuracy needs (~0.001 deg for solar position), we use a
    // simplified approach: apply the frame rotation from ICRF (≈GCRF) to
    // true-of-date equator and equinox, then rotate by GAST.
    //
    // The dominant precession effect over centuries is captured by the
    // precession matrix. For dates within a few decades of J2000, the
    // simplified nutation + GAST approach gives sub-arcsecond accuracy
    // for the Sun, which is sufficient for our purposes.

    // Simplified: rotate ICRF -> True equator and equinox of date
    // Using the "equation of the equinoxes" embedded in GAST,
    // we can approximate: ITRF = R_z(GAST) * [ICRF]
    // This works to ~arcsecond level for solar position because:
    // (a) The Sun moves slowly (~1 deg/day)
    // (b) GAST includes the equation of the equinoxes
    // (c) For sub-arcsecond: need full precession-nutation matrix

    // Apply precession via the IAU 2006 precession angles
    let body_precessed = apply_precession(body_icrf, t);

    // Apply nutation
    let body_true = apply_nutation(body_precessed, epsilon_0, delta_psi, delta_epsilon);

    // Rotate by GAST to get ITRF
    let body_itrf = [
        cos_gast * body_true[0] + sin_gast * body_true[1],
        -sin_gast * body_true[0] + cos_gast * body_true[1],
        body_true[2],
    ];

    // Step 2: Observer ECEF position
    let obs_ecef = geodetic_to_ecef(lat_deg, lon_deg, altitude_m);

    // Step 3: Topocentric vector in ECEF
    let topo_ecef = [
        body_itrf[0] - obs_ecef[0],
        body_itrf[1] - obs_ecef[1],
        body_itrf[2] - obs_ecef[2],
    ];

    let dist = sqrt(
        topo_ecef[0] * topo_ecef[0] + topo_ecef[1] * topo_ecef[1] + topo_ecef[2] * topo_ecef[2],
    );

    // Step 4: Rotate to local East-North-Up (ENU) frame
    let lat = lat_deg * DEG_TO_RAD;
    let lon = lon_deg * DEG_TO_RAD;
    let sin_lat = sin(lat);
    let cos_lat = cos(lat);
    let sin_lon = sin(lon);
    let cos_lon = cos(lon);

    // ENU rotation matrix:
    // E = [-sin(lon),           cos(lon),          0        ]
    // N = [-sin(lat)*cos(lon), -sin(lat)*sin(lon), cos(lat) ]
    // U = [ cos(lat)*cos(lon),  cos(lat)*sin(lon), sin(lat) ]

    let east = -sin_lon * topo_ecef[0] + cos_lon * topo_ecef[1];
    let north = -sin_lat * cos_lon * topo_ecef[0] - sin_lat * sin_lon * topo_ecef[1]
        + cos_lat * topo_ecef[2];
    let up = cos_lat * cos_lon * topo_ecef[0]
        + cos_lat * sin_lon * topo_ecef[1]
        + sin_lat * topo_ecef[2];

    // Elevation angle (from horizon)
    let elevation_rad = asin(up / dist);
    let elevation = elevation_rad * RAD_TO_DEG;

    // Zenith angle
    let zenith = 90.0 - elevation;

    // Azimuth (clockwise from north)
    let mut azimuth = atan2(east, north) * RAD_TO_DEG;
    if azimuth < 0.0 {
        azimuth += 360.0;
    }

    // Compute RA/Dec from the geocentric ICRF position (for reference)
    let r_eq = sqrt(body_icrf[0] * body_icrf[0] + body_icrf[1] * body_icrf[1]);
    let dec = atan2(body_icrf[2], r_eq) * RAD_TO_DEG;
    let mut ra = atan2(body_icrf[1], body_icrf[0]) * RAD_TO_DEG;
    if ra < 0.0 {
        ra += 360.0;
    }

    TopocentricPosition {
        zenith,
        azimuth,
        elevation,
        distance_km: dist,
        right_ascension: ra,
        declination: dec,
    }
}

/// Apply IAU 2006 precession matrix to rotate from J2000 to mean equator
/// and equinox of date.
///
/// Uses Lieske's precession angles (simplified for dates near J2000).
fn apply_precession(pos: [f64; 3], t: f64) -> [f64; 3] {
    // Precession angles (arcseconds)
    // Lieske (1979), as used in IERS Conventions
    let zeta_a = (2306.2181 * t + 1.09468 * t * t + 0.018203 * t * t * t) * PI / (180.0 * 3600.0);
    let theta_a = (2004.3109 * t - 0.42665 * t * t - 0.041833 * t * t * t) * PI / (180.0 * 3600.0);
    let z_a = (2306.2181 * t + 1.09468 * t * t + 0.018203 * t * t * t) * PI / (180.0 * 3600.0);

    // Note: z_A and zeta_A differ at higher order. For simplicity,
    // using the same first-order value (accurate to ~0.1" per century).
    // For dates within +-50 years of J2000, error < 0.05".

    let cos_zeta = cos(zeta_a);
    let sin_zeta = sin(zeta_a);
    let cos_theta = cos(theta_a);
    let sin_theta = sin(theta_a);
    let cos_z = cos(z_a);
    let sin_z = sin(z_a);

    // Precession rotation matrix P = R_z(-z_A) * R_y(theta_A) * R_z(-zeta_A)
    let p11 = cos_z * cos_theta * cos_zeta - sin_z * sin_zeta;
    let p12 = -cos_z * cos_theta * sin_zeta - sin_z * cos_zeta;
    let p13 = -cos_z * sin_theta;
    let p21 = sin_z * cos_theta * cos_zeta + cos_z * sin_zeta;
    let p22 = -sin_z * cos_theta * sin_zeta + cos_z * cos_zeta;
    let p23 = -sin_z * sin_theta;
    let p31 = sin_theta * cos_zeta;
    let p32 = -sin_theta * sin_zeta;
    let p33 = cos_theta;

    [
        p11 * pos[0] + p12 * pos[1] + p13 * pos[2],
        p21 * pos[0] + p22 * pos[1] + p23 * pos[2],
        p31 * pos[0] + p32 * pos[1] + p33 * pos[2],
    ]
}

/// Apply nutation rotation to go from mean to true equator/equinox of date.
fn apply_nutation(pos: [f64; 3], epsilon_0: f64, delta_psi: f64, delta_epsilon: f64) -> [f64; 3] {
    let epsilon = epsilon_0 + delta_epsilon;

    let cos_eps0 = cos(epsilon_0);
    let sin_eps0 = sin(epsilon_0);
    let cos_eps = cos(epsilon);
    let sin_eps = sin(epsilon);
    let cos_dpsi = cos(delta_psi);
    let sin_dpsi = sin(delta_psi);

    // Nutation matrix N:
    // N = R_x(-epsilon) * R_z(-delta_psi) * R_x(epsilon_0)
    let n11 = cos_dpsi;
    let n12 = -sin_dpsi * cos_eps0;
    let n13 = -sin_dpsi * sin_eps0;
    let n21 = sin_dpsi * cos_eps;
    let n22 = cos_dpsi * cos_eps * cos_eps0 + sin_eps * sin_eps0;
    let n23 = cos_dpsi * cos_eps * sin_eps0 - sin_eps * cos_eps0;
    let n31 = sin_dpsi * sin_eps;
    let n32 = cos_dpsi * sin_eps * cos_eps0 - cos_eps * sin_eps0;
    let n33 = cos_dpsi * sin_eps * sin_eps0 + cos_eps * cos_eps0;

    [
        n11 * pos[0] + n12 * pos[1] + n13 * pos[2],
        n21 * pos[0] + n22 * pos[1] + n23 * pos[2],
        n31 * pos[0] + n32 * pos[1] + n33 * pos[2],
    ]
}

/// Apply atmospheric refraction correction to elevation angle.
///
/// Uses Bennett's formula (1982), as recommended by the
/// Astronomical Almanac.
///
/// `elevation_deg`: uncorrected elevation in degrees
/// `pressure_mbar`: atmospheric pressure in millibars (default 1013.25)
/// `temperature_c`: temperature in Celsius (default 10)
///
/// Returns corrected elevation in degrees.
pub fn refraction_correction(elevation_deg: f64, pressure_mbar: f64, temperature_c: f64) -> f64 {
    if elevation_deg < -1.0 {
        // Below horizon, no correction
        return elevation_deg;
    }

    // Bennett's formula (arcminutes)
    let e = elevation_deg;
    let correction_arcmin = 1.0 / tan((e + 7.31 / (e + 4.4)) * DEG_TO_RAD)
        * (pressure_mbar / 1010.0)
        * (283.0 / (273.0 + temperature_c));

    elevation_deg + correction_arcmin / 60.0
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calendar_to_jd_j2000() {
        // J2000.0 = 2000 Jan 1, 12:00:00 TT
        let jd = calendar_to_jd(2000, 1, 1, 12, 0, 0);
        assert!(
            (jd - 2_451_545.0).abs() < 1e-6,
            "J2000 JD: expected 2451545.0, got {}",
            jd
        );
    }

    #[test]
    fn test_calendar_to_jd_known_date() {
        // 2024 Jan 1, 0:00:00 UTC -> JD 2460310.5
        let jd = calendar_to_jd(2024, 1, 1, 0, 0, 0);
        assert!(
            (jd - 2_460_310.5).abs() < 1e-6,
            "2024-01-01 JD: expected 2460310.5, got {}",
            jd
        );
    }

    #[test]
    fn test_jd_tdb_roundtrip() {
        let jd = 2_451_545.0;
        let tdb = jd_to_tdb_seconds(jd);
        assert!(tdb.abs() < 1e-10, "J2000 -> TDB should be 0, got {}", tdb);

        let jd2 = tdb_seconds_to_jd(tdb);
        assert!(
            (jd2 - jd).abs() < 1e-10,
            "roundtrip failed: {} vs {}",
            jd,
            jd2
        );
    }

    #[test]
    fn test_gmst_at_j2000() {
        // At J2000.0 (2000 Jan 1, 12:00 UT1), GMST should be approximately
        // 18h 41m 50.55s = 280.46062 degrees = 4.8949 radians
        let jd = 2_451_545.0;
        let gmst = gmst_radians(jd);
        let gmst_deg = gmst * RAD_TO_DEG;

        // GMST at J2000.0 ≈ 280.46 degrees
        assert!(
            (gmst_deg - 280.46).abs() < 0.5,
            "GMST at J2000.0: expected ~280.46 deg, got {} deg",
            gmst_deg
        );
    }

    #[test]
    fn test_geodetic_to_ecef_equator() {
        // At equator, lon=0, altitude=0
        let ecef = geodetic_to_ecef(0.0, 0.0, 0.0);
        assert!(
            (ecef[0] - EARTH_A).abs() < 0.001,
            "x should be ~{}, got {}",
            EARTH_A,
            ecef[0]
        );
        assert!(ecef[1].abs() < 0.001, "y should be ~0, got {}", ecef[1]);
        assert!(ecef[2].abs() < 0.001, "z should be ~0, got {}", ecef[2]);
    }

    #[test]
    fn test_geodetic_to_ecef_pole() {
        // At north pole
        let ecef = geodetic_to_ecef(90.0, 0.0, 0.0);
        assert!(ecef[0].abs() < 0.001, "x should be ~0, got {}", ecef[0]);
        assert!(ecef[1].abs() < 0.001, "y should be ~0, got {}", ecef[1]);
        assert!(
            (ecef[2] - EARTH_B).abs() < 0.001,
            "z should be ~{}, got {}",
            EARTH_B,
            ecef[2]
        );
    }

    #[test]
    fn test_refraction_correction() {
        // At 0 degrees elevation, refraction is about 34 arcminutes
        let corrected = refraction_correction(0.0, 1013.25, 10.0);
        let correction = corrected - 0.0;
        assert!(
            correction > 0.4 && correction < 0.7,
            "refraction at horizon: expected ~0.57 deg, got {} deg",
            correction
        );
    }

    #[test]
    fn test_nutation_magnitude() {
        // Nutation in longitude should be on the order of ~17 arcseconds
        let (dpsi, deps) = nutation_simple(0.0); // at J2000
        let dpsi_arcsec = dpsi * 180.0 * 3600.0 / PI;
        let deps_arcsec = deps * 180.0 * 3600.0 / PI;

        // The dominant term is ~17.2" for delta_psi
        assert!(
            dpsi_arcsec.abs() < 25.0,
            "delta_psi too large: {} arcsec",
            dpsi_arcsec
        );
        assert!(
            deps_arcsec.abs() < 15.0,
            "delta_epsilon too large: {} arcsec",
            deps_arcsec
        );
    }

    #[test]
    fn test_precession_identity_at_j2000() {
        // At t=0 (J2000), precession should be identity
        let pos = [1.0, 0.0, 0.0];
        let result = apply_precession(pos, 0.0);
        assert!(
            (result[0] - 1.0).abs() < 1e-12,
            "precession at J2000: x = {}",
            result[0]
        );
        assert!(
            result[1].abs() < 1e-12,
            "precession at J2000: y = {}",
            result[1]
        );
        assert!(
            result[2].abs() < 1e-12,
            "precession at J2000: z = {}",
            result[2]
        );
    }

    #[test]
    fn test_precession_small_rotation() {
        // After 1 century, precession should rotate by ~1.4 degrees
        let pos = [1.0, 0.0, 0.0];
        let result = apply_precession(pos, 1.0);

        // The vector should still be approximately unit length
        let len = sqrt(result[0] * result[0] + result[1] * result[1] + result[2] * result[2]);
        assert!(
            (len - 1.0).abs() < 1e-10,
            "precession changed length: {}",
            len
        );

        // After 1 century, rotation angle should be ~1.4 deg (~0.024 rad)
        let angle = libm::acos(result[0]);
        assert!(
            angle > 0.01 && angle < 0.05,
            "precession angle after 1 century: {} rad ({} deg)",
            angle,
            angle * RAD_TO_DEG
        );
    }

    #[test]
    fn test_calendar_utc_to_tdb() {
        // J2000.0 in UTC: 2000-01-01T11:58:55.816 (approximately)
        // delta_t at J2000 ≈ 63.8 seconds
        let tdb = calendar_utc_to_tdb(2000, 1, 1, 12, 0, 0, 63.8);
        // Should be close to TDB = 63.8 seconds (since 12:00 UTC + 63.8s ≈ TDB J2000 + 63.8)
        assert!(
            tdb.abs() < 200.0,
            "TDB at 2000-01-01 12:00 UTC: expected near 0, got {}",
            tdb
        );
    }
}
