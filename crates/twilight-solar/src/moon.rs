//! Low-precision lunar ephemeris (Meeus, Astronomical Algorithms ch. 47,
//! truncated series).
//!
//! Accuracy ~0.3 deg in position — far more than sufficient for sky-
//! brightness modeling (the Krisciunas-Schaefer moonlight model's own
//! scatter is ~20%). Provides geocentric ecliptic coordinates, topocentric
//! altitude/azimuth, phase angle and illuminated fraction.

use libm::{atan2, asin, cos, sin};

const DEG: f64 = core::f64::consts::PI / 180.0;

/// Topocentric lunar state for sky-brightness work.
#[derive(Debug, Clone, Copy)]
pub struct MoonState {
    /// Altitude above the horizon [deg] (no refraction; fine for brightness).
    pub altitude_deg: f64,
    /// Azimuth [deg, 0 = north, clockwise].
    pub azimuth_deg: f64,
    /// Phase angle alpha [deg]: 0 = full moon, 180 = new moon.
    pub phase_angle_deg: f64,
    /// Illuminated fraction of the disk [0..1].
    pub illuminated_fraction: f64,
    /// Distance to the Moon [km]. Geocentric in this Meeus path;
    /// topocentric (observer-to-moon) when produced by the DE440
    /// ephemeris — the difference is bounded by Earth's radius (<1.8%)
    /// and the topocentric value is the physically correct one for
    /// received moonlight.
    pub distance_km: f64,
}

/// Julian day from civil UTC date/time (Gregorian).
fn julian_day(year: i32, month: i32, day: i32, hour_utc: f64) -> f64 {
    let (y, m) = if month <= 2 {
        (year - 1, month + 12)
    } else {
        (year, month)
    };
    let a = (y as f64 / 100.0).floor();
    let b = 2.0 - a + (a / 4.0).floor();
    (365.25 * (y as f64 + 4716.0)).floor() + (30.6001 * (m as f64 + 1.0)).floor()
        + day as f64
        + hour_utc / 24.0
        + b
        - 1524.5
}

/// Geocentric ecliptic longitude/latitude [deg] and distance [km] of the
/// Moon (Meeus ch. 47, principal terms).
fn moon_ecliptic(jd: f64) -> (f64, f64, f64) {
    let t = (jd - 2451545.0) / 36525.0;

    // Mean elements (deg)
    let lp = 218.3164477 + 481267.88123421 * t - 0.0015786 * t * t;
    let d = 297.8501921 + 445267.1114034 * t - 0.0018819 * t * t;
    let m = 357.5291092 + 35999.0502909 * t - 0.0001536 * t * t;
    let mp = 134.9633964 + 477198.8675055 * t + 0.0087414 * t * t;
    let f = 93.2720950 + 483202.0175233 * t - 0.0036539 * t * t;

    let (d, m, mp, f) = (d * DEG, m * DEG, mp * DEG, f * DEG);

    // Principal longitude terms (units 1e-6 deg) — Meeus Table 47.A truncated
    let sum_l = 6288774.0 * sin(mp)
        + 1274027.0 * sin(2.0 * d - mp)
        + 658314.0 * sin(2.0 * d)
        + 213618.0 * sin(2.0 * mp)
        - 185116.0 * sin(m)
        - 114332.0 * sin(2.0 * f)
        + 58793.0 * sin(2.0 * d - 2.0 * mp)
        + 57066.0 * sin(2.0 * d - m - mp)
        + 53322.0 * sin(2.0 * d + mp)
        + 45758.0 * sin(2.0 * d - m)
        - 40923.0 * sin(m - mp)
        - 34720.0 * sin(d)
        - 30383.0 * sin(m + mp);

    // Principal latitude terms (1e-6 deg) — Table 47.B truncated
    let sum_b = 5128122.0 * sin(f)
        + 280602.0 * sin(mp + f)
        + 277693.0 * sin(mp - f)
        + 173237.0 * sin(2.0 * d - f)
        + 55413.0 * sin(2.0 * d - mp + f)
        + 46271.0 * sin(2.0 * d - mp - f)
        + 32573.0 * sin(2.0 * d + f)
        + 17198.0 * sin(2.0 * mp + f);

    // Distance terms (1e-3 km) — Table 47.A
    let sum_r = -20905355.0 * cos(mp)
        - 3699111.0 * cos(2.0 * d - mp)
        - 2955968.0 * cos(2.0 * d)
        - 569925.0 * cos(2.0 * mp)
        + 48888.0 * cos(m)
        - 3149.0 * cos(2.0 * f);

    let lon = lp + sum_l / 1e6;
    let lat = sum_b / 1e6;
    let dist = 385000.56 + sum_r / 1e3;
    (lon.rem_euclid(360.0), lat, dist)
}

/// Sun geocentric ecliptic longitude [deg] (low precision, Meeus ch. 25).
fn sun_ecliptic_lon(jd: f64) -> f64 {
    let t = (jd - 2451545.0) / 36525.0;
    let l0 = 280.46646 + 36000.76983 * t;
    let m = (357.52911 + 35999.05029 * t) * DEG;
    let c = (1.914602 - 0.004817 * t) * sin(m) + 0.019993 * sin(2.0 * m);
    (l0 + c).rem_euclid(360.0)
}

/// Greenwich mean sidereal time [deg].
fn gmst_deg(jd: f64) -> f64 {
    let t = (jd - 2451545.0) / 36525.0;
    (280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * t * t)
        .rem_euclid(360.0)
}

/// Topocentric (approx: geocentric direction + parallax in altitude)
/// lunar state at the given instant and observer.
pub fn moon_state(
    year: i32,
    month: i32,
    day: i32,
    hour_utc: f64,
    lat_deg: f64,
    lon_deg: f64,
) -> MoonState {
    let jd = julian_day(year, month, day, hour_utc);
    let (elon, elat, dist) = moon_ecliptic(jd);
    let eps = 23.4393 * DEG; // mean obliquity (sufficient)

    let (el, eb) = (elon * DEG, elat * DEG);
    // ecliptic -> equatorial
    let ra = atan2(sin(el) * cos(eps) - libm::tan(eb) * sin(eps), cos(el));
    let dec = asin(sin(eb) * cos(eps) + cos(eb) * sin(eps) * sin(el));

    // local hour angle
    let lst = (gmst_deg(jd) + lon_deg) * DEG;
    let ha = lst - ra;

    let phi = lat_deg * DEG;
    let sin_alt = sin(phi) * sin(dec) + cos(phi) * cos(dec) * cos(ha);
    let alt_geo = asin(sin_alt.clamp(-1.0, 1.0));
    // parallax correction (moon is close): horizontal parallax ~ asin(6378/dist)
    let hp = asin(6378.14 / dist);
    let alt = alt_geo - hp * cos(alt_geo);

    let az = atan2(
        sin(ha),
        cos(ha) * sin(phi) - libm::tan(dec) * cos(phi),
    );
    let az_deg = (az / DEG + 180.0).rem_euclid(360.0);

    // Phase angle from sun-moon elongation (sufficient: alpha ~ 180 - elongation)
    let sun_lon = sun_ecliptic_lon(jd) * DEG;
    let cos_elong = cos(eb) * cos(el - sun_lon);
    let elong = libm::acos(cos_elong.clamp(-1.0, 1.0));
    let alpha = core::f64::consts::PI - elong;
    let illum = (1.0 + cos(alpha)) / 2.0;

    MoonState {
        altitude_deg: alt / DEG,
        azimuth_deg: az_deg,
        phase_angle_deg: alpha / DEG,
        illuminated_fraction: illum,
        distance_km: dist,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn julian_day_epoch() {
        // J2000.0: 2000-01-01 12:00 UTC = JD 2451545.0
        assert!((julian_day(2000, 1, 1, 12.0) - 2451545.0).abs() < 1e-9);
    }

    #[test]
    fn moon_distance_in_physical_range() {
        // perigee ~356500, apogee ~406700 km
        for (y, m, d) in [(2026, 1, 1), (2026, 6, 12), (2024, 3, 20)] {
            let (_, _, dist) = moon_ecliptic(julian_day(y, m, d, 0.0));
            assert!(
                (350_000.0..415_000.0).contains(&dist),
                "{y}-{m}-{d}: dist = {dist}"
            );
        }
    }

    #[test]
    fn known_full_moon_is_full() {
        // 2026-01-03 is a full moon (within ~a day): alpha small, illum ~1
        let st = moon_state(2026, 1, 3, 12.0, 0.0, 0.0);
        assert!(
            st.illuminated_fraction > 0.95,
            "illum = {}",
            st.illuminated_fraction
        );
    }

    #[test]
    fn known_new_moon_is_dark() {
        // 2026-01-18/19 new moon: illum ~0
        let st = moon_state(2026, 1, 18, 12.0, 0.0, 0.0);
        assert!(
            st.illuminated_fraction < 0.08,
            "illum = {}",
            st.illuminated_fraction
        );
    }

    #[test]
    fn altitude_and_azimuth_in_range() {
        let st = moon_state(2026, 6, 12, 22.0, 55.65, 12.41);
        assert!((-90.0..=90.0).contains(&st.altitude_deg));
        assert!((0.0..360.0).contains(&st.azimuth_deg));
    }
}
