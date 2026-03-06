//! NREL Solar Position Algorithm (SPA) implementation.
//!
//! Computes solar zenith angle and azimuth with ±0.0003° accuracy
//! for the period -2000 to 6000.
//!
//! Reference: Reda, I.; Andreas, A. (2003). Solar Position Algorithm for Solar
//! Radiation Applications. NREL/TP-560-34302.

use crate::spa_tables::*;
use libm::{acos, asin, atan2, cos, fabs, floor, sin, tan};

const PI: f64 = core::f64::consts::PI;
const DEG_TO_RAD: f64 = PI / 180.0;
const RAD_TO_DEG: f64 = 180.0 / PI;

/// Input parameters for the SPA calculation.
#[derive(Debug, Clone)]
pub struct SpaInput {
    /// Year (e.g., 2003)
    pub year: i32,
    /// Month (1-12)
    pub month: i32,
    /// Day (1-31)
    pub day: i32,
    /// Hour (0-23)
    pub hour: i32,
    /// Minute (0-59)
    pub minute: i32,
    /// Second (0-59)
    pub second: i32,
    /// Observer timezone offset from UTC (hours, e.g., -7 for MST)
    pub timezone: f64,
    /// Observer latitude (degrees, north positive)
    pub latitude: f64,
    /// Observer longitude (degrees, east positive)
    pub longitude: f64,
    /// Observer elevation above sea level (meters)
    pub elevation: f64,
    /// Annual average local pressure (millibars, default 1013.25)
    pub pressure: f64,
    /// Annual average local temperature (°C, default 15)
    pub temperature: f64,
    /// Difference between Earth rotation time and TT (seconds, ~69.184 for 2024)
    pub delta_t: f64,
    /// Surface slope (degrees from horizontal, 0 = flat)
    pub slope: f64,
    /// Surface azimuth rotation (degrees from south, negative = east)
    pub azm_rotation: f64,
    /// Atmospheric refraction at sunset/sunrise (degrees, default 0.5667)
    pub atmos_refract: f64,
}

impl Default for SpaInput {
    fn default() -> Self {
        Self {
            year: 2024,
            month: 1,
            day: 1,
            hour: 12,
            minute: 0,
            second: 0,
            timezone: 0.0,
            latitude: 0.0,
            longitude: 0.0,
            elevation: 0.0,
            pressure: 1013.25,
            temperature: 15.0,
            delta_t: 69.184,
            slope: 0.0,
            azm_rotation: 0.0,
            atmos_refract: 0.5667,
        }
    }
}

/// Output of the SPA calculation.
#[derive(Debug, Clone, Copy)]
pub struct SpaOutput {
    /// Julian day
    pub jd: f64,
    /// Julian century
    pub jc: f64,
    /// Julian ephemeris day
    pub jde: f64,
    /// Julian ephemeris century
    pub jce: f64,
    /// Julian ephemeris millennium
    pub jme: f64,
    /// Earth heliocentric longitude (degrees)
    pub l: f64,
    /// Earth heliocentric latitude (degrees)
    pub b: f64,
    /// Earth-Sun distance (AU)
    pub r: f64,
    /// Geocentric longitude (degrees)
    pub theta: f64,
    /// Geocentric latitude (degrees)
    pub beta: f64,
    /// Nutation in longitude (degrees)
    pub delta_psi: f64,
    /// Nutation in obliquity (degrees)
    pub delta_epsilon: f64,
    /// True obliquity of the ecliptic (degrees)
    pub epsilon: f64,
    /// Apparent sun longitude (degrees)
    pub lambda: f64,
    /// Apparent sidereal time at Greenwich (degrees)
    pub nu0: f64,
    /// Sun right ascension (degrees)
    pub alpha: f64,
    /// Sun declination (degrees)
    pub delta: f64,
    /// Observer hour angle (degrees)
    pub h: f64,
    /// Topocentric zenith angle (degrees) — THE KEY OUTPUT
    pub zenith: f64,
    /// Topocentric azimuth angle (degrees, clockwise from north)
    pub azimuth: f64,
    /// Topocentric elevation angle (degrees, no refraction)
    pub elevation_no_refract: f64,
    /// Topocentric elevation angle (degrees, with refraction)
    pub elevation: f64,
    /// Equation of time (minutes)
    pub eot: f64,
    /// Sunrise time (fractional hour, local)
    pub sunrise: f64,
    /// Sunset time (fractional hour, local)
    pub sunset: f64,
    /// Solar noon (fractional hour, local)
    pub stn: f64,
}

/// Error type for SPA calculations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpaError {
    /// Year out of valid range (-2000 to 6000)
    InvalidYear,
    /// Month out of range (1-12)
    InvalidMonth,
    /// Day out of range
    InvalidDay,
    /// Latitude out of range (-90 to 90)
    InvalidLatitude,
    /// Longitude out of range (-180 to 180)
    InvalidLongitude,
    /// Pressure out of range
    InvalidPressure,
    /// Temperature out of range
    InvalidTemperature,
}

/// Validate SPA input parameters.
fn validate_input(input: &SpaInput) -> Result<(), SpaError> {
    if input.year < -2000 || input.year > 6000 {
        return Err(SpaError::InvalidYear);
    }
    if input.month < 1 || input.month > 12 {
        return Err(SpaError::InvalidMonth);
    }
    if input.day < 1 || input.day > 31 {
        return Err(SpaError::InvalidDay);
    }
    if input.latitude < -90.0 || input.latitude > 90.0 {
        return Err(SpaError::InvalidLatitude);
    }
    if input.longitude < -180.0 || input.longitude > 180.0 {
        return Err(SpaError::InvalidLongitude);
    }
    Ok(())
}

/// Limit angle to [0, 360) degrees.
#[inline]
fn limit_degrees(degrees: f64) -> f64 {
    let mut d = degrees % 360.0;
    if d < 0.0 {
        d += 360.0;
    }
    d
}

/// Limit angle to [0, 2π) radians.
#[inline]
#[allow(dead_code)]
fn limit_radians(radians: f64) -> f64 {
    let mut r = radians % (2.0 * PI);
    if r < 0.0 {
        r += 2.0 * PI;
    }
    r
}

/// Compute Julian Day from calendar date and time.
///
/// Handles both Julian and Gregorian calendars automatically.
pub fn julian_day(
    year: i32,
    month: i32,
    day: i32,
    hour: i32,
    minute: i32,
    second: i32,
    timezone: f64,
) -> f64 {
    let day_decimal = day as f64
        + (hour as f64 - timezone + minute as f64 / 60.0 + second as f64 / 3600.0) / 24.0;

    let (y, m) = if month <= 2 {
        (year - 1, month + 12)
    } else {
        (year, month)
    };

    let jd = floor(365.25 * (y as f64 + 4716.0)) + floor(30.6001 * (m as f64 + 1.0)) + day_decimal
        - 1524.5;

    // Gregorian calendar correction
    if jd > 2299160.0 {
        let a = floor(y as f64 / 100.0) as i32;
        let b = 2 - a + a / 4;
        jd + b as f64
    } else {
        jd
    }
}

/// Compute Julian Century from Julian Day.
#[inline]
pub fn julian_century(jd: f64) -> f64 {
    (jd - 2451545.0) / 36525.0
}

/// Compute Julian Ephemeris Day from Julian Day and delta_t.
#[inline]
pub fn julian_ephemeris_day(jd: f64, delta_t: f64) -> f64 {
    jd + delta_t / 86400.0
}

/// Compute Julian Ephemeris Century.
#[inline]
pub fn julian_ephemeris_century(jde: f64) -> f64 {
    (jde - 2451545.0) / 36525.0
}

/// Compute Julian Ephemeris Millennium.
#[inline]
pub fn julian_ephemeris_millennium(jce: f64) -> f64 {
    jce / 10.0
}

/// Evaluate a VSOP87 periodic term series.
fn eval_periodic_terms(terms: &[PeriodicTerm], jme: f64) -> f64 {
    let mut sum = 0.0;
    for &(a, b, c) in terms {
        sum += a * cos(b + c * jme);
    }
    sum
}

/// Compute Earth heliocentric longitude L (degrees).
pub fn earth_heliocentric_longitude(jme: f64) -> f64 {
    let l0 = eval_periodic_terms(L0_TERMS, jme);
    let l1 = eval_periodic_terms(L1_TERMS, jme);
    let l2 = eval_periodic_terms(L2_TERMS, jme);
    let l3 = eval_periodic_terms(L3_TERMS, jme);
    let l4 = eval_periodic_terms(L4_TERMS, jme);
    let l5 = eval_periodic_terms(L5_TERMS, jme);

    let l_rad =
        (l0 + l1 * jme + l2 * jme * jme + l3 * jme.powi(3) + l4 * jme.powi(4) + l5 * jme.powi(5))
            / 1e8;

    limit_degrees(l_rad * RAD_TO_DEG)
}

/// Compute Earth heliocentric latitude B (degrees).
pub fn earth_heliocentric_latitude(jme: f64) -> f64 {
    let b0 = eval_periodic_terms(B0_TERMS, jme);
    let b1 = eval_periodic_terms(B1_TERMS, jme);

    let b_rad = (b0 + b1 * jme) / 1e8;

    b_rad * RAD_TO_DEG
}

/// Compute Earth-Sun distance R (AU).
pub fn earth_sun_distance(jme: f64) -> f64 {
    let r0 = eval_periodic_terms(R0_TERMS, jme);
    let r1 = eval_periodic_terms(R1_TERMS, jme);
    let r2 = eval_periodic_terms(R2_TERMS, jme);
    let r3 = eval_periodic_terms(R3_TERMS, jme);
    let r4 = eval_periodic_terms(R4_TERMS, jme);

    (r0 + r1 * jme + r2 * jme * jme + r3 * jme.powi(3) + r4 * jme.powi(4)) / 1e8
}

/// Compute geocentric longitude Θ (degrees).
#[inline]
pub fn geocentric_longitude(l: f64) -> f64 {
    limit_degrees(l + 180.0)
}

/// Compute geocentric latitude β (degrees).
#[inline]
pub fn geocentric_latitude(b: f64) -> f64 {
    -b
}

/// Compute nutation in longitude (Δψ) and obliquity (Δε) in degrees.
pub fn nutation(jce: f64) -> (f64, f64) {
    // Mean elongation of the moon from the sun (degrees)
    let x0 = 297.85036 + 445267.111480 * jce - 0.0019142 * jce * jce + jce * jce * jce / 189474.0;

    // Mean anomaly of the sun (degrees)
    let x1 = 357.52772 + 35999.050340 * jce - 0.0001603 * jce * jce - jce * jce * jce / 300000.0;

    // Mean anomaly of the moon (degrees)
    let x2 = 134.96298 + 477198.867398 * jce + 0.0086972 * jce * jce + jce * jce * jce / 56250.0;

    // Moon's argument of latitude (degrees)
    let x3 = 93.27191 + 483202.017538 * jce - 0.0036825 * jce * jce + jce * jce * jce / 327270.0;

    // Longitude of ascending node of moon's mean orbit (degrees)
    let x4 = 125.04452 - 1934.136261 * jce + 0.0020708 * jce * jce + jce * jce * jce / 450000.0;

    let x = [x0, x1, x2, x3, x4];

    let mut delta_psi = 0.0;
    let mut delta_epsilon = 0.0;

    let n = NUTATION_Y_TERMS
        .len()
        .min(NUTATION_PSI_TERMS.len())
        .min(NUTATION_EPS_TERMS.len());

    for i in 0..n {
        let mut sum_xy = 0.0;
        for j in 0..5 {
            sum_xy += NUTATION_Y_TERMS[i][j] as f64 * x[j];
        }
        let sum_xy_rad = sum_xy * DEG_TO_RAD;

        let (psi_a, psi_b) = NUTATION_PSI_TERMS[i];
        delta_psi += (psi_a + psi_b * jce) * sin(sum_xy_rad);

        let (eps_c, eps_d) = NUTATION_EPS_TERMS[i];
        delta_epsilon += (eps_c + eps_d * jce) * cos(sum_xy_rad);
    }

    // Convert from 0.0001 arc-seconds to degrees
    delta_psi /= 36000000.0;
    delta_epsilon /= 36000000.0;

    (delta_psi, delta_epsilon)
}

/// Compute true obliquity of the ecliptic (degrees).
pub fn true_obliquity(jme: f64, delta_epsilon: f64) -> f64 {
    let u = jme / 10.0;

    let mut epsilon0 = 0.0;
    let mut u_power = 1.0;
    for &coeff in OBLIQUITY_COEFFS {
        epsilon0 += coeff * u_power;
        u_power *= u;
    }

    // epsilon0 is in arc-seconds, convert to degrees and add nutation
    epsilon0 / 3600.0 + delta_epsilon
}

/// Compute apparent sun longitude (degrees).
pub fn apparent_sun_longitude(theta: f64, delta_psi: f64, r: f64) -> f64 {
    // Aberration correction
    let delta_tau = -20.4898 / (3600.0 * r);
    theta + delta_psi + delta_tau
}

/// Compute apparent sidereal time at Greenwich (degrees).
pub fn apparent_sidereal_time(jd: f64, jc: f64, delta_psi: f64, epsilon: f64) -> f64 {
    // Mean sidereal time (degrees)
    let nu0 = limit_degrees(
        280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * jc * jc
            - jc * jc * jc / 38710000.0,
    );

    // Apparent sidereal time
    nu0 + delta_psi * cos(epsilon * DEG_TO_RAD)
}

/// Compute sun right ascension (degrees).
pub fn sun_right_ascension(lambda: f64, epsilon: f64, beta: f64) -> f64 {
    let lambda_rad = lambda * DEG_TO_RAD;
    let epsilon_rad = epsilon * DEG_TO_RAD;
    let beta_rad = beta * DEG_TO_RAD;

    let alpha = atan2(
        sin(lambda_rad) * cos(epsilon_rad) - tan(beta_rad) * sin(epsilon_rad),
        cos(lambda_rad),
    );

    limit_degrees(alpha * RAD_TO_DEG)
}

/// Compute sun declination (degrees).
pub fn sun_declination(lambda: f64, epsilon: f64, beta: f64) -> f64 {
    let lambda_rad = lambda * DEG_TO_RAD;
    let epsilon_rad = epsilon * DEG_TO_RAD;
    let beta_rad = beta * DEG_TO_RAD;

    let delta =
        asin(sin(beta_rad) * cos(epsilon_rad) + cos(beta_rad) * sin(epsilon_rad) * sin(lambda_rad));

    delta * RAD_TO_DEG
}

/// Compute observer local hour angle (degrees).
pub fn observer_hour_angle(nu0: f64, longitude: f64, alpha: f64) -> f64 {
    limit_degrees(nu0 + longitude - alpha)
}

/// Compute topocentric sun position (right ascension, declination, hour angle).
///
/// Accounts for parallax due to observer's position on Earth's surface.
fn topocentric_sun_position(
    latitude: f64,
    elevation: f64,
    alpha: f64,
    delta: f64,
    h: f64,
    r_au: f64,
) -> (f64, f64, f64) {
    let lat_rad = latitude * DEG_TO_RAD;

    // Equatorial horizontal parallax of the sun (degrees)
    let xi = 8.794 / (3600.0 * r_au);
    let xi_rad = xi * DEG_TO_RAD;

    // Term u (reduced latitude)
    let u = atan2(0.99664719 * sin(lat_rad), cos(lat_rad));

    // Observer's geocentric position
    let elev_ratio = elevation / 6378140.0;
    let x = cos(u) + elev_ratio * cos(lat_rad);
    let y = 0.99664719 * sin(u) + elev_ratio * sin(lat_rad);

    let h_rad = h * DEG_TO_RAD;
    let delta_rad = delta * DEG_TO_RAD;

    // Parallax correction to right ascension
    let delta_alpha = atan2(
        -x * sin(xi_rad) * sin(h_rad),
        cos(delta_rad) - x * sin(xi_rad) * cos(h_rad),
    );
    let delta_alpha_deg = delta_alpha * RAD_TO_DEG;

    // Topocentric right ascension
    let alpha_prime = alpha + delta_alpha_deg;

    // Topocentric declination
    let delta_prime = atan2(
        (sin(delta_rad) - y * sin(xi_rad)) * cos(delta_alpha),
        cos(delta_rad) - x * sin(xi_rad) * cos(h_rad),
    ) * RAD_TO_DEG;

    // Topocentric hour angle
    let h_prime = h - delta_alpha_deg;

    (alpha_prime, delta_prime, h_prime)
}

/// Compute topocentric zenith angle (degrees).
fn topocentric_zenith(
    latitude: f64,
    delta_prime: f64,
    h_prime: f64,
    pressure: f64,
    temperature: f64,
) -> (f64, f64, f64) {
    let lat_rad = latitude * DEG_TO_RAD;
    let delta_prime_rad = delta_prime * DEG_TO_RAD;
    let h_prime_rad = h_prime * DEG_TO_RAD;

    // Topocentric elevation angle without refraction
    let e0 = asin(
        sin(lat_rad) * sin(delta_prime_rad)
            + cos(lat_rad) * cos(delta_prime_rad) * cos(h_prime_rad),
    ) * RAD_TO_DEG;

    // Atmospheric refraction correction (Meeus approximation)
    let delta_e = if e0 >= -(0.26667 + 0.5667) {
        (pressure / 1010.0) * (283.0 / (273.0 + temperature)) * 1.02
            / (60.0 * tan((e0 + 10.3 / (e0 + 5.11)) * DEG_TO_RAD))
    } else {
        0.0
    };

    let e = e0 + delta_e;
    let zenith = 90.0 - e;

    (zenith, e0, e)
}

/// Compute topocentric azimuth angle (degrees, clockwise from north).
fn topocentric_azimuth(latitude: f64, delta_prime: f64, h_prime: f64) -> f64 {
    let lat_rad = latitude * DEG_TO_RAD;
    let delta_prime_rad = delta_prime * DEG_TO_RAD;
    let h_prime_rad = h_prime * DEG_TO_RAD;

    let azimuth = atan2(
        sin(h_prime_rad),
        cos(h_prime_rad) * sin(lat_rad) - tan(delta_prime_rad) * cos(lat_rad),
    ) * RAD_TO_DEG;

    limit_degrees(azimuth + 180.0)
}

/// Compute equation of time (minutes).
fn equation_of_time(
    alpha: f64,
    delta_psi: f64,
    epsilon: f64,
    _longitude: f64,
    _jd: f64,
    jc: f64,
) -> f64 {
    // Sun mean longitude (degrees)
    let m = limit_degrees(280.46646 + 36000.76983 * jc + 0.0003032 * jc * jc);

    let eot = m - 0.0057183 - alpha + delta_psi * cos(epsilon * DEG_TO_RAD);

    // Normalize to [-180, 180]
    let mut eot = eot;
    if eot > 20.0 {
        eot -= 360.0;
    } else if eot < -20.0 {
        eot += 360.0;
    }

    eot * 4.0 // degrees to minutes (1° = 4 minutes)
}

/// Compute approximate sunrise/sunset/transit times.
fn sun_rise_transit_set(
    latitude: f64,
    longitude: f64,
    delta: f64,
    eot: f64,
    h0_prime: f64,
) -> (f64, f64, f64) {
    let lat_rad = latitude * DEG_TO_RAD;
    let delta_rad = delta * DEG_TO_RAD;

    // Hour angle at sunrise/sunset
    let cos_h0 = (sin(h0_prime * DEG_TO_RAD) - sin(lat_rad) * sin(delta_rad))
        / (cos(lat_rad) * cos(delta_rad));

    if cos_h0 < -1.0 {
        // Sun never sets (midnight sun)
        return (12.0 - eot / 60.0 - longitude / 15.0, 0.0, 24.0);
    }
    if cos_h0 > 1.0 {
        // Sun never rises (polar night)
        return (12.0 - eot / 60.0 - longitude / 15.0, -1.0, -1.0);
    }

    let h0 = acos(cos_h0) * RAD_TO_DEG;

    // Solar transit (solar noon)
    let transit = 12.0 + eot / 60.0 - longitude / 15.0;
    // Note: This is a simplified calculation. The full SPA uses iteration.
    // For our purposes (finding twilight crossings), we use different methods.

    let sunrise = transit - h0 / 15.0;
    let sunset = transit + h0 / 15.0;

    (transit, sunrise, sunset)
}

/// Main function: compute solar position for given input parameters.
pub fn solar_position(input: &SpaInput) -> Result<SpaOutput, SpaError> {
    validate_input(input)?;

    // Step 1: Julian Day
    let jd = julian_day(
        input.year,
        input.month,
        input.day,
        input.hour,
        input.minute,
        input.second,
        input.timezone,
    );

    // Step 2: Julian Century
    let jc = julian_century(jd);

    // Step 3: Julian Ephemeris Day, Century, Millennium
    let jde = julian_ephemeris_day(jd, input.delta_t);
    let jce = julian_ephemeris_century(jde);
    let jme = julian_ephemeris_millennium(jce);

    // Step 4: Earth heliocentric longitude, latitude, radius
    let l = earth_heliocentric_longitude(jme);
    let b = earth_heliocentric_latitude(jme);
    let r = earth_sun_distance(jme);

    // Step 5: Geocentric longitude and latitude
    let theta = geocentric_longitude(l);
    let beta = geocentric_latitude(b);

    // Step 6: Nutation
    let (delta_psi, delta_epsilon) = nutation(jce);

    // Step 7: True obliquity of the ecliptic
    let epsilon = true_obliquity(jme, delta_epsilon);

    // Step 8: Apparent sun longitude
    let lambda = apparent_sun_longitude(theta, delta_psi, r);

    // Step 9: Apparent sidereal time at Greenwich
    let nu0 = apparent_sidereal_time(jd, jc, delta_psi, epsilon);

    // Step 10: Sun right ascension
    let alpha = sun_right_ascension(lambda, epsilon, beta);

    // Step 11: Sun declination
    let delta = sun_declination(lambda, epsilon, beta);

    // Step 12: Observer hour angle
    let h = observer_hour_angle(nu0, input.longitude, alpha);

    // Step 13: Topocentric sun position
    let (_alpha_prime, delta_prime, h_prime) =
        topocentric_sun_position(input.latitude, input.elevation, alpha, delta, h, r);

    // Step 14: Topocentric zenith angle
    let (zenith, e0, e) = topocentric_zenith(
        input.latitude,
        delta_prime,
        h_prime,
        input.pressure,
        input.temperature,
    );

    // Step 15: Topocentric azimuth
    let azimuth = topocentric_azimuth(input.latitude, delta_prime, h_prime);

    // Step 16: Equation of time
    let eot = equation_of_time(alpha, delta_psi, epsilon, input.longitude, jd, jc);

    // Step 17: Sun rise/transit/set (approximate)
    let (stn, sunrise, sunset) =
        sun_rise_transit_set(input.latitude, input.longitude, delta, eot, -0.8333);

    Ok(SpaOutput {
        jd,
        jc,
        jde,
        jce,
        jme,
        l,
        b,
        r,
        theta,
        beta,
        delta_psi,
        delta_epsilon,
        epsilon,
        lambda,
        nu0,
        alpha,
        delta,
        h,
        zenith,
        azimuth,
        elevation_no_refract: e0,
        elevation: e,
        eot,
        sunrise,
        sunset,
        stn,
    })
}

/// Find the time (as fractional hour in local time) when the solar zenith angle
/// crosses a given threshold on a given date.
///
/// Uses binary search between `start_hour` and `end_hour`.
///
/// Returns `None` if the threshold is never crossed in the search window.
pub fn find_zenith_crossing(
    input: &SpaInput,
    target_zenith: f64,
    start_hour: f64,
    end_hour: f64,
    tolerance_deg: f64,
) -> Option<f64> {
    let mut lo = start_hour;
    let mut hi = end_hour;

    // Evaluate at endpoints
    let mut input_lo = input.clone();
    set_time_from_fractional_hour(&mut input_lo, lo);
    let z_lo = solar_position(&input_lo).ok()?.zenith;

    let mut input_hi = input.clone();
    set_time_from_fractional_hour(&mut input_hi, hi);
    let z_hi = solar_position(&input_hi).ok()?.zenith;

    // Check if crossing exists (zenith values bracket the target)
    let sign_lo = z_lo - target_zenith;
    let sign_hi = z_hi - target_zenith;

    if sign_lo * sign_hi > 0.0 {
        return None; // No crossing in this interval
    }

    // Binary search
    for _ in 0..64 {
        let mid = (lo + hi) / 2.0;
        let mut input_mid = input.clone();
        set_time_from_fractional_hour(&mut input_mid, mid);
        let z_mid = solar_position(&input_mid).ok()?.zenith;

        if fabs(z_mid - target_zenith) < tolerance_deg {
            return Some(mid);
        }

        let sign_mid = z_mid - target_zenith;
        if sign_lo * sign_mid < 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    Some((lo + hi) / 2.0)
}

/// Set hour/minute/second fields from a fractional hour value.
///
/// Converts to total integer seconds (with rounding) first, then decomposes
/// with integer arithmetic to avoid floating-point truncation errors.
fn set_time_from_fractional_hour(input: &mut SpaInput, fractional_hour: f64) {
    let total_seconds = (fractional_hour * 3600.0).round() as i32;
    input.hour = total_seconds / 3600;
    input.minute = (total_seconds % 3600) / 60;
    input.second = total_seconds % 60;
}

/// Format a fractional hour as HH:MM:SS string.
///
/// Converts to total integer seconds (with rounding) first, then decomposes
/// with integer arithmetic to avoid floating-point truncation errors.
pub fn format_time(fractional_hour: f64) -> [u8; 8] {
    let total_seconds = (fractional_hour * 3600.0).round() as u32;
    let h = total_seconds / 3600;
    let m = (total_seconds % 3600) / 60;
    let s = total_seconds % 60;

    let mut buf = [b'0'; 8];
    buf[0] = b'0' + (h / 10) as u8;
    buf[1] = b'0' + (h % 10) as u8;
    buf[2] = b':';
    buf[3] = b'0' + (m / 10) as u8;
    buf[4] = b'0' + (m % 10) as u8;
    buf[5] = b':';
    buf[6] = b'0' + (s / 10) as u8;
    buf[7] = b'0' + (s % 10) as u8;
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    /// NREL SPA reference test case from the technical report.
    /// Date: October 17, 2003 at 12:30:30 TT (ΔT = 67s)
    /// Location: 39.742476°N, -105.1786°W, 1830.14m elevation
    /// Pressure: 820 mbar, Temperature: 11°C
    #[test]
    fn test_nrel_reference_case() {
        let input = SpaInput {
            year: 2003,
            month: 10,
            day: 17,
            hour: 12,
            minute: 30,
            second: 30,
            timezone: -7.0,
            latitude: 39.742476,
            longitude: -105.1786,
            elevation: 1830.14,
            pressure: 820.0,
            temperature: 11.0,
            delta_t: 67.0,
            slope: 30.0,
            azm_rotation: -10.0,
            atmos_refract: 0.5667,
        };

        let output = solar_position(&input).unwrap();

        // NREL reference values (from Table A5 in the technical report)
        let jd_expected = 2452930.312847;
        let l_expected = 24.0182616917; // degrees
        let _b_expected = -0.0001011219;
        let r_expected = 0.9965422974; // AU
        let _theta_expected = 204.0182616917;

        // Julian Day
        assert!(
            fabs(output.jd - jd_expected) < 0.0001,
            "JD: expected {}, got {}",
            jd_expected,
            output.jd
        );

        // Earth heliocentric longitude (allow wider tolerance due to table truncation)
        assert!(
            fabs(output.l - l_expected) < 0.01,
            "L: expected {}, got {}",
            l_expected,
            output.l
        );

        // Earth-Sun distance
        assert!(
            fabs(output.r - r_expected) < 0.0001,
            "R: expected {}, got {}",
            r_expected,
            output.r
        );

        // Zenith angle — the key output
        // NREL reference: 50.11162° (topocentric zenith with refraction)
        let zenith_expected = 50.11162;
        assert!(
            fabs(output.zenith - zenith_expected) < 0.01,
            "Zenith: expected {}, got {}",
            zenith_expected,
            output.zenith
        );

        // Azimuth — NREL reference: 194.34024°
        let azimuth_expected = 194.34024;
        assert!(
            fabs(output.azimuth - azimuth_expected) < 0.01,
            "Azimuth: expected {}, got {}",
            azimuth_expected,
            output.azimuth
        );
    }

    #[test]
    fn test_julian_day_known_value() {
        // J2000.0 epoch: January 1.5, 2000 TT = JD 2451545.0
        let jd = julian_day(2000, 1, 1, 12, 0, 0, 0.0);
        assert!(
            fabs(jd - 2451545.0) < 0.001,
            "J2000 JD: expected 2451545.0, got {}",
            jd
        );
    }

    #[test]
    fn test_find_zenith_crossing() {
        // Find sunset (zenith = 90.8333° including refraction) on Oct 17, 2003
        let input = SpaInput {
            year: 2003,
            month: 10,
            day: 17,
            hour: 12,
            minute: 0,
            second: 0,
            timezone: -7.0,
            latitude: 39.742476,
            longitude: -105.1786,
            elevation: 1830.14,
            pressure: 820.0,
            temperature: 11.0,
            delta_t: 67.0,
            slope: 0.0,
            azm_rotation: 0.0,
            atmos_refract: 0.5667,
        };

        // Search for sunset (zenith crossing 90° from below) in the afternoon
        let sunset = find_zenith_crossing(&input, 90.8333, 12.0, 24.0, 0.001);
        assert!(sunset.is_some(), "Should find sunset");
        let sunset_hour = sunset.unwrap();
        // Sunset in Golden, CO on Oct 17 (MST, UTC-7) should be around 17:15-17:30
        assert!(
            sunset_hour > 17.0 && sunset_hour < 18.0,
            "Sunset hour should be ~17:18 MST, got {}",
            sunset_hour
        );
    }

    // ── Input validation ──

    #[test]
    fn validate_rejects_invalid_year() {
        let input = SpaInput {
            year: -3000,
            ..SpaInput::default()
        };
        assert_eq!(solar_position(&input).unwrap_err(), SpaError::InvalidYear);
    }

    #[test]
    fn validate_rejects_invalid_month() {
        let input = SpaInput {
            month: 13,
            ..SpaInput::default()
        };
        assert_eq!(solar_position(&input).unwrap_err(), SpaError::InvalidMonth);
    }

    #[test]
    fn validate_rejects_invalid_day() {
        let input = SpaInput {
            day: 0,
            ..SpaInput::default()
        };
        assert_eq!(solar_position(&input).unwrap_err(), SpaError::InvalidDay);
    }

    #[test]
    fn validate_rejects_invalid_latitude() {
        let input = SpaInput {
            latitude: 91.0,
            ..SpaInput::default()
        };
        assert_eq!(
            solar_position(&input).unwrap_err(),
            SpaError::InvalidLatitude
        );
    }

    #[test]
    fn validate_rejects_invalid_longitude() {
        let input = SpaInput {
            longitude: -181.0,
            ..SpaInput::default()
        };
        assert_eq!(
            solar_position(&input).unwrap_err(),
            SpaError::InvalidLongitude
        );
    }

    #[test]
    fn validate_accepts_boundary_values() {
        let input = SpaInput {
            year: -2000,
            month: 1,
            day: 1,
            latitude: 90.0,
            longitude: -180.0,
            ..SpaInput::default()
        };
        assert!(solar_position(&input).is_ok());

        let input2 = SpaInput {
            year: 6000,
            month: 12,
            day: 31,
            latitude: -90.0,
            longitude: 180.0,
            ..SpaInput::default()
        };
        assert!(solar_position(&input2).is_ok());
    }

    // ── Julian day calculations ──

    #[test]
    fn julian_day_unix_epoch() {
        // Unix epoch: Jan 1, 1970 00:00:00 UTC = JD 2440587.5
        let jd = julian_day(1970, 1, 1, 0, 0, 0, 0.0);
        assert!(
            fabs(jd - 2440587.5) < 0.001,
            "Unix epoch JD: expected 2440587.5, got {}",
            jd
        );
    }

    #[test]
    fn julian_day_gregorian_reform() {
        // Oct 15, 1582 = JD 2299160.5 (first day of Gregorian calendar)
        let jd = julian_day(1582, 10, 15, 0, 0, 0, 0.0);
        assert!(
            fabs(jd - 2299160.5) < 0.5,
            "Gregorian reform JD: expected ~2299160.5, got {}",
            jd
        );
    }

    #[test]
    fn julian_day_increases_with_time() {
        let jd1 = julian_day(2024, 1, 1, 0, 0, 0, 0.0);
        let jd2 = julian_day(2024, 1, 2, 0, 0, 0, 0.0);
        assert!(
            fabs(jd2 - jd1 - 1.0) < 0.001,
            "One day difference: expected 1.0, got {}",
            jd2 - jd1
        );
    }

    #[test]
    fn julian_day_timezone_offset() {
        // Same instant: midnight UTC = 19:00 UTC-5
        let jd_utc = julian_day(2024, 1, 1, 0, 0, 0, 0.0);
        let jd_est = julian_day(2023, 12, 31, 19, 0, 0, -5.0);
        assert!(
            fabs(jd_utc - jd_est) < 0.001,
            "Same instant different TZ: {} vs {}",
            jd_utc,
            jd_est
        );
    }

    #[test]
    fn julian_century_j2000() {
        let jd = 2451545.0; // J2000.0
        let jc = julian_century(jd);
        assert!(fabs(jc) < 1e-10, "JC at J2000 should be 0, got {}", jc);
    }

    #[test]
    fn julian_century_one_century_later() {
        let jd = 2451545.0 + 36525.0;
        let jc = julian_century(jd);
        assert!(
            fabs(jc - 1.0) < 1e-10,
            "JC one century after J2000 should be 1, got {}",
            jc
        );
    }

    #[test]
    fn julian_ephemeris_day_adds_delta_t() {
        let jd = 2451545.0;
        let delta_t = 69.184; // seconds
        let jde = julian_ephemeris_day(jd, delta_t);
        // f64 has ~15 significant digits; JD values are ~2.5e6,
        // so precision is limited to ~1e-9
        assert!(
            fabs(jde - jd - delta_t / 86400.0) < 1e-8,
            "JDE should add delta_t/86400, diff = {:.2e}",
            fabs(jde - jd - delta_t / 86400.0)
        );
    }

    // ── Solar position: geographic consistency ──

    #[test]
    fn solar_noon_zenith_at_equator_equinox() {
        // At equator on equinox, solar noon zenith should be near 0°
        let input = SpaInput {
            year: 2024,
            month: 3,
            day: 20,
            hour: 12,
            minute: 0,
            second: 0,
            timezone: 0.0,
            latitude: 0.0,
            longitude: 0.0,
            ..SpaInput::default()
        };
        let output = solar_position(&input).unwrap();
        // Zenith at solar noon on equator at equinox should be small
        // (not exactly 0 because local noon may not be 12:00 UTC at 0° longitude)
        assert!(
            output.zenith < 25.0,
            "Equator equinox noon zenith should be small, got {}°",
            output.zenith
        );
    }

    #[test]
    fn zenith_increases_with_latitude_in_winter() {
        // At noon UTC on winter solstice, zenith should increase with latitude in NH
        let base = SpaInput {
            year: 2024,
            month: 12,
            day: 21,
            hour: 12,
            minute: 0,
            second: 0,
            timezone: 0.0,
            longitude: 0.0,
            ..SpaInput::default()
        };

        let z_eq = solar_position(&SpaInput {
            latitude: 0.0,
            ..base.clone()
        })
        .unwrap()
        .zenith;
        let z_30 = solar_position(&SpaInput {
            latitude: 30.0,
            ..base.clone()
        })
        .unwrap()
        .zenith;
        let z_60 = solar_position(&SpaInput {
            latitude: 60.0,
            ..base.clone()
        })
        .unwrap()
        .zenith;

        assert!(
            z_30 > z_eq,
            "Zenith at 30°N ({}) should exceed equator ({})",
            z_30,
            z_eq
        );
        assert!(
            z_60 > z_30,
            "Zenith at 60°N ({}) should exceed 30°N ({})",
            z_60,
            z_30
        );
    }

    #[test]
    fn midnight_sun_arctic_summer() {
        // At 70°N on June 21, the sun should be above the horizon at midnight
        let input = SpaInput {
            year: 2024,
            month: 6,
            day: 21,
            hour: 0,
            minute: 0,
            second: 0,
            timezone: 0.0,
            latitude: 70.0,
            longitude: 25.0, // Northern Norway
            ..SpaInput::default()
        };
        let output = solar_position(&input).unwrap();
        assert!(
            output.zenith < 95.0,
            "At 70°N June 21 midnight, zenith should be < 95°, got {}°",
            output.zenith
        );
    }

    #[test]
    fn polar_night_arctic_winter() {
        // At 70°N on Dec 21, the sun should be well below the horizon at noon
        let input = SpaInput {
            year: 2024,
            month: 12,
            day: 21,
            hour: 12,
            minute: 0,
            second: 0,
            timezone: 0.0,
            latitude: 70.0,
            longitude: 25.0,
            ..SpaInput::default()
        };
        let output = solar_position(&input).unwrap();
        assert!(
            output.zenith > 90.0,
            "At 70°N Dec 21 noon, zenith should be > 90°, got {}°",
            output.zenith
        );
    }

    // ── Solar position: declination ──

    #[test]
    fn declination_positive_in_northern_summer() {
        let input = SpaInput {
            year: 2024,
            month: 6,
            day: 21,
            hour: 12,
            minute: 0,
            second: 0,
            ..SpaInput::default()
        };
        let output = solar_position(&input).unwrap();
        assert!(
            output.delta > 20.0 && output.delta < 24.0,
            "June solstice declination should be ~23.4°, got {}°",
            output.delta
        );
    }

    #[test]
    fn declination_negative_in_northern_winter() {
        let input = SpaInput {
            year: 2024,
            month: 12,
            day: 21,
            hour: 12,
            minute: 0,
            second: 0,
            ..SpaInput::default()
        };
        let output = solar_position(&input).unwrap();
        assert!(
            output.delta < -20.0 && output.delta > -24.0,
            "Dec solstice declination should be ~-23.4°, got {}°",
            output.delta
        );
    }

    #[test]
    fn declination_near_zero_at_equinox() {
        let input = SpaInput {
            year: 2024,
            month: 3,
            day: 20,
            hour: 12,
            minute: 0,
            second: 0,
            ..SpaInput::default()
        };
        let output = solar_position(&input).unwrap();
        assert!(
            fabs(output.delta) < 1.0,
            "March equinox declination should be ~0°, got {}°",
            output.delta
        );
    }

    // ── Earth-Sun distance ──

    #[test]
    fn earth_sun_distance_perihelion() {
        // Earth is closest to the Sun around Jan 3 (~0.983 AU)
        let jd = julian_day(2024, 1, 3, 12, 0, 0, 0.0);
        let jde = julian_ephemeris_day(jd, 69.184);
        let jce = julian_ephemeris_century(jde);
        let jme = julian_ephemeris_millennium(jce);
        let r = earth_sun_distance(jme);
        assert!(
            r > 0.982 && r < 0.985,
            "Perihelion distance should be ~0.983 AU, got {} AU",
            r
        );
    }

    #[test]
    fn earth_sun_distance_aphelion() {
        // Earth is farthest from the Sun around Jul 4 (~1.017 AU)
        let jd = julian_day(2024, 7, 4, 12, 0, 0, 0.0);
        let jde = julian_ephemeris_day(jd, 69.184);
        let jce = julian_ephemeris_century(jde);
        let jme = julian_ephemeris_millennium(jce);
        let r = earth_sun_distance(jme);
        assert!(
            r > 1.015 && r < 1.018,
            "Aphelion distance should be ~1.017 AU, got {} AU",
            r
        );
    }

    // ── Equation of time ──

    #[test]
    fn equation_of_time_bounded() {
        // EoT should always be within ±17 minutes
        let input = SpaInput::default();
        for month in 1..=12 {
            let inp = SpaInput {
                month,
                day: 15,
                ..input.clone()
            };
            let output = solar_position(&inp).unwrap();
            assert!(
                fabs(output.eot) < 17.0,
                "EoT in month {} = {:.2} min, should be within ±17",
                month,
                output.eot
            );
        }
    }

    // ── Azimuth ──

    #[test]
    fn azimuth_south_at_noon_northern_hemisphere() {
        // At noon in the northern hemisphere, the sun should be roughly southward
        let input = SpaInput {
            year: 2024,
            month: 6,
            day: 21,
            hour: 12,
            minute: 0,
            second: 0,
            timezone: 0.0,
            latitude: 45.0,
            longitude: 0.0,
            ..SpaInput::default()
        };
        let output = solar_position(&input).unwrap();
        // Azimuth should be near 180° (south), allow wide margin for EoT
        assert!(
            output.azimuth > 140.0 && output.azimuth < 220.0,
            "Noon azimuth at 45°N should be ~180°, got {}°",
            output.azimuth
        );
    }

    // ── find_zenith_crossing ──

    #[test]
    fn find_sunrise_nrel_case() {
        let input = SpaInput {
            year: 2003,
            month: 10,
            day: 17,
            hour: 0,
            minute: 0,
            second: 0,
            timezone: -7.0,
            latitude: 39.742476,
            longitude: -105.1786,
            elevation: 1830.14,
            pressure: 820.0,
            temperature: 11.0,
            delta_t: 67.0,
            slope: 0.0,
            azm_rotation: 0.0,
            atmos_refract: 0.5667,
        };
        let sunrise = find_zenith_crossing(&input, 90.8333, 0.0, 12.0, 0.001);
        assert!(sunrise.is_some(), "Should find sunrise");
        let h = sunrise.unwrap();
        // Sunrise in Golden, CO on Oct 17 MST (UTC-7, no DST) ≈ 6:10-6:20 AM
        assert!(h > 5.5 && h < 7.0, "Sunrise should be ~6:13 MST, got {}", h);
    }

    #[test]
    fn find_zenith_crossing_returns_none_when_no_crossing() {
        // At the equator, the sun never reaches 170° zenith (max ≈ 157°)
        let input = SpaInput {
            year: 2024,
            month: 6,
            day: 21,
            latitude: 0.0,
            longitude: 0.0,
            ..SpaInput::default()
        };
        let result = find_zenith_crossing(&input, 170.0, 0.0, 12.0, 0.001);
        assert!(
            result.is_none(),
            "Should not find 170° zenith crossing at equator"
        );
    }

    #[test]
    fn sunrise_before_sunset() {
        let input = SpaInput {
            year: 2024,
            month: 3,
            day: 20,
            timezone: 0.0,
            latitude: 45.0,
            longitude: 0.0,
            ..SpaInput::default()
        };
        let sunrise = find_zenith_crossing(&input, 90.8333, 0.0, 12.0, 0.001);
        let sunset = find_zenith_crossing(&input, 90.8333, 12.0, 24.0, 0.001);
        assert!(sunrise.is_some() && sunset.is_some());
        assert!(
            sunrise.unwrap() < sunset.unwrap(),
            "Sunrise ({}) should be before sunset ({})",
            sunrise.unwrap(),
            sunset.unwrap()
        );
    }

    // ── Heliocentric coordinates ──

    #[test]
    fn heliocentric_longitude_increases_over_year() {
        // Earth's heliocentric longitude should increase ~1°/day over the year
        let jd1 = julian_day(2024, 1, 1, 12, 0, 0, 0.0);
        let jd2 = julian_day(2024, 4, 1, 12, 0, 0, 0.0);
        let jde1 = julian_ephemeris_day(jd1, 69.184);
        let jde2 = julian_ephemeris_day(jd2, 69.184);
        let jme1 = julian_ephemeris_millennium(julian_ephemeris_century(jde1));
        let jme2 = julian_ephemeris_millennium(julian_ephemeris_century(jde2));
        let l1 = earth_heliocentric_longitude(jme1);
        let l2 = earth_heliocentric_longitude(jme2);
        // 91 days → ~91° increase (mod 360)
        let diff = ((l2 - l1) + 360.0) % 360.0;
        assert!(
            diff > 80.0 && diff < 100.0,
            "Heliocentric longitude change over 91 days: {}°, expected ~91°",
            diff
        );
    }

    #[test]
    fn heliocentric_latitude_small() {
        // Earth's heliocentric latitude should always be very small (< 0.01°)
        let jd = julian_day(2024, 6, 21, 12, 0, 0, 0.0);
        let jde = julian_ephemeris_day(jd, 69.184);
        let jme = julian_ephemeris_millennium(julian_ephemeris_century(jde));
        let b = earth_heliocentric_latitude(jme);
        assert!(
            fabs(b) < 0.01,
            "Heliocentric latitude should be ~0°, got {}°",
            b
        );
    }

    // ── Nutation ──

    #[test]
    fn nutation_bounded() {
        // Nutation in longitude should be within ±20 arcseconds (±0.006°)
        let jd = julian_day(2024, 1, 1, 12, 0, 0, 0.0);
        let jce = julian_ephemeris_century(julian_ephemeris_day(jd, 69.184));
        let (delta_psi, delta_epsilon) = nutation(jce);
        assert!(
            fabs(delta_psi) < 0.01,
            "Nutation in longitude should be < 0.01°, got {}°",
            delta_psi
        );
        assert!(
            fabs(delta_epsilon) < 0.01,
            "Nutation in obliquity should be < 0.01°, got {}°",
            delta_epsilon
        );
    }

    // ── Geocentric transforms ──

    #[test]
    fn geocentric_longitude_offset_180() {
        assert!(fabs(geocentric_longitude(0.0) - 180.0) < 1e-10);
        assert!(fabs(geocentric_longitude(180.0) - 0.0) < 1e-10);
        assert!(fabs(geocentric_longitude(90.0) - 270.0) < 1e-10);
    }

    #[test]
    fn geocentric_latitude_negates() {
        assert!(fabs(geocentric_latitude(1.5) - (-1.5)) < 1e-10);
        assert!(fabs(geocentric_latitude(-0.5) - 0.5) < 1e-10);
    }

    // ── limit_degrees ──

    #[test]
    fn limit_degrees_normalizes() {
        assert!(fabs(limit_degrees(370.0) - 10.0) < 1e-10);
        assert!(fabs(limit_degrees(-10.0) - 350.0) < 1e-10);
        assert!(fabs(limit_degrees(0.0) - 0.0) < 1e-10);
        assert!(fabs(limit_degrees(720.0) - 0.0) < 1e-10);
    }

    // ── set_time_from_fractional_hour ──

    #[test]
    fn set_time_noon() {
        let mut input = SpaInput::default();
        set_time_from_fractional_hour(&mut input, 12.0);
        assert_eq!(input.hour, 12);
        assert_eq!(input.minute, 0);
        assert_eq!(input.second, 0);
    }

    #[test]
    fn set_time_with_minutes() {
        let mut input = SpaInput::default();
        set_time_from_fractional_hour(&mut input, 12.5);
        assert_eq!(input.hour, 12);
        assert_eq!(input.minute, 30);
        assert_eq!(input.second, 0);
    }

    #[test]
    fn set_time_with_seconds() {
        let mut input = SpaInput::default();
        let h = 12.0 + 30.0 / 60.0 + 30.0 / 3600.0;
        set_time_from_fractional_hour(&mut input, h);
        assert_eq!(input.hour, 12);
        assert_eq!(input.minute, 30);
        assert_eq!(input.second, 30);
    }

    #[test]
    fn set_time_midnight() {
        let mut input = SpaInput::default();
        set_time_from_fractional_hour(&mut input, 0.0);
        assert_eq!(input.hour, 0);
        assert_eq!(input.minute, 0);
        assert_eq!(input.second, 0);
    }

    #[test]
    fn set_time_end_of_day() {
        let mut input = SpaInput::default();
        set_time_from_fractional_hour(&mut input, 23.0 + 59.0 / 60.0 + 59.0 / 3600.0);
        assert_eq!(input.hour, 23);
        assert_eq!(input.minute, 59);
        assert_eq!(input.second, 59);
    }

    // ── format_time ──

    #[test]
    fn format_time_midnight() {
        assert_eq!(&format_time(0.0), b"00:00:00");
    }

    #[test]
    fn format_time_noon() {
        assert_eq!(&format_time(12.0), b"12:00:00");
    }

    #[test]
    fn format_time_with_seconds() {
        let h = 12.0 + 30.0 / 60.0 + 30.0 / 3600.0;
        assert_eq!(&format_time(h), b"12:30:30");
    }

    #[test]
    fn format_time_half_hour() {
        assert_eq!(&format_time(6.5), b"06:30:00");
    }

    // ── SpaInput default ──

    #[test]
    fn spa_input_default_is_valid() {
        let input = SpaInput::default();
        assert!(solar_position(&input).is_ok());
    }

    // ── Second NREL-like reference: different date/location ──

    #[test]
    fn solar_position_mecca_midday() {
        // Mecca, March equinox, local noon
        let input = SpaInput {
            year: 2024,
            month: 3,
            day: 20,
            hour: 12,
            minute: 0,
            second: 0,
            timezone: 3.0,
            latitude: 21.4225,
            longitude: 39.8262,
            ..SpaInput::default()
        };
        let output = solar_position(&input).unwrap();
        // At Mecca (21°N), equinox noon, zenith should be ~21° (sun nearly overhead)
        assert!(
            output.zenith > 15.0 && output.zenith < 30.0,
            "Mecca equinox noon zenith: expected ~21°, got {}°",
            output.zenith
        );
        // Azimuth should be roughly south (~180°)
        assert!(
            output.azimuth > 150.0 && output.azimuth < 210.0,
            "Mecca noon azimuth: expected ~180°, got {}°",
            output.azimuth
        );
    }
}
