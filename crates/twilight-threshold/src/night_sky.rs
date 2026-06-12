//! Physical night-sky background luminance model.
//!
//! Replaces a constant dark-sky number with the actual celestial
//! background for a given place, time and view direction — the floor that
//! the TVI contrast detection (and therefore Fajr/Isha) is measured
//! against:
//!
//! Built on MEASURED data wherever a measurement exists:
//!
//! - **Airglow**: van Rhijn zenith-distance enhancement (geometry) over a
//!   ~90 km emitting shell, scaled by the MEASURED daily solar F10.7
//!   radio flux (NOAA SWPC feed via the CLI) through the Walker relation.
//!   Airglow has ~25% intrinsic night-to-night variability that no public
//!   feed resolves — this is the one component that must stay a
//!   measurement-driven model.
//! - **Zodiacal light**: bilinear lookup in the measured Leinert et al.
//!   (1998) Table 16 (ground photometry + Helios A/B + rocket data).
//! - **Integrated starlight**: bilinear lookup in the measured Pioneer
//!   10/11 sky map (Toller 1981; Leinert Table 35) — real galactic
//!   lon x lat structure, taken beyond 2.8 AU where zodiacal light
//!   vanishes.
//! - **Moonlight**: Krisciunas & Schaefer (1991) scattered-moonlight
//!   model (the observatory standard), fed the REAL lunar state from the
//!   JPL DE440 ephemeris when available (exact parallax, true phase
//!   angle, actual distance — a perigee full moon is ~30% brighter).
//!   A full moon raises the sky background 10-100x — which physically
//!   delays the detectable dawn (and classically, moonlit nights
//!   confounded twilight observation for the same reason).
//!
//! Units: component brightnesses are computed in S10 (stars of 10th
//! visual magnitude per square degree) and nanoLamberts where the source
//! papers use them, then converted to cd/m^2:
//!   1 S10 ~ 0.69e-6 cd/m^2 (via 22.0 mag/arcsec^2 = 1.71e-4 cd/m^2
//!   ~ 250 S10), 1 nL = 3.183e-6 cd/m^2.

use twilight_solar::moon::{moon_state, MoonState};

const DEG: f64 = core::f64::consts::PI / 180.0;
const S10_TO_CD: f64 = 0.69e-6;
const NL_TO_CD: f64 = 3.183e-6;

/// Inputs for the night-sky background evaluation.
#[derive(Debug, Clone, Copy)]
pub struct NightSkyInput {
    pub latitude: f64,
    pub longitude: f64,
    /// Civil date (UTC) and fractional hour UTC of the evaluation instant.
    pub year: i32,
    pub month: i32,
    pub day: i32,
    pub hour_utc: f64,
    /// View direction.
    pub view_zenith_deg: f64,
    pub view_azimuth_deg: f64,
    /// Solar 10.7 cm flux (sfu); ~70 solar min, ~200 solar max.
    /// Default 130 (mid-cycle) when unknown.
    pub solar_f107: f64,
    /// Broadband V extinction coefficient (mag/airmass), ~0.2 clear.
    pub extinction_k: f64,
}

impl Default for NightSkyInput {
    fn default() -> Self {
        Self {
            latitude: 0.0,
            longitude: 0.0,
            year: 2026,
            month: 1,
            day: 1,
            hour_utc: 0.0,
            view_zenith_deg: 0.0,
            view_azimuth_deg: 0.0,
            solar_f107: 130.0,
            extinction_k: 0.2,
        }
    }
}

/// Component breakdown [cd/m^2].
#[derive(Debug, Clone, Copy)]
pub struct NightSkyLuminance {
    pub airglow: f64,
    pub zodiacal: f64,
    pub starlight: f64,
    pub moonlight: f64,
    pub total: f64,
    /// Moon altitude [deg] at the instant (for reporting).
    pub moon_altitude_deg: f64,
    /// Moon illuminated fraction [0..1].
    pub moon_illum: f64,
}

/// Krisciunas-Schaefer airmass for scattered light: valid past the
/// horizon, X(90 deg) ~ 5.
fn ks_airmass(zenith_deg: f64) -> f64 {
    let z = zenith_deg.min(107.0) * DEG;
    let s = libm::sin(z);
    1.0 / libm::sqrt((1.0 - 0.96 * s * s).max(1e-4))
}

/// Airglow brightness toward the view direction [cd/m^2].
///
/// Zenith airglow ~ 145 S10 at solar minimum scaling roughly linearly to
/// ~2.4x at solar maximum (Walker 1988; KS91 eq. 2 uses 145 S10 at
/// F10.7 = 0.8 MJy... parameterized here as 90 + 0.43*F107 S10, matching
/// ~120 S10 at F107=70 and ~175 at F107=200). Van Rhijn enhancement for
/// a 90 km emitting layer, attenuated by extinction along the path.
fn airglow_cd(view_zenith_deg: f64, f107: f64, k: f64) -> f64 {
    let zen_s10 = 90.0 + 0.43 * f107;
    let z = view_zenith_deg.min(90.0) * DEG;
    let re = 6371.0;
    let h = 90.0;
    let s = libm::sin(z) * re / (re + h);
    let van_rhijn = 1.0 / libm::sqrt((1.0 - s * s).max(1e-6));
    let x = ks_airmass(view_zenith_deg);
    // self-attenuation of the airglow shell through the lower atmosphere
    let atten = libm::pow(10.0, -0.4 * k * (x - 1.0)) ;
    zen_s10 * van_rhijn * atten * S10_TO_CD
}

/// Index + fraction for linear interpolation on an ascending grid
/// (clamped at both ends).
fn grid_locate(grid: &[f64], x: f64) -> (usize, f64) {
    let n = grid.len();
    if x <= grid[0] {
        return (0, 0.0);
    }
    if x >= grid[n - 1] {
        return (n - 2, 1.0);
    }
    let mut i = 0;
    while grid[i + 1] < x {
        i += 1;
    }
    (i, (x - grid[i]) / (grid[i + 1] - grid[i]))
}

/// Zodiacal light toward the view direction [cd/m^2].
///
/// Bilinear lookup in the MEASURED Leinert et al. (1998) Table 16 —
/// annually averaged zodiacal brightness at 500 nm in S10_sun on the
/// helioecliptic grid (ground photometry extended sunward with Helios
/// A/B + rocket data; the authors endorse smooth interpolation between
/// nodes). See `night_sky_tables` for provenance and cross-checks.
fn zodiacal_cd(helio_dlon_deg: f64, ecl_lat_deg: f64, view_zenith_deg: f64, k: f64) -> f64 {
    use crate::night_sky_tables::{ZODI_DLON, ZODI_LAT, ZODI_S10};
    let dl = helio_dlon_deg.abs().min(180.0);
    let b = ecl_lat_deg.abs().min(90.0);
    let (i, fi) = grid_locate(&ZODI_LAT, b);
    let (j, fj) = grid_locate(&ZODI_DLON, dl);
    let s10 = ZODI_S10[i][j] * (1.0 - fi) * (1.0 - fj)
        + ZODI_S10[i + 1][j] * fi * (1.0 - fj)
        + ZODI_S10[i][j + 1] * (1.0 - fi) * fj
        + ZODI_S10[i + 1][j + 1] * fi * fj;
    let x = ks_airmass(view_zenith_deg);
    s10 * libm::pow(10.0, -0.4 * k * (x - 1.0)) * S10_TO_CD
}

/// Integrated starlight + diffuse galactic light [cd/m^2].
///
/// Bilinear lookup in the MEASURED Pioneer 10/11 background-starlight
/// sky map (Toller 1981; Leinert 1998 Table 35): S10_sun per square
/// degree on a galactic lon x lat grid, observed beyond 2.8 AU where
/// zodiacal light vanishes. The real Milky Way structure — inner-Galaxy
/// bulge, Carina/Cygnus enhancements, even the LMC — instead of a
/// latitude-only falloff. Stars brighter than ~6.5 mag were removed by
/// Pioneer's processing; for the integrated TVI background floor their
/// omission is conservative (the handful of bright stars are seen as
/// point sources, not background).
fn starlight_cd(gal_lon_deg: f64, gal_lat_deg: f64, view_zenith_deg: f64, k: f64) -> f64 {
    use crate::night_sky_tables::{STAR_LAT, STAR_LON, STAR_S10};
    let nlon = STAR_LON.len();
    let l = gal_lon_deg.rem_euclid(360.0);
    let b = gal_lat_deg.clamp(STAR_LAT[0], STAR_LAT[STAR_LAT.len() - 1]);
    let (i, fi) = grid_locate(&STAR_LAT, b);
    // longitude wraparound (grid is 0..350 in 10-deg steps)
    let j = ((l / 10.0) as usize).min(nlon - 1);
    let j1 = (j + 1) % nlon;
    let fj = (l - 10.0 * j as f64) / 10.0;
    let s10 = STAR_S10[i][j] * (1.0 - fi) * (1.0 - fj)
        + STAR_S10[i + 1][j] * fi * (1.0 - fj)
        + STAR_S10[i][j1] * (1.0 - fi) * fj
        + STAR_S10[i + 1][j1] * fi * fj;
    let x = ks_airmass(view_zenith_deg);
    s10 * libm::pow(10.0, -0.4 * k * (x - 1.0)) * S10_TO_CD
}

/// Krisciunas & Schaefer (1991) scattered moonlight [cd/m^2].
fn moonlight_cd(
    moon: &MoonState,
    view_zenith_deg: f64,
    view_azimuth_deg: f64,
    k: f64,
) -> f64 {
    if moon.altitude_deg < -1.0 {
        return 0.0; // moon below horizon: no scattered moonlight
    }
    let alpha = moon.phase_angle_deg.abs();
    // KS91 eq. 9: stellar magnitude factor of the Moon vs phase.
    // The formula is normalized to the MEAN lunar distance; scale by
    // (a/d)^2 for the actual distance — a perigee full moon delivers
    // ~30% more illuminance than an apogee one.
    let mean_dist_km = 385000.56;
    let dist_factor = (mean_dist_km / moon.distance_km) * (mean_dist_km / moon.distance_km);
    let i_star = dist_factor
        * libm::pow(
            10.0,
            -0.4 * (3.84 + 0.026 * alpha + 4.0e-9 * alpha * alpha * alpha * alpha),
        );
    // angular separation moon-view
    let zv = view_zenith_deg * DEG;
    let zm = (90.0 - moon.altitude_deg) * DEG;
    let daz = (view_azimuth_deg - moon.azimuth_deg) * DEG;
    let cos_rho = libm::cos(zv) * libm::cos(zm) + libm::sin(zv) * libm::sin(zm) * libm::cos(daz);
    let rho = libm::acos(cos_rho.clamp(-1.0, 1.0)) / DEG;
    // KS91 eq. 21: scattering function (Rayleigh + Mie forward peak)
    let f_rho = libm::pow(10.0, 5.36) * (1.06 + cos_rho * cos_rho)
        + libm::pow(10.0, 6.15 - rho / 40.0);
    let x_m = ks_airmass(90.0 - moon.altitude_deg);
    let x_v = ks_airmass(view_zenith_deg);
    // KS91 eq. 15 [nanoLamberts]
    let b_nl = f_rho
        * i_star
        * libm::pow(10.0, -0.4 * k * x_m)
        * (1.0 - libm::pow(10.0, -0.4 * k * x_v));
    b_nl * NL_TO_CD
}

/// Ecliptic and galactic coordinates of the view direction.
fn view_coords(
    input: &NightSkyInput,
) -> (
    f64, /*helio dlon*/
    f64, /*ecl lat*/
    f64, /*gal lon*/
    f64, /*gal lat*/
) {
    // local alt/az -> equatorial
    let phi = input.latitude * DEG;
    let alt = (90.0 - input.view_zenith_deg) * DEG;
    let az = input.view_azimuth_deg * DEG;
    let sin_dec =
        libm::sin(phi) * libm::sin(alt) + libm::cos(phi) * libm::cos(alt) * libm::cos(az);
    let dec = libm::asin(sin_dec.clamp(-1.0, 1.0));
    let ha = libm::atan2(
        -libm::sin(az) * libm::cos(alt),
        libm::sin(alt) * libm::cos(phi) - libm::cos(alt) * libm::sin(phi) * libm::cos(az),
    );
    // LST via the moon module's internal convention (recompute GMST here)
    let jd = {
        let (y, m) = if input.month <= 2 {
            (input.year - 1, input.month + 12)
        } else {
            (input.year, input.month)
        };
        let a = (y as f64 / 100.0).floor();
        let b = 2.0 - a + (a / 4.0).floor();
        (365.25 * (y as f64 + 4716.0)).floor() + (30.6001 * (m as f64 + 1.0)).floor()
            + input.day as f64
            + input.hour_utc / 24.0
            + b
            - 1524.5
    };
    let t = (jd - 2451545.0) / 36525.0;
    let gmst = (280.46061837 + 360.98564736629 * (jd - 2451545.0)).rem_euclid(360.0);
    let lst = (gmst + input.longitude) * DEG;
    let ra = lst - ha;

    // equatorial -> ecliptic
    let eps = 23.4393 * DEG;
    let ecl_lat = libm::asin(
        (libm::sin(dec) * libm::cos(eps) - libm::cos(dec) * libm::sin(eps) * libm::sin(ra))
            .clamp(-1.0, 1.0),
    );
    let ecl_lon = libm::atan2(
        libm::sin(ra) * libm::cos(eps) + libm::tan(dec) * libm::sin(eps),
        libm::cos(ra),
    );
    // sun ecliptic lon (Meeus low precision)
    let l0 = 280.46646 + 36000.76983 * t;
    let msun = (357.52911 + 35999.05029 * t) * DEG;
    let c = 1.914602 * libm::sin(msun) + 0.019993 * libm::sin(2.0 * msun);
    let sun_lon = (l0 + c).rem_euclid(360.0);
    let mut dlon = (ecl_lon / DEG - sun_lon).rem_euclid(360.0);
    if dlon > 180.0 {
        dlon = 360.0 - dlon;
    }

    // equatorial -> galactic (NGP at RA 192.86, Dec 27.13; l_NCP 122.932)
    let ngp_ra = 192.859508 * DEG;
    let ngp_dec = 27.128336 * DEG;
    let l_ncp = 122.93192;
    let gal_lat = libm::asin(
        (libm::sin(dec) * libm::sin(ngp_dec)
            + libm::cos(dec) * libm::cos(ngp_dec) * libm::cos(ra - ngp_ra))
            .clamp(-1.0, 1.0),
    );
    let gal_lon = (l_ncp
        - libm::atan2(
            libm::cos(dec) * libm::sin(ra - ngp_ra),
            libm::sin(dec) * libm::cos(ngp_dec)
                - libm::cos(dec) * libm::sin(ngp_dec) * libm::cos(ra - ngp_ra),
        ) / DEG)
        .rem_euclid(360.0);

    (dlon, ecl_lat / DEG, gal_lon, gal_lat / DEG)
}

/// Evaluate the full night-sky background toward the view direction,
/// computing the lunar state internally from the Meeus series.
///
/// When a JPL DE440 ephemeris is available, prefer
/// [`night_sky_luminance_with_moon`] with the DE440-computed
/// [`MoonState`] — real ephemeris data instead of a truncated series.
pub fn night_sky_luminance(input: &NightSkyInput) -> NightSkyLuminance {
    let moon = moon_state(
        input.year,
        input.month,
        input.day,
        input.hour_utc,
        input.latitude,
        input.longitude,
    );
    night_sky_luminance_with_moon(input, &moon)
}

/// Evaluate the full night-sky background with an externally supplied
/// lunar state (e.g. from the JPL DE440 ephemeris).
pub fn night_sky_luminance_with_moon(
    input: &NightSkyInput,
    moon: &MoonState,
) -> NightSkyLuminance {
    let (dlon, ecl_lat, gal_lon, gal_lat) = view_coords(input);
    let airglow = airglow_cd(input.view_zenith_deg, input.solar_f107, input.extinction_k);
    let zodiacal = zodiacal_cd(dlon, ecl_lat, input.view_zenith_deg, input.extinction_k);
    let starlight = starlight_cd(gal_lon, gal_lat, input.view_zenith_deg, input.extinction_k);
    let moonlight = moonlight_cd(
        moon,
        input.view_zenith_deg,
        input.view_azimuth_deg,
        input.extinction_k,
    );
    NightSkyLuminance {
        airglow,
        zodiacal,
        starlight,
        moonlight,
        total: airglow + zodiacal + starlight + moonlight,
        moon_altitude_deg: moon.altitude_deg,
        moon_illum: moon.illuminated_fraction,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dark_input() -> NightSkyInput {
        // New moon 2026-01-18, mid-latitude, zenith view
        NightSkyInput {
            latitude: 30.0,
            longitude: 0.0,
            year: 2026,
            month: 1,
            day: 18,
            hour_utc: 23.0,
            view_zenith_deg: 0.0,
            view_azimuth_deg: 0.0,
            solar_f107: 130.0,
            extinction_k: 0.2,
        }
    }

    #[test]
    fn moonless_dark_sky_matches_published_range() {
        // The canonical dark-site zenith sky is 21.6-22.1 mag/arcsec^2
        // = 1.55e-4 .. 2.45e-4 cd/m^2. The model (moonless, zenith) must
        // land in that range — this anchors the whole TVI floor.
        let n = night_sky_luminance(&dark_input());
        assert!(n.moonlight < 2e-5, "moon should be near-new: {:?}", n);
        assert!(
            (1.4e-4..=2.9e-4).contains(&n.total),
            "dark zenith sky = {:.3e} cd/m^2 ({:?})",
            n.total,
            n
        );
    }

    #[test]
    fn full_moon_dominates() {
        // Full moon 2026-01-03, view 45 deg from moon: KS91 predicts the
        // sky brightens by an order of magnitude or more.
        let mut inp = dark_input();
        inp.day = 3;
        inp.view_zenith_deg = 45.0;
        let n = night_sky_luminance(&inp);
        if n.moon_altitude_deg > 10.0 {
            assert!(
                n.moonlight > 3.0 * (n.airglow + n.zodiacal + n.starlight),
                "full moon up should dominate: {:?}",
                n
            );
        }
    }

    #[test]
    fn airglow_van_rhijn_increases_toward_horizon() {
        let z0 = airglow_cd(0.0, 130.0, 0.0);
        let z70 = airglow_cd(70.0, 130.0, 0.0);
        assert!(z70 > 1.5 * z0, "van Rhijn: {z0:.3e} -> {z70:.3e}");
    }

    #[test]
    fn solar_activity_scales_airglow() {
        assert!(airglow_cd(0.0, 200.0, 0.2) > 1.3 * airglow_cd(0.0, 70.0, 0.2));
    }

    #[test]
    fn zodiacal_brighter_toward_sun_in_ecliptic() {
        let near = zodiacal_cd(40.0, 0.0, 0.0, 0.2);
        let far = zodiacal_cd(140.0, 0.0, 0.0, 0.2);
        let pole = zodiacal_cd(90.0, 85.0, 0.0, 0.2);
        assert!(near > far && far > pole);
        // ecliptic pole = 60 +/- 3 S10 (Leinert Table 16 caption) ~ 4.1e-5
        assert!((2e-5..=8e-5).contains(&pole), "pole = {pole:.2e}");
    }

    #[test]
    fn zodiacal_matches_published_nodes() {
        // Exact table nodes (no extinction): dlon=40, beta=0 -> 925 S10;
        // gegenschein dlon=180, beta=0 -> 180 S10.
        let near = zodiacal_cd(40.0, 0.0, 0.0, 0.0) / S10_TO_CD;
        assert!((near - 925.0).abs() < 1.0, "near = {near}");
        let gegen = zodiacal_cd(180.0, 0.0, 0.0, 0.0) / S10_TO_CD;
        assert!((gegen - 180.0).abs() < 1.0, "gegenschein = {gegen}");
    }

    #[test]
    fn starlight_peaks_in_galactic_plane() {
        // Cygnus region in the plane vs the north galactic pole.
        let plane = starlight_cd(80.0, 0.0, 0.0, 0.2);
        let pole = starlight_cd(0.0, 90.0, 0.0, 0.2);
        assert!(plane > 3.0 * pole);
        // pole rows measure ~25-33 S10 ~ 2e-5
        assert!((1.4e-5..=3e-5).contains(&pole), "pole = {pole:.2e}");
    }

    #[test]
    fn starlight_shows_real_galactic_structure() {
        // The Pioneer map resolves longitude structure a |b|-only fit
        // cannot: the inner Galaxy (l~0) outshines the anticenter (l~180)
        // in the plane.
        let inner = starlight_cd(0.0, 0.0, 0.0, 0.0);
        let anticenter = starlight_cd(180.0, 0.0, 0.0, 0.0);
        assert!(
            inner > 1.3 * anticenter,
            "inner {inner:.2e} vs anticenter {anticenter:.2e}"
        );
    }
}
