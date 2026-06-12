//! JPL DE440 ephemeris backend for high-precision solar position.
//!
//! Uses the DE440 planetary ephemeris (2021) to compute the Sun's
//! geocentric position in the ICRF/J2000 frame, then converts to
//! topocentric zenith angle and azimuth for a given observer.
//!
//! Accuracy: ~0.001 arcsecond (milliarcsecond) for the Sun's position,
//! approximately 1000x more precise than SPA (±0.0003 degrees = ±1.08").
//!
//! The DE440 BSP file (~97 MB) must be provided separately. It is not
//! embedded in the binary due to its size.
//!
//! Download: <https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de440.bsp>
//!
//! Coverage: 1550 CE to 2650 CE.

use crate::earth_rotation::{self, icrf_to_topocentric, TopocentricPosition};
use crate::moon::MoonState;
use crate::spk::{self, SpkFile, EARTH, EARTH_MOON_BARYCENTER, MOON, SUN};
use std::path::Path;

// ── Error type ─────────────────────────────────────────────────────

/// Errors from the DE440 backend.
#[derive(Debug)]
pub enum De440Error {
    /// SPK file error.
    Spk(spk::SpkError),
    /// File not found or not accessible.
    FileNotFound(String),
}

impl From<spk::SpkError> for De440Error {
    fn from(e: spk::SpkError) -> Self {
        De440Error::Spk(e)
    }
}

impl core::fmt::Display for De440Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            De440Error::Spk(e) => write!(f, "SPK error: {}", e),
            De440Error::FileNotFound(path) => write!(f, "DE440 file not found: {}", path),
        }
    }
}

// ── DE440 Ephemeris ────────────────────────────────────────────────

/// JPL DE440 ephemeris handle.
///
/// Wraps an opened SPK file and provides high-level methods
/// for computing solar position.
pub struct De440 {
    spk: SpkFile,
}

impl De440 {
    /// Open a DE440 BSP file.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, De440Error> {
        let p = path.as_ref();
        if !p.exists() {
            return Err(De440Error::FileNotFound(p.to_string_lossy().into_owned()));
        }
        let spk = SpkFile::open(p)?;
        Ok(De440 { spk })
    }

    /// Get the geocentric position of the Sun in ICRF/J2000 (km).
    ///
    /// `tdb_seconds`: seconds past J2000.0 TDB.
    ///
    /// Internally chains:
    ///   Sun(10) wrt SSB(0) - [EMB(3) wrt SSB(0) + Earth(399) wrt EMB(3)]
    pub fn sun_position_icrf(&mut self, tdb_seconds: f64) -> Result<[f64; 3], De440Error> {
        let pos = self.spk.position_chain(SUN, EARTH, tdb_seconds)?;
        Ok(pos)
    }

    /// Get the geocentric position and velocity of the Sun in ICRF/J2000.
    ///
    /// Returns (position_km, velocity_km_s).
    pub fn sun_state_icrf(&mut self, tdb_seconds: f64) -> Result<([f64; 3], [f64; 3]), De440Error> {
        // Chain Sun -> SSB -> Earth
        // sun_wrt_earth = sun_wrt_ssb - earth_wrt_ssb
        // earth_wrt_ssb = emb_wrt_ssb + earth_wrt_emb

        let (sun_pos, sun_vel) = self.spk.state(SUN, 0, tdb_seconds)?;
        let (emb_pos, emb_vel) = self.spk.state(EARTH_MOON_BARYCENTER, 0, tdb_seconds)?;
        let (earth_pos, earth_vel) = self.spk.state(EARTH, EARTH_MOON_BARYCENTER, tdb_seconds)?;

        let pos = [
            sun_pos[0] - emb_pos[0] - earth_pos[0],
            sun_pos[1] - emb_pos[1] - earth_pos[1],
            sun_pos[2] - emb_pos[2] - earth_pos[2],
        ];

        let vel = [
            sun_vel[0] - emb_vel[0] - earth_vel[0],
            sun_vel[1] - emb_vel[1] - earth_vel[1],
            sun_vel[2] - emb_vel[2] - earth_vel[2],
        ];

        Ok((pos, vel))
    }

    /// Get the geocentric position of the Moon in ICRF/J2000 (km).
    ///
    /// DE440 stores Moon(301) and Earth(399) relative to the Earth-Moon
    /// barycenter (3); the segment chain resolves the geocentric vector.
    /// The DE440 lunar orbit is fit to laser ranging - sub-meter accuracy,
    /// vs ~0.3 deg for the truncated Meeus series.
    pub fn moon_position_icrf(&mut self, tdb_seconds: f64) -> Result<[f64; 3], De440Error> {
        Ok(self.spk.position_chain(MOON, EARTH, tdb_seconds)?)
    }

    /// Topocentric lunar state from the real JPL ephemeris - drop-in
    /// replacement for the Meeus-based [`crate::moon::moon_state`].
    ///
    /// The topocentric conversion subtracts the observer's WGS84 position
    /// from the geocentric vector, so the Moon's ~1 deg horizontal
    /// parallax is handled exactly (for the Sun it is negligible; for the
    /// Moon it moves the apparent altitude by up to a degree). The phase
    /// angle is the true Sun-Moon-Earth angle computed from the two ICRF
    /// vectors, not an ecliptic-elongation approximation.
    #[allow(clippy::too_many_arguments)] // Calendar components + observer location are all independent
    pub fn moon_state(
        &mut self,
        year: i32,
        month: i32,
        day: i32,
        hour_utc: f64,
        delta_t: f64,
        latitude: f64,
        longitude: f64,
        elevation: f64,
    ) -> Result<MoonState, De440Error> {
        let total_seconds = hour_utc * 3600.0;
        let hour = (total_seconds / 3600.0) as i32;
        let minute = ((total_seconds - hour as f64 * 3600.0) / 60.0) as i32;
        let second = (total_seconds - hour as f64 * 3600.0 - minute as f64 * 60.0) as i32;
        let jd_utc = earth_rotation::calendar_to_jd(year, month, day, hour, minute, second);
        let jd_tdb = earth_rotation::utc_jd_to_tdb_jd(jd_utc, delta_t);
        let tdb_seconds = earth_rotation::jd_to_tdb_seconds(jd_tdb);

        let moon_icrf = self.moon_position_icrf(tdb_seconds)?;
        let sun_icrf = self.sun_position_icrf(tdb_seconds)?;

        // UT1 ≈ UTC (same convention as solar_position)
        let topo = icrf_to_topocentric(moon_icrf, jd_utc, jd_tdb, latitude, longitude, elevation);

        // Phase angle at the Moon between the Sun and the Earth (geocentric
        // observer is sufficient: the topocentric refinement to alpha is
        // < 0.002 deg).
        let sm = [
            sun_icrf[0] - moon_icrf[0],
            sun_icrf[1] - moon_icrf[1],
            sun_icrf[2] - moon_icrf[2],
        ];
        let em = [-moon_icrf[0], -moon_icrf[1], -moon_icrf[2]];
        let dot = sm[0] * em[0] + sm[1] * em[1] + sm[2] * em[2];
        let nsm = libm::sqrt(sm[0] * sm[0] + sm[1] * sm[1] + sm[2] * sm[2]);
        let nem = libm::sqrt(em[0] * em[0] + em[1] * em[1] + em[2] * em[2]);
        let alpha = libm::acos((dot / (nsm * nem)).clamp(-1.0, 1.0));

        Ok(MoonState {
            altitude_deg: topo.elevation,
            azimuth_deg: topo.azimuth,
            phase_angle_deg: alpha.to_degrees(),
            illuminated_fraction: (1.0 + libm::cos(alpha)) / 2.0,
            distance_km: topo.distance_km,
        })
    }

    /// Compute topocentric solar position for a given observer.
    ///
    /// # Arguments
    /// * `year`, `month`, `day`, `hour`, `minute`, `second` - UTC date/time
    /// * `delta_t` - TT-UTC offset in seconds (~69.184 for 2024)
    /// * `latitude` - observer geodetic latitude (degrees, north positive)
    /// * `longitude` - observer geodetic longitude (degrees, east positive)
    /// * `elevation` - observer altitude above WGS84 (meters)
    ///
    /// # Returns
    /// [`TopocentricPosition`] with zenith, azimuth, elevation, distance.
    #[allow(clippy::too_many_arguments)] // Calendar components (y/m/d/h/m/s) + delta_t + observer (lat/lon/elev) are all independent
    pub fn solar_position(
        &mut self,
        year: i32,
        month: i32,
        day: i32,
        hour: i32,
        minute: i32,
        second: i32,
        delta_t: f64,
        latitude: f64,
        longitude: f64,
        elevation: f64,
    ) -> Result<TopocentricPosition, De440Error> {
        // Convert UTC to TDB
        let jd_utc = earth_rotation::calendar_to_jd(year, month, day, hour, minute, second);
        let jd_tdb = earth_rotation::utc_jd_to_tdb_jd(jd_utc, delta_t);
        let tdb_seconds = earth_rotation::jd_to_tdb_seconds(jd_tdb);

        // UT1 ≈ UTC for our purposes (UT1-UTC < 0.9s)
        let jd_ut1 = jd_utc;

        // Get geocentric Sun position in ICRF
        let sun_icrf = self.sun_position_icrf(tdb_seconds)?;

        // Convert to topocentric
        let topo = icrf_to_topocentric(sun_icrf, jd_ut1, jd_tdb, latitude, longitude, elevation);

        Ok(topo)
    }

    /// Compute topocentric solar position at a given UTC fractional hour.
    ///
    /// The fractional hour is decomposed into signed h/m/s components and
    /// fed to the pure Julian-day conversion, so values outside [0, 24)
    /// (negative for timezone-east queries, >= 24 for past-midnight
    /// windows) land on the correct adjacent civil day without explicit
    /// calendar rollover. Seconds precision is preserved.
    #[allow(clippy::too_many_arguments)] // Calendar components + observer location are all independent
    pub fn solar_position_at_hour(
        &mut self,
        year: i32,
        month: i32,
        day: i32,
        fractional_hour: f64,
        delta_t: f64,
        latitude: f64,
        longitude: f64,
        elevation: f64,
    ) -> Result<TopocentricPosition, De440Error> {
        let total_seconds = fractional_hour * 3600.0;
        let hour = (total_seconds / 3600.0) as i32;
        let minute = ((total_seconds - hour as f64 * 3600.0) / 60.0) as i32;
        let second = (total_seconds - hour as f64 * 3600.0 - minute as f64 * 60.0) as i32;

        self.solar_position(
            year, month, day, hour, minute, second, delta_t, latitude, longitude, elevation,
        )
    }

    /// Compute solar zenith angle (degrees) at a given UTC fractional hour.
    ///
    /// Convenience method matching the SPA pipeline interface.
    #[allow(clippy::too_many_arguments)] // Calendar components + observer location are all independent
    pub fn zenith_at_hour(
        &mut self,
        year: i32,
        month: i32,
        day: i32,
        fractional_hour: f64,
        delta_t: f64,
        latitude: f64,
        longitude: f64,
        elevation: f64,
    ) -> Result<f64, De440Error> {
        Ok(self
            .solar_position_at_hour(
                year,
                month,
                day,
                fractional_hour,
                delta_t,
                latitude,
                longitude,
                elevation,
            )?
            .zenith)
    }

    /// Find the UTC fractional hour when the solar zenith angle crosses
    /// a target value. Uses bisection search.
    ///
    /// Mirrors `spa::find_zenith_crossing()` for API compatibility.
    ///
    /// `target_zenith`: target zenith angle in degrees
    /// `start_hour`, `end_hour`: search range (fractional hours, local UTC)
    /// `tolerance`: convergence tolerance in fractional hours
    #[allow(clippy::too_many_arguments)] // Calendar components + observer location + search params are all independent
    pub fn find_zenith_crossing(
        &mut self,
        year: i32,
        month: i32,
        day: i32,
        target_zenith: f64,
        start_hour: f64,
        end_hour: f64,
        tolerance: f64,
        delta_t: f64,
        latitude: f64,
        longitude: f64,
        elevation: f64,
    ) -> Result<Option<f64>, De440Error> {
        let z_start = self.zenith_at_hour(
            year, month, day, start_hour, delta_t, latitude, longitude, elevation,
        )?;
        let z_end = self.zenith_at_hour(
            year, month, day, end_hour, delta_t, latitude, longitude, elevation,
        )?;

        // Check if a crossing exists
        let sign_start = z_start > target_zenith;
        let sign_end = z_end > target_zenith;

        if sign_start == sign_end {
            return Ok(None); // no crossing in this interval
        }

        let mut lo = start_hour;
        let mut hi = end_hour;

        while (hi - lo) > tolerance {
            let mid = (lo + hi) / 2.0;
            let z_mid = self.zenith_at_hour(
                year, month, day, mid, delta_t, latitude, longitude, elevation,
            )?;

            let sign_mid = z_mid > target_zenith;
            if sign_mid == sign_start {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        Ok(Some((lo + hi) / 2.0))
    }

    /// Get the number of segments in the loaded SPK file.
    pub fn segment_count(&self) -> usize {
        self.spk.segments().len()
    }

    /// Get coverage range (TDB seconds past J2000) for Sun-Earth data.
    pub fn sun_earth_coverage(&self) -> Option<(f64, f64)> {
        // Find the Sun wrt SSB segment (usually has the broadest coverage)
        self.spk
            .segments()
            .iter()
            .find(|s| s.target == SUN)
            .map(|s| (s.start_epoch, s.end_epoch))
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn get_de440_path() -> Option<String> {
        let path = std::env::var("DE440_PATH")
            .unwrap_or_else(|_| "/Users/mostafamahdi/twilight/data/de440.bsp".to_string());

        if std::path::Path::new(&path).exists() {
            Some(path)
        } else {
            None
        }
    }

    #[test]
    #[ignore]
    fn test_de440_open() {
        let path = match get_de440_path() {
            Some(p) => p,
            None => {
                eprintln!("DE440 not available, skipping");
                return;
            }
        };

        let de = De440::open(&path).expect("failed to open DE440");
        println!("DE440 loaded: {} segments", de.segment_count());

        if let Some((start, end)) = de.sun_earth_coverage() {
            let start_jd = earth_rotation::tdb_seconds_to_jd(start);
            let end_jd = earth_rotation::tdb_seconds_to_jd(end);
            println!("Sun coverage: JD {:.1} to JD {:.1}", start_jd, end_jd);
        }
    }

    #[test]
    fn test_de440_moon_distance_physical() {
        let path = match get_de440_path() {
            Some(p) => p,
            None => {
                eprintln!("DE440 not available, skipping");
                return;
            }
        };
        let mut de = De440::open(&path).expect("failed to open DE440");
        // Geocentric lunar distance stays within perigee..apogee bounds.
        for (y, m, d) in [(2026, 1, 3), (2026, 6, 12), (2024, 3, 20)] {
            let jd = earth_rotation::calendar_to_jd(y, m, d, 0, 0, 0);
            let tdb = earth_rotation::jd_to_tdb_seconds(
                earth_rotation::utc_jd_to_tdb_jd(jd, 69.2),
            );
            let p = de.moon_position_icrf(tdb).expect("moon position");
            let r = libm::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
            assert!(
                (356_000.0..407_000.0).contains(&r),
                "{y}-{m}-{d}: lunar distance {r:.0} km out of physical range"
            );
        }
    }

    #[test]
    fn test_de440_moon_agrees_with_meeus() {
        // The truncated Meeus series is good to ~0.3 deg geocentric; the
        // topocentric Meeus path adds a simplified parallax. DE440 must
        // agree with it within those error bars at several epochs - any
        // larger discrepancy means a frame/chain bug, not series error.
        let path = match get_de440_path() {
            Some(p) => p,
            None => {
                eprintln!("DE440 not available, skipping");
                return;
            }
        };
        let mut de = De440::open(&path).expect("failed to open DE440");
        for (y, m, d, h) in [
            (2026, 1, 3, 22.0),
            (2026, 1, 18, 5.0),
            (2026, 6, 12, 1.5),
            (2025, 9, 7, 19.0),
        ] {
            let jpl = de
                .moon_state(y, m, d, h, 69.2, 55.65, 12.41, 0.0)
                .expect("de440 moon state");
            let meeus = crate::moon::moon_state(y, m, d, h, 55.65, 12.41);
            assert!(
                (jpl.altitude_deg - meeus.altitude_deg).abs() < 1.0,
                "{y}-{m}-{d} {h}h: altitude JPL {:.2} vs Meeus {:.2}",
                jpl.altitude_deg,
                meeus.altitude_deg
            );
            // Azimuth comparison only away from the zenith (degenerate there)
            if jpl.altitude_deg.abs() < 80.0 {
                let daz = (jpl.azimuth_deg - meeus.azimuth_deg + 540.0).rem_euclid(360.0) - 180.0;
                assert!(
                    daz.abs() < 1.5,
                    "{y}-{m}-{d} {h}h: azimuth JPL {:.2} vs Meeus {:.2}",
                    jpl.azimuth_deg,
                    meeus.azimuth_deg
                );
            }
            assert!(
                (jpl.illuminated_fraction - meeus.illuminated_fraction).abs() < 0.03,
                "{y}-{m}-{d} {h}h: illum JPL {:.3} vs Meeus {:.3}",
                jpl.illuminated_fraction,
                meeus.illuminated_fraction
            );
            // JPL distance is TOPOCENTRIC, Meeus is geocentric: they may
            // differ by up to one Earth radius (parallax) plus the Meeus
            // series error.
            assert!(
                (jpl.distance_km - meeus.distance_km).abs() < 6371.0 + 1500.0,
                "{y}-{m}-{d} {h}h: distance JPL {:.0} vs Meeus {:.0}",
                jpl.distance_km,
                meeus.distance_km
            );
        }
    }

    #[test]
    fn test_de440_moon_known_phases() {
        // 2026-01-03 full moon, 2026-01-18 new moon (UTC) - the real
        // ephemeris must reproduce the almanac.
        let path = match get_de440_path() {
            Some(p) => p,
            None => {
                eprintln!("DE440 not available, skipping");
                return;
            }
        };
        let mut de = De440::open(&path).expect("failed to open DE440");
        let full = de.moon_state(2026, 1, 3, 12.0, 69.2, 0.0, 0.0, 0.0).unwrap();
        assert!(full.illuminated_fraction > 0.97, "full: {:?}", full);
        let new = de.moon_state(2026, 1, 18, 12.0, 69.2, 0.0, 0.0, 0.0).unwrap();
        assert!(new.illuminated_fraction < 0.05, "new: {:?}", new);
    }

    #[test]
    fn test_solar_position_at_hour_rolls_civil_day() {
        let path = match get_de440_path() {
            Some(p) => p,
            None => {
                eprintln!("DE440 not available, skipping");
                return;
            }
        };
        let mut de = De440::open(&path).unwrap();
        let (lat, lon, elev, dt) = (21.4225, 39.8262, 277.0, 69.184);

        // Negative UTC hour (timezone east of Greenwich) = previous civil day.
        let neg = de
            .solar_position_at_hour(2024, 6, 15, -2.5, dt, lat, lon, elev)
            .unwrap();
        let prev = de
            .solar_position(2024, 6, 14, 21, 30, 0, dt, lat, lon, elev)
            .unwrap();
        assert!((neg.zenith - prev.zenith).abs() < 1e-9);
        assert!((neg.azimuth - prev.azimuth).abs() < 1e-9);

        // Hour >= 24 = next civil day.
        let over = de
            .solar_position_at_hour(2024, 6, 15, 25.5, dt, lat, lon, elev)
            .unwrap();
        let next = de
            .solar_position(2024, 6, 16, 1, 30, 0, dt, lat, lon, elev)
            .unwrap();
        assert!((over.zenith - next.zenith).abs() < 1e-9);
        assert!((over.azimuth - next.azimuth).abs() < 1e-9);

        // Seconds are preserved: 12h + 90s is not the same as 12h.
        let with_sec = de
            .solar_position_at_hour(2024, 6, 15, 12.0 + 90.0 / 3600.0, dt, lat, lon, elev)
            .unwrap();
        let exact = de
            .solar_position(2024, 6, 15, 12, 1, 30, dt, lat, lon, elev)
            .unwrap();
        assert!((with_sec.zenith - exact.zenith).abs() < 1e-9);
    }

    #[test]
    #[ignore]
    fn test_de440_sun_position_j2000() {
        let path = match get_de440_path() {
            Some(p) => p,
            None => return,
        };

        let mut de = De440::open(&path).unwrap();
        let sun_pos = de.sun_position_icrf(0.0).unwrap();

        println!(
            "Sun wrt Earth at J2000.0: [{:.3}, {:.3}, {:.3}] km",
            sun_pos[0], sun_pos[1], sun_pos[2]
        );

        let dist = (sun_pos[0].powi(2) + sun_pos[1].powi(2) + sun_pos[2].powi(2)).sqrt();
        println!("Distance: {:.3} km = {:.6} AU", dist, dist / 149_597_870.7);

        assert!(
            dist > 145_000_000.0 && dist < 155_000_000.0,
            "Sun-Earth distance at J2000 should be ~1 AU, got {} km",
            dist
        );
    }

    #[test]
    #[ignore]
    fn test_de440_solar_position_topocentric() {
        let path = match get_de440_path() {
            Some(p) => p,
            None => return,
        };

        let mut de = De440::open(&path).unwrap();

        // 2024-03-20 12:00 UTC (near vernal equinox)
        // Mecca: 21.4225°N, 39.8262°E, 277m
        let topo = de
            .solar_position(2024, 3, 20, 12, 0, 0, 69.184, 21.4225, 39.8262, 277.0)
            .unwrap();

        println!("Solar position at Mecca, 2024-03-20 12:00 UTC:");
        println!("  Zenith:    {:.4}°", topo.zenith);
        println!("  Azimuth:   {:.4}°", topo.azimuth);
        println!("  Elevation: {:.4}°", topo.elevation);
        println!("  Distance:  {:.0} km", topo.distance_km);
        println!("  RA:        {:.4}°", topo.right_ascension);
        println!("  Dec:       {:.4}°", topo.declination);

        // At Mecca noon UTC (15:00 local), Sun should be high
        // Zenith should be reasonable (0-90 range for daytime)
        assert!(
            topo.zenith > 0.0 && topo.zenith < 90.0,
            "zenith {} not in daytime range",
            topo.zenith
        );
    }

    #[test]
    #[ignore]
    fn test_de440_vs_spa_comparison() {
        use crate::spa::{self, SpaInput};

        let path = match get_de440_path() {
            Some(p) => p,
            None => return,
        };

        let mut de = De440::open(&path).unwrap();

        // Test location: Mecca, 2024-06-15 12:00 UTC
        let lat = 21.4225;
        let lon = 39.8262;
        let elev = 277.0;
        let delta_t = 69.184;

        // DE440
        let de440_topo = de
            .solar_position(2024, 6, 15, 12, 0, 0, delta_t, lat, lon, elev)
            .unwrap();

        // SPA
        let spa_input = SpaInput {
            year: 2024,
            month: 6,
            day: 15,
            hour: 12,
            minute: 0,
            second: 0,
            timezone: 0.0,
            latitude: lat,
            longitude: lon,
            elevation: elev,
            delta_t,
            ..Default::default()
        };
        let spa_output = spa::solar_position(&spa_input).unwrap();

        println!("\nDE440 vs SPA comparison (2024-06-15 12:00 UTC, Mecca):");
        println!(
            "  Zenith:  DE440={:.6}°  SPA={:.6}°  diff={:.6}°",
            de440_topo.zenith,
            spa_output.zenith,
            (de440_topo.zenith - spa_output.zenith).abs()
        );
        println!(
            "  Azimuth: DE440={:.6}°  SPA={:.6}°  diff={:.6}°",
            de440_topo.azimuth,
            spa_output.azimuth,
            (de440_topo.azimuth - spa_output.azimuth).abs()
        );

        // The difference should be small (< 0.01 degrees ideally, but
        // our simplified Earth rotation may introduce up to ~0.05°)
        let zenith_diff = (de440_topo.zenith - spa_output.zenith).abs();
        assert!(
            zenith_diff < 1.0,
            "zenith difference too large: {} degrees",
            zenith_diff
        );
    }

    // ── Horizons validation tests ──────────────────────────────────

    /// Validate Sun position at J2000.0 against JPL Horizons DE441 output.
    ///
    /// Horizons query: Sun (10) wrt Earth (399), geocentric, ICRF, TDB.
    /// Reference: JD 2451545.0 (2000-01-01 12:00:00 TDB)
    ///   X =  2.649903367743050E+07 km
    ///   Y = -1.327574173383451E+08 km
    ///   Z = -5.755671847054072E+07 km
    ///
    /// Tolerance: 1 km (DE440 vs DE441 differences are < 0.01 km)
    #[test]
    #[ignore]
    fn test_horizons_sun_position_j2000() {
        let path = match get_de440_path() {
            Some(p) => p,
            None => return,
        };

        let mut de = De440::open(&path).unwrap();
        let pos = de.sun_position_icrf(0.0).unwrap();

        let horizons_x = 2.649_903_367_743_05e7;
        let horizons_y = -1.327574173383451e8;
        let horizons_z = -5.755671847054072e7;

        let dx = (pos[0] - horizons_x).abs();
        let dy = (pos[1] - horizons_y).abs();
        let dz = (pos[2] - horizons_z).abs();
        let dr = (dx * dx + dy * dy + dz * dz).sqrt();

        println!("J2000.0 Sun position validation:");
        println!(
            "  X: DE440={:.6}  Horizons={:.6}  diff={:.6} km",
            pos[0], horizons_x, dx
        );
        println!(
            "  Y: DE440={:.6}  Horizons={:.6}  diff={:.6} km",
            pos[1], horizons_y, dy
        );
        println!(
            "  Z: DE440={:.6}  Horizons={:.6}  diff={:.6} km",
            pos[2], horizons_z, dz
        );
        println!("  Total position error: {:.6} km", dr);

        // DE440 vs DE441 should agree to < 1 km
        assert!(dr < 1.0, "position error {} km exceeds 1 km tolerance", dr);
    }

    /// Validate Sun position at 2024-03-18 12:00 TDB against Horizons.
    ///
    /// Reference: JD 2460388.0 (2024-03-18 12:00:00 TDB)
    ///   X =  1.488262271675500E+08 km
    ///   Y = -4.653065462862011E+06 km
    ///   Z = -2.017521535139262E+06 km
    #[test]
    #[ignore]
    fn test_horizons_sun_position_2024() {
        let path = match get_de440_path() {
            Some(p) => p,
            None => return,
        };

        let mut de = De440::open(&path).unwrap();

        // TDB seconds past J2000 for JD 2460388.0
        let tdb = (2_460_388.0 - 2_451_545.0) * 86400.0;
        let pos = de.sun_position_icrf(tdb).unwrap();

        let horizons_x = 1.488_262_271_675_5e8;
        let horizons_y = -4.653065462862011e6;
        let horizons_z = -2.017521535139262e6;

        let dx = (pos[0] - horizons_x).abs();
        let dy = (pos[1] - horizons_y).abs();
        let dz = (pos[2] - horizons_z).abs();
        let dr = (dx * dx + dy * dy + dz * dz).sqrt();

        println!("2024-03-18 Sun position validation:");
        println!(
            "  X: DE440={:.6}  Horizons={:.6}  diff={:.6} km",
            pos[0], horizons_x, dx
        );
        println!(
            "  Y: DE440={:.6}  Horizons={:.6}  diff={:.6} km",
            pos[1], horizons_y, dy
        );
        println!(
            "  Z: DE440={:.6}  Horizons={:.6}  diff={:.6} km",
            pos[2], horizons_z, dz
        );
        println!("  Total position error: {:.6} km", dr);

        // Near-contemporary epoch: DE440 vs DE441 agree very well
        assert!(dr < 1.0, "position error {} km exceeds 1 km tolerance", dr);

        // Verify distance is ~1 AU (near equinox, ~149.6M km)
        let dist = (pos[0].powi(2) + pos[1].powi(2) + pos[2].powi(2)).sqrt();
        assert!(
            (dist / 149_597_870.7 - 1.0).abs() < 0.02,
            "distance {} AU not near 1 AU",
            dist / 149_597_870.7
        );
    }

    /// Validate Sun-Earth distance at perihelion and aphelion.
    #[test]
    #[ignore]
    fn test_sun_earth_distance_perihelion_aphelion() {
        let path = match get_de440_path() {
            Some(p) => p,
            None => return,
        };

        let mut de = De440::open(&path).unwrap();

        // 2024 perihelion: ~Jan 3 (0.9833 AU)
        let tdb_jan3 = earth_rotation::calendar_utc_to_tdb(2024, 1, 3, 0, 0, 0, 69.184);
        let pos_peri = de.sun_position_icrf(tdb_jan3).unwrap();
        let dist_peri = (pos_peri[0].powi(2) + pos_peri[1].powi(2) + pos_peri[2].powi(2)).sqrt();
        let au_peri = dist_peri / 149_597_870.7;

        // 2024 aphelion: ~Jul 5 (1.0167 AU)
        let tdb_jul5 = earth_rotation::calendar_utc_to_tdb(2024, 7, 5, 0, 0, 0, 69.184);
        let pos_aph = de.sun_position_icrf(tdb_jul5).unwrap();
        let dist_aph = (pos_aph[0].powi(2) + pos_aph[1].powi(2) + pos_aph[2].powi(2)).sqrt();
        let au_aph = dist_aph / 149_597_870.7;

        println!("Perihelion 2024: {:.6} AU ({:.0} km)", au_peri, dist_peri);
        println!("Aphelion 2024:   {:.6} AU ({:.0} km)", au_aph, dist_aph);

        // Perihelion: ~0.983 AU
        assert!(
            au_peri > 0.980 && au_peri < 0.986,
            "perihelion distance {} AU out of range",
            au_peri
        );

        // Aphelion: ~1.017 AU
        assert!(
            au_aph > 1.014 && au_aph < 1.020,
            "aphelion distance {} AU out of range",
            au_aph
        );

        // Aphelion > Perihelion
        assert!(
            au_aph > au_peri,
            "aphelion {} should be > perihelion {}",
            au_aph,
            au_peri
        );
    }

    /// Test multiple epochs across a year to verify consistency.
    #[test]
    #[ignore]
    fn test_de440_vs_spa_multiple_epochs() {
        use crate::spa::{self, SpaInput};

        let path = match get_de440_path() {
            Some(p) => p,
            None => return,
        };

        let mut de = De440::open(&path).unwrap();

        // Test at the start of each month in 2024, noon UTC, Mecca
        let lat = 21.4225;
        let lon = 39.8262;
        let delta_t = 69.184;

        let mut max_diff = 0.0f64;

        for month in 1..=12 {
            let de_topo = de
                .solar_position(2024, month, 15, 12, 0, 0, delta_t, lat, lon, 0.0)
                .unwrap();

            let spa_input = SpaInput {
                year: 2024,
                month,
                day: 15,
                hour: 12,
                minute: 0,
                second: 0,
                timezone: 0.0,
                latitude: lat,
                longitude: lon,
                delta_t,
                ..Default::default()
            };
            let spa_output = spa::solar_position(&spa_input).unwrap();

            let diff = (de_topo.zenith - spa_output.zenith).abs();
            if diff > max_diff {
                max_diff = diff;
            }

            println!(
                "  2024-{:02}-15: DE440={:.4}° SPA={:.4}° diff={:.4}°",
                month, de_topo.zenith, spa_output.zenith, diff
            );
        }

        println!("Max zenith difference across 12 months: {:.4}°", max_diff);

        // The difference should be consistently small (< 0.1 degrees)
        // The main source of error is our simplified precession/nutation.
        assert!(
            max_diff < 0.1,
            "max zenith difference {} exceeds 0.1 degrees",
            max_diff
        );
    }
}
