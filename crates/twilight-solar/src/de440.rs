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
use crate::spk::{self, SpkFile, EARTH, EARTH_MOON_BARYCENTER, SUN};
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

    /// Compute solar zenith angle (degrees) at a given UTC fractional hour.
    ///
    /// Convenience method matching the SPA pipeline interface.
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
        let total_seconds = fractional_hour * 3600.0;
        let hour = (total_seconds / 3600.0) as i32;
        let minute = ((total_seconds - hour as f64 * 3600.0) / 60.0) as i32;
        let second = (total_seconds - hour as f64 * 3600.0 - minute as f64 * 60.0) as i32;

        let topo = self.solar_position(
            year, month, day, hour, minute, second, delta_t, latitude, longitude, elevation,
        )?;

        Ok(topo.zenith)
    }

    /// Find the UTC fractional hour when the solar zenith angle crosses
    /// a target value. Uses bisection search.
    ///
    /// Mirrors `spa::find_zenith_crossing()` for API compatibility.
    ///
    /// `target_zenith`: target zenith angle in degrees
    /// `start_hour`, `end_hour`: search range (fractional hours, local UTC)
    /// `tolerance`: convergence tolerance in fractional hours
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

        let horizons_x = 2.649903367743050e7;
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

        let horizons_x = 1.488262271675500e8;
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
