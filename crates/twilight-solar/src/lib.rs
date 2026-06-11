//! Solar position algorithms for twilight computation.
//!
//! Provides two backends for computing solar zenith angle and azimuth:
//!
//! - **SPA** (NREL Solar Position Algorithm): Analytical, fast, no external data.
//!   Accuracy: +/-0.0003 deg for the period -2000 to 6000.
//!
//! - **DE440** (JPL Development Ephemeris 440): requires the DE440 BSP file
//!   (~97 MB). The geometric ICRF solar position from the SPK reader is
//!   milliarcsecond-level vs JPL Horizons, but the delivered topocentric
//!   zenith/azimuth chain uses UT1≈UTC and simplified precession-nutation,
//!   so end-to-end accuracy is arcsecond-level (tens of arcseconds worst
//!   case) — far more than sufficient for prayer times.
//!   Coverage: 1550 to 2650 CE.
//!
//! The SPA backend is always available and serves as the default. The DE440
//! backend is selected at runtime by passing a path to the BSP file (there
//! is no cargo feature flag).

pub mod spa;
pub mod spa_tables;

pub mod de440;
pub mod earth_rotation;
pub mod spk;
