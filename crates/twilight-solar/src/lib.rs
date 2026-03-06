//! Solar position algorithms for twilight computation.
//!
//! Provides two backends for computing solar zenith angle and azimuth:
//!
//! - **SPA** (NREL Solar Position Algorithm): Analytical, fast, no external data.
//!   Accuracy: +/-0.0003 deg for the period -2000 to 6000.
//!
//! - **DE440** (JPL Development Ephemeris 440): Highest precision, requires the
//!   DE440 BSP file (~97 MB). Accuracy: ~0.001 arcsecond (milliarcsecond).
//!   Coverage: 1550 to 2650 CE.
//!
//! The SPA backend is always available and serves as the default. The DE440
//! backend requires the `de440` feature flag and a path to the BSP file.

pub mod spa;
pub mod spa_tables;

pub mod de440;
pub mod earth_rotation;
pub mod spk;
