# twilight-solar

Solar position computation with two backends.

## Backends

**NREL SPA** (default). Implementation of the Solar Position Algorithm by Reda & Andreas (2003), NREL/TP-560-34302. Accuracy: +/-0.0003 degrees for the period -2000 to 6000 CE. Full VSOP87 planetary theory with 5th-order polynomial series. 63 nutation terms for longitude and obliquity corrections. Topocentric parallax correction for observer elevation. Meeus atmospheric refraction approximation.

**JPL DE440** (optional). Pure Rust reader for JPL Development Ephemeris binary SPK files. Parses the DAF (Double precision Array File) container format and evaluates Chebyshev polynomial position/velocity coefficients. Full IAU precession-nutation (Lieske 1979 + Wahr 1981) and ICRF-to-topocentric frame conversion including Earth rotation, geodetic-to-ECEF, and sidereal time. Validated to 8 meters against JPL Horizons for the Sun-Earth vector. Requires the ~114 MB `de440.bsp` data file (not included in the repository).

SPA is the automatic fallback when no DE440 file is available.

## What it computes

Given a date, time, timezone, and observer location:

- Topocentric solar zenith angle
- Topocentric azimuth (clockwise from north)
- Sun declination and right ascension
- Earth-Sun distance in AU
- Equation of time
- Approximate sunrise, sunset, and solar noon

## Key functions

- `solar_position`. Full pipeline, returns `SpaOutput` with all intermediate values.
- `find_zenith_crossing`. Binary search for the time when SZA crosses a target angle. Used to convert threshold depression angles back to clock time.
- `De440::open`. Opens a DE440 BSP file and provides `solar_position_topocentric` for high-precision queries.

## Data

`spa_tables.rs` contains the VSOP87 periodic term tables (L0..L5, B0..B1, R0..R4), nutation Y/psi/epsilon coefficient tables, and obliquity polynomial coefficients. About 2,500 lines of constant data.

## Tests

73 tests (63 run, 10 ignored). NREL reference case (Oct 17, 2003 Golden CO), Julian day edge cases, declination at solstices/equinoxes, Earth-Sun distance at perihelion/aphelion, equation of time bounds, midnight sun and polar night detection, input validation, zenith crossing search, time formatting, Chebyshev polynomial evaluation. DE440-specific tests (ignored without the BSP file): DE440 vs SPA comparison, DE440 vs Horizons validation, Sun-Earth distance at orbital extremes.
