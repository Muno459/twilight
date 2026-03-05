# twilight-solar

Implementation of the NREL Solar Position Algorithm (SPA) following Reda & Andreas (2003), NREL/TP-560-34302.

## Accuracy

±0.0003 degrees for the period -2000 to 6000 CE.

## What it computes

Given a date, time, timezone, and observer location:

- Topocentric solar zenith angle (the key output)
- Topocentric azimuth (clockwise from north)
- Sun declination and right ascension
- Earth-Sun distance in AU
- Equation of time
- Approximate sunrise, sunset, and solar noon

## Implementation

The full VSOP87 planetary theory (heliocentric longitude, latitude, radius) with 5th-order polynomial series. 63 nutation terms for longitude and obliquity corrections. Topocentric parallax correction for observer elevation. Meeus atmospheric refraction approximation.

Key internal functions:

- `julian_day`. calendar date to JD, handles Julian/Gregorian transition
- `earth_heliocentric_longitude`. VSOP87 L0..L5 series evaluation
- `nutation`. 63-term nutation in longitude and obliquity
- `solar_position`. full pipeline, returns `SpaOutput` with all intermediate values
- `find_zenith_crossing`. binary search for the time when SZA crosses a target (used to convert threshold angles back to clock time)

## Data

`spa_tables.rs` contains the VSOP87 periodic term tables (L0..L5, B0..B1, R0..R4), nutation Y/psi/epsilon coefficient tables, and obliquity polynomial coefficients. This is about 2,500 lines of constant data.

## Tests

47 tests: NREL reference case (Oct 17, 2003 Golden CO), Julian day edge cases, declination at solstices/equinoxes, Earth-Sun distance at perihelion/aphelion, equation of time bounds, midnight sun and polar night detection, input validation, zenith crossing search, and time formatting.
