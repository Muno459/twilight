# twilight-data

Embedded atmospheric data tables and the atmosphere builder. All data is compiled into the binary. no runtime file I/O, no network requests.

## Modules

**`atmosphere_profiles`**. US Standard Atmosphere 1976 profiles. Temperature, pressure, and number density as functions of altitude from 0–100 km. Ozone number density profile with peak near 22 km. Linear interpolation between grid points. Correctly implements the tropospheric lapse rate (-6.5 K/km) and stratospheric isothermal layer.

Reference values at sea level:
- Temperature: 288.15 K
- Pressure: 1013.25 hPa
- Number density: 2.547e25 molecules/m³

**`ozone_xsec`**. Ozone absorption cross-sections from Serdyuchenko et al. (2014) at 293 K. 41 wavelength bins from 380–780 nm at 10 nm spacing. Covers the Chappuis band (peak near 600 nm) and tail of the Huggins band (below 400 nm). Linear interpolation for intermediate wavelengths.

**`solar_spectrum`**. TSIS-1 Hybrid Solar Reference Spectrum (HSRS). Spectral solar irradiance at top of atmosphere in W/m²/nm. 41 bins, 380–780 nm. Peak near 450 nm. Used to weight MCRT radiance into physical units.

**`builder`**. Constructs `AtmosphereModel` from the profile data. `build_clear_sky` creates a 50-shell atmosphere with Rayleigh scattering and ozone absorption at each shell and wavelength. Supports optional cloud layer insertion with configurable optical depth, base/top altitude, single-scattering albedo, and asymmetry parameter.

## Tests

64 tests covering profile reference values, monotonicity, interpolation, cross-section validation, solar spectrum integration, builder shell geometry, wavelength grids, extinction properties, and cloud layer effects.
