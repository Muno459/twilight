# twilight-data

Embedded atmospheric data tables and the atmosphere builder. All data is compiled into the binary. No runtime file I/O, no network requests.

## Modules

**`atmosphere_profiles`**. US Standard Atmosphere 1976 profiles. Temperature, pressure, number density, and ozone density as functions of altitude from 0 to 100 km. Linear interpolation between grid points. Implements the tropospheric lapse rate (-6.5 K/km) and stratospheric isothermal layer.

Reference values at sea level:
- Temperature: 288.15 K
- Pressure: 1013.25 hPa
- Number density: 2.547e25 molecules/m3

**`ozone_xsec`**. Ozone absorption cross-sections from Serdyuchenko et al. (2014) at 293 K. 41 wavelength bins from 380 to 780 nm at 10 nm spacing. Covers the Chappuis band (peak near 600 nm) and tail of the Huggins band (below 400 nm). Note: the builder no longer uses this module for gas absorption (replaced by multi-temperature cross-sections in `twilight-core`). Retained for backward compatibility.

**`solar_spectrum`**. TSIS-1 Hybrid Solar Reference Spectrum (HSRS). Spectral solar irradiance at top of atmosphere in W/m2/nm. 41 bins, 380 to 780 nm. Peak near 450 nm.

**`aerosol`**. OPAC aerosol climatology. Six types: continental clean, continental average, urban, maritime clean, maritime polluted, desert. Each type provides AOD at 550nm, Angstrom exponent, single-scattering albedo, asymmetry parameter, and scale height. Extinction profile as a function of altitude via exponential decay. Also supports custom `AerosolProperties` with arbitrary parameters for weather-derived aerosols.

**`cloud`**. Cloud optical models. Six types: thin cirrus, thick cirrus, altostratus, stratus, stratocumulus, cumulus. Each provides base/top altitude, optical depth, SSA, and asymmetry parameter. Also supports custom `CloudProperties` for weather-derived clouds with fractional coverage scaling.

**`builder`**. Constructs `AtmosphereModel` from all of the above:

- `build_clear_sky`: 50-shell atmosphere with Rayleigh scattering only (no gas absorption). Base layer for adding further components.
- `build_full`: Rayleigh + gas absorption (all five species from `twilight-core`) + optional aerosol + optional cloud.
- `build_full_with_gas`: Same as `build_full` but accepts optional O3 column override (Dobson Units) and NO2 surface density override (molecules/m3) for live weather integration.

The gas absorption pipeline: `apply_gas_absorption_standard` builds a standard gas profile from US Std 1976, optionally scales the O3 column and NO2 profile to match observations, then calls `apply_gas_absorption` from `twilight-core` to fold absorption into shell optics.

## Tests

153 tests. Profile reference values, monotonicity, interpolation, cross-section validation, solar spectrum integration, builder shell geometry, wavelength grids, clear sky properties (SSA=1, Rayleigh fraction=1), gas absorption integration (extinction increase, SSA reduction, scattering coefficient preservation), all 6 aerosol types, all 6 cloud types, combined aerosol+cloud, `build_full_with_gas` with O3/NO2 overrides (6 tests).
