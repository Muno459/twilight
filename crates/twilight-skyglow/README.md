# twilight-skyglow

Light pollution skyglow model. Estimates how artificial lighting adds to the natural twilight sky brightness, shifting the apparent onset and end of twilight.

## Approach

The model follows Garstang (1986, 1989) radiative transfer for skyglow. Light emitted upward from ground sources scatters off atmospheric molecules (Rayleigh) and aerosols (Mie/HG) back toward the observer. The scattered radiance depends on source distance, emission angle, aerosol optical depth, and wavelength.

The spectral dimension matters: LED streetlights (strong blue peak at 450nm) scatter far more efficiently via Rayleigh than HPS sodium lamps (narrow yellow peak at 589nm). A city that has converted to LED produces a bluer, brighter skyglow dome for the same total lumen output.

## Modules

**`garstang`**. Core RT computation. `zenith_brightness` integrates scattered radiance from a set of ground light sources at given distances and fluxes. Rayleigh and Mie scattering with wavelength-dependent optical depth profiles. `bin_sources` aggregates distributed radiance (e.g. from VIIRS satellite data) into discrete source bins by distance and azimuth. Slant optical depth computation through the aerosol and molecular layers.

**`spectrum`**. Spectral lamp profiles. LED emission at 3000K, 4000K, and 5000K color temperatures (blue peak + phosphor broadband). HPS emission (narrow 589nm sodium line + broadband). Mixed spectra with configurable LED fraction. Blue-light fraction and Rayleigh scattering effectiveness metrics for each lamp type.

**`angular`**. Directional skyglow variation. Azimuthal enhancement near bright sources (city centers), zenith-to-horizon brightness gradient, and twilight observation geometry factors. The enhancement decays with angular distance from the source and increases toward the horizon.

**`bortle`**. Bortle Dark-Sky Scale (1-9) mapping. Converts between Bortle class, zenith luminance (mcd/m2), sky quality meter readings (mag/arcsec2), naked-eye limiting magnitude, and VIIRS nighttime radiance (nW/cm2/sr). Estimates the prayer time shift in minutes from zenith luminance: negligible at dark sites (Bortle 1-3), several minutes in suburban areas (Bortle 5-6), significant in city centers (Bortle 8-9).

## Usage

The main entry point is `quick_estimate_at_angle`, which takes a VIIRS-equivalent radiance, LED fraction, and elevation angle, and returns a `SkyglowResult` with spectral radiance, Bortle class, zenith brightness, and blue-light fraction.

The CLI exposes this via `--bortle <class>` or `--skyglow` (with `--radiance` for direct VIIRS input).

## Tests

69 tests. Garstang RT (zenith brightness vs distance, source additivity, flux scaling, AOD dependence, empty/zero cases), spectral profiles (LED blue peak, HPS sodium peak, mixed interpolation, Rayleigh effectiveness, blue fraction bounds), angular model (azimuthal decay, horizon enhancement, twilight factor), Bortle mapping (monotonicity, luminance/SQM/NELM conversions, roundtrips, prayer shift estimation).
