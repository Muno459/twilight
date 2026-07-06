# twilight-skyglow

Light pollution skyglow model. Estimates how artificial lighting adds to the natural twilight sky brightness, shifting the apparent onset and end of twilight.

## Status

The Garstang-style RT integration in `garstang` now feeds the prayer-time
pipeline in ONE specific role: `slant_brightness` (the slant-LOS
generalization of the zenith integral) supplies the AZIMUTHAL STRUCTURE of
the khayt skyglow veil through `DirectionalSkyglow` / `directional_veils`,
with per-patch veils normalized so their all-azimuth mean equals the
isotropic atlas-rail value (structure from Garstang+VIIRS, amplitude from
the Falchi/atlas calibration). Its ABSOLUTE magnitudes remain unvalidated
against published skyglow measurements and are never consumed directly;
the amplitude rail stays on the zenith-luminance estimate
(`quick_estimate_at_angle`).

## Approach

The model follows Garstang (1986, 1989) radiative transfer for skyglow. Light emitted upward from ground sources scatters off atmospheric molecules (Rayleigh) and aerosols (Mie/HG) back toward the observer. The scattered radiance depends on source distance, emission angle, aerosol optical depth, and wavelength.

The spectral dimension matters: LED streetlights (strong blue peak at 450nm) scatter far more efficiently via Rayleigh than HPS sodium lamps (narrow yellow peak at 589nm). A city that has converted to LED produces a bluer, brighter skyglow dome for the same total lumen output.

## Modules

**`garstang`**. Core RT computation. `zenith_brightness` integrates scattered radiance from a set of ground light sources at given distances and fluxes; `slant_brightness` generalizes the same integral to an arbitrary azimuth/elevation line of sight (unit-gated to reduce to the zenith integral at elevation 90). Rayleigh and Mie scattering with wavelength-dependent optical depth profiles. `bin_sources` aggregates distributed radiance (e.g. from VIIRS satellite data) into discrete source bins by distance and azimuth. Slant optical depth computation through the aerosol and molecular layers.

**`spectrum`**. Spectral lamp profiles. LED emission at 3000K, 4000K, and 5000K color temperatures (blue peak + phosphor broadband). HPS emission (narrow 589nm sodium line + broadband). Mixed spectra with configurable LED fraction. Blue-light fraction and Rayleigh scattering effectiveness metrics for each lamp type.

**`angular`**. Directional skyglow variation. Azimuthal enhancement near bright sources (city centers), zenith-to-horizon brightness gradient, and twilight observation geometry factors. The enhancement decays with angular distance from the source and increases toward the horizon.

**`bortle`**. Bortle Dark-Sky Scale (1-9) mapping. Converts between Bortle class, zenith luminance (mcd/m2), sky quality meter readings (mag/arcsec2), naked-eye limiting magnitude, and VIIRS nighttime radiance (nW/cm2/sr). Includes a rough heuristic for the prayer-time shift from zenith luminance; this heuristic is uncalibrated and disconnected from the RT pipeline - use the full pipeline (skyglow radiance added before threshold crossing) for any real estimate.

## Usage

The main entry point is `quick_estimate_at_angle`, which takes a VIIRS-equivalent radiance, LED fraction, and elevation angle, and returns a `SkyglowResult` with spectral radiance, Bortle class, zenith brightness, and blue-light fraction.

For azimuthally-resolved veils, `dnb::DnbGrid` serves a whole VIIRS Black Marble night grid as a `RadianceSource`, `DirectionalSkyglow::from_radiance_source` bins it into ground sources around the observer, and `directional_veils` returns per-azimuth (mesopic, red) veils on the same photometric rail as the isotropic path.

The CLI exposes this via `--bortle <class>` or `--skyglow` (with `--radiance` for direct VIIRS input), plus `--skyglow-directional` for the per-patch khayt veils.

## Tests

Garstang RT (zenith brightness vs distance, source additivity, flux scaling, AOD dependence, empty/zero cases), spectral profiles (LED blue peak, HPS sodium peak, mixed interpolation, Rayleigh effectiveness, blue fraction bounds), angular model (azimuthal decay, horizon enhancement, twilight factor), Bortle mapping (monotonicity, luminance/SQM/NELM conversions, roundtrips, prayer shift estimation).
