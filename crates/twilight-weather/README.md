# twilight-weather

Live atmospheric conditions from the [Open-Meteo](https://open-meteo.com/) API. Free, no API key, no account required.

## What it fetches

Two API calls, both to Open-Meteo:

1. **Weather API**: temperature, surface pressure, visibility, cloud cover by altitude (low/mid/high/total), and AOD at 550nm from CAMS.
2. **Air Quality API**: surface O3 and NO2 concentrations in ug/m3 from CAMS global forecasts, plus dust concentration.

Both return current conditions for a given latitude/longitude.

## Modules

**`api`**. HTTP client using `ureq`. Constructs Open-Meteo URLs, parses JSON responses via `serde`, and returns raw `WeatherConditions` (AOD, visibility, cloud cover fractions, dust, O3, NO2) and `AirQualityCurrent` structs.

**`mapping`**. Converts raw weather observations into parameters the MCRT engine can use:

- *Aerosol properties*: AOD at 550nm mapped to extinction profile. Angstrom exponent inferred from dust concentration (low dust = fine urban/continental particles with high Angstrom; high dust = coarse desert particles with low Angstrom). Single-scattering albedo and asymmetry parameter selected by aerosol regime.
- *Cloud properties*: Cloud cover by altitude mapped to cloud layer type (stratus, stratocumulus, altostratus, cirrus) with optical depth scaled by fractional coverage. Low clouds take priority over high clouds when both are present.
- *Gas composition*: Surface O3 mapped to total column O3 in Dobson Units via empirical linear relationship (baseline 300 DU at 60 ug/m3, clamped to 220-450 DU). Surface NO2 converted to number density in molecules/m3 for tropospheric profile scaling.
- *Description*: Human-readable summary of conditions for CLI output.

## Tests

45 tests. API URL formatting, aerosol mapping (AOD thresholds, desert/urban/continental regimes, Angstrom ranges), cloud mapping (altitude priority, optical depth scaling, type selection), gas composition (O3 column estimation, O3 clamping, NO2 unit conversion, edge cases), and description formatting.
