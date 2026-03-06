# twilight-cli

Command-line interface for the twilight engine. Three subcommands.

## `solar`

Solar position and conventional fixed-angle twilight times.

```bash
twilight-cli solar --lat 21.4225 --lon 39.8262 --date 2024-03-20 --tz 3.0
```

Shows zenith, azimuth, declination, equation of time at noon, then a table of morning/evening times for all standard conventions (civil, nautical, astronomical, MWL 18, Egypt 15, Umm al-Qura 19.5, etc). Add `--de440 data/de440.bsp` for JPL DE440 ephemeris comparison.

## `mcrt`

Run the MCRT engine directly and inspect spectral radiance and luminance across a range of solar zenith angles.

```bash
twilight-cli mcrt --lat 21.4225 --lon 39.8262 --sza-start 90 --sza-end 108 --sza-step 2
```

Outputs two tables: spectral radiance at selected wavelengths (450, 550, 650, 700 nm), and luminance analysis (photopic, scotopic, mesopic, spectral centroid, twilight color classification). Add `--polarized` for full Stokes vector output. Add `--weather` for live atmospheric conditions.

## `pray`

Full physically-based prayer time computation.

```bash
twilight-cli pray --lat 21.4225 --lon 39.8262 --date 2024-03-20 --tz 3.0
```

Runs the two-pass adaptive MCRT scan, reports Fajr, Isha al-abyad, and Isha al-ahmar times with their equivalent depression angles, and compares against all conventional fixed-angle methods. Add `--verbose` for the full SZA-by-SZA twilight analysis table.

## Key options

| Flag | Description |
|---|---|
| `--weather` | Live weather from Open-Meteo (AOD, clouds, O3/NO2) |
| `--terrain` | Terrain masking from Copernicus GLO-30 DEM |
| `--bortle <1-9>` | Light pollution (Bortle dark-sky class) |
| `--aerosol <type>` | Manual aerosol (continental-clean, urban, desert, ...) |
| `--cloud <type>` | Manual cloud (thin-cirrus, stratus, cumulus, ...) |
| `--scattering <mode>` | single, mc, or hybrid |
| `--photons <N>` | MC photons per wavelength per SZA step |
| `--polarized` | Full Stokes vector polarized RT |
| `--de440 <path>` | Use JPL DE440 ephemeris file |
| `--elevation <m>` | Observer elevation in meters |
| `--albedo <0-1>` | Surface reflectance (default 0.15) |
| `--tz <hours>` | Timezone offset from UTC |
| `--delta-t <sec>` | TT-UT1 in seconds (default 69.184) |
