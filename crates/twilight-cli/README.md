# twilight-cli

Command-line interface for the twilight engine. Three subcommands.

## `solar`

Solar position and conventional fixed-angle twilight times.

```bash
twilight solar --lat 21.4225 --lon 39.8262 --date 2024-03-20 --tz 3.0
```

Shows zenith, azimuth, declination, equation of time at noon, then a table of morning/evening times for all standard conventions (civil, nautical, astronomical, MWL 18, Egypt 15, Umm al-Qura 19.5, etc).

## `mcrt`

Run the MCRT engine directly and inspect spectral radiance and luminance across a range of solar zenith angles.

```bash
twilight mcrt --lat 21.4225 --lon 39.8262 --sza-start 90 --sza-end 108 --sza-step 2
```

Outputs two tables: spectral radiance at selected wavelengths (450, 550, 650, 700 nm), and luminance analysis (photopic, scotopic, mesopic, spectral centroid, twilight color classification).

## `pray`

Full physically-based prayer time computation.

```bash
twilight pray --lat 21.4225 --lon 39.8262 --date 2024-03-20 --tz 3.0
```

Runs the two-pass adaptive MCRT scan, reports Fajr, Isha al-abyad, and Isha al-ahmar times with their equivalent depression angles, and compares against all conventional fixed-angle methods. Add `--verbose` for the full SZA-by-SZA twilight analysis table.

## Options

All subcommands accept `--elevation` (meters) and `--delta-t` (TT-UT1 in seconds, default 69.184). The `pray` subcommand also takes `--albedo` (surface reflectance, default 0.15) and `--sza-step` (scan resolution in degrees, default 0.5).
