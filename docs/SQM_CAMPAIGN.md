# SQM Field-Calibration Campaign

This is the protocol for the project's #1 open task: end-to-end field
validation of the twilight radiative-transfer engine against real sky
measurements. Everything the engine computes funnels into one number,
the zenith sky brightness through the night, and a Sky Quality Meter
measures exactly that number for about 200 EUR. Anyone can run this
protocol; no knowledge of the codebase is required beyond the two CLI
commands shown below.

The deliverable of the campaign is a table of "engine minus meter"
offsets binned by solar depression angle. That table is the direct
calibration input for the engine's twilight thresholds.

## 1. Hardware

| Item | Model | Cost | Notes |
|---|---|---|---|
| Sky Quality Meter | Unihedron SQM-LU-DL | ~250 EUR | USB, built-in datalogger, runs unattended on a battery pack. Preferred. |
| or | Unihedron SQM-LE | ~150-250 EUR | Ethernet, needs a powered host (Raspberry Pi works) logging once per minute. |
| Optional second unit | SQM-LU-DL + long-pass red filter (>600 nm cut-on, e.g. a 2-inch RG610/OG590 photographic or astronomy filter) | ~250 EUR + 30-80 EUR | The red channel. See section 7. |
| Weatherproof housing | Unihedron housing or any vented enclosure with a clean glass/acrylic window | 0-50 EUR | Mind condensation; a small desiccant pack helps. |
| Battery/power | USB power bank or 12 V + converter | | One full night per charge minimum. |

Both meters measure sky surface brightness in mag/arcsec^2 over a ~20
degree (LU) full-width half-maximum cone. The narrower LU/LU-DL beam is
preferred over the older L model because the prediction is for the
zenith point.

The optional red-filtered unit is the single highest-value addition:
calibrated measurements of the RED twilight channel (shafaq al-ahmar,
the Shafi'i/Maliki/Hanbali Isha criterion) are the weakest dataset in
the entire published literature. Even 5 nights of red-channel decay
curves would be a novel contribution.

## 2. Mounting

- Point the meter at the ZENITH (straight up), level within a few
  degrees (check with a bubble level on the housing).
- Unobstructed sky: no roof edges, trees, or masts inside roughly 40
  degrees of zenith.
- No direct artificial light on the front lens: not under a street
  lamp, no porch lights, no passing headlights. Skyglow scattered in
  the atmosphere is fine (the engine models it); direct illumination of
  the optics is not.
- Fix the mount so it does not move between nights. Re-aiming between
  nights adds scatter that looks like model error.

## 3. Site characterization (once per site)

Record in a site log, kept with the data:

- Coordinates (5 decimal places) and elevation.
- Your Bortle class estimate (1-9), plus the basis for it (naked-eye
  limiting magnitude, Milky Way visibility). If you can, also note the
  VIIRS radiance for the site from a light-pollution map; pass it to
  the CLI as `--radiance` for a more precise skyglow input than the
  Bortle class.
- Horizon photos: 8 compass directions plus one straight up through
  the meter's aperture. These document obstructions and light domes.
- Meter serial number(s), firmware, filter (if any), and the housing
  window in front of the lens (an extra window dims readings by
  ~0.1 mag; note it so the offset is attributable).

## 4. Cadence and night selection

- Log at 1 reading per minute, from before sunset to after sunrise.
  The interesting structure (twilight decay, threshold crossings) is
  minutes wide; 1/min resolves it with margin.
- Target at least 10 CLEAR nights. The comparison is only meaningful
  when the sky is cloud-free or near it; even thin cirrus moves twilight
  brightness by more than the effects being calibrated.
- Mix the nights deliberately:
  - at least 4 moonless nights (moon below horizon or < 10% illuminated)
  - at least 3 moonlit nights (the engine's Krisciunas-Schaefer moon
    model needs validation too; a full moon raises the floor 10-100x)
  - the rest as they come.
- Note air quality if anything unusual (dust event, smoke, high humidity).

## 5. The run matrix (per night)

Before each night, generate the prediction:

```
twilight-cli sqm predict --lat 54.82 --lon 9.36 --date 2026-06-13 \
    --bortle 4 --out night_2026-06-13_pred.csv
```

- `--date` is the evening the night STARTS; the curve runs from local
  sunset to the next sunrise.
- Use `--weather` to fold in the live aerosol/cloud forecast, or omit
  it for the US Standard clear-sky baseline (run both if you want to
  separate weather error from model error).
- Skyglow: pass the site's `--radiance X` (preferred) or `--bortle N`,
  or `--skyglow` for the satellite atlas value.
- The summary printed alongside the CSV gives the predicted darkest
  magnitude and the predicted twilight-end / dawn-start times (the
  points where the curve is within 0.1 mag of the night floor).

After the night, retrieve the meter log and compare:

```
twilight-cli sqm compare --lat 54.82 --lon 9.36 --date 2026-06-13 \
    --bortle 4 --log sqm_2026-06-13.dat
```

The log is read in either the native Unihedron format (semicolon
separated, `#` headers, UTC timestamp first, magnitude last) or a plain
2-column CSV `timestamp_iso,mag`; the format is autodetected. Bare
timestamps are interpreted as UTC. The command exits nonzero if fewer
than 10 readings align with the predicted night, so it can gate a
scripted pipeline.

Keep per night: the raw meter log, the prediction CSV, the compare
report, and a one-line weather/moon/cloud note. Repeat `sqm compare`
with `--weather` variants later as needed; the raw log is the asset.

## 6. What each comparison closes

The compare report's central table is the offset (sim minus measured,
in magnitudes) binned by solar depression:

| Bin | What it tests | What a bias means |
|---|---|---|
| 0-6 deg | Bright twilight: MCRT single/multiple scattering, aerosol model | Radiative-transfer or aerosol error; should be near zero on clear nights |
| 6-12 deg | The Isha/Fajr working range: scattering at large SZA + the onset of the celestial floor | This is the money bin: a bias here moves prayer times directly, ~1 depression-degree per several 0.1 mag at these depths |
| 12-18 deg | Deep twilight: multiple scattering dominates, celestial background significant | Tests the hardest RT regime and the airglow/zodiacal model jointly |
| 18+ deg | Night floor: airglow + zodiacal + starlight + moon + skyglow only | Pure celestial-background and skyglow calibration; the MCRT plays no role here |

Two distinct calibrations come out of the campaign:

1. ABSOLUTE SCALE AT TWILIGHT DEPTHS. The depression-binned bias is a
   direct measurement of how much the engine's predicted sky brightness
   is off at each twilight depth, with the instrument as the standard.
   This is independent of any human observer and closes the "is the
   physics right" question.

2. THRESHOLD VS DETECTION. The engine's prayer times come from a
   contrast-detection model on top of the brightness curve. SQM data
   alone cannot test that layer; combining the SQM curve with HUMAN dawn
   observations from the same site and night does. The instrument tells
   you what the sky did; the observer tells you when the eye called it.
   The gap between them, expressed in depression degrees, is exactly
   the quantity the khayt edge factors encode.

### One-page observer protocol (for calibration nights with a human observer)

- Pick a morning with a clear eastern horizon. Know the dawn azimuth
  in advance (the `pray` command prints solar geometry; roughly
  northeast in summer, southeast in winter at mid-northern latitudes).
- Be in position 60+ minutes before the conventional 18-degree Fajr
  time. Dark-adapt for at least 20 minutes before any judgment: no
  phone, no flashlight, no car headlights. If you must log on paper,
  use a deep-red light kept away from the eyes.
- Look toward the dawn azimuth, scanning roughly 20 degrees either
  side, at 3-10 degrees above the horizon.
- Record two times, to the minute, by a clock checked against NTP/GPS:
  - FIRST DISTINCTNESS: the first moment a whitish brightening along
    the horizon is unmistakably there (you would bet on it; not "maybe").
  - SPREAD: the moment the brightening is clearly wide, spanning tens of
    degrees of azimuth rather than a narrow vertical wedge. The
    narrow-wedge-only state is the false dawn (al-fajr al-kadhib); the
    wide band is the true dawn (tabayyun in 2:187).
- Also record: observer age (pupil size and scotopic sensitivity vary),
  eyeglasses, cloud at the horizon, and any stray light incidents that
  broke dark adaptation.
- Do not look at the SQM display or any prediction before recording;
  the report must be blind to be usable.

## 7. The red channel (optional second meter)

Mount the red-filtered unit beside the primary, same aiming. The
filtered readings are not absolute (the filter eats a fixed number of
magnitudes), but the SHAPE of the red decay curve and the TIME its
slope flattens to the night floor measure the disappearance of shafaq
al-ahmar instrumentally. Calibrate the filter offset on a moonless
night floor against the unfiltered unit, then compare the red curve's
flattening time against the engine's Isha (al-ahmar) prediction.
Literature data for this channel is nearly nonexistent; treat these
logs as publishable.

## 8. Feeding results back into the engine

The engine constants this campaign calibrates live in
`crates/twilight-cpu/src/khayt.rs` (`KhaytParams::default`), and each
documents its current provenance in comments there:

- `edge_factor_appearance` (currently 45.0): the morning
  edge-discernibility factor, pinned today to the desert campaigns
  (KACST 14.6 +- 0.3, Hail 14.0 +- 0.3, Aswan camera 14.9 deg). Your
  observer first-distinctness/spread times, combined with the
  same-night measured SQM curve, re-derive this factor with a known
  instrument-grade brightness scale under it.
- `edge_factor_disappearance` (currently 4.0): the evening factor,
  pinned to the classical 17 deg white-shafaq convention and SQM
  twilight-end statistics (17.99 +- 0.16 deg).
- `k_contrast` / `k_contrast_red` (0.4): extended-source contrast
  multipliers; the red channel data constrains `k_contrast_red`.

Procedure:

1. If the 18+ deg bin shows a bias: fix the background first. Check the
   skyglow input, then the airglow level (F10.7 of the night vs the 130
   sfu default used by `sqm predict`). Nothing else can be calibrated
   while the floor is off.
2. If the 6-18 deg bins show a bias on clear nights: that is an
   RT/atmosphere discrepancy. Re-run the comparison with `--weather`
   and with `--scattering hybrid` before touching anything; record the
   residual bias in the night log.
3. With the brightness scale verified (or its bias known and recorded),
   convert each observer report to a solar depression angle, compute
   the band/reference contrast the engine predicts at that instant, and
   adjust the edge factors so the predicted detection matches the
   observed times. Update the constants AND their justification
   comments in khayt.rs; the out-of-sample check documented there
   (Padborg/UK June mornings should land at OpenFajr's summer 12.3-12.7
   deg without retuning) must still pass.

## 9. Quick checklist

- [ ] Meter zenith-mounted, level, unobstructed, no direct lamps
- [ ] Site log: coordinates, Bortle estimate, VIIRS radiance, horizon photos
- [ ] Logging at 1/min, clock NTP/GPS-checked
- [ ] `sqm predict` run and saved before the night
- [ ] Night is clear; moon state noted
- [ ] Observer protocol followed on human-calibration mornings (20 min dark adaptation, blind, two times recorded)
- [ ] `sqm compare` run and report saved after the night
- [ ] 10+ clear nights accumulated (4+ moonless, 3+ moonlit)
