# twilight-threshold

Converts spectral radiance from the MCRT engine into physically-meaningful luminance values and determines Fajr/Isha prayer times by finding perceptual thresholds.

## Modules

**`vision`**. CIE standard observer functions. Photopic V(λ) (CIE 1924, 81 points, 380–780 nm at 5 nm) representing cone-mediated daylight vision with peak at 555 nm. Scotopic V'(λ) (CIE 1951) representing rod-mediated dark-adapted vision with peak at 507 nm. The Purkinje shift between these two functions is central to twilight perception.

**`luminance`**. Spectral integration of radiance weighted by vision functions. Photopic luminance (K_m = 683 lm/W), scotopic luminance (K'_m = 1700 lm/W), and CIE 2010 mesopic luminance that smoothly transitions between the two as light level decreases. Also computes:
- Red-band luminance (λ > 600 nm) for *shafaq al-ahmar* detection
- Blue-band luminance (λ < 500 nm)
- Spectral centroid (radiance-weighted mean wavelength) for color classification

**`threshold`**. The prayer time determination logic. `analyze_twilight` takes spectral radiance at a given SZA and produces a `TwilightAnalysis` with luminance values, spectral centroid, and color classification (Blue, White, Orange, Red, Dark). `determine_prayer_times` takes a sequence of analyses across SZA and finds threshold crossings:
- Fajr: mesopic luminance crosses upward through the Fajr threshold
- Isha al-abyad: mesopic luminance crosses downward (white glow disappears)
- Isha al-ahmar: red-band luminance crosses downward (red glow disappears)

Crossing interpolation uses logarithmic interpolation in luminance space for sub-step precision.

## Tests

72 tests covering vision function properties (peak values, ranges, Purkinje shift), luminance computation (scaling, spectral weighting, band isolation), color classification logic, threshold crossing interpolation, and end-to-end prayer time determination from synthetic spectra.
