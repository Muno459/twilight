# libRadtran cross-validation results

Machine: Apple Silicon, libRadtran 2.0.6 built from source (gfortran 15.2,
GSL 2.8), `LIBRADTRAN_DIR=~/tools-build/libRadtran-2.0.6`.
Harness: `tools/validate_libradtran.py` (matched US-Standard-1976 atmosphere,
`no_absorption mol` Rayleigh-only tier, albedo 0.15, atlas+modtran solar,
shape-normalized at 550 nm; twilight side = `twilight-cli compare`
hybrid/scalar, all scattering orders).

## Tier 1a — twilight hybrid vs DISORT (pseudospherical, 16 streams)

Sky radiance, 401 wavelengths × {zenith, 75°} view × {0°, 90°, 180°} relative
azimuth. Median / p90 relative error per SZA (2026-06-12 run):

| SZA | median | p90 |
|----:|------:|-----:|
| 60° | 2.8% | 8.2% |
| 70° | 2.7% | 8.2% |
| 80° | 2.7% | 8.4% |
| 85° | 2.6% | 8.1% |
| 90° | 3.5% | 11.6% |
| 95° | 14.2% | 44.6% |

**Conclusion:** twilight's full transport agrees with DISORT at the few-percent
level everywhere DISORT itself is trustworthy (≤90°). The SZA 95 row is
dominated by pseudospherical DISORT's own breakdown past the terminator
(plane-parallel heritage), not necessarily by twilight — the spherical
referee for that regime is tier 1b.

## Tier 1b — twilight hybrid vs MYSTIC (spherical 1D MC) — PRELIMINARY

Zenith radiance at 450/550/650 nm (550 = shape anchor), `mc_spherical 1D`,
2×10⁶ photons + VROOM:

| SZA | 450 nm | 650 nm | notes |
|----:|------:|------:|------|
| 95° | +14% | −8% | twilight slightly blue-rich vs MYSTIC |
| 98° | −19% | −55% | |
| 100° | +226% | −31% | sign flips ⇒ noise-dominated |
| 102° | +91% | (n/a) | MYSTIC partially photon-starved |
| 104° | anchor only | | MYSTIC reaches, barely |
| 106° | — | — | MYSTIC returns zero (needs ≥10⁸ photons) |

**Status: not yet conclusive.** Known systematics to resolve before this
tier can pass/fail the engine:
1. **Refraction**: twilight traces refracted shadow rays; this MYSTIC config
   does not — refraction matters increasingly past 96°.
2. **Profile granularity**: twilight's 41-level/100 km grid vs afglus
   (120 km, finer layers) — the deep-twilight signal comes from the
   high-altitude tail where the profiles differ most. This is also where
   the engine's 100 km ceiling question lives: MYSTIC (cap 120 km) still
   sees signal at 104° where twilight's own single-scatter is zero — the
   hybrid's MS term carries twilight there, consistent with the ceiling
   costing real signal at SZA ≥ 104°.
3. **MC noise on both sides**: twilight ran 500 rays/step single-seed in
   this comparison; MYSTIC at 2M photons is itself noisy below ~10⁻⁹.

Next steps (tracked): matched-profile run (twilight grid exported to
`atmosphere_file`), `mc_photons 1e8` overnight runs for SZA ≥ 102°,
refraction-off twilight runs for apples-to-apples, then absolute (non-shape)
comparison with a TSIS solar file on the uvspec side.

## Reproduce

```bash
export LIBRADTRAN_DIR=~/tools-build/libRadtran-2.0.6
cargo build --release -p twilight-cli
python3 tools/validate_libradtran.py --tier 1a --shape-only
python3 tools/validate_libradtran.py --tier 1b --shape-only
```
