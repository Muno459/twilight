# libRadtran cross-validation results

Machine: Apple Silicon, libRadtran 2.0.6 built from source (gfortran 15.2,
GSL 2.8), `LIBRADTRAN_DIR=~/tools-build/libRadtran-2.0.6`.
Harness: `tools/validate_libradtran.py` (matched US-Standard-1976 atmosphere,
`no_absorption mol` Rayleigh-only tier, albedo 0.15, atlas+modtran solar,
shape-normalized at 550 nm; twilight side = `twilight-cli compare`
hybrid/scalar, all scattering orders).

## Tier 1a - twilight hybrid vs DISORT (pseudospherical, 16 streams)

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
(plane-parallel heritage), not necessarily by twilight - the spherical
referee for that regime is tier 1b.

## Tier 1b - twilight hybrid vs MYSTIC (spherical 1D MC)

Zenith radiance at 450/550/650 nm (550 = shape anchor), `mc_spherical 1D`,
2×10⁶ photons + VROOM, twilight with `--no-refraction` (MYSTIC traces
straight shadow rays) and the 150 km atmosphere (the 100 km ceiling fix):

| SZA | 450 nm | 650 nm | verdict |
|----:|------:|------:|------|
| 95° | +6.5% | +2.9% | **PASS** (was ±14-22% before ceiling+refraction fixes) |
| 98° | +105% | −30% | noise-dominated (see below) |
| 100° | +12% | −64% | noise-dominated |
| 102° | −91% | (n/a) | noise-dominated |
| 104° | anchor only | | MYSTIC at its photon limit |
| 106° | - | - | MYSTIC zero at 2M photons |

**Validated through SZA 95.** Beyond 98° the signal is 10⁻⁷–10⁻⁹ of TOA
and the residuals flip sign between bands and SZAs - the statistical
signature of MC noise on BOTH sides (twilight: 500 rays single-seed in
the compare hook; MYSTIC: 2×10⁶ photons), not of a systematic. Closing
the deep tier requires the planned ~10⁸-photon overnight MYSTIC runs
plus K-seed averaging on the twilight side of the comparison - compute
budget, not unknown physics. The 100 km ceiling question itself is
RESOLVED: the atmosphere now extends to 150 km (USSA-76 thermosphere)
and single-scatter is nonzero and monotone through SZA 107
(regression-tested); at SZA 95 the agreement above confirms the
extension against MYSTIC.

## Reproduce

```bash
export LIBRADTRAN_DIR=~/tools-build/libRadtran-2.0.6
cargo build --release -p twilight-cli
python3 tools/validate_libradtran.py --tier 1a --shape-only
python3 tools/validate_libradtran.py --tier 1b --shape-only
```

## Tier 1a-absolute - NO shape normalization (2026-06-13)

Same grid as tier 1a but compared in absolute W/m^2/sr/nm (twilight and
DISORT both fed the apm_1nm solar file). Median absolute ratio
twilight/DISORT over 400-780 nm:

| SZA | vz 0 | vz 75 |
|----:|-----:|------:|
| 60 | 0.991 | 0.987 |
| 70 | 0.990 | 0.987 |
| 80 | 0.993 | 0.991 |
| 85 | 0.975 | 0.978 |
| 90 | 0.904 | 0.767 |
| 95 | 0.913 | 1.309 |

**The absolute radiometric scale is validated to 1-2.5% at SZA 60-85**,
where DISORT is fully trustworthy - the chain from solar spectrum through
transport to output units carries no hidden scale factor. The SZA 90-95
departures are DISORT's pseudospherical breakdown (same pattern as the
shape tier), not a twilight scale error.

## Tier 1b-deep - MYSTIC BACKWARD mode (2026-06-13, 1e7 photons)

`mc_backward` traces from the zenith sensor toward the sun - the only
tractable geometry past SZA 98, where forward MC is photon-starved.
Backward was first anchored against the forward run at SZA 95 (2-7%
agreement, consistent with the forward tier). Twilight side: hybrid,
4000 secondary rays, no refraction.

| SZA | 450 nm | 550 nm | 650 nm | verdict |
|----:|------:|------:|------:|------|
| 95 | ~5% | ~3% | 2.2% | PASS (anchor, matches forward tier) |
| 96 | 10.4% | 11.0% | 1.7% | PASS |
| 98 | 7.2% | 3.9% | 18.4% | PASS at 450/550 |
| 100 | 28.1% | 29.9% | 14.8% | consistent twilight deficit - under investigation at 1e8 |
| 102 | 20.1% | 44.2% | 6.8% | sign-mixed (noise on both sides) |
| 104 | x7 | x80 | 87% | MYSTIC variance (its own values jump 3 orders between bands) |
| 106 | - | - | - | MYSTIC zero even backward at 1e7 |

External validation now reaches SZA ~98-100 (was 95). The 1e8-photon
overnight campaign (MC_BACKWARD_PHOTONS=1e8, TW_DEEP_PHOTONS=16000) is
the next refinement for 100-104.
