# The khayt appearance edge factor: three independent attacks on the one tuned number

> **Frame note (2026-08-12).** This document analyses the constant in the
> pre-hyperaccuracy transport frame, where its value was 45. The production
> value is now **56** (`crates/twilight-cpu/src/khayt.rs`), reached in two
> steps: a three-site cluster protocol that selected 70, then the
> 2026-07-07 full-campaign refit to 56.5 (RMS 0.133 deg, leave-one-out mean
> 56.4, range 55.2-57.0) after the 70 was found to carry a 0.3-0.4 deg
> out-of-sample bias under the leverage of the Hail outlier. The structural
> findings below (cross-site invariance, no latitude trend) are properties
> of the inversion and are what the README cites; the absolute figures
> quoted here (geometric mean 41.6 against a calibrated 45.0) belong to the
> old frame and do not transfer. Re-run
> `python tools/confirm_edge_factor.py` for the current-frame ladder.

The khayt criterion carries exactly one calibrated psychophysics
constant: `KhaytParams::edge_factor_appearance = 45.0`, applied on
`k_contrast = 0.4`, i.e. an **effective 18x** multiplier over the
Blackwell TVI disc threshold encoded in threshold.rs. It was calibrated
ONCE against the Mecca/desert cluster and then survived Birmingham
(out of sample, full year) and a Padborg observation. Psychophysics is
the softest layer of the engine, so this document attacks the constant
three independent ways:

1. **Implied-factor inversion** (the decisive one): for every
   khayt-matched naked-eye campaign, measure the engine's
   factor-to-depression response and invert the published observation
   into the factor that would exactly reproduce it. A tight cluster
   pins the constant; scatter or a latitude trend falsifies it.
2. **Literature decomposition**: can published psychophysics build the
   18x out of named, cited components, or is it an unexplained fudge?
3. **Zodiacal cross-prediction**: the same detection machinery drives
   the false-dawn (kadhib) verdict on the zodiacal wedge, which was
   never tuned; its skyglow cutoff is checked against the published
   Bortle-scale zodiacal-light visibility limit.

Machinery: `tools/criterion_edge_factor.py` (runner + analysis; raw
engine stdout cached under `validation/criterion_runs/edge_factor/`).
The engine gained a calibration-analysis knob for this study:
`TWILIGHT_KHAYT_EDGE_APPEARANCE` overrides the appearance factor in
`KhaytParams::default()` (khayt.rs, documented there, unit-tested,
production semantics unchanged when unset). All runs release build,
hybrid SS+MC, GPU (Metal), SZA step 0.5, defaults otherwise.

## 0. A finding on the way: HEAD has drifted from the calibration engine

The plan was to reuse the pristine-sweep cache for the factor-45
points. A fresh-binary verification row stopped that: HEAD
(b776776 + working tree) does not reproduce the criterion-sites sweep.
Same site, same date, same factor 45, both GPU deterministic:

| Site (date) | f45 HEAD | f45 pristine cache | drift |
|---|---|---|---|
| riyadh_desert 2004-10-15 | 14.99 | 14.44 | +0.55 |
| hail 2015-01-15 | 14.65 | 14.46 | +0.19 |
| sinai 2011-04-15 | 13.99 | 14.84 | -0.85 |
| fayum 2018-12-09 | 15.04 | 14.73 | +0.31 |
| matrouh 2012-04-15 | 13.99 | 14.13 | -0.14 |
| kottamia 2010-04-15 | 15.01 | 14.51 | +0.50 |
| bahariya 2007-10-15 | 14.98 | 14.79 | +0.19 |
| wadi_natrun 2017-01-15 | 13.87 | 13.58 | +0.29 |
| tubruq 2010-01-15 | 15.17 | 14.82 | +0.35 |
| sanaa 2003-11-24 | 15.29 | 14.84 | +0.45 |
| birmingham 2015-04-20 | 15.30 | 14.73 | +0.57 |
| birmingham 2015-09-23 | 15.01 | 14.87 | +0.14 |
| birmingham 2015-06-30 | 12.67 | 12.50 | +0.17 |
| **mean** | | | **+0.21 deg** [-0.85..+0.57] |

This is radiative-transfer drift (the b776776 hyperaccuracy merge and
subsequent in-flight work), not psychophysics: the celestial
backgrounds printed by both engines agree to <1%, MC sigma is ~0.02
deg, and the drift is site-dependent (not a global offset). Two
consequences for this analysis: (a) every ladder point, factor 45
included, was rerun fresh on one pinned binary (single-engine ladders);
(b) implied factors are reported in two frames below. The drift itself
is a real, quantified warning: **the transport under the criterion has
moved by a mean +0.21 deg since calibration, which is the same order
as the calibration residuals.** If HEAD's transport is the better one
(the merge claims hyperaccuracy), the psychophysics constant should be
recalibrated against the same campaigns on the frozen HEAD engine; the
calibration-frame analysis below says by how much (spoiler: from 45 to
~50, a shift smaller than the cross-site spread).

## 1. Attack 1: implied-factor inversion across 13 campaigns

### Method

- For each khayt-matched naked-eye campaign of
  RESULTS_CRITERION_SITES.md: one date (the pristine-sweep date whose
  factor-45 khayt sits closest to the site's factor-45 seasonal mean),
  engine run at edge factors 25 / 45 / 60 / 80 (hail also 35 as a
  log-linearity probe; sanaa and the Birmingham June row 25/45/80).
- Response curves dep(log10 f) are close to log-linear away from
  cliffs (median |slope| ~2.2 deg/dex); inversion is monotone
  piecewise-linear between bracketing ladder points, log-linear fit
  only for extrapolation.
- Seasonal adjustment: campaign values are multi-year means, the
  ladder is one date, so the inversion target is
  `obs + (old-engine f45 at date - old-engine site seasonal mean)`,
  a differential that is drift-insensitive to first order.
- Two frames: **HEAD** (invert on the fresh curve directly) and
  **CALIBRATION** (shift each site's target by its own measured f45
  drift, recovering the inversion against the engine the 45 was tuned
  on). The calibration frame is the clean universality test: engine
  drift cancels per site.

### Response ladders (khayt fajr depression, deg)

| Site | date | f25 | f35 | f45 | f60 | f80 |
|---|---|---|---|---|---|---|
| riyadh_desert | 2004-10-15 | 15.44 | | 14.99 | 14.78 | 14.61 |
| hail | 2015-01-15 | 15.11 | 15.02 | 14.65 | 14.37 | 13.99 |
| sinai | 2011-04-15 | 15.59 | | 13.99 | 13.94 | 13.97 |
| fayum | 2018-12-09 | 15.30 | | 15.04 | 14.93 | 14.16 |
| matrouh | 2012-04-15 | 14.90 | | 13.99 | 13.96 | 14.00 |
| kottamia | 2010-04-15 | 15.49 | | 15.01 | 13.99 | 13.96 |
| bahariya | 2007-10-15 | 15.63 | | 14.98 | 14.62 | 13.98 |
| wadi_natrun | 2017-01-15 | 14.28 | | 13.87 | 13.56 | 13.31 |
| tubruq | 2010-01-15 | 15.42 | | 15.17 | 14.71 | 14.35 |
| sanaa | 2003-11-24 | 15.67 | | 15.29 | | 14.69 |
| birmingham_apr | 2015-04-20 | 15.59 | | 15.30 | 15.00 | 14.36 |
| birmingham_sep | 2015-09-23 | 15.40 | | 15.01 | 14.70 | 14.64 |
| birmingham_jun | 2015-06-30 | 12.86 | | 12.67 | | 12.45 |
| birmingham_dec (--skyglow, mesopic veil) | 2015-12-10 | 13.00 | | 12.30 | | 11.76 |

Reproducibility spot check: hail f45 run twice independently gives
14.65 both times (deterministic GPU seeding).

Structural note, stated honestly: at sinai, matrouh, and kottamia the
response SATURATES near 13.94-14.0 deg for f >= 45-60 (a margin-curve
cliff; the same class as the Birmingham June turnaround). Those sites
constrain the factor mostly from the deep (small-factor) side: sinai's
observation is compatible with ANY f >= ~40, and its quoted implied
factor is the crossing of the steep branch.

### Implied factors

obs = campaign naked-eye angle; CI from the campaign's published SD
where one exists, propagated through the local response slope.

| Site | lat N | obs (deg) | implied factor, HEAD frame | implied factor, CALIBRATION frame |
|---|---|---|---|---|
| riyadh_desert (KACST) | 25.8 | 14.6 +- 0.3 | 80.0 | 36.0 [24..54] |
| hail | 27.5 | 14.01 +- 0.32 | 71.2 | 61.7 [45..79] |
| north sinai | 31.1 | 14.61 | 32.7 | 44.7 (>= ~40 plateau) |
| fayum (wadi al hitan) | 29.3 | 14.8 | 41.1 | 26.0 |
| matrouh | 31.0 | 14.5 | 34.1 | 37.3 |
| kottamia | 29.9 | 14.66 +- 0.2 | 49.4 | 36.7 [29..45] |
| bahariya | 28.7 | 14.6 | 55.7 | 47.8 |
| wadi al-natrun | 30.5 | 14.57 | 24.7 | 17.5 (extrapolated; the sweep's known outlier) |
| tubruq desert | 32.1 | 14.68 | 68.0 | 53.2 |
| sanaa (Sultan bracket) | 15.5 | 13.2 < tabayyun < 18.8 | consistent | bounds [0.6..492]: consistent, non-constraining |
| birmingham apr 20 (panel 15.0) | 52.4 | 15.0 | 60.0 | 26.0 |
| birmingham sep 23 (panel 14.6) | 52.4 | 14.6 | 78.6 | 57.8 (extrapolated just past f80) |
| birmingham dec 10 (panel 12.9, measured-atlas veil) | 52.4 | 12.9 | 27.2 | n/a (veil model changed between engines; HEAD frame only) |
| birmingham jun 30 (panel 12.3) | 52.4 | 12.3 | factor-INSENSITIVE (see below) | same |

The dec-glow row is the only VEILED inversion and it is veil-model
sensitive: under the superseded photopic-as-mesopic veil the same
panel value inverted to ~61, under the corrected mesopic veil to 27.
Both straddle 45; the row is a consistency check on the skyglow path,
not a pin on the psychophysics constant, and it stays out of the
distribution statistics.

Distribution over the 8 desert-campaign sites with in-ladder
inversions (wadi al-natrun's extrapolated outlier excluded, listed
separately):

```
factor:   15    20     30     40  45  50    60    70    80
          |-----|------|------|---+---|-----|-----|-----|   (log axis)
wadi_natrun    17.5*                                        * extrapolated
fayum                 26.0
birmingham_apr        26.0p                                 p panel row
riyadh                       36.0
kottamia                     36.7
matrouh                       37.3
sinai                             44.7  (plateau: >=~40)
bahariya                              47.8
tubruq                                   53.2
birmingham_sep                              57.8p*
hail                                          61.7
                                 ^ calibrated 45
```

- **CALIBRATION frame, desert n=8: geometric mean 41.6, spread 0.117
  dex = x/1.31, range 26..62.** The calibrated 45 sits within 1
  standard error of the ensemble mean (SE 0.041 dex; geo-mean CI
  37.8..45.8). In effective-multiplier terms: 16.6x with 1-SE band
  15.1..18.3, calibrated 18.0 inside.
- **No latitude trend: Pearson r(log10 f, lat) = +0.07** across
  25.8..32.1 N (and the two non-plateau Birmingham panel rows at
  52.4 N straddle 45 from both sides, 26 and 58). Universality is not
  falsified. The HEAD frame shows r = -0.59, but that trend is
  manufactured by the site-dependent transport drift of section 0,
  not by psychophysics: it disappears in the calibration frame.
- **The cross-site spread is observation-noise-sized.** 0.117 dex
  through a median response slope of ~2.2 deg/dex is 0.26 deg of
  depression, the same size as the campaigns' own quoted SDs
  (0.2..0.32 deg); the per-site implied-factor CIs from those SDs
  (x/1.24 .. x/1.5) cover the cluster spread (x/1.31). There is no
  evidence for intrinsic site-to-site psychophysics variation beyond
  observational noise.
- **With the outlier included** (n=9): geo-mean 37.8, x/1.47. Wadi
  al-Natrun (implied 17.5) was already the pristine sweep's one
  unexplained miss (-0.74 deg, single published number, no
  uncertainty, no background story); it is 2.4 sigma from the cluster
  in log-factor.
- **HEAD frame, desert n=8: geo-mean 51.4, x/1.41.** If HEAD's
  transport is adopted as the better one, the constant should be
  recalibrated to ~50 (equivalently effective ~20x); that shift
  (0.09 dex) is smaller than the cross-site spread and within the
  Attack 2 bracket, i.e. the psychophysics claim is stable under the
  transport revision.

### Leave-one-out: recalibrate on each single desert site

For each site: adopt its implied factor as if IT had been the sole
calibration (as Mecca once was), predict all other desert sites, and
report the maximum prediction movement (deg) and the residuals against
the seasonally adjusted observations. Calibration frame.

| Calibrated on | implied f | max shift elsewhere (deg) | mean abs delta (deg) | max abs delta (deg) |
|---|---|---|---|---|
| riyadh_desert | 36.0 | 0.60 | 0.35 | 0.64 |
| hail | 61.7 | 1.02 | 0.52 | 1.19 |
| sinai | 44.7 | 0.01 | 0.30 | 0.73 |
| fayum | 26.0 | 1.49 | 0.61 | 1.47 |
| matrouh | 37.3 | 0.51 | 0.33 | 0.61 |
| kottamia | 36.7 | 0.56 | 0.34 | 0.62 |
| bahariya | 47.8 | 0.22 | 0.32 | 0.80 |
| wadi_natrun | 17.5 | 1.85 | 0.98 | 1.83 |
| tubruq | 53.2 | 0.59 | 0.38 | 0.92 |

Reading: recalibrating the whole criterion on any single mainstream
desert campaign (factors 36..62) moves predictions at the other sites
by at most 0.2..1.0 deg and leaves the ensemble mean residual at
0.30..0.52 deg, versus ~0.30 deg at the 45 baseline. Only the two
extreme single-site calibrations (fayum 26, wadi_natrun 17.5) degrade
other sites past 1 deg. The single-anchor calibration history (Mecca
once, then frozen) was therefore not lucky: almost any of these
campaigns used alone would have produced a criterion within ~0.5 deg
of the current one everywhere tested.

### Birmingham June: the trough does not depend on the tuned number

The compressed-twilight row (Jun 30, panel 12.3): engine 12.86 / 12.67
/ 12.45 at factors 25 / 45 / 80. Slope -0.8 deg/dex: a FACTOR OF TWO
change in the psychophysics constant moves the June verdict by ~0.24
deg. The celebrated June agreement (engine 12.5-12.7 vs panel
12.3-12.6, the UK scholars' hand rule) is transport + geometry at the
twilight floor cliff, not a fit artifact of the edge factor. The same
insensitivity protects the Padborg observation-confirmed Fajr.
Conversely this also means the June rows cannot pin the factor, and
they are excluded from the distribution above.

## 2. Attack 2: literature decomposition of the effective 18x

What the engine actually applies: threshold contrast
`18 x C_TVI(L_ref)` where C_TVI is the threshold.rs table (0.70 at
1e-4 cd/m^2, 0.35 at 1e-3, ... 0.017 photopic). At the desert
reference adaptation (2.2e-4..1e-3 cd/m^2) this makes the dawn band
distinct at an excess of ~6..10x the reference sky, matching the
khayt.rs docstring's calibration note (excess/L_ref ~ 10.8).

### The laboratory baseline, quantified

Crumey 2014 (MNRAS 442:2600, arXiv:1405.4209, read in full) is the
modern authority on Blackwell 1946 and its field application. The
Blackwell conditions (Crumey sec. 1.2): 19 highly trained observers,
~20/20, unconstrained BINOCULAR vision, effectively infinite viewing
time, forced choice among 8 positions (or yes/no at low light), 50%
detection probability, uniform sharp-edged achromatic discs (0.595
arcmin .. 6 deg), uniform backgrounds, target location and timing
known. Best case on every axis.

Evaluating Crumey's fits of Blackwell's own table 8 (his eqs. 23-44)
against the engine's TVI table at the relevant adaptations:

| adaptation (cd/m^2) | engine C_TVI | Blackwell 6-deg disc (50%, lab) | engine/lab | effective khayt (18 x C_TVI) | over lab |
|---|---|---|---|---|---|
| 2.2e-4 | 0.552 | 0.0700 | 7.9x | 9.94 | 142x |
| 3.5e-4 | 0.480 | 0.0603 | 8.0x | 8.64 | 143x |
| 1.0e-3 | 0.350 | 0.0426 | 8.2x | 6.30 | 148x |

So the full bridge from Blackwell's laboratory 50% threshold for a
6-deg disc to the engine's field tabayyun threshold is a stable
**~145x**, which the engine books in three layers: the TVI table is
itself a practical-visibility tabulation sitting **~8x** above raw
Blackwell at scotopic levels; `k_contrast 0.4` credits the band's
size; `edge_factor 45` carries the rest.

### Published components

| # | Component | Published value/range | Source |
|---|---|---|---|
| 1 | 50% forced-choice -> "confident of having seen" | **1.62** ("observers were confident of having seen the target only in cases where the resulting detection probability was 90 per cent or greater, corresponding to f = 1.62 ... thresholds should be multiplied by at least this much to give realistic values") | Blackwell 1952, quoted in Crumey sec. 1.2 |
| 2 | Forced-choice -> "common-sense seeing" (self-aware, adjustment method) | **2.4** | Blackwell & Blackwell 1971, quoted in Crumey sec. 1.2 |
| 3 | Twilight star visibility field factor (the closest classic to this task) | **l = 2** proposed; **1.814** fits exactly | Tousey & Hulburt 1948; Tousey & Koomen 1953 built the twilight-star charts on it; Crumey sec. 1.4 |
| 4 | Overall field factor for real naked-eye astronomical detection | **F = 1.4 .. 2.4 typical, F = 2 notional** (dark-sky limiting-magnitude surveys); F = 0.94 for a 7.0-mag exceptional observer | Crumey sec. 3.1 |
| 5 | Field factors for non-ideal targets/conditions (non-uniform target, non-uniform background, glare, observer state); age alone spans 1..6.92 | **multiplicative, task-specific, up to ~7 from age alone**; road-lighting practice requires Visibility Level **VL ~ 7** over threshold for reliable real-task detection | Taylor 1964 (the field-factor framework); Blackwell & Blackwell 1971 age multipliers; CIE/Adrian VL practice |
| 6 | Soft-edged (blurred) extended target: perceived contrast/detectability loss vs sharp edge | factor **~2** class (no canonical scotopic value; "blurred edges look faint") | Georgeson et al. 2007; khayt.rs already books ~2x edge clawback inside k=0.4 |
| 7 | Slowly brightening stimulus: no onset transient, no flicker enhancement | abrupt onset > gradual sensitivity in temporal CSF; slow ramps forfeit the transient channel (factor ~1.5..3 class) | temporal contrast sensitivity literature (de Lange / Kelly tradition) |
| 8 | Binocular summation sqrt(2) | **NOT claimable**: Blackwell's data are already binocular (Lythgoe & Phillips 1938: 1.4 C_binoc = 0.5(C_L + C_R)) | Crumey secs. 1.2, 1.6.4; flagged to prevent double-booking |

### Assembling the product, honestly

Layer A (the TVI table's ~8x over raw Blackwell): almost exactly
Blackwell's own two correction factors, 1.62 x 2.4 = 3.9, times the
Tousey-Hulburt twilight field allowance l ~ 1.8..2: 3.9 x 2 = 7.8 ~ 8.
The engine's "instrument-grade" table is thus a defensible practical
tabulation, not raw lab data.

Layer B (k_contrast 0.4): the size credit for a degrees-wide band vs
the table's disc rows (pure size 0.08-0.26 at scotopic per the
Blackwell size curves and Crumey's asymptote, clawed back ~2x by the
soft edge, component 6). Published and booked.

Layer C (edge_factor 45, equivalently the residual ~18x over the
table x k, or ~36x over Blackwell-with-confidence): the product of
components 4-7 for THIS task: a task-focused but real-world observer
(F ~ 1.4..2.4), judging a borderless gradient (2..5), growing slowly
with no transient (1.5..3), against a structured night reference
(zodiacal column, Milky Way; part of Taylor's non-uniform-background
factor, 1.5..3), to legal-certainty distinctness rather than 50%
detection (already partly in layer A; the road-lighting VL ~ 7
precedent shows real tasks demand multiples of threshold). Credible
product range: 1.4 x 2 x 1.5 x 1.5 ~ **6** (everything at its
kindest) to 2.4 x 5 x 3 x 3 ~ **108** (everything at its harshest),
center of mass ~ 2 x 3 x 2 x 2 ~ **24-40**.

**Verdict for Attack 2: the decomposition brackets the constant but
does not pin it.** The calibrated 45 (and the Attack 1 cluster 26..62)
sits inside the published bracket [~6, ~108] and close to its center
of mass; every layer of the engine's stack corresponds to a named,
cited effect, and the only sub-1 term (size) is booked at its
published value. But the honest width of the literature bracket is a
factor of ~4 either way. Psychophysics literature justifies ORDER
15-20x effective and rules out both ~2x (a dawn band would be
"visible" 2.5 deg deeper than any eye campaign reports; this is
exactly the documented instrument-vs-eye gap) and ~100x (fajr would
land shallower than every desert campaign). It cannot select 18 over
12 or 30; the field campaigns do that.

## 3. Attack 3: zodiacal-visibility cross-prediction (kadhib vs Bortle)

The false-dawn (kadhib) verdict uses the same contrast machinery
(central-patch margin on the zodiacal wedge, spread test failing) with
the same 18x appearance threshold; nothing about it was ever tuned
(the zodiacal wedge enters through the Leinert 1998 tables). Published
naked-eye facts to cross-predict (Bortle 2001, Sky & Telescope):

- Class 1-3: zodiacal light obvious to striking (casts shadows at 1).
- Class 4: "zodiacal light clearly evident but doesn't extend even
  halfway to the zenith".
- Class 5: "only hints of zodiacal light on the best nights".
- Class 6: "zodiacal light no longer visible".

So the published naked-eye cutoff is the Bortle 5 to 6 transition.

Engine test: Mecca coordinates, 2015-12-21 (the pristine run detects
kadhib), rising artificial skyglow via `--skyglow --radiance R`
(R in nW/cm^2/sr; the same Bortle radiance equivalents as
RESULTS_CRITERION_SITES.md sec. 9.3: R=2 ~ B4, R=6 ~ B5, R=15 ~ B6).
Engine: current HEAD binary with the corrected MESOPIC veil (the veil
is read from the calibrated skyglow spectrum's mesopic band, so veil
magnitudes here supersede the photopic-veil numbers in earlier notes).
Kadhib onset depressions from the printed onset clock time via an
independent NOAA solar-position check that reproduces the engine's
printed (time, depression) pairs to <=0.01 deg.

| veil (VIIRS nW) | Bortle | khayt sadiq dep | kadhib dep (stdout) | kadhib lead over sadiq (deg) |
|---|---|---|---|---|
| pristine | 1 | 15.08 | (margin-derived 15.69) | 0.48 |
| 0.5 | 2 | 14.24 | 14.71 | 0.46 |
| 2.0 | 4 | 14.03 | 14.29 | 0.39 |
| 6.0 | 5 | 13.58 | 14.05 | 0.36 |
| 15.0 | 6 | 13.20 | 13.62 | 0.33 |

The ladder (Mecca frame, 2015-12-21, factor fixed at the production
45, post-mesopic-veil engine): the kadhib lead over the sadiq, i.e.
how long the vertical zodiacal column is seen as a DISTINCT earlier
event, shrinks monotonically with the veil: 0.48 deg pristine, 0.46 at
Bortle 2, 0.39 at Bortle 4, 0.36 at Bortle 5, 0.33 at Bortle 6 (about
2 minutes of clock time). Directionally this matches the published
behavior (zodiacal light plainly visible at Bortle 1-3, lost by
Bortle 4-5): the distinct false-dawn phase compresses toward merger
with the true dawn as the urban veil rises. What the ladder does NOT
show is a hard extinction of the kadhib event by Bortle 5-6: the
engine still detects a (brief) central-first crossing there. That is
not necessarily wrong: the literature describes the free-standing
zodiacal cone against a fully dark sky, while the engine's kadhib
event rides immediately ahead of dawn where the wedge is brightest;
but the honest statement is that the cross-prediction is PARTIAL:
trend confirmed, cutoff untested by this observable.

### Attack 3 re-run at the production constant (2026-08-16)

The ladder above is the f = 45 frame. Re-run on the shipped engine
(f = 56 default, CPU reference path, same site/date/protocol via
`python tools/criterion_edge_factor.py --attack3 --cpu`), with the lead
quantified at the machinery level as the separation between the
central-patch and spread margin crossings (the stdout kadhib onset
applies an additional reporting gate and is not monotone at B6):

| veil (VIIRS nW) | Bortle | khayt sadiq dep | margin central dep | lead (deg) |
|---|---|---|---|---|
| pristine | 1 | 14.54 | 15.20 | 0.50 |
| 0.5 | 2 | 14.01 | 14.41 | 0.38 |
| 2.0 | 4 | 13.68 | 14.04 | 0.33 |
| 6.0 | 5 | 13.37 | 13.69 | 0.31 |
| 15.0 | 6 | 12.98 | 13.37 | 0.29 |

Monotone compression 0.50 -> 0.29, same qualitative verdict as the f=45
frame (trend confirmed, hard extinction not reproduced, cross-prediction
PARTIAL). NOAA time-to-depression self-check residuals <= 0.006 deg.
This is the ladder the application paper quotes.

Two tool fixes made for this re-run, both affecting anyone reproducing
on Windows: `CAL_FACTOR` updated 45 -> 56 (it must track the shipped
default, since run_one only sets the override env var when the factor
differs from it), and engine stdout is now decoded as UTF-8 explicitly
(the locale-codepage default mangled the degree sign and silently made
every depression parse as None).

## 4. Confidence verdict

> **Edge factor pinned by the field, bracketed by the literature,
> cross-validated on the zodiacal wedge.** Implied-factor inversion
> across 8 independent desert campaigns (25.8..32.1 N, data
> 1907..2019): geometric mean 41.6 vs calibrated 45.0, cross-site
> spread x/1.31 (0.117 dex ~ 0.26 deg, the size of the campaigns' own
> quoted uncertainties), **no latitude trend** (r = +0.07); calibrated
> 45 within 1 SE of the ensemble; effective multiplier 18x pinned to
> **16.6x, 1-SE band 15.1..18.3, site spread 12.7..21.8**. One known
> outlier (Wadi Al-Natrun, implied 17.5, the pristine sweep's
> unexplained miss). Leave-one-out: calibrating on any single
> mainstream campaign lands within 0.2..1.0 deg of the current
> criterion everywhere else. Decomposes as (Blackwell confident-seeing
> 1.62 x common-sense 2.4 ~ 3.9) x (twilight field factor ~2) x
> (soft-edge, slow-ramp, structured-background field factors ~6..25,
> Taylor/Crumey framework): published bracket ~[6, 108] effective,
> containing 18x near its center of mass; the literature alone pins
> only to a factor ~2-3. Zodiacal cross-prediction:
> Attack 3 verdict: PARTIAL support. The psychophysics layer reproduces
the monotone suppression of false-dawn distinctness with urban
skyglow (0.48 to 0.33 deg lead across Bortle 1 to 6), the qualitative
signature the visibility literature demands, on a phenomenon the
constant was never tuned on; it does not reproduce (and this
observable cannot cleanly test) the hard Bortle 4-5 visibility cutoff
of the free-standing zodiacal cone. Caveats that must ride along: HEAD transport
> has drifted +0.21 deg mean from the calibration engine (recalibrate
> to ~50 if HEAD is adopted, a sub-spread shift); three sites
> saturate at f >= 45-60 and bound the factor only from below; the
> compressed-twilight (June/high-latitude) verdicts are factor-
> insensitive (0.24 deg per factor of 2) and neither support nor
> constrain the constant.

## 5. Reproducibility

- `python3 tools/criterion_edge_factor.py` reruns Attack 1 (cache =
  raw stdout under `validation/criterion_runs/edge_factor/`; delete a
  file to force); `--analyze` re-parses without running;
  `--attack3` runs the zodiacal ladder. `TWILIGHT_CLI=<path>` pins the
  binary; the Attack 1 ladders in this document ran on a snapshot of
  HEAD b776776 + the khayt.rs knob (copied out of the tree so parallel
  rebuilds cannot change engines mid-sweep); the Attack 3 ladder and
  the Birmingham dec-glow ladder ran on the post-mesopic-veil current
  binary, as labeled.
- The env knob: `TWILIGHT_KHAYT_EDGE_APPEARANCE=<f>` (khayt.rs,
  `edge_appearance_override`, unit test
  `edge_appearance_env_override_is_picked_up`). Unset = production
  default 45.0.
- Superseded runs (dec-glow ladder under the earlier photopic-as-
  mesopic veil) are kept under
  `validation/criterion_runs/edge_factor/superseded_photopic_veil/`.
- Sources quoted in Attack 2: Crumey 2014 MNRAS 442:2600
  (arXiv:1405.4209) secs. 1.2, 1.4, 1.6.4, 3.1 (Blackwell 1946;
  Blackwell 1952; Blackwell & Blackwell 1971; Taylor 1960a/b, 1964;
  Lythgoe & Phillips 1938; Tousey & Hulburt 1948); Tousey & Koomen
  1953 JOSA 43:177; Knoll, Tousey & Hulburt 1946 JOSA 36:480;
  Georgeson et al. 2007 (blurred-edge contrast); CIE/Adrian visibility
  level practice (VL ~ 7); Bortle 2001 Sky & Telescope dark-sky scale.
