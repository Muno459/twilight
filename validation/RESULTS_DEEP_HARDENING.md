# Deep-regime hardening: where the referee tier is actually limited

This document records three measurements made against the merged deep
tier (16 PASS, `deep_regime_results.csv`). None of them overturns a
published number; together they change what the tier is able to claim,
and they identify where the next compute should go.

Reproduce with `python tools/pool_mystic_replicas.py` and
`cargo test -p twilight-cpu --release -- --ignored --nocapture
g_s3_eq1d_paired`.

## 1. The referee, not the engine, sets the band width

The gate is

    |m - r| <= 3*sqrt(se_m^2 + se_r^2) + 0.05*r

with a cell counted as constraining only when that band is below `0.5*r`.
Decomposing the two error terms across all 16 cells:

| term | range across the tier |
|---|---|
| model relative SE (`se_m/m`) | 2.5 to 5.9 percent (one field cell 9.9) |
| referee relative SE (`se_r/r`) | **5.9 to 10.7 percent** |

In **15 of the 16 cells the MYSTIC reference is the dominant term**. The
single exception is tau* 3 / SZA 103 / 550 nm on the field path, where
the two are level (9.9 against 9.7).

The consequence is the practical one: additional model seeds barely move
the gate. Driving `se_m` to zero everywhere would only shrink the bands
from 0.25-0.46 to 0.12-0.35. Compute spent on the engine is compute
mostly wasted; the reference is the limiting instrument.

A second consequence is about what "16/16 PASS" means. The bands are 25
to 46 percent wide, so the tier currently establishes agreement at the
tens-of-percent level. The clear-sky DISORT anchor is a percent-level
statement. These are different classes of claim and should be worded
differently.

## 2. Pooling the replicas the campaign already produced

`artifacts/2026-07_deep_closure/mystic_replicas/` holds two fresh-seed
1e9-photon replicas for each of four tau* = 3 references. The cached run
and its replicas are three independent estimates of the same quantity, so
they pool by inverse variance:

| cell | n | referee SE was | pooled | chi2/dof |
|---|---|---|---|---|
| tau3 SZA101 450 | 3 | 6.4 % | 3.6 % | 0.29 |
| tau3 SZA101 550 | 3 | 6.1 % | 3.6 % | 0.63 |
| tau3 SZA103 550 | 3 | 9.7 % | 5.3 % | 0.53 |
| tau3 SZA103 650 | 3 | 10.6 % | 5.7 % | **4.56** |

Regating the six affected tier rows against the pooled reference:

| cell | ratio was | ratio now | band was | band now |
|---|---|---|---|---|
| tau3 101 450 1d | 1.057 | 1.018 | 0.277 | 0.211 |
| tau3 101 550 1d | 0.871 | 0.911 | 0.287 | 0.242 |
| tau3 101 550 field | 0.907 | 0.948 | 0.293 | 0.249 |
| tau3 103 550 1d | 1.125 | 1.059 | 0.376 | 0.261 |
| tau3 103 550 field | 0.992 | 0.933 | 0.464 | 0.370 |
| tau3 103 650 1d | 1.088 | 0.886 | 0.400 | 0.257 |

All six still PASS, the mean band tightens 24 percent, and four of the
six ratios move toward unity. This is free: the data was already on disk.

### The chi2 result is a finding in its own right

Three of the four cells scatter consistently with their reported sigmas
(chi2/dof 0.29 to 0.63). The fourth, tau* 3 / SZA 103 / 650 nm, gives
**chi2/dof = 4.56**: the three 1e9-photon estimates disagree about 2.1
times more than their own error bars permit.

This is the deepest, reddest, thinnest-signal cell, which is exactly
where the heavy tail is expected to break the central-limit assumption
behind a reported Monte Carlo standard error. The campaign record
already noted this cell as "a low draw"; the chi2 makes the stronger and
more useful statement, that **the referee's quoted sigma is optimistic
there**, and by roughly a factor of two.

Since the tier's own gate is built from `se_r`, any cell whose `se_r` is
understated has a band that is too narrow. Passing anyway is therefore
conservative, not lenient. But the 3-sigma Gaussian band should not be
presented as if the underlying distribution were established as normal.

## 3. The field-vs-1D equivalence gate could not fail

`g_s3_eq1d_deep` (cited as G-FC-2 in `docs/FIELD_CONNECTIONS_PLAN.md`) is
the gate that certifies the 3D-field representation against the 1D deck.
Measured on this tree:

```
SZA 101: 1D 9.83988e-8 (se 6.10e-8) field 9.84274e-8 (se 6.10e-8)
         ratio 1.000  diff 2.86e-11  band 2.61e-7
SZA 103: 1D 1.76513e-8 (se 1.50e-8) field 1.76513e-8 (se 1.50e-8)
         ratio 1.000  diff 3.77e-14  band 6.39e-8
```

The band is **2.65x the mean at SZA 101 and 3.62x at SZA 103**. A gate
that accepts a factor-of-three discrepancy cannot fail for any physically
plausible error, which by this project's own standard ("a gate that
cannot fail is treated as a bug") makes it the third such gate found.

Two separate causes, both fixable:

1. **The wrong error model.** Both arms are driven from the same
   `seed_salt` sequence, so their samples are PAIRED. The gate combines
   them as `3*sqrt(se_a^2 + se_b^2)`, the formula for independent
   samples, discarding the shared randomness that cancels in the
   difference. The arms in fact agree to between 1e-4 and 1e-6 relative
   while each carries 62 to 85 percent individual SE.
2. **Coverage.** It runs tau* = 3, scalar, 550 nm only. The deep-tier
   field rows that sit high against the referee are tau* = **1** and
   **polarized**, a configuration no equivalence gate touched.

`g_s3_eq1d_paired` replaces it: same physics, paired per-seed relative
difference, crossed over tau* in {1, 3} and scalar/Stokes chains.

All eight configurations pass (4 seeds x 1500 photons, deliberately
small: the paired difference needs no budget):

| configuration | paired relative difference | paired SE | band |
|---|---|---|---|
| tau*3 SZA 101 scalar | +8.98e-8 | 3.68e-11 | 2.00e-3 |
| tau*3 SZA 103 scalar | +5.14e-8 | 3.28e-8 | 2.00e-3 |
| tau*1 SZA 101 scalar | +6.13e-4 | 5.33e-4 | 3.60e-3 |
| tau*1 SZA 103 scalar | -3.88e-4 | 3.36e-4 | 3.01e-3 |
| tau*3 SZA 103 stokes | +1.11e-3 | 8.79e-4 | 4.64e-3 |
| tau*3 SZA 101 stokes | +4.11e-3 | 4.81e-3 | 1.64e-2 |
| tau*1 SZA 103 stokes | -6.68e-3 | 1.04e-2 | 3.31e-2 |
| tau*1 SZA 101 stokes | -2.86e-2 | 1.78e-2 | 5.54e-2 |

The band is 0.2 to 5 percent rather than 265 percent, and in the scalar
tau* = 3 cells the two representations agree to **1e-8**, machine
precision.

One structural observation falls out. Pairing is enormously effective in
the scalar chains (paired SE 3.7e-11 to 5.3e-4) and roughly a thousand
times weaker in the Stokes chains (8.8e-4 to 1.8e-2). The polarized
field path evidently consumes its random stream in a different order
from the polarized 1D path once Stokes rotations enter the voxel
traversal, so the shared randomness stops cancelling. That is not a bug
- every polarized cell is still consistent with zero difference, the
largest being 1.6 sigma - but it is the reason the polarized arms need
more seeds than the scalar arms to reach the same resolving power, and
it is worth stating rather than discovering later.

### What this says about the tau* = 1 field rows

The two tau* = 1 field cells sit at ratios 1.179 and 1.186 against the
referee while their 1d counterparts sit at 0.995 and 0.988 - an apparent
19 percent gap between two representations of the same deck. The paired
gate shows those representations agree to better than 0.1 percent.

The gap is therefore **not** an equivalence defect. What remains: the
tier's field and 1d rows differ by roughly two combined standard errors
(model 3.6 to 4.9 percent, referee 6.1 to 8.5 percent), and the two
cells are the same wavelength and share the campaign's seed
configuration, so they are not two independent draws. A correlated
two-sigma fluctuation is the leading explanation, and tightening the
referee (section 2) is what would settle it.

## 4. SASKTRAN2 does not reach this regime out of the box

Every external referee used so far (DISORT, MYSTIC) comes from
libRadtran, and in the deep tier both the referee and the engine are
Monte Carlo. SASKTRAN2 is attractive because it is independent on both
axes: different lineage, and a DETERMINISTIC solver whose error modes are
unrelated to Monte Carlo variance. It advertises exactly the needed
capabilities - `GeometryType.Spherical`, solar/LOS/multiple-scatter
refraction, polarized output.

It was installed (2026.6.0) and driven from a ground observer looking at
zenith. `tools/sasktran2_probe.py` records the working invocation, which
is not the obvious one: `GroundViewingSolar` is documented as looking AT
the ground and **segfaults** when handed an upward `cos_viewing_zenith`;
the correct class is `SolarAnglesObserverLocation` with
`cos_viewing_zenith = +1` and `relative_azimuth` in radians.

The result is negative. With successive orders plus refraction, 550 nm
zenith radiance goes

| SZA | 80 | 90 | 95 | 101 | 103 | 105 |
|---|---|---|---|---|---|---|
| radiance | 5.46e-3 | 1.80e-3 | 1.32e-3 | 1.35e-3 | 1.34e-3 | 1.34e-3 |

It **flattens past SZA 95** and changes by 0.3 percent per 2 degrees out
to 105, where the physical decay is orders of magnitude. The solver has
left its valid domain well before the regime of interest. Switching the
multiple-scatter source to discrete ordinates and disabling refraction
instead gives 1.13e-12 at SZA 101 - nine orders of magnitude from the
successive-orders answer for the same geometry, which is itself the
diagnosis. Raising `num_sza` to tune the terminator resolution segfaults
at small values.

The negative result is worth keeping, because it is direct empirical
support for the introduction's premise: independent codes give out near
SZA 100, and past that the references are the bottleneck rather than the
model. It should be cited that way rather than quietly dropped. A
genuine second referee will need either a purpose-configured SASKTRAN2
(with its authors' guidance on the terminator regime) or SCIATRAN.

## 5. How much does the calibrated constant actually matter?

The appearance edge factor has moved twice (45 in the pre-hyperaccuracy
frame, 70 under the three-site cluster protocol, 56 after the
2026-07-07 full-campaign refit). A reader is entitled to ask whether a
criterion whose one constant moved by 55 percent is measuring physics or
fitting data. The answer is a derivative, and it had not been reported.

Mecca, 2026-06-13, clear sky, production build, overriding only
`TWILIGHT_KHAYT_EDGE_APPEARANCE`:

| f | 40 | 45 | 50 | 56 | 60 | 70 | 80 |
|---|---|---|---|---|---|---|---|
| Fajr depression | 15.02 | 15.01 | 15.02 | 14.72 | 14.65 | 14.54 | 14.47 |

**Doubling the constant from 40 to 80 moves Fajr by 0.55 degrees**, so
d(depression)/df is about -0.014 deg per unit. The production change
that this repository actually made, 70 to 56, moves Mecca Fajr by
**0.18 degrees**, roughly 43 seconds of clock time. Against campaign
observations whose own quoted uncertainties are +-0.3 degrees, the
criterion is far less sensitive to its calibrated constant than the
measurements are to their own noise. That is the robustness argument the
papers should be making explicitly, and it is a strong one.

### A caveat that came out of the same sweep

Sampling finely across the middle of the range:

| f | 50 | 51 | 52 | 53 | 54 | 55 | 56 |
|---|---|---|---|---|---|---|---|
| depression | 15.02 | 15.02 | 15.01 | 15.00 | 14.73 | 14.78 | 14.72 |

Neighbouring points agree to 0.01-0.02 degrees over f = 50 to 53 - the
runs reuse seeds, so the Monte Carlo noise is common-mode and largely
cancels along the ladder - and then the response **steps by 0.27 degrees
between f = 53 and f = 54**, with a small non-monotonic recovery at 55.
A smooth 0.02-degree neighbourhood either side makes a 0.27-degree jump
structural rather than statistical.

The likely mechanism is that the criterion is not a smooth functional of
f: `spread_required = 5` demands that an INTEGER number of band patches
pass simultaneously, so the detection time jumps whenever f crosses a
value that flips the fifth patch. The production value 56 and the
leave-one-out range 55.2-57.0 sit above the step, not on it, so nothing
published is affected. But a piecewise-constant response should be
disclosed rather than found by a referee, and the honest framing is that
the criterion is smooth in f except at patch-count boundaries.

## Where the next compute goes

1. Pool every deep cell, not only the four that have replicas. Replicas
   are cheaper per unit of band than model seeds by the ratio in
   section 1.
2. Treat the 650 nm deep cells as a distribution problem before a
   budget problem: report the per-seed empirical spread, not only a
   Gaussian SE.
3. Give the polarized equivalence arms more seeds than the scalar ones;
   section 3 measures why they need it.
4. A second referee remains the highest-value addition, and section 4
   shows it is not a pip install away.
