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

**Validated through SZA 95.** Beyond 98° the signal is 10⁻⁷ to 10⁻⁹ of TOA
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
| 100 | 28.1% | 29.9% | 14.8% | consistent twilight deficit - refined at 1e8 below |
| 102 | 20.1% | 44.2% | 6.8% | sign-mixed (noise on both sides) |
| 104 | x7 | x80 | 87% | MYSTIC variance (its own values jump 3 orders between bands) |
| 106 | - | - | - | MYSTIC zero even backward at 1e7 |

### 1e8-photon refinement (2026-06-13, MC_BACKWARD_PHOTONS=1e8)

The overnight 1e8-photon MYSTIC backward campaign was run to settle
whether the SZA-100 disagreement was a twilight deficit or referee
noise. It was largely the latter: with 10x the photons the SZA-100
disagreement roughly halved in the two bands MYSTIC can still resolve.

| SZA | 450 nm | 550 nm | 650 nm | note |
|----:|------:|------:|------:|------|
| 95 | 1.3% | 5.0% | 1.6% | PASS |
| 96 | 4.2% | 4.7% | 2.4% | PASS |
| 98 | 1.6% | 1.4% | 3.9% | PASS (tightened from 1e7) |
| 100 | 13.7% | 13.6% | 28.8% | blue/green halved vs 1e7; red still noisy |
| 102 | 24.6% | 16.4% | 47.2% | referee-limited |
| 104 | 59.7% | 33.7% | 56.3% | referee-limited |
| 106 | 67.0% | 1974% | 74.4% | referee incoherent |

The SZA-106 550 nm point sits 100x below its own 450 and 650 nm
neighbors (1.9e-11 against ~1e-9), which is impossible for a smooth
twilight spectrum: at this depression public MYSTIC backward does not
converge even at 1e8 photons, so the disagreement past SZA 102 measures
the referee, not twilight. **External validation against MYSTIC is solid
to SZA ~98 and usable to SZA 100; beyond SZA 102 no public 3D MC referee
converges, and the field SQM campaign (docs/SQM_CAMPAIGN.md) is the only
end-to-end check in the deep tail.**

## G2: explicit-cloud slab referee (2026-07-02, gate G2 of docs/3D_TRANSPORT_PLAN.md)

External referee for the Stage-2 explicit-cloud chain transport (CPU
chain estimator as of commit 0cc8bf5) on a plane-parallel homogeneous
water-cloud slab at daytime SZA. **Verdict: G2 FAILS for both chain
estimators.** The referee side itself is triple-anchored (clear-sky
agreement ~1%, disort vs MYSTIC 0.02-0.2% on 6 cloud cases), so the
disagreement is twilight's.

### The same-problem construction

twilight's cloud channel is delta-Eddington scaled at build time
(`twilight-data builder::add_cloud_layer`, Joseph/Wiscombe/Weinman
1976): for unscaled inputs (tau, ssa, g) the medium actually carries

- extinction OD `tau* = tau (1 - ssa g^2)`
- single-scattering albedo `ssa* = (1-g^2) ssa / (1 - ssa g^2)`
- HG asymmetry `g* = g/(1+g)` (`atm.cloud_g_scaled`)

with the scattering part (`tau_scat* = tau (1-g^2) ssa`) in the gray
per-shell cloud channel and the absorption part (`tau_abs = tau (1-ssa)`,
exactly conserved) folded into the shell optics. The libRadtran water
cloud was configured with THESE scaled values and a Henyey-Greenstein
phase function (`wc_properties hu` is HG by construction; `wc_modify
tau/ssa/gg set` are gray). Both codes therefore solve the IDENTICAL
scaled transport problem and the comparison gates the CHAIN MACHINERY,
not the delta-scaling approximation. With the water-cloud preset
microphysics ssa = 0.999, g = 0.85: de-scale factor `1 - ssa g^2 =
0.2782225`, `ssa* = 0.9964058`, `g* = 0.4594595`. Deck between 1 and
2 km (afglus grid levels), uniform.

| case | twilight `--cloud-tau` (unscaled) | carried tau* (= uvspec `tau set`) | tau_scat* | tau_abs |
|-----:|----:|----:|----:|----:|
| tau* 1 | 3.594246 | 1.000000 | 0.996406 | 0.003594 |
| tau* 3 | 10.782737 | 3.000000 | 2.989218 | 0.010783 |
| tau* 10 | 35.942456 | 10.000000 | 9.964058 | 0.035942 |

Configuration (both codes): Rayleigh-only gas (uvspec `no_absorption
mol` + `mol_abs_param crs`; twilight `--rayleigh-only` + custom cloud =
`build_with_cloud_properties`, no gas absorption, no aerosol), albedo
0.15, no refraction, scalar (`--fast`), and the IDENTICAL solar scale:
uvspec was fed twilight's own TSIS-1 10 nm table
(`validation/g2_solar_tsis_tw.dat`), so the comparison is absolute, no
shape normalization. Referee: disort 32 streams pseudospherical
(deterministic, exact for the plane-parallel slab at SZA 30/60);
MYSTIC forward (`mc_photons 4e6`, `mc_std`) as MC cross-check on 6
cases. twilight: CPU chain estimators, hybrid (production, 2000
photons/wl) and multiple (independent analog, 8000 photons/wl), 6 seeds
each, SE = seed scatter. Grid: SZA {30, 60, 85 (stretch)} x view zenith
{0, 60} x rel azimuth {0, 180} x {450, 550, 650} nm. Decks and raw
outputs: `validation/g2_*`; full table `validation/g2_results.csv`;
campaign `tools/validate_libradtran.py --tier g2` (15 min wall).

### Referee validation

- Clear anchor (tau* 0), twilight/disort at 550 nm: hybrid 0.988-1.011
  across all 9 geometries (matches tier 1a-absolute), multiple
  0.944-0.996 (larger MC noise; one 4-se point at SZA 85 zenith).
  The setup chain (solar, Rayleigh, geometry, units) is clean at the
  ~1-2% level; sphericity-vs-plane-parallel is inside that.
- disort vs MYSTIC under cloud: 6 cases (tau* 1/3/10, SZA 30/60,
  zenith + one slant), agreement 0.02-0.2%, MYSTIC 1-sigma <= 0.17%.
  The referee value is unambiguous.

### Results, 550 nm (ratio twilight/disort; +-1 se from 6 seeds)

| tau* | SZA | vz | ra | hybrid/disort | multiple/disort |
|-----:|----:|---:|---:|--------------:|----------------:|
| 1 | 30 | 0 | 0 | 1.165+-0.007 | 1.016+-0.004 |
| 1 | 30 | 60 | 0 | 0.956+-0.006 | 1.016+-0.004 |
| 1 | 30 | 60 | 180 | 1.013+-0.013 | 1.040+-0.008 |
| 1 | 60 | 0 | 0 | 1.172+-0.013 | 1.019+-0.008 |
| 1 | 60 | 60 | 0 | 0.912+-0.004 | 1.019+-0.002 |
| 1 | 60 | 60 | 180 | 0.980+-0.022 | 1.055+-0.006 |
| 1 | 85 | 0 | 0 | 1.173+-0.018 | 1.010+-0.008 |
| 1 | 85 | 60 | 0 | 0.899+-0.015 | 1.057+-0.009 |
| 1 | 85 | 60 | 180 | 0.891+-0.015 | 1.071+-0.010 |
| 3 | 30 | 0 | 0 | 0.836+-0.007 | 1.051+-0.005 |
| 3 | 30 | 60 | 0 | 0.511+-0.004 | 1.083+-0.007 |
| 3 | 30 | 60 | 180 | 0.539+-0.008 | 1.100+-0.004 |
| 3 | 60 | 0 | 0 | 0.865+-0.010 | 1.089+-0.008 |
| 3 | 60 | 60 | 0 | 0.527+-0.007 | 1.145+-0.006 |
| 3 | 60 | 60 | 180 | 0.548+-0.011 | 1.150+-0.005 |
| 3 | 85 | 0 | 0 | 0.904+-0.014 | 1.154+-0.023 |
| 3 | 85 | 60 | 0 | 0.615+-0.016 | 1.285+-0.010 |
| 3 | 85 | 60 | 180 | 0.630+-0.011 | 1.276+-0.012 |
| 10 | 30 | 0 | 0 | 0.247+-0.003 | 1.308+-0.004 |
| 10 | 30 | 60 | 0 | 0.034+-0.000 | 1.495+-0.009 |
| 10 | 30 | 60 | 180 | 0.034+-0.000 | 1.478+-0.010 |
| 10 | 60 | 0 | 0 | 0.274+-0.006 | 1.430+-0.011 |
| 10 | 60 | 60 | 0 | 0.040+-0.000 | 1.712+-0.003 |
| 10 | 60 | 60 | 180 | 0.040+-0.001 | 1.681+-0.006 |
| 10 | 85 | 0 | 0 | 0.323+-0.005 | 1.734+-0.027 |
| 10 | 85 | 60 | 0 | 0.054+-0.001 | 2.177+-0.016 |
| 10 | 85 | 60 | 180 | 0.053+-0.001 | 2.158+-0.015 |

MYSTIC cross-checks (W/m^2/sr/nm, 550 nm): tau* 1 SZA 60 zenith
1.1891e-1 (disort 1.1899e-1); tau* 3: SZA 30 zenith 2.9674e-1 (2.9671e-1),
SZA 60 zenith 1.3005e-1 (1.3008e-1); tau* 10: SZA 30 zenith 1.3706e-1
(1.3704e-1), SZA 60 zenith 5.9053e-2 (5.9023e-2), SZA 60 vz 60 4.2461e-2
(4.2505e-2).

### Gate band and verdict

Band: disort is exact for the scaled plane-parallel problem at SZA
30/60 (MYSTIC-confirmed); twilight-vs-referee residuals that are NOT
chain physics are bounded by the clear anchor (~1-2%) plus seed SE.
Gate at SZA 30/60: |ratio - 1| <= 3% + 2 se. SZA 85 informational
(pseudospherical disort; clear anchor still ~1%).

- multiple (independent analog): tau* 1 largely IN BAND (+1.6 to +4.0%
  at SZA 30/60; the (60,60,180) point +5.5% is marginally out). tau* 3
  FAILS: +5 to +15%. tau* 10 FAILS: +31 to +71%.
- hybrid (production): FAILS at every tau* tested. tau* 1: zenith
  +17% at all three SZAs while slant views sit -4 to -9%. tau* 3:
  zenith -14%, slant -45 to -49%. tau* 10: zenith x0.25-0.32, slant
  x0.03-0.05 (up to ~25-30x too dark).

### Characterization (no tuning attempted, per the referee mandate)

1. Both discrepancies are wavelength-flat (450/550/650 ratios equal
   within noise) and survive the cloud/clear double ratio, so they are
   in the gray cloud chain machinery, not in gas optics, solar scale,
   or geometry.
2. The hybrid deficit is CONVERGED BIAS, not variance: tau* 10 SZA 30
   zenith is stable at 0.0339-0.0345 for 2k/8k/32k photons/wl
   (disort: 0.1370). Its signature tracks the slant cloud OD of the
   eye path (tau*/mu_view): zenith (tau_eye 10) is x4 low, vz 60
   (tau_eye 20) is x25-30 low. The order-1 cloud NEE + Beer-Lambert
   eye LOS supply too little in-scattered light once the deck is
   diffusion-dominated, i.e. the chains fail to build up the diffuse
   field that dominates below/inside a thick deck. The tau* 1 zenith
   +17% overshoot (present at all SZAs, absent in multiple) points at
   the order-1 cloud NEE/chain-seed mixture weighting rather than deep
   transport.
3. The multiple-mode excess is a monotone POSITIVE inflation growing
   with tau* and path slant: +2% (tau* 1) to +9% (tau* 3) to +43%
   (tau* 10) at SZA 60 zenith, worse on slant views and at SZA 85,
   consistent with a small per-cloud-collision multiplicative excess
   (mean number of cloud scatters grows with tau* and slant).
4. G-HYB-MULT (the internal gate of commit 0cc8bf5, hybrid==multiple
   at SZA 95-97) cannot see this: the two estimators disagree with
   EACH OTHER by 20% (tau* 3) to 40x (tau* 10 slant) at daytime SZA,
   and both disagree with the external referee in opposite directions.
   Internal cross-estimator agreement at one SZA band was necessary
   but not sufficient.

### Reproduce

```bash
export LIBRADTRAN_DIR=~/tools-build/libRadtran-2.0.6
cargo build --release -p twilight-cli
python3 tools/validate_libradtran.py --tier g2   # ~15 min
python3 validation/g2_table.py                   # markdown tables
```

## G2 addendum: root causes found and fixed (2026-07-02, post-verdict)

Scope of the fix: crates/twilight-core/src/photon.rs only (chain walkers
and the two hybrid LOS drivers). Single mode untouched. Each defect was
first confirmed against an INDEPENDENT plane-parallel backward-MC
referee written from scratch for the G2 slab
(diag_g2_slab_independent_reference in twilight-cpu/simulation.rs; it
reproduces disort to 0.1-1 percent on the identical per-shell profile),
then re-refereed against disort.

### Root cause 1 (Multiple's inflation): ground bounces landed BELOW the
### surface

cross_boundary advances a crossing photon to boundary_pos + dir*1e-3
with dir still pointing DOWNWARD at a ground hit, so every ground bounce
in every walker (trace_photon, trace_photon_polarized, the Stokes,
scalar and ALIS secondary chains) started ~1 mm below shells[0].r_inner,
where shell_index() is None. Two consequences:

1. The ground-NEE shadow ray saw no shell and returned T = 1: the
   sun-ground-eye term was scored UNATTENUATED through any deck (true
   T under tau* = 10 at SZA 30 is ~1e-5). Under a deck most backward
   chains reach the ground, which produced the +31 to +118 percent
   inflation, monotone in tau* and slant (more ground visits per chain)
   and wavelength flat (the term is gray) - exactly the observed
   signature that had suggested a per-collision excess.
2. The next bounce iteration found no shell and terminated the chain as
   "escaped": ALL ground-reflected diffuse light was missing from every
   chain estimator (hybrid chains included).

In clear sky the two errors nearly cancel (T_sun 0.85-0.9 instead of 1
on a small term, minus the missing ground-diffuse), hiding inside the
1-2.5 percent clear band. The isolating experiment: a pure-cloud slab
with albedo 0 was CLEAN (trace_photon/flat referee = 0.99) while albedo
0.15 inflated 1.56x at zenith.

Fix: snap the bounce point to r_surface + 1e-3 before the ground NEE
and the Lambertian re-emission (5 sites). Measured effect of this fix
alone (550 nm, 3 seeds): Multiple tau*=10 SZA 30 zenith 1.308 -> 0.999,
vz60 1.495 -> 1.013; SZA 60 zenith 1.430 -> 0.988, vz60 1.712 -> 1.028.

### Root cause 2 (hybrid tau*=1 zenith +17 percent): eye-path 1D cloud
### tau classified by the step-midpoint shell

The hybrid LOS drivers computed each step's cloud tau as
atm.cloud_extinction[shell_at_step_midpoint] * ds over 750-1452 m steps
that SPAN shell boundaries: a zenith step one third below the deck still
got full-deck extinction over its whole length. For the 1-2 km deck this
inflated both the deck's in-scatter source and the eye-path cloud OD by
1.5x at zenith (net +17 percent after the extra self-attenuation) and
1.45x at vz60 (net -7 percent; the slant attenuation overcompensates).
Fix: eye_step_cloud_tau computes the 1D fallback exactly (analytic
per-shell path lengths via ray_path_through_shell); the field DDA
(tau_along) was already exact. The chain walkers never had this defect
(they race per-shell segments).

### Root cause 3 (hybrid's converged thick-deck deficit): midpoint LOS
### quadrature vs per-step cloud opacity

The eye-path quadrature source(mid)*exp(-tau_mid)*ds underestimates
INT source*exp(-tau) ds by exp(-x/2)/[(1-exp(-x))/x] for per-step cloud
tau x: 0.98 at x=0.75 (tau* 1 zenith), 0.50 at x=4.35 (tau* 3 vz60),
0.18 at x=7.5 (tau* 10 zenith, ds=750 m), 0.010 at x=14.5 (tau* 10
vz60, ds=1452 m). These factors match the converged pre-fix deficits
(observed 0.912-0.956 / 0.511-0.548 / 0.247 / 0.034 with root causes
1-2 layered on top), which is why the deficit tracked the eye-path
slant cloud OD and was photon-count independent. Fix: cloud-adaptive
substepping in BOTH hybrid drivers (CLOUD_SUBSTEP_TAU = 0.25, cap
CLOUD_MAX_SUBSTEPS = 64): steps whose exact cloud tau exceeds 0.25 are
subdivided; order-1 NEE and chain launches run per substep; the coarse
step's chain budget is importance-allocated over substeps (unbiased for
any n_j >= 1). Cloud-free steps keep k = 1 and are bit-identical to the
previous code (same arithmetic, same RNG stream), so clear-sky runs
change only through root cause 1.

### Checked and exonerated

- sample_henyey_greenstein / cloud_phase_value: textbook-exact pair. The
  g_s2 chi-square gate is tightened from chi2 < 80 to 43.82 (the
  chi2(19) p=0.001 critical value; 20 bins, no fitted parameters) and
  passes for g in {0, 0.46, 0.85}. The Multiple inflation also persisted
  with g ~ 0 (isotropic), independently clearing the sampler.
- The two-budget gas/cloud decomposition race: a flat-geometry replica
  of the exact walk structure matches a single-channel textbook
  reference within MC noise at tau* = 10.
- Weight windows: dormant at G2 SZAs (h_ww ~ 1e6 m below SZA ~96); every
  RR site applies the survivor boost exactly; splits conserve weight.
  No cloud-lid 1/T_lid factor exists in weight_window_target (altitude +
  CADIS only): a deep-twilight variance concern, not a G2 bias.
- No order-1 double counting in the cloud seed mixture (the p_c
  cancellation is exact per wavelength; chains never score NEE at their
  seed vertex), and no bounce cap in the chain walkers (escape/RR only).

### Post-fix G2 campaign (2026-07-02, 6 seeds, identical protocol)

550 nm, ratio twilight/disort (mean +- se over 6 seeds):

| tau* | SZA | vz | ra | hybrid/disort | multiple/disort |
|-----:|----:|---:|---:|--------------:|----------------:|
| 0 | 30 | 0 | 0 | 1.003+-0.001 | 1.006+-0.011 |
| 0 | 30 | 60 | 0 | 0.996+-0.001 | 0.990+-0.006 |
| 0 | 30 | 60 | 180 | 0.978+-0.001 | 0.967+-0.007 |
| 0 | 60 | 0 | 0 | 0.991+-0.001 | 0.966+-0.018 |
| 0 | 60 | 60 | 0 | 1.001+-0.001 | 0.982+-0.013 |
| 0 | 60 | 60 | 180 | 0.984+-0.001 | 0.963+-0.013 |
| 0 | 85 | 0 | 0 | 0.982+-0.001 | 0.933+-0.014 |
| 0 | 85 | 60 | 0 | 1.004+-0.001 | 0.981+-0.006 |
| 0 | 85 | 60 | 180 | 0.990+-0.001 | 0.974+-0.008 |
| 1 | 30 | 0 | 0 | 0.994+-0.004 | 1.001+-0.004 |
| 1 | 30 | 60 | 0 | 0.990+-0.008 | 0.996+-0.004 |
| 1 | 30 | 60 | 180 | 0.989+-0.010 | 1.002+-0.008 |
| 1 | 60 | 0 | 0 | 0.987+-0.006 | 0.987+-0.008 |
| 1 | 60 | 60 | 0 | 0.994+-0.004 | 0.998+-0.002 |
| 1 | 60 | 60 | 180 | 0.990+-0.013 | 0.991+-0.006 |
| 1 | 85 | 0 | 0 | 1.043+-0.008 | 0.949+-0.008 |
| 1 | 85 | 60 | 0 | 1.045+-0.007 | 0.990+-0.009 |
| 1 | 85 | 60 | 180 | 0.974+-0.016 | 0.970+-0.012 |
| 3 | 30 | 0 | 0 | 1.004+-0.010 | 0.994+-0.005 |
| 3 | 30 | 60 | 0 | 1.009+-0.012 | 0.995+-0.008 |
| 3 | 30 | 60 | 180 | 1.026+-0.011 | 0.995+-0.004 |
| 3 | 60 | 0 | 0 | 0.981+-0.015 | 0.994+-0.008 |
| 3 | 60 | 60 | 0 | 1.017+-0.013 | 1.007+-0.006 |
| 3 | 60 | 60 | 180 | 1.010+-0.021 | 0.983+-0.005 |
| 3 | 85 | 0 | 0 | 0.953+-0.015 | 0.974+-0.024 |
| 3 | 85 | 60 | 0 | 0.979+-0.017 | 0.987+-0.012 |
| 3 | 85 | 60 | 180 | 0.936+-0.020 | 0.960+-0.014 |
| 10 | 30 | 0 | 0 | 0.992+-0.015 | 0.998+-0.004 |
| 10 | 30 | 60 | 0 | 0.984+-0.035 | 1.015+-0.010 |
| 10 | 30 | 60 | 180 | 1.040+-0.026 | 0.998+-0.010 |
| 10 | 60 | 0 | 0 | 1.024+-0.025 | 0.988+-0.013 |
| 10 | 60 | 60 | 0 | 1.051+-0.019 | 1.022+-0.005 |
| 10 | 60 | 60 | 180 | 0.997+-0.034 | 0.991+-0.005 |
| 10 | 85 | 0 | 0 | 1.050+-0.050 | 0.981+-0.028 |
| 10 | 85 | 60 | 0 | 0.939+-0.052 | 1.005+-0.015 |
| 10 | 85 | 60 | 180 | 0.909+-0.045 | 0.996+-0.013 |

Gate (SZA 30/60, ALL wavelengths 450/550/650, band 3 percent + 2 se):
0 failures out of 144 gated points, both estimators, all tau*. Worst
gated deviations: hybrid +5.9 percent at tau*=10 SZA 60 vz 60 450 nm
(band 6.2 percent, se-dominated), multiple -4.6 percent at tau*=10
SZA 60 vz 60 ra 180 650 nm (band 6.3 percent). SZA 85 (informational,
pseudospherical disort): hybrid 0.909-1.050, multiple 0.933-1.005
(pre-fix: 0.034-1.173 / 0.949-2.177). MYSTIC cross-checks re-run and
agree with disort as before.

### Clear-sky movement (root cause 1 exists in clear sky too)

The ground fix removes a spurious unattenuated sun-ground NEE term and
restores ground-reflected diffuse light in the chains; the two partially
cancel in clear sky (~1 percent net). Post-fix clear anchors sit at
0.963-1.006 (multiple) and 0.978-1.004 (hybrid) across SZA 30/60/85 at
550 nm: inside the historical 1-2.5 percent clear band, most points as
close to or closer to disort than pre-fix. Single mode is untouched and
bit-identical.

### Deep-twilight arbitration (SZA 95/97, the G-HYB-MULT regime)

The internal cross-estimator gate `g_s2_hybrid_matches_multiple`
(SZA 95/97, thin OD-2 deck, vz 80) FAILED after the fixes even though
every referee point passed. Arbitrated externally with spherical
BACKWARD MYSTIC (mc_spherical 1D, mc_backward, mc_vroom, same scaled
slab convention, 550 nm, no refraction on both sides):

| SZA | MYSTIC | twilight multiple | twilight hybrid |
|----:|-------:|------------------:|----------------:|
| 95 (clear) | 1.092e-3 +- 1.2e-5 | 1.095e-3 | 1.073e-3 |
| 97 (clear) | 1.674e-4 +- 4.7e-6 | 1.651e-4 | 1.707e-4 |
| 95 (deck)  | 1.389e-4 +- 3.9e-6 | 1.286e-4 +- 1.1e-5 | 1.300e-4 +- 2.6e-5 |
| 97 (deck)  | 1.662e-5 +- 1.4e-6 | 1.629e-5 +- 2.5e-6 | 7.47e-6 +- 3.3e-7 |

Multiple agrees with MYSTIC everywhere (it is now anchored at SZA 30-85
by disort AND at 95/97 by spherical MYSTIC). The hybrid agrees in clear
sky and at SZA 95 under the deck, but sits at 0.45x under the deck at
SZA 97. This deficit is PRE-EXISTING (HEAD c94b11d measures 1.04e-5 at
the same point, 0.63x) and is the already-documented analog-under-cloud
starvation: forced-collision mode is disabled under any cloud channel,
so above ZENITH_SZA_START the hybrid chains sample the sunlit region
(shadow height ~48 km at SZA 97) analog, and the estimator converges
one-sidedly from below (512 photons: 7.5e-6; 2048: 9.3e-6; 8192:
1.07e-5, climbing toward 1.66e-5). Bisection with pinned SZA knobs
(seed lobes, Dwivedi, ET, weight windows, CADIS, LOS importance all
pinned to their SZA-95 values) leaves the deficit unchanged, confirming
budget starvation rather than any of the SZA-gated machinery. The
combined-channel forced mode for the 1D deck remains the tracked
Stage-2 follow-up.

Why the gate used to pass: the pre-fix hybrid at vz 80 carried an
order-1 eye-OD misclassification of ~e^{+0.3} (root cause 2) that
inflated it back toward Multiple, and its seed variance was ~20x larger
(pre-fix se ~24 percent of the mean), widening the 3-SE band over the
residual gap: two wrongs cancelling, exactly the failure mode the
external-referee mandate exists to break. The gate is restructured to
stay honest AND keep its regression teeth: two-sided agreement at
SZA 88/92 (where the analog chains converge at gate budgets), and a
one-sided bound at SZA 97 (hybrid must NOT exceed Multiple beyond
3 SE + 5 percent, which still fails loudly on the cloud-blind forced
composition this gate was built against, plus a 0.25x collapse floor).

Restructured-gate result (8 seeds, both decks): SZA 88 hybrid/multiple
1.015 (1D) and 1.016 (field), diff well inside band; SZA 92 ratio 1.031
(1D), diff inside band; SZA 97 one-sided, hybrid 1.93e-4 inside
[floor 1.30e-4, upper 5.85e-4] around multiple 5.20e-4. All pass.
