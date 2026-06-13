# 3D Cloud Transport: Implementation Plan

Status: PLANNED (this document is the gate to starting).
Reviewed: design adversarially attacked and amended 2026-06-13; full
touch-point inventory with file:line references in the session archive
(every consumer of `cloud_extinction`, `cloud_diffuse_transmittance`,
and `cloud_g_scaled` is enumerated there).

## Goal

Replace the engine's two remaining transport approximations together:

1. Clouds are transported as horizontally uniform layers, while the
   measured field (cloud3d, MODIS/SEVIRI) is fully 3D.
2. Cloud internal scattering is a closed-form two-stream transmittance
   (Eddington T_diff), not explicit Monte Carlo.

Target state: "3D clouds, 1D gases". The molecular/aerosol atmosphere
stays 1D spherical (it is horizontally uniform to excellent
approximation); clouds become a voxel field traversed explicitly by the
photon chains.

## Core architecture (post-review)

### Decomposition tracking, not naive Woodcock

The single most important design decision. The medium is split into two
channels:

- GAS CHANNEL: the per-shell molecular/aerosol extinction, sampled
  ANALYTICALLY exactly as today. The exponential transform, forced
  collisions, VSPG, and every ALIS per-wavelength ratio stay
  byte-identical: none of that machinery needs to know clouds exist.
- CLOUD CHANNEL: the gray (wavelength-flat), delta-Eddington-scaled
  cloud scattering extinction, delta-tracked with null collisions
  against macro-cell majorants.

Why this beats tracking the total extinction:

- Naive per-shell majorants are unusable: one dense cumulus voxel
  (sigma* ~ 0.014/m) in a shell of clear air (~1e-5/m) gives ~1000x
  oversampling across the whole shell: thousands of null collisions per
  clear-air segment.
- The exponential transform's stretched density sigma' = sigma(1 -
  alpha cos) does not compose with null-collision tracking on the total
  (at alpha = 0.5 the stretched density is not even a majorant inside
  dense voxels). Keeping ET on the gas channel only sidesteps the
  incompatibility entirely, exactly where the deep-twilight machinery
  is needed most (SZA >= 100).
- ALIS: cloud extinction is gray, so null collisions on the cloud
  channel have per-wavelength ratio identically 1: the spectral-MIS
  weights (wr, pr) are untouched by the tracking loop. Tracking the
  total would put (sigma_maj - sigma_w)/(sigma_maj - sigma_h) firefly
  factors on every null event near saturated voxels.

### The field

`Cloud3DField` in twilight-core (no_std: a view over caller-owned
slices; twilight-data owns ingestion and derivation):

```
Cloud3DField<'a> {
    data: &'a [f32],              // delta-scaled cloud SCATTERING extinction
    macrocell_majorants: &'a [f32], // ~16x16 km x level tiles, max per tile
    column_tau_prefix: &'a [f32], // vertical tau* prefix sums per column
    background_column: &'a [f32], // 1D fallback profile (per level)
    g_water: f32, g_ice: f32,     // per-type asymmetry (delta-scaled)
    dims, origin, spacing,        // equal-angle local grid + fine z-grid
}
```

- VERTICAL AXIS DECOUPLED from transport shells: dedicated z-grid for
  0-16 km at 250-500 m (~48-64 levels), preserving cloud3d's 240 m
  resolution. At grazing twilight geometry a few hundred meters of
  cloud-top error displaces the shadow edge by 10-15 km horizontally;
  km-scale shells would smear exactly what matters. A per-(transport
  shell, macro-tile) majorant table bridges to the shell walker.
- HORIZONTAL: observer-centered all-azimuth core, ~256 km radius at
  2 km (the khayt fan needs all azimuths), PLUS a sunward shadow strip:
  the deep-twilight action is at the terminator, 1,100 km (SZA 100) to
  1,780 km (SZA 106) sunward. The strip is coarse 2.5D (column tau* +
  cloud-top height at 8-16 km resolution) and is consulted only by
  shadow rays and BDPT entry legs, which need integrated tau, not voxel
  structure. Memory: core ~20-40 MB f32 + strip a few MB; trivial for
  unified memory.
- THE FIELD OWNS ALL CLOUD when present: the builder zeroes
  cloud_extinction[] / cloud optics in the shells, and out-of-extent
  queries are answered by the embedded background column inside the
  accessor. No transport branching, no seam at the field edge.
- Regridding is CONSERVATIVE in column tau (preserve the integral of
  sigma dz per column), never point-sampled.
- Sources: cloud3d ice (80x240 m, SEVIRI/GOES) + water-deck extrusion
  from CWP/CTH/phase, partitioned by retrieval phase flag (water-phase
  pixels -> extrusion; ice-phase -> cloud3d; mixed -> cloud3d ice +
  max(0, CWP - IWP) water residual). SEVIRI 15-min products preferred
  over MODIS overpasses (4-8 h stale at twilight); field timestamps are
  recorded and stale fields are refused beyond a configured limit.

### Traversal

One shared DDA routine (Amanatides-Woo in shell/angle coordinates;
radial crossings already have exact closed forms via
`next_shell_boundary`) used by ALL of: deterministic-leg line
integrals, delta-tracking flights, forced-collision scouts AND the
tau(s) inversion in `advance_to_optical_depth*`. Bit-identical tau
functions everywhere or the forced-mode sampler pdf will not match its
weights (the audit's bias class). No fixed-step marching: the field is
piecewise constant, DDA integrates it exactly and cheaper.

VSPG segments stay PER SHELL CROSSING (importance is altitude-only)
with tau bounds from the exact in-shell DDA integral; segment counts do
not grow and the VSPG_MAX_SEGMENTS=128 overflow (which silently drops
tau ranges: a latent bias) gets an assertion regardless.

### Local optics

A single `LocalOptics { sigma_s_ray, sigma_s_aer, sigma_s_cloud,
sigma_a, g_aer, g_cloud_star }` with ONE paired (sample_phase,
eval_phase) over the 3-lobe mixture, used by: the seed sampler,
seed_mixture_pdf, the seed numerator, NEE phase evaluation (the
hand-inlined copy in the scalar chain is deleted), Dwivedi and guide
MIS denominators, ALIS phase ratios, and BDPT vertex evaluation.
Component selection is by SCATTERING fraction (the field stores
scattering extinction). The Stokes chain's cloud convention is declared
explicitly: cloud scatters as a depolarizing HG (phase on the I-term,
full depolarization), not left emergent. A chi-square
sampled-vs-evaluated gate runs per estimator variant and on GPU parity.

## Stages and gates

### Stage 0: field + builders + geometry gate (~900 LOC)

- Cloud3DField + accessor + majorant/prefix derivation (twilight-core).
- Builders in twilight-data: cloud3d-to-field, water extrusion,
  synthetic generators (uniform, single cube, checkerboard).
- Builder split: when a field is present, shells carry no cloud.
- GATE G1 (deterministic, exact): with a horizontally constant field,
  the 3D ray-marched slant tau on every deterministic leg equals the 1D
  analytic slant tau to quadrature precision. Pure geometry; no MC.

### Stage 1: deterministic legs read the field (~700 LOC)

- Eye LOS, NEE shadow rays, BDPT connections, trace_transmittance (which
  today has NO cloud term at all: a standing gap) compute cloud tau* by
  DDA through the field.
- CRITICAL AMENDMENT vs the draft: Stage 1 KEEPS the T_diff functional
  form, fed with the 3D tau. Swapping deterministic legs to Beer-Lambert
  before chains scatter explicitly would delete all cloud-diffused light
  (the documented "prayer times degenerate to sunrise" failure, ~8x at
  OD 10). T_diff(3D tau) captures the real geometric win immediately:
  sun rays through actual cloud GAPS get T_diff(0) = 1.
- Touch points (from the inventory): 5 fns in single_scatter.rs, the
  two hybrid LOS walks, the ALIS BDPT re-walk, trace_transmittance,
  plus Option<&Cloud3DField> plumbing through simulation/pipeline/CLI.
- Stage 1 is shippable alone and strictly better than today.

### Stage 2: chains in the 3D medium (~1,500-1,800 LOC, the hard one)

- Decomposition tracking in all FOUR chain implementations in lockstep:
  scalar, Stokes, ALIS, and the BDPT light subpath. Until the light
  subpath converts, bdpt_strength is forced to 0 when a field is
  present (NEE-only: unbiased, just noisier) so the blend never mixes
  two different models of the same integral.
- Explicit in-cloud scattering via LocalOptics; T_diff retired as
  physics. The Eddington FORMULA survives as an importance heuristic:
  the weight-window target gains a cloud-lid factor (1/T_lid from the
  column prefix sums) so chains are not split under thick decks where
  escape probability is nil; NEE gets a one-lookup early-out when the
  slant cloud tau lower bound exceeds ~15.
- The 1D fallback path also moves to explicit scattering (same
  estimator for in-field and out-of-field cloud).
- Hardest single item (per the inventory): the ALIS chain's forced-mode
  and ratio sites under the new walker.
- GATE G2 (external referee): explicit-MS vs MYSTIC/SHDOM on a
  plane-parallel homogeneous slab and a synthetic cube at daytime SZA,
  within MC noise + a few percent.
- GATE G3 (documented-delta ledger): the expected difference between
  Stage-2 output and the old T_diff pipeline under standard decks is
  PREDICTED from G2 (sign and magnitude), and the measured delta must
  match the prediction. Agreement with the old numbers is explicitly
  not required: explicit MC will correctly disagree with the
  diffusion-limit closed form by 10-40% at tau* 1-3.
- GATE (variance): cloudy Padborg scene, CV at fixed wall-clock must
  not regress more than a stated factor vs Stage 1.
- A chi-square sampler-vs-pdf gate per estimator variant; an
  ALIS-vs-per-wavelength statistical gate with a synthetic 3D cloud.

### Stage 3: Metal GPU port

- Field + majorants + prefixes as new packed buffers (BUFFER_VERSION 4;
  the shader-side version gate from the hardening pass will catch any
  drift loudly).
- The lockstep Blelloch cloud prefix scan in the hybrid kernel dies
  (nonuniform field): each thread integrates its own segment by DDA.
- f32 error budget derived BEFORE the parity tolerances are written
  (tau prefix sums over 1,000 km paths, ALIS ratio products).
- RNG discipline: null-collision loops draw only from rng.tau; the
  CPU/GPU determinism contract is re-derived, not assumed.

### Stage 4: validation and ship

- G1/G2/G3 + GPU parity all green.
- Honesty note that carries into the README: NO external 3D spherical
  twilight referee exists (public MYSTIC does 3D plane-parallel only).
  The twilight-regime check is: G3 ledger + asymptotic limits (thin and
  thick deck against analytic single-scatter and diffusion bounds) +
  the SQM field campaign (docs/SQM_CAMPAIGN.md) as the only true
  end-to-end referee, with cloud-field timestamps logged.
- Perf: pray run < ~5 min GPU on the reference machine, measured not
  asserted.

## Open decisions (resolve at Stage 0)

1. Temporal validity: one field per run vs per-scan-step fields vs wind
   advection. Broken cloud advects ~10 km per 10 min; a threshold scan
   sweeps tens of minutes. Start with one field + recorded timestamp +
   documented error; revisit after SQM data exists.
2. Cloud ABSORPTION in voxels: today absorption is folded into shell
   optics (horizontally uniform). Either move it into the field (second
   channel) or document with numbers that visible-band cloud absorption
   is negligible. Decide with a worked number at Stage 0.
3. Non-hybrid estimators (trace_photon, mc_scatter_spectrum) and
   twilight-skyglow: declared 1D-only initially; revisit if anything
   consumes them in production.
4. Path guide: training and MIS use the local mixture; the 32-bin
   angular resolution may be too coarse for gap directions. Defer
   until Stage 2 measurements exist.
5. Forced-collision gate under decks (tau_max > 20 disables forced
   mode): intended regime switch or needs a cloud-aware gate. Measure
   at Stage 2.

## Effort

| Stage | LOC | Risk |
|---|---|---|
| 0 field + builders + G1 | ~900 | low |
| 1 deterministic legs | ~700 | low (shippable alone) |
| 2 chains | ~1,500-1,800 | high (ALIS forced-mode is the hardest single item) |
| 3 GPU | ~800-1,200 | medium (parity discipline) |
| 4 validation | harness reuse | scheduled compute |

Test hygiene throughout (per project convention): heavy MC gates are
#[ignore] with targeted filters; full suites at commit gates only.
