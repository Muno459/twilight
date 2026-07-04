# Deep-regime closure: field-forced chains and the SZA 101-103 referee (2026-07-04)

Follow-on to validation/RESULTS_FORCED_CLOUD.md, working tree on top of
b776776 (plus an unrelated GPU-seed commit 1201940 by the wgpu port).
This campaign closes the last variance-limited regime of the external
referee program: thick cloud decks at SZA >= 101, where both sides of
every prior comparison were MC-noise-limited (MYSTIC 1e8 backward SE
10-19 percent, twilight hybrid seed CV 21-74 percent at 16k photons,
LOW-POWER verdicts across the board). Three fronts:

1. collapse the twilight-side variance under 3D FIELDS by extending
   forced-collision mode to the per-wavelength chains (majorant-combined
   truncated null-collision delta tracking; ALIS deliberately stays
   analog under fields),
2. shrink the referee side with an ultra-budget MYSTIC campaign
   (3e8 photons per case, 1e9 on the SZA 103 rows),
3. add physics-constraint gates that resolve bias below single-point
   noise without any referee in the loop (monotonicity ladders and a
   chi2 smoothness test), plus estimator-level distribution gates that
   referee the new flight machinery analytically.

## 1. The estimator, as implemented

### Setting

Under a 3D field the cloud extinction sigma_c(x) varies inside a shell
segment, so the exact combined-channel fold of the 1D forced mode
(RESULTS_FORCED_CLOUD.md) does not apply: the scout cannot invert a
non-piecewise-constant tau. The July-02 campaign therefore left field
chains ANALOG, and the field rows at SZA >= 97 sat in the documented
0.37-0.45x starvation class (deeper: collapse).

### Majorant-combined channel

The scout/advance machinery now folds a MAJORANT-combined channel

    sigma_m(s) = sigma_gas(shell) + c_maj(shell),

where c_maj is a PER-SHELL bound on the field's sigma_at over the
shell's radial band: for each field z level, the max over the
macrocell tile maxima (or the raw voxel row when no macrocell data
exists) joined with background_column, then the max over the levels a
shell band touches, with level bounds floored and clamped OUTWARD so a
band endpoint exactly on a level boundary includes both neighbours
(Cloud3DField::level_max_sigma / band_max_sigma; the f32 -> f64 casts
are exact so domination survives the widening). The majorant is
deliberately lat/lon-independent: conservative (a ray through a clear
part of an occupied shell band sees null collisions, never bias) and
cheap (ONE macrocell scan per driver call, `field_shell_majorants`,
reused by every chain of that call). Gas needs no majorant: it is
exactly shell-constant already. sigma_m is again piecewise constant per
shell, so the existing scout / VSPG-segment / advance machinery inverts
the majorant-combined tau EXACTLY, unchanged.

### Truncated null-collision flight

In majorant-tau coordinates u = tau_m(s) in [0, T] (T the scouted
boundary tau), the true first-collision law is delta tracking: unit-rate
Poisson events, each accepted REAL with a(u) = sigma_t(u)/sigma_m(u),
else NULL. Forcing removes the escape atom PER STAGE: from budget
position u_k (u_0 = 0),

    weight *= (1 - e^{-(T - u_k)}),
    du ~ truncated Exp(1) on (0, T - u_k],  u_{k+1} = u_k + du,
    advance the walk by du, classify with ONE uniform xi:
      REAL CLOUD  if xi * sigma_m < sigma_c(x),
      REAL GAS    if xi * sigma_m < sigma_c(x) + sigma_gas,
      NULL        otherwise -> next stage.

Stage 0 of the scalar chain keeps the VSPG segment proposal (its exact
density correction only reshapes the stage-0 draw); the Stokes chain
keeps its plain truncated draw. Null re-draws and the classification
uniform come from rng.tau; the classification is drawn ONLY in shells
with c_maj > 0, so clear-sky and 1D streams are bit-identical to the
pre-field code (verified below).

### Expectation preservation (the telescoping proof)

The joint density of nulls at u_1 < ... < u_k and a real event at u is

    prod_{j=0..k} [e^{-(u_{j+1}-u_j)} / (1 - e^{-(T-u_j)})]
      x prod_{j=1..k} [1 - a(u_j)] x a(u),

while the accumulated weight is prod_{j=0..k} (1 - e^{-(T-u_j)}): every
stage normalizer cancels EXACTLY against its forced factor, leaving

    weight x density = e^{-u} prod_j (1 - a(u_j)) a(u).

Summing over k and integrating the ordered nulls over [0, u] gives
sum_k (1/k!) [INT_0^u (1-a)]^k = e^{u - INT_0^u a}, so

    E[weight at a real collision at u] = a(u) e^{-INT_0^u a} du,

the exact analog delta-tracking first-arrival law, for every u and
every path functional, with total real-collision weight (1 - e^{-tau_t})
in expectation (tau_t the TRUE combined tau). Vertex typing by the
sigma_c : sigma_gas split cancels the per-type coefficients as in the
1D mode. Full derivation in photon.rs at trace_secondary_chain_scalar's
`use_forced` block.

### Scope: per-wavelength chains only; ALIS stays analog

The chains this lands in are trace_secondary_chain (production Stokes)
and trace_secondary_chain_scalar: single-wavelength, so a null carries
NO weight ratio (the null material cancels exactly). The ALIS tracer
would need per-wavelength null ratios
(sigma_m - sigma_t_w)/(sigma_m - sigma_t_h) under an ALL-wavelength
majorant, with heavy tails whenever sigma_m approaches sigma_t_w: real
new machinery with real new failure modes, deliberately NOT built. ALIS
keeps its `field.is_none()` gate (analog under fields, unbiased), with
the doc note updated in place.

### Backstops (exact-to-f64 acceptance class)

- fp-exhausted budget (remaining T - u <= 1e-12): the chain is killed;
  the correct continuation carries weight * (1 - e^{-1e-12}), an
  expectation loss below f64 resolution (same class as
  FORCED_TAU_CUTOFF = 20).
- FIELD_NULL_EVENT_LIMIT = 512 events: the expected null count per
  flight is below T < 20; P(>512) is a Poisson(20) tail. Kill on hit.

### VSPG segment-buffer overflow fix (review round 2, same files)

scout_with_vspg_segments and _alis silently DROPPED segments past
VSPG_MAX_SEGMENTS = 128 while continuing to accumulate tau, so
vspg_sample_from_segments normalized by p_sum < 1 - e^{-tau_max}: head
collisions over-weighted, the tail never sampled. Only reachable through
reflection-multiplied crossings (a full 64-shell double crossing is
exactly 128), hence never in production geometry; still fixed exactly:
on overflow the LAST segment's tau_hi extends across the overflow tau
at neutral importance 1.0 (any positive importance is unbiased; only
the TILING of [0, tau_max] matters), so p_sum telescopes exactly, and a
relaxed atomic counter VSPG_OVERFLOW_EVENTS makes overflow observable.
New regression test `vspg_overflow_extends_last_segment_to_tile_tau_max`
builds a total-internal-reflection waveguide (grazing ray inside an
n = 1.5 shell traverses it once per scout iteration, 200 total) and
pins p_sum == 1 - e^{-tau_max} to 1e-12 relative on BOTH scout
variants: PASS (pre-fix it fails by the dropped tail).

## 2. Estimator gates (unit level, twilight-core default suite)

All in photon.rs tests, non-ignored (seconds each), all PASS:

- `field_forced_flight_matches_analytic_first_collision_law` (G-FF-LAW,
  the decisive null-loop referee): 400k forced flights along a fixed
  slant ray through a fine checkerboard deck, binned by path length,
  against direct fine-step integration of sigma_at: every bin of
  dP_cloud(t) = sigma_c(t) e^{-tau_t} dt and dP_gas likewise within
  5 per-bin SEs (+3e-4 relative floor), for the tight majorant AND a
  x3-inflated one (the law is majorant-invariant); total collision
  mass matches 1 - e^{-tau_t} to 3e-3 in both runs.
- `field_forced_uniform_matches_1d_forced_chains` (G-FF-EQ1D, chain
  level): on a uniform field whose majorant EQUALS sigma (nulls
  impossible by construction), field-forced chains match the
  already-gated exact 1D combined-channel chains: means agree to
  0.04 percent at matched seeds (band 3 SE + 2 percent).
- `field_forced_majorant_invariance` (G-FF-MAJ-INV): inflating the
  majorants (x4 uniform, x3 checkerboard) forces heavy null traffic;
  means agree within 3 combined SE + 2 percent on both fields.
- `field_shell_majorants_dominate_sigma_at`: pointwise domination of
  sigma_at by the per-shell majorants over every shell band (uniform:
  equality; checkerboard: 200 random probes per shell), and exact zero
  for shells outside the field z range.

## 3. Bit-identity vs HEAD (RNG discipline, empirical)

Reference dumps (`bitcheck_dump`, full f64 bit patterns of per-
wavelength radiance at 41 wavelengths per row set) run in this tree
and in a b776776 worktree with the identical harness block grafted;
1066 rows per tree, byte-compared:

| surface | SZAs | rows | result |
|:--|:--|--:|:--|
| clear-sky ALIS hybrid | 30/85/97/101/106 | 205 | IDENTICAL |
| clear-sky Stokes hybrid | 85/97/101 | 123 | IDENTICAL |
| 1D deck ALIS hybrid | 60/85/97/100 | 164 | IDENTICAL |
| 1D deck Stokes hybrid | 60/85/97/100 | 164 | IDENTICAL |
| 1D deck Multiple | 97 | 41 | IDENTICAL |
| field Multiple | 97 | 41 | IDENTICAL |
| field ALIS + Stokes, below gate | 88/92 | 164 | IDENTICAL |
| field ALIS, deep (stays analog) | 97/101 | 82 | IDENTICAL |
| field Stokes, deep (THE change) | 97/101 | 82 | 82/82 moved |

Every unchanged path is bit-identical, including the 1D forced regime
(the combined channel reads the same `atm.cloud_extinction` values
through the new caller-provided-channel plumbing) and ALIS under
fields at deep SZA. The only moved surface is exactly the one the
campaign redefines: per-wavelength Stokes chains under a field at
local SZA >= 96.

## 4. Variance ledger (the headline)

Seed statistics of the production Stokes hybrid at 550 nm on the
synthetic tau* = 3 uniform deck FIELD (the G3 deck as a 3D field),
12 seeds x 16000 photons, IDENTICAL protocol and budget in both trees:
FORCED = this tree (field-forced chains), ANALOG = b776776 worktree
(field chains analog, the pre-campaign estimator).

| SZA | forced mean | forced CV | analog mean | analog CV | reading |
|--:|--:|--:|--:|--:|:--|
| 99 | 4.222e-7 | 69% | 8.317e-7 | 195% | CV collapse 2.8x (same distribution center: the analog mean is 3 lottery seeds, range 1.7e-8..5.5e-6) |
| 101 | 5.846e-8 | 131% | 5.482e-9 | 74% | analog is the COLLAPSE class: mean 10.7x below forced (0.045x of the referee) with FALSE-precision CV around it; forced recovers to 0.48x of the referee with an honest tail |
| 103 | 2.013e-9 | 109% | 7.652e-10 | 119% | same class, deeper: forced recovers the mean 2.6x |

The naive CV ratio is only meaningful where both estimators are
centered (SZA 99: 2.8x collapse). At 101/103 the analog CV is false
precision around a collapsed mean, exactly the pathology the July-02
campaign documented for the pre-fix 1D path ("CVs 4.8-29.3% around a
mean 6x too low"); the honest ledger metric there is MEAN RECOVERY
toward the referee (10.7x at 101, 2.6x at 103) plus the forced side's
honest heavy-tail CV. The field-forced estimator thereby lands the
field path in the same class as the validated 1D forced path
(field 0.48x vs 1d 0.40x of the referee at tau*3/101/550, comparable
CVs), where before it was in collapse.

REAL PADBORG SEVIRI FIELD (the production medium: patchy broken cloud
over 49-60N from tools/cloud3d_seviri.py, /tmp/padborg_field.bin),
Stokes hybrid, 2000 photons x 8 seeds, Padborg observer, zenith view.
FORCED (this tree) all three referee wavelengths; ANALOG (b776776)
at 550 nm:

| SZA | wl | forced mean | forced CV | analog mean | analog CV | reading |
|--:|--:|--:|--:|--:|--:|:--|
| 99 | 450 | 5.893e-7 | 43% | | | |
| 99 | 550 | 4.916e-7 | 68% | 4.671e-7 | 82% | centered both; CV gain 1.2x |
| 99 | 650 | 1.958e-7 | 56% | | | |
| 101 | 450 | 1.565e-7 | 90% | | | |
| 101 | 550 | 4.734e-8 | 76% | 1.946e-8 | 128% | analog partially collapsed: forced recovers the mean 2.4x AND cuts CV 1.7x |
| 101 | 650 | 1.793e-8 | 65% | | | |
| 103 | 450 | 3.150e-8 | 161% | | | |
| 103 | 550 | 2.484e-8 | 174% | 3.767e-9 | 166% | analog in the collapse class: forced recovers the mean 6.6x |
| 103 | 650 | 5.698e-9 | 149% | | | |

A matched-budget 4k-photon synthetic A/B (queue jobs) corroborates the
16k table: the analog side at 4k reads 0.02-0.04x of the referee at
SZA 101-103 (collapse class) while the forced side stays in the
0.3-0.5x heavy-tail class with honest CVs.

NOTE on CV as a metric under collapse: a starved analog estimator can
show a SMALLER CV than the forced one around a mean an order of
magnitude too low (synthetic SZA 101: analog CV 74% around 0.045x of
the referee vs forced CV 131% around 0.48x). The ledger therefore
reports mean recovery and CV together; CV ratios alone are only
meaningful where both estimators are centered (SZA 99).

## 5. Ultra-deep referee campaign (tools/validate_libradtran.py --tier deep)

Referee: the same delta-scaled uniform 1-2 km deck construction as G3
(wc_properties hu + wc_modify tau/ssa/gg set), spherical 1D backward
MYSTIC, mc_vroom, mc_std, TSIS-1 solar table, zenith view, no
refraction. 3e8 photons per case at SZA 101, 1e9 at SZA 103 (the
mixed-budget cache keeps both generations valid). Twilight side:
path=1d is the CLI ALIS hybrid (the same estimator and protocol as the
July-02 G3 rows this extends), path=field is the production Stokes
per-wavelength chain on the equivalent horizontally uniform field via
the deep_referee_runner harness, 12 seeds x 16000 photons at 550 nm.

Verdicts: PASS/FAIL with band = 3 x sqrt(se_tw^2 + se_MYSTIC^2) +
5% x MYSTIC; LOW-POWER when the band exceeds half the referee value.
Full CSV: validation/deep_regime_results.csv. Summary (550 nm rows;
450/650 on the 1d path in the CSV):

| tau* | SZA | path | tw/MYSTIC | band/ref | verdict | tw seed CV | budget |
|--:|--:|:--|--:|--:|:--|--:|:--|
| 1 | 101 | 1d | 0.66 | 0.41 | PASS | 99% | 40 seeds x 16k |
| 1 | 101 | field | 1.08 | 1.35 | LOW-POWER | 137% | 12 x 16k |
| 1 | 103 | 1d | 1.27 | 1.08 | LOW-POWER | 166% | 40 x 16k |
| 1 | 103 | field | 0.73 | 1.06 | LOW-POWER | 154% | 12 x 16k |
| 3 | 101 | 1d | 0.40 | 0.67 | LOW-POWER | 172% | 12 x 16k |
| 3 | 101 | field | 0.48 | 0.62 | LOW-POWER | 131% | 12 x 16k |
| 3 | 103 | 1d/field | (referee 1e9 pending at assembly time) | | | | |

READING, honestly: the REFEREE is no longer the limit anywhere
(SE 6-7% at 3e8 for SZA 101, 8-10% at 1e9 for 103, vs 10-19% before).
The twilight side's heavy-tail seed CV (100-260% at 16k photons) is
now the binding noise. Moving rows out of LOW-POWER therefore takes
seed count, and the tau* = 1 SZA 101 rows DID move: at 40 seeds all
three wavelengths gate as PASS (0.59-0.66x with bands 0.38-0.49; the
one-sided-low deficit at fixed photon budget is the same converges-
from-below class the July-02 campaign measured and probed). The
tau* = 1 SZA 103 rows are now CENTERED (1.10-1.27x) but their CV
(166-179%) keeps them LOW-POWER at 40 seeds. tau* = 3 stays
LOW-POWER at every affordable budget: the chains under a tau* = 3
deck at SZA >= 101 remain tail-limited, the program's standing
residual (budget or cloud-capable BDPT are the known follow-ups).
The FIELD path now sits in the same class as the validated 1d path
(0.48x vs 0.40x at tau*3/101) instead of the collapse class: that
parity is this campaign's contribution; see the variance ledger.

## 6. Physics-constraint and estimator gates

- G-S3-EQ1D-DEEP (gate 4c, uniform-3D-field == 1D-deck at depth, both
  paths forced, per-wavelength scalar chain, 8 seeds x 8000, 550 nm):
  SZA 101: 1D 7.926e-8 vs field 7.921e-8 (ratio 0.999);
  SZA 103: 1D 2.207e-8 vs field 2.204e-8 (ratio 0.999). PASS. The two
  representations share seeds (common random numbers), so the 6e-4
  relative agreement is a high-power structural check: any
  representation-dependent term would break it loudly.
- G-S3-MONO (gate 4a): tau* ladder at SZA 97, 650 nm:
  tau*1 4.446e-6 > tau*3 2.804e-6, resolved at 2 combined SE. SZA
  ladder at tau* = 3, 550 nm: 99 (3.141e-7) > 101 (9.296e-8, k = 32)
  > 103 (2.158e-8, k = 32), both rungs resolved at 2 SE. PASS.
  FINDING kept in the gate header: clear sky is NOT a rung at 650 nm:
  the referee's own tau*1 row (4.663e-6) exceeds the validated
  clear-sky zenith radiance (2.774e-6 +- 0.057e-6): a thin low deck
  REDIRECTS the bright solar-horizon light into the dim red zenith
  and brightens it; the first draft failed on that wrong premise and
  the failure is physics, not estimator. Also documented: the referee
  shows tau*1 ~ tau*3 within its SE at SZA >= 101 (sidelight
  scattered in by the thicker deck compensates extinction), so a tau*
  ladder is only asserted where the referee resolves it (SZA 97).
- G-S3-CHI2 (gate 4b): three drafts preserved as findings; the
  instrument (chi2 of weighted-cubic log-fit residuals vs per-point
  seed SEs) is valid exactly where the seed SE is faithful. Draft 2
  (95-104, k = 6, ramped budgets) measured chi2 34.0/dof 9 with the
  SZA 103 point at 1.41e-9 +- 25% CLAIMED against the 1e9 referee's
  2.72e-8 (19x low, all seeds clustered under the unsampled tail) and
  a non-monotone 103 -> 104 lottery jump: SE-unfaithfulness, not
  transport bias. Draft 3 (dense 95-99.8, k = 6) measured chi2
  63.7/dof 12 with sign-ALTERNATING residuals and non-monotone
  neighbor jumps: seed SEs understated ~2.3x at k = 6 even at
  moderate depth. Final gate: 10 points over 95-99.5 at k = 12
  (seed-SE faithful regime, containing the ZENITH_SZA_START = 96
  forced-mode seam): chi2 11.6 on dof 6 (99.9% bound 22.46), max
  |z| 2.09, per-point se/m 0.09-0.59: residuals fully consistent with
  the per-point SEs. PASS: no hidden bias bends the twilight decay
  curve across the forced-mode turn-on.
- G-S3-CB (gate 1b, the decisive estimator gate): checkerboard
  fields (8 km cells, clear background) at SZA 97/100/103, 550 nm,
  field-forced per-wavelength scalar hybrid (8 seeds x 8000) vs the
  fully analog Multiple (8 seeds x 4e5, trajectory-independent,
  externally anchored family). tau* = 1 gated TWO-SIDED, tau* = 3
  gated one-sided [0.35x, +3SE+5%] (the heavy-tail budget class, see
  residuals). First row measured: tau*1 SZA 97 ratio 0.871, diff
  3.53e-6 within band 5.76e-6, PASS; the remaining rows were still
  computing at report time on a load-200+ box (log:
  scratchpad/queue_mine/g_s3_cb_v3.log; the gate is wired into the
  ignored suite for the orchestrator's rerun).
  TWO PRESERVED FINDINGS from earlier drafts: (i) at ZENITH view over
  a broken tau*1 deck with the observer under a clear cell, the hybrid
  reads 0.499x of Multiple with a false-tight 1.1% seed SE (all LOS
  seeds gas-only; the direction lobes rarely sample the off-axis cloud
  couplings): recorded as production residual 3. (ii) tau*3
  checkerboard at zenith measured 0.618x (8x8000), the same tail class
  as the uniform tau*3 rows.

## 7. Suite status and review-round-2 gate rewrites

Review-round-2 items (all in this campaign's files):

- VSPG overflow: fixed in both fused scouts (last-segment extension +
  VSPG_OVERFLOW_EVENTS counter); TIR-waveguide regression test pins
  p_sum == 1 - e^{-tau_max} to 1e-12 relative on 200-traversal walks
  in BOTH scout variants. PASS (fails pre-fix by the dropped tail).
- G-HYB-MULT budgets: 16 seeds everywhere, 2048 photons regime 1,
  48 seeds x 8192 at SZA 97. Measured 1D-deck rows: SZA 88 ratio
  1.019 band/ref 0.08, SZA 92 ratio 0.980 band 0.11, SZA 97 ratio
  1.005 band 0.18 (the <= ~20 percent review target, met). The
  3D-field SZA-97 branch is upgraded from one-sided (analog era) to
  two-sided.
- G-FORCED-1D: rewritten as a scale-free ratio gate,
  |hyb/mul - 1| < 3 x CV-derived SE + 5 percent. Measured: SZA 97
  ratio 1.149 within band 0.389; SZA 100 ratio 1.099 within band
  0.553 (budgets 8192 x 8 and 16384 x 24 hybrid seeds; a 4096 x 8
  probe drew a tail seed and produced a 1.64 band, which is why the
  SZA-100 budget is what it is). PASS, and inflation of any size now
  fails by construction.

Standing regression gates re-run on this tree: g_s2_eq1d reproduces
its recorded values (1D 2.56203e-3 vs field 2.56202e-3), G-S2 default
suites green (twilight-core 397 passed incl. the new field-forced and
overflow gates; twilight-cpu default suite green), cargo check
--workspace --all-targets clean, cargo clippy zero warnings.

Re-run standing gates, all green: g_s2_alis (diff 4.76e-4 < band
8.10e-3, reproducing the recorded July value), g_s2_gap_mc
(clear 3.245e-2 > gap 3.147e-2 > uniform 2.586e-3, reproduced),
stratus_twilight gate, g_s2_eq1d (reproduced to 6 digits).

FINAL SUITE PASS (this tree, 2026-07-04 23:00): twilight-core 397
passed / 0 failed (includes the four new field-forced estimator gates
and the VSPG overflow regression); twilight-cpu 103 passed / 0 failed
(16 ignored = the heavy MC gates run individually above); cargo check
--workspace --all-targets clean; cargo clippy --workspace: zero
warnings in every crate this campaign touches (2 pre-existing
warnings live in twilight-gpu/src/tests.rs, the wgpu port's
uncommitted work, outside this campaign's ownership).

STILL IN FLIGHT at report time (all detached with logs, harmless to
re-run): G-HYB-MULT k = 48 field-deck rows
(queue_mine/g_s2_hybmult_k48.log; its 1D rows are the ones recorded
above), G-S3-CB rows 2-6 (g_s3_cb_v3.log), the 1e9 top-up of the
tau* = 3 SZA 103 MYSTIC trio (validation/deep_topup_1e9.log; the
table falls back to the completed 3e8 generation for those three
referee cells until it lands).

## 8. Residuals, stated precisely

1. HEAVY-TAIL BUDGET DEFICIT UNDER tau* = 3 AT SZA >= 101 (both deck
   representations, both chain families): the twilight means at 16k
   photons read 0.37-0.48x of the ultra-budget referee with honest
   seed CVs of 100-260%, converging from below as budget grows (the
   July-02 probe class). The referee side no longer limits anything
   (SE 6-10%); closing these rows needs either campaign-scale seed
   counts (the tau* = 1 SZA 101 rows DID close at 40 seeds) or
   cloud-capable BDPT (the standing follow-up).
2. SEED-SE UNFAITHFULNESS in the tail regime: at k <= 8 seeds the
   seed SE understates the sampling error wherever the tail is
   unsampled (measured: chi2 drafts 2-3; the 103-point false-precision
   cross-check vs the 1e9 referee). Every gate this campaign ships
   either sizes k for faithfulness (chi2 k = 12, mono k = 32,
   G-HYB-MULT k = 48 at SZA 97) or gates one-sided with floors.
3. ZENITH VIEW OVER A BROKEN DECK (new finding): with the observer
   under a CLEAR cell of a checkerboard tau* = 1 deck at SZA 97, the
   hybrid reads 0.499x of the analog Multiple with a FALSE-TIGHT 1.1%
   seed SE: all LOS seeds are gas-only and the chain direction lobes
   rarely sample the down-and-sideways couplings to off-axis cloud.
   Unbiasedness at that geometry is pinned by the flight-law gates;
   the deficit is importance-sampling starvation. Production zenith
   scans over broken decks inherit this at chain budgets; the slant
   views (vz 80, the khayt geometry class) do not thread a single
   cell column and are gated two-sided instead. Follow-up candidates:
   a lateral-cloud seed lobe (aim chains at the nearest occupied
   macrocell) or cloud-capable BDPT.
4. ALIS UNDER FIELDS stays analog (per-wavelength null ratios under an
   all-wavelength majorant are the deferred machinery); the production
   scalar path under fields therefore keeps analog-class variance. The
   polarized (Stokes) production path is field-forced.
5. The per-shell majorant is lat/lon-independent: rays through clear
   parts of occupied shell bands pay null-collision overhead, and
   grazing chords through decked shells exceed FORCED_TAU_CUTOFF and
   fall back to analog (exactly as the 1D forced mode always did).
   Macrocell-local per-segment majorants are the known refinement if
   field-forced acceptance ever becomes rate-limiting.
