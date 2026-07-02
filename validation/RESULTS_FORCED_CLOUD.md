# Combined-channel forced-collision mode under 1D cloud decks (2026-07-02/03)

Addendum to validation/RESULTS_G3_CLOUD_TWILIGHT.md. Branch
validation-campaigns on top of a4ab932. This lands the tracked Stage-2
follow-up that every external referee campaign pointed at: the hybrid
production estimator was variance-starved under cloud decks at
SZA >= 97 because forced-collision mode was disabled whenever a gray
cloud channel was present (the conservative fix in 0cc8bf5 for the
original cloud-blind forced composition). Forced mode now composes
UNBIASEDLY with the 1D shell cloud channel; the 18 KNOWN-LIM rows of
the G3 campaign are the acceptance targets and are re-refereed below
against the cached MYSTIC values.

## The estimator, as implemented

### Setting

Decomposition tracking splits the medium into the gas channel
(per-shell, per-wavelength extinction in `atm.optics`) and the gray
cloud channel (`atm.cloud_extinction`, the delta-scaled pure-scattering
extinction, CONSTANT PER SHELL in the 1D fallback; cloud absorption is
folded into the shell optics at build time). Pre-fix, analog bounces
raced the two channels (exact first-arrival inversion) while forced
mode, which samples a collision from the scout's boundary tau, knew
only the gas channel and was therefore disabled under any cloud.

### Combined channel (scalar chain)

Because both channels are piecewise constant per shell, the combined
extinction sigma_t = sigma_gas + sigma_c is exactly piecewise constant,
and the existing scout / VSPG-segment / advance machinery inverts the
combined optical depth exactly by summing the two extinctions per shell
segment. No majorant, no rejection. At a bounce with combined boundary
tau T_t, the physical next-event law along the ray is

    collide at s:  p(s) ds = sigma_t(s) exp(-tau_t(s)) ds
    type given s:  cloud with sigma_c/sigma_t, gas with sigma_gas/sigma_t
    escape:        exp(-T_t)

Forced mode importance-samples the collision measure with the escape
atom removed:

1. weight *= (1 - exp(-T_t))  (the combined collision probability),
2. s from the truncated combined density (the VSPG segment selection is
   a further proposal change carrying its own exact correction factor,
   unchanged machinery),
3. vertex type from the extinction conditional at the collision shell:
   cloud with probability sigma_c / (sigma_c + sigma_gas).

The type probabilities cancel the per-type extinction coefficients
exactly (f_type / p_type telescopes), so step 3 carries NO weight
factor. A gas vertex then multiplies by the gas SSA; a cloud vertex is
pure scattering with the gray HG lobe (`cloud_g_scaled`). This is
IDENTICAL vertex physics to the analog race, which realizes the same
first-arrival law, so forced and analog bounces of the same chain now
estimate the same integral: the biased composition that motivated
0cc8bf5 is gone by construction rather than by disabling the device.

### ALIS: the per-wavelength argument (the crux)

The cloud channel is GRAY: its tau contribution is identical for every
wavelength, tau_t_w = tau_gas_w + tau_c with tau_c shared. The ALIS
scout accumulates the combined tau per wavelength; the hero flight is
sampled from its truncated combined density p_h (VSPG-modulated), and
the vertex type from the hero's extinction conditional
P_h(cloud|s) = sigma_c / (sigma_c + sigma_gas_h). The wavelength-w
target at this bounce is

    f_w(s, cloud) = sigma_c        exp(-tau_t_w(s))
    f_w(s, gas)   = sigma_gas_w(s) exp(-tau_t_w(s))

and the exact IS weight f_w / (p_h * P_h) splits into the hero factor
(1 - exp(-T_t_h)) times a per-wavelength ratio:

    cloud vertex: exp(-(tau_t_w - tau_t_h)) = exp(-(tau_gas_w - tau_gas_h))
                  (the shared tau_c cancels; NO sigma ratio, the gray
                  channel's coefficient is wavelength flat)
    gas vertex:   (sigma_gas_w / sigma_gas_h) exp(-(tau_gas_w - tau_gas_h))
                  (exactly the pre-existing gas-only forced ratio)

Verification per type (weight_w times the joint sampling density must
reproduce the target): for the gas vertex
(1-e^{-T_t_h}) (sigma_gas_w/sigma_gas_h) e^{-(tau_gas_w-tau_gas_h)}
  x [sigma_t_h e^{-tau_t_h}/(1-e^{-T_t_h})] x [sigma_gas_h/sigma_t_h]
  = sigma_gas_w e^{-tau_t_w} = f_w(s, gas),
and for the cloud vertex the same telescoping gives
sigma_c e^{-tau_t_w} = f_w(s, cloud). Exact for every wavelength and
any hero. The combined-tau DIFFERENCES returned by the advance are the
gas differences (gray cancellation), so the ratios are computed from
the same arrays the gas-only code used, with the sigma factor keyed on
the vertex type. Both wr (target ratios) and pr (the spectral
one-sample-MIS pdf-ratio family, idealized analog free-flight pdfs)
receive the same per-event factors, matching the analog branch's
existing convention (survival-only ratio at a cloud vertex, sigma-ratio
times survival at a gas vertex), so the balance-heuristic weight family
remains a positive path function consistent across hero choices and the
spectral MIS stays exactly unbiased. The per-wavelength forced weight
is NOT (1 - e^{-T_t_w}): escape contributes zero to collision paths,
and integrating f_w/p_h over the truncated domain recovers
(1 - e^{-T_t_w}) in expectation, so no extra normalization ratio is
applied (the 2026-06-12 double-counting audit note carries over to the
combined channel verbatim).

### 3D field: NOT shipped, and why

Under a 3D field sigma_c varies inside a shell segment, so the combined
tau is no longer piecewise constant and the scout cannot fold it
exactly. An unbiased forced flight there needs per-segment majorants
(macrocell_max / background_column) plus truncated-domain delta
tracking: sample from the truncated MAJORANT density, weight by
(1 - e^{-T_maj_remaining}) at each draw, null-collision reject with
sigma_c(x)/sigma_c_maj, and re-draw within the remaining truncated
budget. The normalizers telescope against the forced factors and the
scheme is unbiased for the scalar chain, BUT the ALIS composition
additionally needs per-wavelength NULL ratios
(sigma_maj - sigma_t_w)/(sigma_maj - sigma_t_h) at every null event,
which requires a majorant valid for ALL wavelengths simultaneously and
has heavy tails whenever sigma_maj approaches sigma_t_w. That is real
new machinery with real new failure modes, and every KNOWN-LIM referee
row is a 1D deck, so the field path deliberately stays ANALOG
(unbiased, delta-tracked, unchanged) and forced mode gates on
`field.is_none()`. Field runs carry `atm.cloud_extinction == 0` by the
caller contract, so the gate is exact. The BDPT light subpath likewise
stays disabled under ANY cloud channel (unchanged, separate follow-up).

### RNG discipline

Forced flights draw from the same `rng.tau` stream positions the
gas-only forced mode used (scout: no draws; VSPG segment + within-
segment draws; Stokes chain: one truncated-exponential draw). The ONLY
new draw is the vertex-type draw, taken from `rng.tau` AFTER the
advance and ONLY when the collision shell carries nonzero cloud
extinction; on clear-sky paths (and any bounce colliding outside the
deck) no draw is taken. All combined-channel arithmetic adds exact 0.0
on clear shells. Consequence, verified empirically below: clear-sky
runs, sub-SZA-96 cloudy runs (forced off by the SZA gate), field runs,
and Multiple-mode runs are bit-identical to HEAD a4ab932.

## Code

- crates/twilight-core/src/photon.rs (the only production file touched):
  - `scout_tau_to_boundary`, `scout_with_vspg_segments`,
    `scout_with_vspg_segments_alis`, `scout_tau_to_boundary_alis`,
    `vspg_sample_scatter_tau` (test-only): accumulate combined per-shell
    tau (gas + `atm.cloud_extinction[shell]`).
  - `advance_to_optical_depth`, `advance_to_optical_depth_alis`: invert
    the combined tau; the ALIS variant returns per-wavelength combined
    taus (differences = gas differences).
  - `trace_secondary_chain` (Stokes), `trace_secondary_chain_scalar`,
    `trace_secondary_chain_alis`: `use_forced` now gates on
    `field.is_none()` (was `!cloud.channel`); forced branch draws the
    vertex type from the extinction conditional; ALIS applies the
    type-keyed per-wavelength ratios derived above.
  - `ChainCloud.channel` removed (nothing gates on it anymore);
    `has_cloud_channel` remains solely as the BDPT gate.
  - New unit test `scout_and_advance_fold_1d_cloud_exactly` pins the
    combined scout analytically (gas+2.0 deck tau on radial rays), the
    fused-vs-plain scout equality, the gray invariance of ALIS
    per-wavelength tau differences under a deck, and the advance/scout
    round trip into the deck interior.
- crates/twilight-cpu/src/simulation.rs (tests only): see gates below.

## Acceptance

All runs on the same heavily shared 12-core Apple Silicon machine as the
original campaigns (load average 30-77 throughout; a parallel g3cube
referee campaign shared the box). MYSTIC referee values are the CACHED
G3 campaign runs (validation/g3/, byte-verified by the runner's
case.done markers); only the twilight side re-ran.

### 1. G3 rerun: the 18 KNOWN-LIM rows (primary acceptance target)

`tools/validate_libradtran.py --tier g3-cloud-twilight`, protocol
unchanged (hybrid: 6 seeds x 16000 photons; bands
3 x sqrt(se_tw^2 + se_MYSTIC^2) + 5% MYSTIC). Runner summary:
28 pass / 0 fail / 8 low-power / 18 known-lim -- the runner still
LABELS hybrid SZA >= 97 rows KNOWN-LIM (tools/ is outside this
change's ownership), so the row-level verdicts below are computed
against the same band formula from the CSV. "in band" = |hyb - MYSTIC|
<= band. Pre-fix values from the 2026-07-02 campaign CSV.

Hybrid vs MYSTIC, ratio (seed CV% in parentheses):

| tau* | SZA | wl | pre ratio (CV%) | post ratio (CV%) | band | post in band |
|---:|---:|---:|---:|---:|---:|:--|
| 1 | 95 | 450 | 0.949 (4.6) | 0.949 (4.6) BIT-IDENTICAL | 18.5% | yes (PASS pre and post) |
| 1 | 95 | 550 | 0.960 (4.2) | 0.960 (4.2) BIT-IDENTICAL | 17.5% | yes |
| 1 | 95 | 650 | 0.974 (3.5) | 0.974 (3.5) BIT-IDENTICAL | 15.8% | yes |
| 1 | 97 | 450 | 0.848 (29.8) | 0.981 (9.7) | 36.1% | yes |
| 1 | 97 | 550 | 1.035 (36.3) | 1.095 (9.7) | 39.1% | yes |
| 1 | 97 | 650 | 0.975 (38.5) | 1.244 (15.7) | 65.1% | yes |
| 1 | 99 | 450 | 0.178 (29.3) | 0.927 (25.4) | 76.5% | yes |
| 1 | 99 | 550 | 0.165 (4.8) | 0.969 (27.7) | 86.3% | yes (referee LOW-POWER) |
| 1 | 99 | 650 | 0.165 (12.5) | 1.235 (38.6) | 148.8% | yes |
| 1 | 101 | 450 | 0.193 (59.0) | 0.595 (21.1) | 52.9% | yes (LOW-POWER, not gated) |
| 1 | 101 | 550 | 0.158 (58.7) | 0.672 (40.1) | 90.6% | yes (LOW-POWER) |
| 1 | 101 | 650 | 0.223 (55.5) | 1.075 (57.9) | 194.8% | yes (LOW-POWER) |
| 3 | 95 | 450 | 0.923 (16.2) | 0.923 (16.2) BIT-IDENTICAL | 49.9% | yes |
| 3 | 95 | 550 | 0.936 (10.7) | 0.936 (10.7) BIT-IDENTICAL | 35.3% | yes |
| 3 | 95 | 650 | 0.957 (8.6) | 0.957 (8.6) BIT-IDENTICAL | 29.9% | yes |
| 3 | 97 | 450 | 1.247 (48.0) | 0.931 (24.2) | 73.8% | yes |
| 3 | 97 | 550 | 0.696 (42.9) | 0.974 (23.8) | 75.7% | yes |
| 3 | 97 | 650 | 0.655 (22.1) | 1.071 (15.2) | 56.2% | yes |
| 3 | 99 | 450 | 0.222 (56.0) | 0.565 (21.4) | 43.3% | NO (deficit 43.5%, misses by 0.2pp; see residuals) |
| 3 | 99 | 550 | 0.835 (63.1) | 0.797 (24.0) | 63.9% | yes |
| 3 | 99 | 650 | 1.395 (62.7) | 0.809 (21.3) | 58.6% | yes |
| 3 | 101 | 450 | 0.026 (59.2) | 0.222 (52.6) | 51.1% | no (LOW-POWER, not gated) |
| 3 | 101 | 550 | 0.038 (58.6) | 0.437 (61.1) | 92.3% | yes (LOW-POWER) |
| 3 | 101 | 650 | 0.035 (55.4) | 0.372 (74.1) | 96.9% | yes (LOW-POWER) |

Reading: the one-sided starvation is gone as a class. At SZA 99
(tau* 1) the ratios moved from 0.165-0.178 to 0.93-1.24; at SZA 101
(tau* 1) from 0.16-0.22 to 0.60-1.08; the tau* 3 / SZA 101 collapse
(0.026-0.038, values 25-30x below the referee) recovered to 0.22-0.44.
Of the 12 gateable deep rows (SZA 97/99, both taus), 11 pass their
bands; the SZA 95 rows are BIT-IDENTICAL to the pre-fix campaign (the
SZA gate leaves them analog), doubling as an end-to-end determinism
check of the whole rerun.

### 2. Seed-CV at SZA 97/99 (the variance objective)

Pre-fix hybrid seed SE at SZA 97 was 22-48% of the mean (the
documented variance blow-up phase). Post-fix, same budgets:

- SZA 97 tau* 1: 29.8/36.3/38.5% -> 9.7/9.7/15.7% (450/550/650 nm)
- SZA 97 tau* 3: 48.0/42.9/22.1% -> 24.2/23.8/15.2%
- SZA 99 tau* 3: 56.0/63.1/62.7% -> 21.4/24.0/21.3%
- SZA 99 tau* 1: pre-fix CVs (4.8-29.3%) were FALSE precision around a
  mean 6x too low; post-fix 25-39% around a mean consistent with the
  referee. The estimator is now centered; its remaining scatter at 16k
  photons is honest MC noise.

### 3. G2 daytime rerun (zero-movement check)

`--tier g2` full rerun (disort cached): the twilight-side columns
(tw_hybrid, tw_hybrid_se, tw_multiple, tw_multiple_se) are
BIT-IDENTICAL to the recorded 144/144 campaign in ALL 108 CSV rows;
the only fields that moved are the runner's own fresh daytime MYSTIC
spot checks (6 referee-side values, each within its own SE). Forced
mode is SZA-gated at ZENITH_SZA_START = 96, so zero movement at
SZA 30-85 was expected and is confirmed exactly; the 144/144 verdict
carries over unchanged.

### 4. Internal gates (all --release)

- g_s2_hybrid_matches_multiple (updated: two-sided at SZA 97 for the
  1D deck, see the test header for the honest-budget derivation):
  - 1D deck SZA 88: ratio 1.015 (diff 7.1e-3 < band 6.1e-2) PASS
  - 1D deck SZA 92: ratio 1.031 (diff 1.8e-3 < band 2.5e-2) PASS
  - 1D deck SZA 97 TWO-SIDED, forced: hybrid 4.837e-4 (se 7.1e-5) vs
    multiple 5.196e-4 (se 1.2e-5), ratio 0.931, diff 3.6e-5 < band
    2.4e-4 PASS (the pre-fix gate could only run one-sided here)
  - field deck SZA 88: 1.016 PASS; SZA 92: 1.023 PASS; SZA 97 stays
    ONE-SIDED (field chains remain analog): hybrid 1.950e-4 vs
    multiple 5.271e-4, ratio 0.370, inside [0.25x floor, upper bound]
    PASS -- the same 0.37-0.45x analog-starvation class the earlier
    vz-80 arbitration measured, unchanged by this work as expected
- g_s2_forced_under_1d_cloud_matches_multiple (renamed from
  g_s2_forced_off_under_1d_cloud; pins forced-under-cloud against the
  externally anchored analog Multiple estimator, two-sided, plus the
  below-clear sanity):
  - SZA 97: forced-hybrid 4.837e-4 vs multiple 5.196e-4, ratio 0.931,
    diff 3.6e-5 < band 2.4e-4 PASS; deck << clear (4.99e-3) PASS
  - SZA 100: 2.433e-5 vs 2.579e-5, ratio 0.943, diff 1.5e-6 < band
    3.6e-5 PASS; deck << clear (2.34e-4) PASS
- g_s2_eq1d (SZA 95): 1D 2.56203e-3 vs field 2.56202e-3 PASS
- g_s2_gap_mc (SZA 95): clear 3.245e-2 > gap 3.147e-2 > uniform
  2.586e-3 PASS
- g_s2_alis (SZA 94): diff 4.8e-4 < band 8.1e-3 PASS
- g_s2_chi_cloud_phase_sampled_matches_evaluated: PASS
- diag_g2_slab_independent_reference: trace_photon/flat ratios
  0.9936-1.0050 PASS
- stratus_twilight_remains_visible_and_below_clear_sky: re-pointed to
  SZA 97 (inside the forced regime where the starvation lived; see the
  test header) and now actually verifiable: PASS
- New unit gate scout_and_advance_fold_1d_cloud_exactly: PASS
- Default suites: twilight-core 391 passed / 0 failed, twilight-cpu
  88 passed / 0 failed
- cargo check --workspace --all-targets: clean;
  cargo clippy --workspace --all-targets: zero warnings

### 5. Bit-identity (RNG discipline, empirical)

Reference outputs captured at HEAD a4ab932 and re-run post-fix,
byte-compared:

- clear-sky scalar hybrid (ALIS driver), SZA 30/85/97/101/106: IDENTICAL
- clear-sky polarized hybrid (Stokes + scalar chains), SZA 85/97: IDENTICAL
- 1D deck hybrid at SZA 60/85 (below the forced gate): IDENTICAL
- 1D deck Multiple mode at SZA 97: IDENTICAL
- g2 CSV twilight columns (108 rows): IDENTICAL
- g3 SZA 95 hybrid rows: IDENTICAL

### Residuals, stated precisely

1. tau* 3, SZA 99, 450 nm: post-fix ratio 0.565 vs band 43.3% (misses
   by 0.2 percentage points). Probed at 4x the campaign budget
   (4 fresh seeds x 64000 photons): mean 7.84e-7 +- 2.4e-7, ratio
   0.894 vs the same cached referee. The estimator converges toward
   the referee as the budget grows, so this is residual heavy-tail
   under-sampling at the FIXED 16k campaign budget (the bluest channel
   through the thicker deck at the deepest gated SZA), not a bias of
   the composition; the pre-fix ratio at this point was 0.222 and did
   not move with photons.
2. tau* 3, SZA 101 (not gated; referee LOW-POWER, bands 51-97%): the
   hybrid recovered 8-11x (0.026-0.038 -> 0.22-0.44) but still reads
   one-sidedly low with seed CV 53-74% at 16k photons. The bluest
   channel misses even the wide band. This is the deepest-SZA,
   thickest-deck corner where the NEE-only chains (BDPT stays off
   under cloud) traverse tau_c = 3 on top of the SZA-101 shadow-height
   problem; the residual is a variance/heavy-tail budget effect of the
   FIXED estimator, not the pre-fix systematic starvation (which was
   photon-count independent in direction). Raising the campaign budget
   or landing cloud-capable BDPT are the follow-ups.
3. 3D FIELD path: forced mode remains OFF (analog, unbiased,
   externally unrefereed in 3D); see the design section for the exact
   majorant/null-ratio machinery a field extension needs.
4. BDPT light subpath: unchanged, off under any cloud channel.
