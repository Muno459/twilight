# Cloud-coupled BDPT (light-subpath) — engineering plan

Status: PHASE 1 IMPLEMENTED (unbiased foundation), PHASE 2 IN PROGRESS (variance).
Author pass: field-mapped against the current tree, cross-verified by four independent
code surveys plus direct reads, then implemented and measured.

## EMPIRICAL RESULT (2026-07-08) — READ FIRST

The Phase 1 estimator port (combined-channel vertex-type draw in `trace_light_subpath`,
gate relaxed to `field.is_none()`) is DONE and builds clean. Measured behavior at the
deep 1D-deck cells (wl 550, `hybrid_scatter_radiance_alis`, refereed vs cached MYSTIC):

- UNBIASED where resolvable: SZA 101 / tau*=3 tracks MYSTIC (1.54e-7 vs 1.22e-7, within
  ~1 se); SZA 103 / tau*=1 climbs to ratio 0.72 (1.42e-8 vs 1.96e-8) at 128 seeds. The
  code mirrors the validated backward-chain `use_forced` math exactly, so unbiasedness is
  expected; SZA 103 / tau*=3 mean climbs monotonically toward MYSTIC as seeds grow
  (5.5e-10 @16 -> 1.45e-9 @64 -> 4.0e-9 @128 seeds; ratio 0.27, still under-resolved).
- BUT HEAVY-TAILED at thick decks: SZA 103 / tau*=3 shows cv 1.4 -> 4.8 -> 8.9 as seeds
  grow (cv GROWING with N = heavy/near-infinite-variance tail), min 1.3e-10 / med 3.4e-10
  / max 4.05e-7 (one seed = 27x the true value, 100x the sample mean). The tail scales
  with deck optical thickness: tau*=1 cv 3.4 / max-mean 26 vs tau*=3 cv 8.9 / max-mean 100.

CONCLUSION: enabling BDPT under the deck is a CORRECT, UNBIASED foundation and helps thin
decks, but it does NOT by itself close the thick-deck LOW-POWER cells: it trades NEE's heavy
tail for a BDPT connection-singularity heavy tail. The deep-cell variance is still open.

DIAGNOSIS: the spikes are rare high-weight connections (the classic BDPT near-connection
1/r^2 singularity, g_term_ds). A near-LOS light vertex -> huge connection weight, amplified
by the forward-peaked gray-cloud HG lobe (g~0.85) at in-deck (cloud) vertices. NEE (backward
chain) samples exactly those near-LOS deck vertices WELL (no 1/r^2 spike; sun at infinity),
while the BDPT connect strategy samples them BADLY. So the principled fix is per-path MIS
between the BDPT connection and NEE (balance heuristic on the two path densities), which
down-weights the spiky near-connections and defers them to NEE -- unbiased, and it CANCELS
the singularity. This is the pdf/reverse-density machinery the codebase deliberately never
built (the current w_bdpt/w_back is a fixed SZA sigmoid, not per-path MIS).

REVISED FRAMING vs section 1: the forward/adjoint pdf MIS is NOT needed for UNBIASEDNESS
(the convex blend is unbiased regardless), but it IS needed for VARIANCE (to close the
cells). So the "real BDPT MIS" is back on the critical path -- for variance, not bias --
and refraction stays off in the referee so no ray-bending Jacobian is required for the gate.

ATTRIBUTION (BDPT_FORCE_GAS_VERTICES toggle, SZA 103 / tau*=3, 64 seeds): baseline (cloud
vertices ON) and FORCE_GAS=1 (cloud vertices scored as gas) are BYTE-IDENTICAL -- same
mean 7.43e-9, same cv 6.79, same max 4.05e-7. => across 64 seeds x 4096 subpaths NOT ONE
vertex was a cloud vertex. VSPG biases the light-subpath scatter to 50-80 km; the deck is
1-2 km; the forced-scatter never lands in the deck. So:
  - The combined-channel vertex-type draw (Phase 1) is CORRECT but DORMANT at these cells
    (old gas-only scoring is numerically identical). It is the necessary partner to a
    deck-aware light importance, not a standalone fix.
  - The heavy tail is the GENERIC gas-vertex BDPT connection singularity, amplified in the
    thick-deck regime by attenuation reshaping the signal.
  - ROOT CAUSE: under a thick deck the multiple-scatter signal is dominated by light that
    scattered INSIDE the deck, but the light subpath (VSPG->high altitude) never samples the
    deck. BDPT is structurally mismatched to the thick-deck regime.

THE REAL FIX (Phase 2, substantial) = TWO coupled estimator changes:
  (i) DECK-AWARE light-subpath importance: make VSPG (or an added strategy) place a
      controlled fraction of light vertices INSIDE the deck where the thick-deck signal
      originates. This activates the (already-built) cloud-vertex scoring and lets BDPT
      sample the dominant signal directly instead of via rare high connections.
  (ii) PER-PATH connection<->NEE MIS (balance heuristic on the two path densities): tame the
      near-connection 1/r^2 spikes that (i) will INCREASE (in-deck vertices connect to nearby
      in-deck LOS points). Needs the reverse/area-measure pdf machinery the tree never built.
Both are needed; either alone is insufficient. This is a multi-day estimator-research effort
with uncertain payoff (BDPT may still struggle with thick-deck high-order diffusion).

WHAT PHASE 1 DELIVERS AS-IS (mergeable, honest): unbiased BDPT under 1D decks (helps thin
decks; removes the documented `!cloud_channel` limitation), clear-sky byte-identical, 3D
fields still correctly gated off. It does NOT close the thick-deck LOW-POWER cells. Those
remain LOW-POWER (honest) pending Phase 2, or can be brute-forced at campaign-scale seeds.

## PHASE 2 (i) RESULT (2026-07-08) — DECK-AWARE IMPORTANCE IS FUTILE (measured)

Implemented deck-aware light-subpath importance (BDPT_DECK_IMPORTANCE_BOOST multiplies VSPG
importance for cloud shells in the light-subpath scout only; weight correction keeps it
unbiased; clear-sky bit-identical). Added BDPT_CLOUD_VERTEX_COUNT (vertices landing in a
cloud shell). Measured at SZA 103 / tau*=3, 32 seeds, wl 550:

  deck_boost  cloud_verts   mean         cv     max
  1           27852         3.6963e-10   0.941  1.88e-9
  1000        63135         3.6963e-10   0.941  1.88e-9
  100000      63964         3.6963e-10   0.941  1.88e-9

The deck IS reached (tens of thousands of in-deck vertices), but boosting deck sampling
2.3x leaves the radiance, cv, AND max BYTE-IDENTICAL. => in-deck light vertices contribute
~0 to the observer. Mechanism: a single forward scatter only reaches the 1-2 km deck on
DESCENDING rays, which carry no twilight signal; the signal-carrying rays graze the high
atmosphere and never reach the deck. So deck-aware importance on a 1-vertex light subpath
cannot help. (Corrects the earlier "never reaches the deck" read: it reaches a USELESS part
of the deck.) Instrumentation left in, default-off (bit-identical).

CONSEQUENCE: the tractable full-fix path (deck importance + connection MIS on 1-vertex BDPT)
is dead. The only remaining BDPT avenue is 2-VERTEX light subpaths (sun -> high scatter ->
observer-side deck -> connect) + connection MIS. 2-vertex is known to reach more signal but
regressed CV 2.7-8.6x (photon.rs:1630, reverted twice); the MIS is the unbuilt piece meant
to tame that. This is a materially larger, more speculative build. Strong signal that BDPT
is the wrong tool for thick-deck diffusion; a diffusion-aware forced/analog importance may
be the better-matched estimator (the "reconsider the estimator" option).

## PHASE 2 (2-VERTEX) RESULT (2026-07-08) — 2-VERTEX IS ALSO FUTILE; FIX IS BOUNCE-0 MIS

Added a runtime BDPT_ACTIVE_LIGHT_VERTS toggle (const cap 2, default 1 = bit-identical;
drives the light loop AND both backward-chain `bdpt_covered` sites). Measured SZA 103 /
tau*=3, 48 seeds, wl 550:

  verts  deck_boost  cloud_verts  mean        cv     max
  1      1           41456        9.77e-9     5.94   4.05e-7
  2      1           163280       9.16e-9     6.31   4.05e-7
  2      1000        278339       9.16e-9     6.31   4.05e-7

The 2nd vertex (even boosted into the deck) yields 4-7x more in-deck vertices but leaves the
mean, cv, AND the exact max (4.05e-7) unchanged => the 2nd vertex contributes ~0. The entire
signal AND the entire heavy tail are the BOUNCE-0 (order-2) connection, dominated by a single
near-connection spike (max 4.05e-7 = 27x the true value, ~90% of the 48-seed mean).

=> THE FIX IS NOT 2-VERTEX. It is a per-path balance-heuristic MIS between the two strategies
that sample the order-2 path sun->X->Y->observer (X = scatter vertex, Y = LOS point):
  - BDPT-connect: X ~ p_L(X) (light subpath, forward from sun), connect to Y.
  - Backward+NEE: Y from eye LOS, X ~ p_E(X|Y) (backward chain scatters Y->X), NEE X->sun.
  Y is sampled identically (deterministic eye LOS) in both, so its density cancels.
  w_bdpt(X) = p_L(X) / (p_L(X) + p_E(X|Y));  w_nee(X) = p_E / (p_L + p_E).  Sum = 1 => unbiased.
Near-connection X (close to Y): p_E >> p_L (backward from Y samples nearby X densely), so
w_bdpt -> 0 (the spike is suppressed) and NEE (which samples it well) takes over. Far X:
p_L dominates, BDPT keeps it. This CANCELS the singularity, unbiased, replacing the current
fixed SZA-sigmoid w_bdpt/w_back with a real per-connection balance heuristic.

WHAT THIS NEEDS (the green-field piece): both strategies must evaluate BOTH densities at their
X, in the same (volume) measure. BDPT-connect has X,Y so both are computable. Backward+NEE
has X,Y from its walk and must evaluate p_L(X) at an arbitrary X (light-subpath volume pdf:
entry pdf x forced-scatter density x VSPG x phase). pdf_fwd (currently accumulated, dead) is
meant to BE p_L along the light subpath; the backward side needs it as a closed-form of X.
Refraction is OFF in the referee so no ray-bending Jacobian. This is the multi-day core.

## PHASE 2 MIS DESIGN — LOCKED (2026-07-08), ready to implement

STRUCTURE (verified by mapping): backward chain starts at the LOS quadrature vertex Y
(=scatter_pos). Its bounce_idx=k NEE contributes an order-(k+2) path. bounce_idx=0 NEE and
the BDPT bounce-0 connection BOTH estimate the identical order-2 path sun->X->Y->observer;
Y is the deterministic eye LOS vertex in both, so its density cancels. Current w_bdpt/w_back
= fixed SZA sigmoid (bdpt_strength), a convex blend (unbiased, not per-path). pdf_fwd is an
isotropic PLACEHOLDER (sigma_h*INV_4PI), never used.

TWO TECHNIQUES sample X (order-2 vertex), volume measure at X, sigma_h(X) common (cancels):
  p_nee(X)  = phase_eye * INV_4PI * e^{-tau_YX} / d^2            [backward extends Y->X]
              (phase_eye = eye-scatter phase at Y into Y->X dir; d=|X-Y|; tau_YX=connection OD)
  p_bdpt(X) = e^{-tau_light} / ( entry_weight * (1 - e^{-tau_max_light}) )   [collimated light beam]
              (light ray is -sun_dir; entry = X projected to TOA along +sun_dir;
               entry_weight = BDPT_R_DELTA*2*BDPT_PHI_HALF_WIDTH*toa_r_sq*r_frac;
               tau_light = OD entry->X; tau_max_light = OD entry->far boundary.
               MEASURE: entry_pdf is per TOA area; /cos_inc converts to per-perp-area of the
               collimated beam, and entry_pdf = cos_inc/entry_weight, so cos_inc cancels ->
               1/entry_weight. No direction/phase factor: light travels straight to its first
               scatter, so no INV_4PI on this side -- a REAL asymmetry vs p_nee, not a common factor.)
  SUPPORT: p_bdpt(X) MUST be 0 when X's back-projected entry falls outside the light sampling
  window (r_frac in [0.97,1], |dphi|<=pi/16 about the observer azimuth). Else the backward
  side down-weights NEE (w_back<1) with no BDPT term to compensate -> bias low. Enforce the
  window test in the density fn.

WEIGHT (balance heuristic, partition of unity => unbiased):
  w_bdpt(X,Y) = p_bdpt / (p_bdpt + p_nee);   w_back(X,Y) = p_nee / (p_bdpt + p_nee) = 1 - w_bdpt.
SPIKE CANCELLATION (why it works): BDPT contrib ~ g_term_ds ~ 1/d^2. w_bdpt ~ p_bdpt/(p_bdpt +
C/d^2). contrib*w_bdpt ~ (1/d^2)*(p_bdpt d^2 /C) = p_bdpt/C = BOUNDED as d->0. The near
connection is deferred to NEE (which has no 1/d^2 blow-up: sun at infinity). Far X: p_nee ~
1/d^2 small, w_bdpt->1, BDPT keeps it.

IMPLEMENTATION STEPS (bias-critical; validate each):
  1. fn light_subpath_density(atm, x_pos, sun_dir, observer_azimuth params, hero_wl) -> f64
     = p_bdpt(X): back-project to TOA entry along +sun_dir; window support test (return 0 if
     outside); walk OD entry->X (tau_light) and entry->boundary (tau_max_light), combined
     channel; return e^{-tau_light}/(entry_weight*(1-e^{-tau_max_light})). (Omit sigma_h; it
     cancels. VSPG bias omitted in v1 = approximation, still unbiased, refine for variance.)
  2. fn nee_extend_density(phase_eye, tau_YX, d) -> f64 = phase_eye*INV_4PI*e^{-tau_YX}/d^2.
  3. BDPT connection (photon.rs ~6611): replace `w_bdpt` (fixed) with per-connection
     w_bdpt(X,Y) using lv.pos=X, scatter_pos=Y (have phase_eye, t_conn=e^{-tau_YX}, d, and
     p_bdpt via the light fn). Keep sigma_h out of both (cancels).
  4. Backward NEE bounce_idx=0 (photon.rs ~5638, trace_secondary_chain_alis): replace
     nee_weight (=w_back fixed) with 1 - w_bdpt(X,Y) where X = this bounce-0 vertex, Y =
     start_pos (seed). Compute p_bdpt(X) via light fn, p_nee via the chain's own Y->X geometry.
     ONLY for bounce_idx=0 (order-2); higher bounces keep weight 1.0 (BDPT doesn't cover them).
  5. Gate the new weight to bdpt_active (1D deck / clear); when BDPT inactive, w_back=1 as now.
  6. VALIDATE: (a) mean matches the old fixed-blend mean AND MYSTIC (unbiasedness); (b) cv /
     max-over-mean collapse at SZA 103 tau*=3 (the win); (c) clear-sky bit-identical (BDPT off
     there / window support); (d) existing cloud gates still pass.
NOTE: the order-3 (bounce_idx=1) asymmetry the mapping flagged is pre-existing and out of
scope here (BDPT only covers order-2 with 1-vertex); leave bounce_idx=1 on the fixed w_back
or set active_light_verts appropriately.

## PRE-BUILD GATE (2026-07-08): convergence probe = the estimator is UNBIASED

Before building, tested the scary alternative (the 0.27x-of-MYSTIC mean = a genuinely
missing contribution, which MIS would freeze into a confident wrong-low answer):
swept photons at SZA103/tau3 and watched the grand mean.
  128x6000 (768k paths):  m=4.02e-9  ratio 0.27   (one spike = 79% of the mean)
  64x10000 (640k paths):  m=2.39e-8  ratio 1.58   (kills biased-low: can't jump above)
  256x10000 (2.56M paths): m=1.19e-8 +- 0.45e-8, ratio 0.79 (MYSTIC at +0.7 SE)
Pooled ~4M paths: 1.23e-8 vs MYSTIC 1.51e-8. Unbiased at the resolvable ~25% precision;
the deep-cell problem is PURE VARIANCE (max sample 7.85e-7 = 52x truth). Correctness
check for the MIS build: the high-N mean must stay ~1.2e-8 while cv collapses; a mean
shift = a partition-of-unity bug.

## IMPLEMENTED (2026-07-08): per-path bounce-0 balance-heuristic MIS

All in photon.rs (twilight-core), fast suites green (404 core + 103 cpu).

1. SIGN BUGFIX found during the density derivation: the BDPT connection's eye phase used
   cos_theta_eye = connection_dir . (-view_dir) = -cos(Theta). Physical convention (chain
   seed cos_seed_view = dir . view_dir; order-1 NEE sun_dir . view_dir; chain NEE
   sun_dir . current_dir) requires connection_dir . view_dir. Fixed. Invisible for pure
   Rayleigh (referee cells: deep_atm_1d has NO aerosols, so this does not move the MYSTIC
   comparison), wrong for HG aerosol shells in production; also made the two order-2
   estimators disagree on the integrand, which would bias ANY blend.
2. Shared weight machinery: bdpt_entry_frame() (entry disk basis + pref_phi, factored out
   of trace_light_subpath verbatim); BDPT_R_DELTA / BDPT_PHI_HALF_WIDTH promoted to module
   consts; BdptMisCtx (per run: sun frame, toa_radius, ref_wl = num_wl/2, n_light) +
   BdptMisSeg (per coarse LOS step: segment endpoints, n_eye = that step's chain count);
   bdpt_light_vertex_density() = p_L(X) (support-gated back-projection to the TOA entry
   window, T_ref(E->X) / (entry_weight (1 - T_ref(E->B)))); bdpt_mis_weight() = w_bdpt =
   n_L p_L / (n_L p_L + n_E p_E) with p_E = q_seed(Y_c, omega) T_ref(Y_c->X) / d^2, Y_c =
   CLOSEST point of the coarse segment (weight constant across the step => partition of
   unity exact per (X, step) despite the two sides' different in-step quadratures; also
   bounds the g_term_ds near-edge singularity).
3. Driver (hybrid_scatter_radiance_alis): builds mis_ctx when bdpt_active && verts==1;
   chains receive (ctx, seg) per coarse step; the connection pass caches p_L per vertex
   per batch, rebuilds the SAME seg (identical n_eye expression) per step, and applies the
   per-connection weight in place of the fixed w_bdpt. Fixed sigmoid blend remains only
   for the BDPT_VERTS=2 diagnostic.
4. Chain (trace_secondary_chain_alis): new mis param; bounce-0 NEE weight becomes
   1 - bdpt_mis_weight(X, step) (exact complement); ground_bounced flag => ground-bounced
   bounce-0 paths (sun->X->ground->Y) keep FULL NEE weight (BDPT cannot sample them; the
   old fixed blend down-weighted them with nothing compensating = small pre-existing bias,
   now fixed); per-path mode never skips NEE.
5. Cleanup: dead pdf_fwd field + isotropic-placeholder tracking removed from LightVertex
   and trace_light_subpath.
Deliberate weight-internal approximations (identical on both sides, variance-only by the
partition argument): VSPG within-beam redistribution and Dwivedi flight stretch omitted;
fixed ref channel; plain forced-flight densities.
RNG discipline: weights consume no RNG; below SZA 99 nothing changes (byte-identity);
at SZA >= 99 values change by design (weights + sign fix), streams unchanged.

## MEASURED (2026-07-08): MIS unbiased; the tail attribution was WRONG

SZA103/tau3, 128x10000, per-path MIS active:
  m=1.2088e-8 (ratio 0.801 vs MYSTIC; pre-MIS 256x10000 was 1.1887e-8) -> UNBIASED, the
  partition of unity holds as designed.
  cv=6.603, max=7.85e-7 max/mean=64.9 -> NO variance collapse, and the max sample is
  BIT-FOR-BIT the pre-MIS value (same seed, same 3 sig figs).
The identical max is decisive: the MIS changed the weights of bounce-0 main NEE, BDPT
connections, and ground-bounced bounce-0 (0.34 -> {0..1, or 1.0}), with RNG streams
untouched. A spike surviving unchanged sits in NONE of those classes. The deep-cell
heavy tail is therefore NOT the order-2 near-connection singularity (that attribution,
inherited from the earlier diagnostic round, is falsified) -- it lives in bounce >= 1
NEE and/or split-particle NEE: a higher-order backward-chain weight blow-up
(VSPG x forced x window-survival products).
KEEP the per-path MIS: measured-unbiased, kills a real (if not dominant) variance class,
and carries two genuine bugfixes (eye-phase sign, ground-bounce down-weighting).
NEXT: scoring-only RNG-neutral attribution gates added (BDPT_DIAG_* statics, BDPT_NEE_MIN/
MAX, BDPT_SPLIT_OFF, BDPT_GROUND_OFF, BDPT_CONN_OFF envs + BDPT_SEED_ONLY single-seed
rerun with argmax_seed reported): decompose the spike seed's value by term, then fix the
REAL tail class. No more inferred attributions -- measured only.

## ATTRIBUTED (2026-07-08): the tail is rare sun-run mass concentration at order 8-11

Scoring-only RNG-neutral gate decomposition of the spike seed (idx 5, 7.8465e-7 = 52x
MYSTIC): connections 0.005%, order-1 0.005%, bounce-0 NEE ~0.001%, bounce-1 0.05%,
bounces 2-5 ~0, bounces 10+ ~0, BOUNCE 6-9 = 99.96%. Split/ground terms ~0.
Weight anatomy of the dominant NEE (probe): hero_w=0.24, wr=1.19, t_suns=0.37,
rr_factor=17, vspg=0.24, et=1.0, bounce 9, alt 7.7 km. ALL O(1): NOT a weight blow-up.
The chain physically walked ~1500 km to sunlit air (local slant tau ~1 at 550) and scored
ordinary-weighted NEEs from there. Median seed = 2.6e-10 = 2% of MYSTIC; 29/64 seeds ~0;
the cell's whole signal flows through the ~2% of seeds whose chains complete a sun-run.
Unbiased, but the signal probability is ~50x smaller than its contribution share.
Path guide can't fix it: it is trained only from light vertices (terminator corridor,
high alt), so the deep-shadow cells a wandering chain occupies are untrained (uniform).
Weight windows can't fix it: splitting at a collision cannot smooth the deterministic
arrival NEE (k copies x W/k = same total).

## NEXT BUILD: chain-vertex <-> light-vertex connections (full BDPT s=1, t>=2)

The structural fix, unlocked by the per-path MIS machinery:
- Registry of light vertices (reservoir, ~512) kept from the (already existing) guide
  training pass; per-vertex p_L precomputed once (bdpt_light_vertex_density).
- At each main-chain vertex X_k (bounce k), connect to ONE registry vertex lv (uniform
  pick v1, x N_reg/1 weighting): path sun->lv->X_k->...->Y->obs, order k+3 = the SAME
  integral as the bounce-(k+1) NEE. Pairing per order: conn-at-X_k <-> NEE-at-X_{k+1}.
  Orders 3,4,5,... all covered -- the order 8-11 tail no longer needs physical sun-runs
  (the connection crosses the 1500 km in one deterministic transmittance segment).
- Balance weight (shared function, both sides): w_conn = n_L p_L(lv) / (n_L p_L(lv) +
  n_ext p_ext(lv|X_k)), p_ext = (chain direction-mixture pdf at X_k toward lv) x
  T_ref(X_k->lv)/d^2 (sigma cancels). Near-connection 1/d^2 cancels exactly as bounce-0.
  NEE-at-X_{k+1} weight = 1 - w_conn(lv:=X_{k+1}, X_k=prev vertex) when the X_k->X_{k+1}
  flight was direct; ground-bounced flights keep NEE weight 1 (disjoint path family).
- Spectral: chain hero_weight x wr[w] and lv.hero_weight x lv.weight_ratio[w] are both
  absolute per-w weights -- hero mismatch between sides is a non-issue.
- RNG: connection picks use a conn_rng derived via splitmix64 from the chain streams
  WITHOUT advancing them -- all existing paths bit-identical, only scores add.
- Bounce-0 keeps the existing LOS-step pairing (order 2). BDPT_VERTS=2 stays diagnostic.
- Cost: ~1 extra transmittance walk per chain bounce (~ the NEE shadow ray) => <~2x.
VALIDATION: same protocol (mean vs MYSTIC 1.51e-8 at SZA103/tau3; cv/max collapse; the
tracking cells; fast suites; sub-99 byte-identity).

## MEASURED (2026-07-08): chain-vertex connections TRANSFORM the deep cell

Built (photon.rs): BDPT_REG_MAX=512 registry (dedicated subpath pass, per-vertex p_L
precomputed), one uniform registry pick per main-chain bounce, order-(k+3) balance
pairing conn-at-X_k <-> NEE-at-X_{k+1} via bdpt_chain_conn_weight (shared both sides;
prefix pdfs cancel so this IS the exact Veach balance for the family), per-flight ground
flag, conn_rng derived non-consumingly (all paths bit-identical, scores only add).

SZA103/tau3 (10k photons/seed): median seed 2.6e-10 -> 1.2-1.9e-8 (~50-70x, AT truth
scale); near-zero seeds 29/64 -> 0/128; cv 4.7-8.9 -> 0.86-2.1; max/mean 33-66 -> 4-12.
Pooled 128 seeds: 1.46x +- 0.23 MYSTIC (+2.0 sigma, tail-skewed; trimming the single
residual outlier -> 1.26x). SZA101/tau3 32 seeds: ratio 0.983 +- 0.13, cv 0.75 -> NO
systematic bias; the SZA103 excess is residual tail statistics.

WIDE-REGISTRY EXPERIMENT (negative, reverted): widening the registry window 13x in area
(r 0.10 / phi pi/4) to cover residual sun-run seeds DILUTED every connection (median
fell 5x, starved seeds returned) while the tail survived unchanged -- the balance weight
CORRECTLY keeps NEE dominant at efficient last hops (p_ext >> p_L for a walk already
standing next to sunlit air). The residual tail is the genuine deep-eye-prefix path
family; no last-vertex connection samples it efficiently; window width is not the lever.
Candidate future levers: 2-vertex light subpaths FOR CHAIN CONNECTIONS (sun-side carries
the lateral penetration; different from the measured-futile order-2/3 LOS 2-vertex),
or corridor-cell guide training. Neither needed for closure (below).

CLOSURE IMPLICATION: the deep referee's SZA-103 LOW-POWER verdicts were se_tw/my ~
0.5-0.6 at feasible seeds. cv ~1 (vs 4.7-8.9) cuts se_tw/my to ~0.03-0.06 at the same
1024-seed budget -- the SZA-103 cells become GATEABLE. Re-run the deep tier (cached
MYSTIC refs) as the closure gate.

## FINAL VALIDATION (2026-07-08, dedicated narrow registry, all suites green + clippy 0)

  SZA101/tau3: ratio 0.983 +- 0.13, cv 0.75  (32 seeds)     ON MYSTIC
  SZA103/tau1: ratio 0.994 +- 0.10, cv 0.58  (32 seeds)     ON MYSTIC (pre-build 0.72, cv 3.4)
  SZA103/tau3: pooled 128 seeds 2.09e-8 = 1.39x +- 0.22 (+1.8 sigma), batch ratios
    1.24 / 1.88 / 1.24 / 1.18, batch cv 0.90 / 2.38 / 0.94 / 0.97, medians ~1.2e-8,
    all 128 seeds alive (0 starved). The +1.8 sigma is the residual deep-eye-prefix
    tail (one 3.97e-7 seed carries most of batch 2's excess); no systematic bias
    resolvable given both tracking cells sit at 0.98-0.99.
Fast suites: 404 core + 103 cpu green; clippy clean. NOT pushed (standing constraint).

## STOKES PORT (2026-07-08, user chose "port to STOKES" = the rigorous referee closure)

The chain-connection estimator ported to the POLARIZED chain (trace_secondary_chain +
hybrid_scatter_radiance polarized branch = hybrid_perwl = the deep referee estimator):
- StokesConnArgs (ctx + registry, no LOS pairing: the polarized driver has no LOS
  connections, so order 2 keeps plain NEE; connections cover orders >= 3 = where the
  deep signal lives). Per-wavelength driver => registry traced at THE wavelength
  (hero == wavelength_idx), weight-internal densities exact (no ref-channel proxy).
- Eye-side vertex: full Mueller treatment via scatter_stokes_fast, exactly the NEE
  code with -sun_dir replaced by the beam direction (lv -> X_k). Light-side vertex:
  scalar phase intensity P11(theta_lv)/4pi; the incident-beam polarization coupling is
  dropped -- the SAME documented I-approximation class as the chain's unpolarized seed.
- is_main/prev-vertex/per-flight-ground bookkeeping mirrors the ALIS chain; conn_rng
  derived non-consumingly (all existing draws bit-identical; sub-99/field/unpolarized
  paths completely unchanged).
MEASURED on the REFEREE PATH (hybrid_perwl, deep_config polarized, 24 seeds x 10k):
  SZA103/tau3: med 1.52e-8 vs MYSTIC 1.51e-8, mean ratio 1.30 +- 0.23, cv 0.864
  SZA103/tau1: med 1.92e-8 vs MYSTIC 1.96e-8, mean ratio 1.32 +- 0.28, cv 1.05
  SZA101/tau3: ratio 0.939 +- 0.16, cv 0.838
  (History: these cells were LOW-POWER with se_tw/my ~ 0.5-0.6 at 1024 seeds.)
At the deep tier's 1024-seed budget, cv ~0.9 => se_tw/my ~ 3%: the SZA-103 cells are
now GATEABLE. NEXT: re-run validate_libradtran.py --tier deep (cached MYSTIC refs;
AWS c7a pattern from the 2026-07-08 campaign) to flip the referee table rows.
Suites green (404 core + 103 cpu), clippy: only pre-existing harness warnings remain.

## ATTRIBUTION CAMPAIGN (2026-07-08/09, AWS): the +20% was a REGISTRY BUG -- FOUND + FIXED

Arms: (1) conn-vs-analog-Multiple at 99.5/100.5 -- inconclusive (Multiple cv ~4);
(2) STOKES 512-seed at the FAIL cells -- 1.465/1.281, agrees with ALIS (but shares the
conn design, circular); (3) OLD estimator (fixed blend, BDPT_VERTS=2 => mis off) at
2048 seeds: FAIL cells read 1.003 +- 0.10 and 1.032 +- 0.16 of the refs -- but its
median/mean 0.03-0.07 makes underconvergence un-rule-outable there. DECISIVE ARM:
partition cross-check at SZA 99.5 where BOTH estimators converge and E[new] = E[old]
by construction: OLD 3.806e-7 +- 0.148 (2048 seeds, cv 1.76) vs NEW 4.908e-7 +- 0.178
(512 seeds, cv 0.82) = +29% at 4.8 sigma => PARTITION VIOLATION => bug hunt => FOUND:

THE REGISTRY SUCCESS-CONDITIONING BUG: the registry pass traces BDPT_REG_MAX=512
subpaths but keeps only those yielding a vertex (n_verts > 0, weight > 0, p_L > 0);
the uniform pick then estimates the SUCCESS-CONDITIONED mean, overcounting connections
by 1/P(success) = 512/reg_len. Predicts the whole observed pattern: +29% at 99.5,
growing with depth (+34% at 101/450, +43% at 103/550: deeper => more grazing rim beams
fail => smaller reg_len), varying per wavelength (tau_max is hero-dependent). The LOS
connection never had it (divides by the FULL subpath count). Fix: scale every conn
score by reg_frac = reg_len/512 (BdptChainMisArgs + StokesConnArgs, both chains).
The MYSTIC refs' shape-incoherence remains real but SECONDARY (per-cell scatter, not
the global offset). Verification + full Stage A re-run with the fixed estimator queued
on the attribution box; MYSTIC fresh-seed replicas of the tau3 refs run alongside.

FIX VERIFIED (2026-07-09 ~22:55 UTC): fixed NEW at 99.5 = 3.8916e-7 +- 0.138 vs OLD
3.8058e-7 +- 0.148 => +2.3% at 0.4 sigma, partition RESTORED (pre-fix +29% at 4.8
sigma). Measured reg_frac = 3.8916/4.9082 = 0.793 => ~21% of registry subpaths fail
at this geometry (the a-priori geometric estimate of <1% was wrong; measurement rules).
Same argmax seed / cv pre-and-post: a pure normalization rescale, as expected.
CAVEAT logged: the PRE-fix 100.5 pair had agreed (OLD 1.610 vs NEW-prefix 1.596e-7),
which under reg_frac~0.79 implies OLD sat ~+28% high there (a tail fluctuation of the
cv-2.6 heavy estimator, nominal +3.8 sigma) OR reg_frac is strongly SZA-dependent;
the fixed Stage A verdicts + (optionally) a fixed 100.5 rerun disambiguate.

## CLOSURE (2026-07-09): DEEP TIER FULLY GATED, 1d 12/12 PASS

Fixed-estimator Stage A re-run (1024 seeds x 16k photons, box i-03469fb9611601a2c,
terminated; total AWS spend across all three campaigns ~ $45):
  tau1 1d: 0.984 / 0.995 / 0.902 / 0.931 / 0.988 / 0.970   all PASS
  tau3 1d: 1.057 / 0.871 / 0.920 / 0.939 / 1.125 / 1.088   all PASS
Official tier assembly (validate_libradtran.py --tier deep, cached refs):
  **14 PASS / 0 FAIL / 0 1d-LOW-POWER** (2 remaining LOW-POWER are the SZA-103 FIELD
  rows: the untouched 3D-field STOKES path at its documented seed cap -- connections
  gate on field.is_none(); future item). Shipped state was 7 PASS / 7 LOW-POWER.
MYSTIC REPLICAS (8 fresh-seed 1e9 reruns of 4 tau3 refs, archived under
validation/deep/mystic_replicas_2026-07-09): reproduce within ~1 sigma EXCEPT the
cached tau3/103/650 ref, which sits 3.3 sigma BELOW its two replicas (a low draw --
the cached shape-incoherence was real per-draw ref scatter). Fixed-estimator ratios
PASS against BOTH cached and replica-pooled refs (pooled: 1.017/0.909/1.058/0.855).
validation/deep_regime_results.csv regenerated (pre-fix backup in session scratchpad
pre_fix_backup/). Remaining documented items: field SZA-103 rows (seed-budget), paper
table/figure regeneration, the 6-9% shape tension at tau3/101 (within pooled-ref
noise). NOT pushed (standing constraint).

## PAPERS + CALIBRATION (2026-07-09, closing the loop)

P1 deep-regime sections rewritten to the new state (tables/figure regenerated from the
new CSV; methods now describe the connection estimator incl. the attempt-count
normalization; limitations re-scoped to the two 3D-field cells; builds clean, 32 pp).
Engine README updated. CRITERION A/B on a dedicated box (TWILIGHT_BDPT_CONN_OFF env +
BDPT_CHAIN_CONN_DISABLE atomic added as a clean whole-estimator kill-switch): fajr
detection depressions at f=56, connections ON vs OFF: Kottamia +0.02 deg, Aswan -0.02,
Birmingham -0.01, Riyadh -0.30 but under 1 sigma of its own +-1.4 min crossing fit.
NO significant calibration shift: f=56 and all Paper 2 campaign numbers stand. (The
criterion sites are clear-sky, where the pre-connection estimator was already
converged; the connection estimator's fix mattered under thick decks, which the
criterion never sees.) All AWS instances terminated.

## DEEP-TIER 1024-SEED RE-RUN (2026-07-08 evening, AWS c7a.24xlarge ~2.1h, terminated)
[SUPERSEDED by the registry-bug fix above -- these verdicts carried the +1/P(success)
conn overcount; see the post-fix re-run below for the standing table.]

New-estimator tree rsynced (no git push), stale 1d caches cleared, MYSTIC refs cached,
DEEP_WORKERS=48 x RAYON 2, 16k photons, 1024 seeds, 1d both taus (field rows unaffected:
connections gate on field.is_none()). Per-seed caches: scratchpad rerun_1d_tau{1,3}.json.
VERDICTS (band = 3 sqrt(se_tw^2+se_my^2) + 0.05 my; old verdict in parens):
  tau1 101/450 1.215 PASS(LOW-POWER)  101/550 1.224 PASS(PASS)  101/650 1.100 PASS(LOW-POWER)
  tau1 103/450 1.142 PASS(LOW-POWER)  103/550 1.203 PASS(LOW-POWER) 103/650 1.177 PASS(LOW-POWER)
  tau3 101/450 1.341 FAIL(LOW-POWER)  101/550 1.094 PASS(FAIL!)  101/650 1.148 PASS(PASS)
  tau3 103/450 1.189 PASS(LOW-POWER)  103/550 1.426 FAIL(LOW-POWER) 103/650 1.380 PASS(LOW-POWER)
  => 1d rows: 10 PASS / 2 FAIL / 0 LOW-POWER (was 5/0/7). ZERO LOW-POWER: the closure
  goal (resolvability) is achieved. THE HONEST RESIDUAL: every ratio sits above 1
  (1.09-1.43, mean ~1.22); at the new precision two thick-deck cells resolve HIGH
  beyond band. This is a real, newly-visible systematic -- candidates: (a) residual
  estimator bias at thick decks (the deep-prefix tail's mean contribution, or a subtle
  conn-scoring excess the SZA101 diagnostic could not resolve), (b) the cached deep
  MYSTIC references themselves (public MYSTIC backward documented incoherent past SZA
  102; refs carry their own convergence risk), or both. Previously this offset hid
  INSIDE the LOW-POWER noise. NOT tuned away; investigation is the next work item.
  Official validation/deep_regime_results.csv NOT yet overwritten (paper tables stay
  at the shipped state until the 2 FAILs are understood).

Status: SCOPED foundation below is implemented; the sections describe the original plan.

## 0. Why this exists

The deep-twilight LOW-POWER cells — SZA 101-103, tau*=3 (and tau*=1/SZA-103), 1D uniform
deck — are variance-limited, not bias-limited. Under a deck the hybrid estimator falls back
to NEE-only (backward chains, `w_back = 1`), which is unbiased but noisy; the seed standard
error never shrinks past the deep-referee band, so the MYSTIC gate reports LOW-POWER instead
of PASS. The repo already names the fix: `g_s3_field_forced_matches_multiple_checkerboard`
header (`simulation.rs:2883-2885`) and `RESULTS_DEEP_REGIME.md:434-441` both call closure of
these rows "the standing budget/BDPT follow-up." This document is that follow-up.

BDPT (light subpath launched from the sun, connected to the eye-LOS march) collapses the
heavy tail because it samples the sun-lit contribution directly instead of waiting for a
backward chain to random-walk up to a sun-lit altitude. It already exists and works for
clear sky. It is deliberately gated OFF under any cloud. The task is to make the light
subpath compose unbiasedly with the gray cloud channel so the gate can be relaxed under a
1D deck.

## 1. The reframing (what this is NOT)

Two assumptions from the initial framing are FALSE against the code, and both make the job
smaller:

1. **No forward/adjoint ray-bending Jacobian is required to close the target cells.**
   - The BDPT "MIS" (`w_bdpt`/`w_back`, `photon.rs:5961-5970`) is a fixed SZA sigmoid
     convex blend, `w_bdpt = sigmoid((sza-102)/1.5)`, of two independent estimators of the
     *same* multiple-scatter integral. It is NOT a pdf-ratio heuristic. `pdf_fwd`
     (`photon.rs:1682`) is dead code; there is no reverse/adjoint pdf anywhere. A convex
     blend of two consistent estimators is unbiased iff each estimator is individually
     unbiased. So the task reduces to: make the light-subpath estimator individually
     unbiased under the deck. No adjoint pdf, no area-measure Jacobian, no balance heuristic
     to build.
   - The MYSTIC-refereed deep configuration runs with refraction OFF (`deep_atm_1d` forces
     `refractive_index[*] = 1.0`, `simulation.rs:2440-2442`; the libRadtran decks carry no
     refraction either). With `n = 1` rays are straight, so the existing straight-line
     connection geometry is exact and the ray-bending Jacobian is irrelevant to the gate.

   The ray-bending Jacobian is a PRODUCTION concern only (real refracting atmosphere with
   tropospheric connection endpoints), fully decoupled from the deep-tier acceptance gate.
   It is deferred to Phase 3 and is NOT on the critical path to closing the cells.

2. **The fix is a port, not green-field.** The backward chains already solved the identical
   composition problem in the `ac673c7` "combined-channel forced mode" (`photon.rs:3854-3991`).
   The light subpath's scout/advance machinery is ALREADY combined-channel (it samples the
   combined gas+gray-cloud tau under a deck); it merely mis-scores every vertex as gas. The
   fix is to give it the same vertex-type draw the chains gained.

## 2. Scope boundary (honest)

| Surface | Chain | BDPT? | This plan closes it? |
|---|---|---|---|
| 1D deck, `compare --fast` (ALIS scalar) | `hybrid_scatter_radiance_alis` | yes (gated off under cloud) | **YES — primary target** |
| 1D deck, `deep_referee_runner` | `hybrid_scatter_radiance` polarized STOKES | no BDPT in chain | no (needs Stokes port, Phase 4) |
| 3D field, any | ALIS is analog-only under fields; Stokes is field-forced | no | no (ALIS null ratios blow up under an all-wl majorant, `photon.rs:5205-5215`) |

The primary, tractable deliverable is **unbiased BDPT under a 1D deck in the ALIS chain**,
refereed against MYSTIC and against the analog `Multiple` (`trace_photon`) estimator, at
SZA 101-103, tau* = 1,3. This closes the `tab:variance` (1D) LOW-POWER cells.

The Stokes chain (`deep_referee_runner`, the polarized production quantity) has no BDPT at
all. Porting BDPT into the Stokes hybrid, and any 3D-field extension, are explicitly OUT of
this plan's critical path (Phase 4, separate effort). A cross-check that ALIS-BDPT and Stokes
agree within band at these cells (Phase 3) is how we argue the variance win transfers to the
production quantity without claiming to have moved the Stokes numbers directly.

## 3. The estimator change (the core)

Mirror the ALIS combined-channel vertex-type draw (`photon.rs:5340-5377`) inside
`trace_light_subpath` (`photon.rs:1738`). At each recorded light vertex, in the shell
`scatter_shell`:

```
sigma_c = atm.cloud_extinction[scatter_shell]           // gray cloud extinction (shell-constant, pure scatter)
sigma_h = atm.optics[scatter_shell][hero_wl].extinction // hero-wavelength GAS extinction
if sigma_c > 0.0 && draw(rng) < sigma_c / (sigma_c + sigma_h) {   // GUARDED: draw only when cloud present
    cloud_vertex = true
    g_cloud = atm.cloud_g_scaled                        // 1D: shell-constant gray HG asymmetry
}
```

Then the vertex physics and spectral reweight split by type:

- **Cloud vertex:** pure scattering (no absorption); connection `phase_light` uses the gray
  HG lobe at `g_cloud`; spectral ratio is transmittance-only (the gray cloud tau cancels in
  every per-wavelength difference):
  `for w: ratio = exp(-(taus_at_pos[w] - taus_at_pos[hero_wl])); weight_ratio[w] *= ratio`.
- **Gas vertex (existing behavior):** apply gas SSA (absorption); connection `phase_light`
  uses the gas Rayleigh+aerosol phase; spectral ratio carries the gas extinction ratio too:
  `for w: ratio = (sigma_w/sigma_h) * exp(-(taus_at_pos[w] - taus_at_pos[hero_wl]))`.

The forced-scatter weight `(1 - exp(-tau_max))` is unchanged — `tau_max` is already the
combined tau because the scout/advance machinery is combined-channel. The type-selection
probability cancels the per-type coefficient exactly (`beta_total * p_c = beta_cloud`,
`photon.rs:3888-3890`), so no extra weight factor appears; this is why the mixture stays
unbiased and low-variance.

### Edit surface
1. `LightVertex` struct (`photon.rs:1683-1701`): add `is_cloud: bool` and `g_cloud: f64`
   (or a small enum + asymmetry) so the connection can pick the correct phase function.
2. `trace_light_subpath` (`photon.rs:1738-2067`): compute `sigma_c/sigma_h`, do the guarded
   vertex-type draw on the correct RNG stream, set the new fields, and apply the type-correct
   `weight_ratio` update (mirror `photon.rs:5356-5377`). Use `atm.cloud_extinction[shell]`
   and the per-wavelength `taus_at_pos` already available to the ALIS walk.
3. Connection loop (`photon.rs:6468-6539`): when `lv.is_cloud`, compute `phase_light` from
   the gray HG lobe at `lv.g_cloud` instead of the gas phase; the eye-side `phase_eye` and
   `t_conn`/`t_obs` (already `BeerLambert`, cloud-attenuated) are unchanged.
4. Gate (`photon.rs:5959`): change `!cloud_channel` to `field.is_none()` so BDPT is active
   for clear sky AND 1D decks, but stays off for 3D fields (where ALIS forced mode itself is
   analog-only). Keep the `secondary_rays > 0` and `sza >= BDPT_SZA_START` conditions.

## 4. Invariants (must hold — these are the correctness contract)

1. **Clear-sky byte-identity.** The vertex-type RNG draw is taken ONLY when `sigma_c > 0`
   (guard exactly like `photon.rs:3874-3875`, `5344-5345`). Under clear sky no new draw
   happens, so every clear-sky RNG stream is bit-identical and all clear-sky gates keep
   their current values. Verified by `bitcheck_dump` (`simulation.rs:2789`) run in two
   trees, hex-diffed (BDPT clear-sky surfaces must show zero diff).
2. **No prior deck-BDPT stream to preserve.** BDPT was OFF under any deck before, so there
   is no pre-existing cloudy-BDPT RNG sequence to keep; the only byte-identity obligation is
   clear sky.
3. **RNG stream discipline.** Type draw goes on the same stream the ALIS forced walk uses
   for classification (`local_rng.tau` convention, `photon.rs:5348`); `rng.dir` untouched.
4. **wr/pr lockstep + tau_c cancellation.** Whatever spectral arrays the light subpath
   carries (`weight_ratio`, and `pr` if present) receive the identical ratio on every
   branch; the gray cloud tau must cancel in every per-wavelength difference.
5. **Connection model consistency.** The connection legs are `BeerLambert` cloud-attenuated
   (`transmittance_between_points_spectrum`); the subpath must propose under the same
   combined model — which, post-fix, it does.

## 5. Acceptance gate (the definition of done)

Three layers, increasing rigor:

- **A. Same-scene A/B unit gate** — new `#[ignore]` test
  `g_bdpt_under_1d_cloud_matches_multiple`, mirroring `g_s2_forced_under_1d_cloud_matches_multiple`
  (`simulation.rs:1049`) and `g_s3_field_forced_matches_multiple_checkerboard`
  (`simulation.rs:3022`). Estimator A = ALIS hybrid with BDPT-on under the deck (new). Estimator
  B = analog `multiple_perwl` (`trace_photon`, the independent reference). Scenes: `deep_atm_1d`
  at tau* = 1,3, SZA 101/103. Assert two-sided `band = 3*se + 0.05*max(mA,mB)` where both
  converge; the BDPT arm must satisfy the band at SZA 103 / tau*=3, the row G-S3-CB only
  reports today. Reuse `perwl_mean_se` (`multiple: bool` switch), `deep_config`, `deep_field`.
- **B. Variance-reduction ledger** — BDPT-on vs BDPT-off (analog) seed-CV at the same cells,
  same photon budget. Success = BDPT-on seed-SE small enough that
  `band = 3*sqrt(se_tw^2 + se_my^2) + 0.05*my <= 0.5*my` (i.e. the cell leaves LOW-POWER)
  AND `|m_bdpt - m_my| <= band` (PASS), using the exact `validate_libradtran.py:1192-1202`
  formula and the cached MYSTIC values already in `validation/`.
- **C. MYSTIC referee** — run the BDPT-on ALIS arm through the deep harness against the cached
  3e8-1e9-photon MYSTIC decks. Requires either (i) an `DEEP_ESTIMATOR`/`DEEP_POLARIZED=0` env
  in `deep_referee_runner` so it can run the ALIS chain (it currently hardcodes
  `deep_config(photons, true)` = Stokes), or (ii) driving `compare --fast` (ALIS) at the deep
  grid, which is the surface `write_deep_csv.py` already consumes for
  `validation/deep_regime_results.csv`. Path (ii) matches how the current 1D deep cells were
  produced; prefer it for provenance continuity.

Gate passes when the currently-LOW-POWER 1D cells (tau*=3 SZA-103, tau*=1 SZA-103) reach PASS
under the unchanged band formula, at a seed/photon budget comparable to the clear-sky cells —
i.e. closed by the estimator, not by brute-force seed count.

## 6. Parity / regression obligations

- Clear-sky ALIS results: bit-identical (bitcheck) — hard requirement.
- All existing cloud gates (G-HYB-MULT, G-FORCED-1D, G-S3-CB, G-EQ1D, G-ALIS): must still
  pass; BDPT-on must not perturb the analog `Multiple` reference (it doesn't touch it) and
  must keep the hybrid within the same bands.
- GPU parity is NOT part of this gate: the GPU crate has neither ALIS nor BDPT nor `Multiple`.
  BDPT-under-cloud is CPU-only, refereed by MYSTIC and the analog CPU `Multiple`.
- The only existing BDPT test (`bdpt_light_subpath_vertex_diagnostic`, `photon.rs:9114`) asserts
  nothing physical; this plan adds the first correctness gate for BDPT.

## 7. Phasing

- **Phase 1 — estimator port (core).** Struct + `trace_light_subpath` vertex-type draw +
  connection phase + gate relaxation. Build clean; `bitcheck` clear-sky identity.
- **Phase 2 — acceptance gate A + B.** Write the A/B unit gate and the variance ledger; show
  the target 1D cells close. This is the definition of done for the primary deliverable.
- **Phase 3 — MYSTIC referee (gate C) + production refraction check.** Referee the ALIS-BDPT
  arm against cached MYSTIC; separately confirm whether production (refraction-on) connection
  endpoints under a deck stay high-altitude enough for straight-line, else scope the
  ray-bending Jacobian (deferred, non-blocking for validation).
- **Phase 4 (out of critical path) — Stokes BDPT + 3D field.** Port BDPT into
  `hybrid_scatter_radiance` (polarized) so the production Stokes deep cells and the field
  cells can also benefit. Much larger; separate plan.

## 8. Key file references

- BDPT: `crates/twilight-core/src/photon.rs` — `trace_light_subpath` :1738, `LightVertex`
  :1683, connection :6468-6539, MIS blend / gate :5952-5970, `BDPT_MAX_LIGHT_VERTICES=1` :1633.
- ac673c7 template: `photon.rs:3854-3991` (derivation), `:5340-5377` (ALIS instance).
- Deep harness: `crates/twilight-cpu/src/simulation.rs` — `deep_referee_runner` :2605,
  `hybrid_perwl` :2506, `multiple_perwl` :2539, `perwl_mean_se` :2570, `deep_atm_1d` :2434,
  `deep_field` :2451, `deep_deck_props` :2418 (base 1km/top 2km, g 0.85, ssa 0.999),
  `bitcheck_dump` :2789.
- Referee band: `tools/validate_libradtran.py:1192-1202`; mirror `tools/write_deep_csv.py:59-62`.
- Precedent gates: `g_s2_forced_under_1d_cloud_matches_multiple` :1049,
  `g_s3_field_forced_matches_multiple_checkerboard` :3022 (KNOWN-LIM row this closes).
</content>
</invoke>
