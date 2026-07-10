# Voxel-field connection estimator: design

Status: IMPLEMENTED on branch field-conns (98bebc9, 2026-07-10), gate
verdicts pending. The design below was written first and the
implementation followed it: Cloud3DField::advance_to_combined_tau (exact
piecewise-constant combined inversion), scout/advance field parameters
(chains pass None), voxel vertex-type draw, field threading through the
densities and weights, and the los_pairing flag (bounce-0 LOS pairing
stays 1D-only). All activation predicates are identical to main when no
field is present; fast suites and clippy green. The G-FC gate ladder
(bitcheck vs main, partition invariant under a uniform field at SZA 99.5,
field-vs-1D equivalence at 101, field referee cells at 128 seeds) is
queued on the verification box; the branch does NOT merge to main until
all four pass.

## Goal

Extend the chain-vertex <-> light-vertex connection estimator (orders >= 3,
SZA >= 99) from the 1D combined channel to heterogeneous 3D cloud fields.
The field path currently runs analog chains and gates its deep referee
cells only through large seed campaigns (the SZA-103 550nm cells needed
1024 and 512 seeds x 16k photons, ~700 core-hours per tau), and production
broken-deck khayt scans at deep SZA carry the same heavy tail. The 1D
experience says the connections change the estimator's shape: median seed
from ~2% of truth to truth scale, cv from 5-9 to ~1.

## The key structural insight

The documented blocker for forced-mode CHAINS under a field (ac673c7
addendum: exact per-wavelength null-collision ratios need an
all-wavelength majorant and blow up in heavy tails) does NOT apply to the
LIGHT SUBPATH. The light beam is one straight ray from a TOA entry point:
its combined optical depth profile tau(s) is EXACTLY integrable by the
existing DDA (`Cloud3DField::tau_along`) plus the analytic per-shell gas
sums, and the forced-collision position is an EXACT inversion of that
profile (`advance_to_tau` for the cloud part; a two-channel stepped
inversion for gas+cloud combined). No delta tracking, no majorant, no null
collisions, no new tail. The gray field cancels in per-wavelength ratio
differences exactly as the 1D gray deck does (the whole ac673c7 ALIS
crux), so the light subpath's spectral bookkeeping carries over verbatim.

## What changes, piece by piece

1. `trace_light_subpath` under `field = Some(f)`:
   - scout: combined tau to the boundary = per-shell gas walk + one
     `f.tau_along(entry, -sun_dir, s_max)`; ground truncation as in 1D.
   - forced position: invert tau(s) = tau_gas(s) + tau_cloud(s). Stepped
     inversion over the DDA segments (each segment has constant cloud
     extinction and analytic gas), exact to numerical tolerance.
   - vertex type draw: sigma_c from the voxel at the collision point,
     g from `f.g_at(pos)`; the conditional and weight bookkeeping are the
     1D code unchanged.
   - VSPG: keep the altitude x SZA importance as in 1D (cloud-blind);
     the 1D deck experiments showed deck-aware importance is futile.
2. `bdpt_light_vertex_density` under a field: the back-projection and
   window support tests are geometry-only (unchanged). T(E -> X) and
   T(E -> B) already go through `transmittance_between_points_spectrum`,
   which is field-capable (DDA chord). One new argument (`field`) threaded
   through; the 1e-9 truncation floor unchanged.
3. Chain-side scoring: the Stokes connection block already passes `field`
   to its transmittance; the ALIS block uses the module path and needs the
   same argument. `bdpt_chain_conn_weight`'s T_ref likewise.
4. Gating: `conn_active` / `mis_ctx` drop the `field.is_none()` condition
   once 1 to 3 land. The bounce-0 LOS pairing stays OFF under fields
   (order 2 keeps plain NEE, as the Stokes path already does; the LOS
   connection machinery is a separate, optional follow-up).
5. Chains stay ANALOG under fields. The connections do not require forced
   chains: they fire at whatever collisions the analog walk produces, and
   the NEE complements apply at those same collisions.

## What stays exact

The partition-of-unity argument is untouched: conn-at-X_k and
NEE-at-X_{k+1} pair per order through one shared weight function, so any
density approximation inside the weight (including ignoring field
structure in q_dir, or the VSPG omission) is variance-only. The attempt
count normalization (reg_frac) carries over as is; expect a LOWER success
fraction under broken fields (beams extinguished in cloud towers still
yield vertices via forced collision, so failures remain the tau_max
degenerate rim only; verify by logging reg_len).

## Risks and measured lessons to respect

- Do NOT widen the registry entry window a priori. The 1D experiment
  measured that widening dilutes connection density ~linearly in area
  while the balance weight correctly refuses to transfer efficient-last-
  hop paths; window changes are a measured decision, made only if the
  broken-field residual tail demands them.
- Broken fields decorrelate the azimuthal importance of entries from the
  observer's projection (cloud gaps move the bright corridors). If the
  uniform-field gate passes but the checkerboard gate keeps a tail,
  registry importance (picking lv by contribution estimates instead of
  uniformly) is the first lever; it is unbiased for any positive pick
  distribution if the pick probability divides the score.
- The June-solstice/high-latitude evening detection interacts with any
  estimator change through the cone gate (the bracket_index lesson):
  rerun the persistent-twilight khayt checks after landing.

## Gates (all must pass before the flag flips on)

- G-FC-0 bit-identity: below SZA 99, and field-Multiple everywhere, byte
  identical to HEAD (the gate machinery exists).
- G-FC-1 partition invariant: uniform thin field at SZA 99.5, 2048-seed
  fixed-blend arm vs 512-seed connection arm, means within 2 sigma (the
  invariant that caught the reg_frac bug; run it FIRST).
- G-FC-2 equivalence: uniform field vs its equivalent 1D deck, connection
  estimators on both, per-wavelength agreement inside MC bands (the
  G-EQ1D pattern, but across the estimator pair rather than trajectory
  replicas).
- G-FC-3 referee: the deep field cells (tau* 1 and 3, SZA 101/103,
  550 nm) vs the cached MYSTIC references at 128 and 256 seeds; the win
  condition is gating at <= 256 seeds where 512 to 1024 were needed.
- G-FC-4 broken deck: the checkerboard zenith-starvation row moves toward
  its referee or stays within bands; no new tail class (max/mean and
  n<1%mean tracked per seed).

## Cost estimate

Light-subpath DDA: ~10^2 voxel segments per beam x 512 registry subpaths
per driver call, microseconds each: negligible against the chains.
Per-bounce connection transmittance already dominates the 1D overhead
(~2x chains); the field chord adds the DDA factor (~1.5-2x on the
transmittance), so expect ~2.5-3x analog-chain cost at deep SZA, repaid
by orders-of-magnitude seed reduction.

## Out of scope

Forced-mode field CHAINS (the ac673c7 addendum blocker stands), the LOS
order-2 pairing under fields, GPU kernels (CPU-routed under fields by
design), and any change to the field referee protocol.
