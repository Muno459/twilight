# twilight — Comprehensive Review (Opus 4.8 sweep)

**Date:** 2026-06-11 · **Scope:** physics correctness, MC variance/CV, Metal GPU, fabrication ("AI slop"), integration, libRadtran validation
**Method:** 14 parallel specialist reviewers → adversarial verification of every critical/high finding → integration + validation strategists → completeness critic. 42 agents, 808 tool uses. All findings cite file:line and were re-read by an independent verifier. Empirical artifacts (builds/tests run on this machine) in `.review-artifacts/`.

**Bottom line:** the deterministic single-scatter core, the solar engine, and the gas-absorption *data* are largely sound — but the project's headline claims outrun the code. 3 of 4 GPU backends are fabrications that cannot compile; the Garstang skyglow model, Snell's-law refraction, and the LBL Voigt machinery are dead code in production; prayer thresholds rest on fabricated "SQM calibration"; the default `pray` path finds threshold crossings on an un-averaged noisy MC curve; and there is one real physics bug in the phase-function angle that corrupts all hazy/cloudy results.

---

## 1. CRITICAL (confirmed by adversarial verification)

| # | Finding | Where |
|---|---------|-------|
| C1 | **Vulkan backend cannot compile** — `include_bytes!` of 4 nonexistent `.spv` files; no GLSL source, no build.rs | `vulkan.rs:50-53` |
| C2 | **CUDA backend cannot compile** — `include_str!("../shaders/twilight.cu")`, file doesn't exist | `cuda.rs:25` |
| C3 | **WebGPU backend cannot compile** — `twilight.wgsl` doesn't exist; also assumes `subgroupAdd` without requesting the wgpu feature | `wgpu_backend.rs:16` |
| C4 | **Raw VIIRS upward radiance injected as observer sky radiance** — skips the entire Garstang propagation; overstates skyglow by orders of magnitude | `skyglow/spectrum.rs:198-257`, `pipeline.rs:692-699` |
| C5 | **Garstang RT model is dead code** — the README headline skyglow feature is never called by the pipeline; documented `compute_skyglow` doesn't exist | `garstang.rs:98-267,354-452` |
| C6 | **Pipeline panics at midnight-sun latitudes** — empty coarse scan → `0..(0usize-1)` underflow → index-out-of-bounds. Reproduced: Tromsø 69.6°N, 2024-06-21 | `threshold.rs:190-191`, `pipeline.rs:556-624` |

## 2. HIGH (confirmed)

| # | Finding | Where |
|---|---------|-------|
| H1 | **Phase-function angle is the supplement of the true scattering angle** (`cos_theta = sun_dir·(−view_dir)` instead of `sun_dir·view_dir`). Rayleigh is symmetric → clear sky unaffected; but the forward-peaked HG term is evaluated backwards → aerosol/cloud forward-scatter suppressed by 1–2 orders of magnitude in the twilight glow. **The single most important physics fix.** Same convention error mirrored in `photon.rs:425,730` (NEE). | `single_scatter.rs:219-235,408` |
| H2 | **Prayer thresholds claim "calibrated against SQM observations" — no SQM data, fitting code, or derivation exists anywhere.** The constants that *define Fajr and Isha* are unsourced magic numbers. | `threshold.rs:54-93` |
| H3 | **MC threshold crossing on a noisy curve** — each SZA gets ONE un-averaged MC estimate (default hybrid, 100 rays); 5–7% CV near the crossing band; first-crossing latch makes a noise dip register as the crossing → jittery prayer minute | `simulation.rs:256-310`, `threshold.rs:190-237` |
| H4 | **Hybrid seed-scatter estimator is structurally biased** — Stokes-normalize trick removes the phase-function magnitude; zenith/terminator branches estimate a *different integral* than the phase branch (not valid MIS) | `photon.rs:2618-2676` |
| H5 | **`hybrid_los_prefix` Metal kernel is dead + its "optimization" comment is false** — tau prefix recomputed O(steps²) per chunk; this is the dominant cost tripping the GPU watchdog | `twilight.metal:1844-2020`, `metal.rs:103-106` |
| H6 | **README GPU section fabricated**: "four hand-tuned backends", instruction-level feature table for shaders that don't exist, "cross-backend parity ensures identical physics" (parity test silently no-ops with <2 backends) | `README.md:131,140,142-156,239,251` |
| H7 | **Surface O3 → column DU mapping is invented** — 0.5 DU/(µg/m³) linear proxy; surface O3 is essentially uncorrelated with the stratospheric column. README sells it as "today's actual O3 column" | `weather/mapping.rs:186-197` |
| H8 | **Fabricated VIIRS data sources** — embedded lookup, S3 COG tiles, MCRT ground tracer: none exist | `skyglow/lib.rs:10-27` |

## 3. Empirical test evidence (run on this machine, M-series, 2026-06-11)

- `cargo build --release --workspace` (default) ✅
- `--features cuda` ❌ / `--features vulkan` ❌ / `--features webgpu` ❌ — missing shader files (see `gpu-feature-builds.txt`)
- `cargo test -p twilight-gpu --features metal`: **130 passed, 4 FAILED** (`metal-tests.txt`):
  - `hybrid_scatter_v2` killed by macOS GPU watchdog (`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`) ×2
  - **GPU CV at SZA 100° ≈ 73%** (σ=5.5e-4 on mean 7.5e-4) vs CPU CV 1.4%; one seed 4× outlier; GPU/CPU ratio 1.32 out of tolerance
  - Batched hybrid **0.7× slower than serial** (313 ms vs 234 ms) — the batching is a pessimization
- Note: a new "parity" test accepts GPU within **0.05×–20×** of CPU — a tolerance so wide it can't fail meaningfully (`tests.rs`, split-dispatch test)

## 4. Variance / CV / efficiency assessment (the user's #1 ask)

Where the CV comes from, in order of leverage:

1. **Single sample per SZA.** Each SZA's luminance is one MC estimate seeded from `sza_deg.to_bits()` — noise is uncorrelated across adjacent SZAs, and the crossing is interpolated between two noisy points exactly where luminance is 5e-4–3e-3 cd/m² (worst relative error).
2. **The hybrid already computes order-1 exactly** — but doesn't exploit it as a control variate. Only the (small) multiple-scatter correction needs to be stochastic; estimate `MS = total − SS_exact` with MC and the CV of the *sum* collapses.
3. **Pure-MC mode is an analog walk** (no Russian roulette, no forced scattering, no importance sampling) — at SZA>100° photons must random-walk ~1000+ km to escape the shadow; nearly all weight dies. This mode is unusable for production and should be labeled debug-only.
4. **The hybrid's biasing stack (zenith power-cosine, Dwivedi, terminator shaping, forced_tau_min, VSPG boost) is both biased (H4) and high-variance** — branches estimating different integrals inflate spread; importance-weight cap at 200 hints at the weight-tail problem.
5. **GPU v2 kernel** adds its own variance (73% CV at SZA 100°) on top — RNG decorrelation and the isfinite-masking of NaNs (which silently drops samples) are suspects.

**Recommended CV/efficiency program (in order):**
1. Fix H1 (phase angle) first — no point variance-tuning a biased integrand.
2. Make `pray` deterministic-by-default (single-scatter), hybrid opt-in — prayer times must be reproducible.
3. Control variate: hybrid returns `SS_exact + MC(MS)`; the noise then only scales the MS fraction (~5–20% of signal at twilight) → ~5–20× CV reduction for free.
4. Fix the seed-scatter MIS (H4) so all branches estimate the same integral with weights `P_phase/p_branch` — removes both bias and the weight-spread variance.
5. K-seed averaging + sample-variance tracking per SZA → report a confidence interval on the crossing minute; adaptively raise rays where `|L − threshold|` is small.
6. Fit a smooth monotone log-luminance(SZA) model (e.g. isotonic or low-order polynomial) and find the crossing on the fit, not the raw pair.
7. Replace first-crossing latch with all-crossings + CV gate.
8. GPU: wire the `hybrid_los_prefix` pre-pass for real (kills the O(steps²) recompute AND the watchdog timeout), fix RNG seeding per ray, root-cause the v2 NaNs that the isfinite guards currently hide.

## 5. Other notable findings (medium, selected)

- **Refraction is dead code in production** (critic finding): `compute_refractive_indices` is only called in tests; production builder leaves n=1.0 everywhere → README "Snell's law at every shell boundary" is false in every real run.
- **Mesopic model is not CIE 191:2010** as labeled (missing M(m) normalization, constant K-ratio); scotopic V′(λ) table corrupted (peak 0.982 vs 1.0, 10–20% errors); `luminance_scotopic = photopic × 1.5` in two places.
- **LBL Voigt machinery (Humlicek/“Weideman”) is dead code** at runtime (absorption is prebaked); the “Weideman 1994” function is actually a Laplace continued fraction (false attribution); O2 partition exponent self-contradictory.
- **Terrain never shifts Fajr/Isha** — only sunrise/sunset; GeoTIFF tag parser is dead code with a hardcoded geo-transform; Danish LiDAR backend is an unfinished placeholder.
- **Clouds**: single-term HG instead of Mie (no forward peak/glory); SSA=0.999 for water cloud; `twilight-clouds` crate is an empty placeholder; thick low cloud collapses dawn to an implausible ~9–10° depression.
- **Weather**: NO2 surface override rescales the whole column 16–130×; maritime aerosol types unreachable.
- **Solar**: DE440-vs-Horizons validation tests are all `#[ignore]`d; "±0.001″" claim not achieved by delivered topocentric path (~25″).
- **Slop markers**: chat-transcript comments ("Let me use the known 2-term…"), dead v1 Metal kernels, dead solar/vision GPU LUTs ("reserved for Phase 11f"), perf numbers with no benchmark harness, "978 tests" badge unreachable in any single build.

## 6. libRadtran validation plan (runnable)

Three tiers, each isolating one physics block (full decks in `opus-sweep.json → libradtran`):

1. **Tier-1 — single-scatter vs DISORT (single-scattering output), Rayleigh-only**, matched US-Std-1976 + 347 DU O3 + TSIS solar + albedo 0.15, zenith + principal-plane radiance at SZA 90–108°, 380–780 nm/10 nm. Target |rel err| < 3–5%.
2. **Tier-2 — column optics**: direct/diffuse irradiance at SZA 60–85° sweeping O3 ∈ {220, 347, 600} DU and aerosol ∈ {none, continental, urban, desert} — tests gas absorption + aerosol extinction.
3. **Tier-3 — full field vs DISORT multi-stream AND MYSTIC backward-MC** at SZA 90–108°, then convolve both with V(λ)/V′(λ) → luminance-vs-SZA → **directly compare the implied threshold-crossing SZAs** (the photometric bottom line for prayer times).

Engineering: add `twilight-cli compare` subcommand (the engine already computes the exact observable; only an output hook is missing — `simulation.rs:127,145`, `single_scatter.rs:361`) + `tools/validate_libradtran.py` to generate decks, run `uvspec`, and emit a pass/fail table.

## 7. Priority roadmap

1. **Fix H1** (phase angle sign) + add the toward-sun > away-from-sun HG test.
2. **Fix C6** (polar panic) — guard `analyses.len() < 2`, short-circuit persistent twilight.
3. **De-fabricate**: delete or honestly gate CUDA/Vulkan/WebGPU; rewrite README GPU section, skyglow claims, O3-column claim, SQM-calibration comment; fix the test badge.
4. **CV program** (§4 items 2–7) — deterministic default, control variate, MIS fix, seed averaging, crossing-on-fit.
5. **Metal**: wire `hybrid_los_prefix` pre-pass; root-cause v2 NaNs; restore meaningful parity tolerance; fix batch-slower-than-serial.
6. **Wire the dead physics or drop the claims**: refraction n(λ) population in the production builder; Garstang into the pipeline (replacing raw-VIIRS injection); terrain into Fajr/Isha.
7. **Vision layer**: correct V′(λ) table; real CIE 191:2010 mesopic or relabel; source the thresholds (published twilight photometry / SQM campaign — this is the heart of the fiqh question).
8. **Clouds**: Mie phase functions (precomputed tables, e.g. from libRadtran's mie tool or Hu-Stamnes parameterization), SSA≈1.0, real `twilight-clouds` crate or fold into `twilight-data`.
9. **libRadtran harness** (§6) in CI as the permanent truth anchor.

---
*Fable 5 cross-model sweep: running (run `wf_16692828-9a0`); comparison addendum to follow.*
