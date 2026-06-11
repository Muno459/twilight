# Fable 5 Cross-Model Sweep — Addendum to REVIEW.md

**Run:** `wf_16692828-9a0` (resumed once after session-limit hit) · 14 dimensions, same prompts/protocol as the Opus 4.8 sweep, all agents on `fable`.
**Yield:** 112 findings vs Opus's 75. Same big picture, plus 9 major new confirmed discoveries — Fable's verifiers *executed code* (ran the binary, replicated algorithms in Python vs scipy) rather than only re-reading it.

## Cross-model agreement (both sweeps, independently)

Both models independently found, with matching file:line:
- **Phase-angle supplement bug** (`sun_dir·(−view_dir)`) — Opus rated high, Fable rated **critical** with the sharper observation that *all* estimator sites are consistently wrong (single_scatter ×2, photon.rs ×7 incl. NEE + hybrid order-1 + Metal), masked because every test uses `rayleigh_fraction = 1.0`. Fable quantified: ~123× aerosol phase underestimate at production geometry.
- Fabricated CUDA/Vulkan/WebGPU backends + README table + parity claim (also proven empirically by direct builds).
- SQM-calibration fabrication on the prayer thresholds; corrupted scotopic V′(λ) table; mesopic ≠ CIE 191:2010.
- Polar-day panic; MC-noise threshold crossing; Metal watchdog kill; 0.05×–20× "parity" test.
- Garstang dead code + raw-VIIRS-as-sky-radiance; terrain never shifts Fajr/Isha; surface-O3→column fudge; clouds via single-term HG; `twilight-clouds` empty husk; refraction never enabled in production (Opus critic flagged it; Fable's core-physics reviewer confirmed with grep evidence: `compute_refractive_indices` only called from `#[cfg(test)]`).

**Conclusion: the two models agree on every major finding. Nothing in the Opus report was contradicted.**

## New in Fable 5 (confirmed; not in the Opus report)

1. **CLOUDY PRAYER TIMES ARE BROKEN OUTRIGHT (critical, verifier ran the binary).** `pray --cloud stratus` → radiance 2.1e-15 (≈0) with non-monotonic fireflies; **Fajr returned at SZA 90.00 = sunrise** (clear sky: 105.46). The estimator weights every LOS contribution by direct-beam `exp(−τ)` (e⁻³⁸ through OD-10 stratus) and the MC chains that should rebuild the ~50% diffuse transmittance are never sampled (importance branches tuned for clear-sky Rayleigh). The headline use-case — "your prayer app doesn't know if it's cloudy" — is the one twilight gets wrong by 60–90+ minutes. Fix: delta-Eddington/two-stream for the cloud layer coupled to MC, or refuse OD>1 clouds. `photon.rs:2458-2469, 2973`
2. **Doppler half-width exactly 100× too small** — m/s vs cm/s mixed in `sqrt(2RT·ln2/M)/c`; verified ratio 100.000; a unit test *asserts the wrong value*. `gas_absorption.rs:501-526,1294-1309`
3. **Voigt continued-fraction blow-up near the real axis** — replicated in Python vs `scipy.wofz`: rel. error up to 7.8e4 at y=1e-6; only y≳0.3 reaches 1%. `gas_absorption.rs:370-439`
4. **Equation-of-time sign inverted in `sun_rise_transit_set`** — canonical NREL case off by **29 minutes** (±33 max). `spa.rs:544-578`
5. **O3 column doc lie: profile actually integrates to 546 DU, not "~347"** — so `--weather` with a measured ~300 DU column *cuts* O3 45% instead of roughly matching; the only test brackets 100–600 DU. `weather/lib.rs:73-75`
6. **Metal RNG seed collapse (root cause found)** — `metal.rs:284` keeps only the low 32 bits of the seed; `sza_deg.to_bits()` has all-zero low 32 bits for every 0.5°-grid SZA → **base_seed = 0 for the entire prayer scan** (verified arithmetically). Compounds with:
7. **Untruncated importance sampler in the Metal shader** — the CPU bounds importance weights at 200 via a *truncated* power-cosine proposal (`photon.rs:901,913-926,4774-4814`); the shader never ported the truncation → unbounded f32 weights → overflow/NaN → the `isfinite` guards silently zero them (downward bias) → the 73% CV + 4× firefly we measured. Together with the dead `hybrid_los_prefix` (O(steps²) recompute → watchdog), the full Metal pathology is now explained.
8. **MC estimator defects (beyond Opus's MIS-bias finding):** (a) orders-2+ source function never applies `p(ω′→view)/4π` — the view-phase coupling is missing entirely (critical, partial-confirmed); (b) scalar+ALIS secondary chains *kill the whole chain at the first shell-boundary crossing* instead of continuing with a fresh exponential sample (memoryless property violated; the Stokes chain does it right); (c) three-branch weights double-count the hemisphere component (Σ weights = 0.75/0.25 instead of 0.5/0.5).
9. **Garstang has an extra ÷4π** (phase functions already unit-normalized) — in both `garstang.rs:261` and `twilight.metal:2253`; plus uplight fraction re-applied to satellite-measured upward flux (double count).
10. **DE440 "sub-arcsecond" measured at 24–89 arcsec** delivered topocentric (UT1≈UTC alone is up to 13.5″; 4-term nutation; ζ_A=z_A shortcut). Geometric ICRF is genuinely mm-level — the fabrication is in the delivered-accuracy claim.
11. **Danish LiDAR backend is a guessed, untested endpoint** (code comment admits it; two different Danish data platforms conflated) — README ticks it as done. (critical, confirmed)
12. **The smoking gun:** literal abandoned LLM reasoning shipped in production docs — `mapping.rs:199-211` contains "… wait]" mid-derivation followed by a dimensionally-wrong "Actually:" formula. Also `exp-mult` README "precision technique" that the shader explicitly argues against and does not use; fictional `de440` feature flag; "AFGL 6 atmospheres" where 5 of 6 silently fall back to US Standard.

## Revised priority order (merging both sweeps)

1. **Phase-angle fix** (every estimator site + Metal shader) — invalidates all aerosol/cloud results until fixed.
2. **Cloud transport** — delta-Eddington/two-stream layer coupling (or hard-refuse OD>1) so cloudy Fajr/Isha is *possible*; this is the product's stated reason to exist.
3. **Polar panic guard** + EoT sign + O3 546-DU recalibration + Doppler 100× (each is a small, well-bounded fix with a regression test).
4. **MC correctness set:** view-phase coupling in orders 2+, boundary-crossing continuation, branch-weight normalization, SSA-before-NEE — *then* the CV program from REVIEW.md §4 (control variate, K-seed averaging, crossing-on-fit).
5. **Metal:** port the truncated proposal (kills NaNs at the source — then delete the isfinite masking), fix the 32-bit seed packing (use splitmix64 of the full 64-bit seed), wire `hybrid_los_prefix`, restore a meaningful parity tolerance.
6. **De-fabricate** (single PR): delete vapor backends + Danish LiDAR guess + Garstang/VIIRS claims + SQM claim + DE440 accuracy + AFGL profiles + README GPU section/table/badges + perf numbers + "… wait]" residue.
7. **Re-source the thresholds** (the fiqh-critical constants) from published twilight photometry, and stand up the libRadtran harness (REVIEW.md §6) as the permanent truth anchor.

## Test-suite honesty (test-quality dimension, resumed run)

More substantive than typical slop — only ~6.5% of 1,090 `#[test]` fns are assertion-weak, and solar/threshold/data crates carry real literature-anchored golden values (NREL Table A5, Horizons vectors, CIE peaks, US-Std values). But the headline is dishonest:
- **"978 passing / ~14 seconds" matches no buildable configuration** — ~1,050 actual default-feature tests, README's own table sums to 991, runtime is minutes-to-tens-of-minutes, and the badge is a hardcoded shields.io static with **no CI workflow behind it**.
- **On real Apple Silicon the suite is RED** (the 4 Metal failures we reproduced).
- 30 GPU tests pass as **silent no-ops**; 19 "parity" tests are structurally unrunnable; 15 `#[ignore]`d tests (incl. the advertised DE440-vs-Horizons validation) never run.
- **No test in the workspace ever asserts that a Fajr or Isha time is produced.** The end-to-end product output is untested.
- The radiance engine has zero external golden values — all self-consistency.

## Completeness critic (resumed run) — two fundamental discoveries

1. **The 100 km atmosphere ceiling truncates exactly the regime prayer times live in.** Measured: single-scatter radiance is **exactly 0.0 for SZA ≥ 104°**, and at SZA 102° the engine's luminance is **20–40× below published twilight photometry** (Rozenberg, Patat 2006) — while the engine's own Fajr/Isha crossings sit at SZA 104–105°. The headline outputs are computed in the zone where the model collapses; the reported depressions (~14.6–15.1°) may be artifacts of the ceiling + the luminance deficit rather than physics. Deep twilight is dominated by high-altitude scattering and multiple scattering — a 100 km cap plus weak MS handling cannot represent 15–18° depressions faithfully. **This must be resolved before trusting any absolute prayer time from the engine.**
2. **The polarized Stokes/Mueller path was never audited — and it is the DEFAULT production path** (`pipeline.rs:121 polarized: true`; `--fast` is the scalar opt-out). Given the confirmed supplement-angle bug in the scalar phase code, this is the most likely hiding place for another sign/convention error.
3. Measured systematic: `--scattering single` vs `hybrid` moves Fajr by ~2 min and Isha by ~3 min (same site/date) — a mode-dependent bias on top of the MC noise.
4. **Scope correction:** the "fabricated 100× ozone table" (`ozone_xsec.rs`) is *dead code*; the **live** O3 table in `gas_absorption_data.rs` is correct Serdyuchenko data. The dead file should be deleted, but production O3 absorption is not 100× wrong. (The Doppler 100× and Voigt blow-up findings concern the also-dead LBL path.)
5. **Positive controls (don't over-purge):** `no_std`/`forbid(unsafe_code)` claims are true; identical-args hybrid runs are bit-for-bit deterministic; the README Mecca clear-sky numbers approximately reproduce (Fajr 05:23:11 vs claimed 05:23); Rayleigh-only Tier-1 physics is solid.

## Final revised top-priority list (both sweeps + critic)

1. **Phase-angle fix** (all 9 sites + Metal) — prerequisite for everything aerosol/cloud.
2. **Audit the polarized path** (default-on!) for the same class of convention error.
3. **Raise/replace the 100 km ceiling and validate absolute luminance vs published twilight photometry** (Patat 2006, Spitschan 2016) — without this, no Fajr/Isha number is trustworthy at 15–18° depressions. The libRadtran harness is the tool for this.
4. **Cloud transport** (delta-Eddington/two-stream coupling) — currently 60–90 min wrong under cloud.
5. Bounded fixes: polar panic, EoT sign, 546-DU recalibration, NO2 column, Metal seed/proposal/los-prefix.
6. MC correctness set → then the CV/efficiency program.
7. De-fabrication PR + honest README + delete dead code (incl. `ozone_xsec.rs`, LBL path or fix it, vapor backends).
8. Add the missing **end-to-end test that asserts Fajr/Isha values**, golden-pinned.
