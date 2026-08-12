# Campaign record: the deep-twilight closure (2026-07-08 to 2026-07-11)

This file makes the repository self-contained: it consolidates the
context that accumulated across the working sessions so that nothing
essential lives only in a session transcript. Detailed derivations and
per-increment verdicts: docs/BDPT_UNDER_CLOUD_PLAN.md (the estimator
campaign) and docs/FIELD_CONNECTIONS_PLAN.md (the voxel-field
extension). Calibration analysis: twilight-papers repository,
CALIBRATION_MEAN_VS_2SD_RESOLVED.md.

## What the engine is

twilight: a backward Monte Carlo radiative-transfer engine computing
absolute spectral sky radiance through deep twilight (solar zenith
angles to ~108 degrees) in a spherically stratified refracting
atmosphere to 150 km, 41 wavelengths, full Stokes polarization, measured
3D cloud fields walked voxel by voxel. On top of the radiance sits a
psychophysical detection layer (the khayt criterion) computing the
Quranic dawn (fajr al-sadiq), false dawn (al-kadhib), and the evening
shafaq events as contrast-detection problems. Four implementations:
scalar hero-wavelength (ALIS) CPU chain (the production `--fast` path),
polarized Stokes CPU chain (the default path and the deep referee's
estimator), Metal GPU, WGSL GPU.

## The problem this campaign solved

The deep referee tier (cloud decks at SZA 101/103 against 1e9-photon
spherical backward MYSTIC references) stood at 7 PASS / 0 FAIL /
7 LOW-POWER: the deepest cells were statistically unresolvable at any
practical seed budget because the backward-chain estimator's signal
traveled through rare "sun-run" walks (chains physically random-walking
~1500 km to sunlit air before scoring); the median seed carried ~2
percent of the true radiance and single seeds carried 50x.

## What was built (all merged to main)

1. Bidirectional connection estimator: every main-chain collision at
   SZA >= 99 connects to a registry of 512 light subpaths traced from
   the sunlit terminator, covering every scattering order >= 3 from the
   sunlit side; each order pairs the connection against its next-event
   counterpart through a shared per-path balance-heuristic MIS whose two
   weights sum to one (unbiased by construction). Both CPU chains.
2. Three real bugs found and fixed on the way, each by a designed
   falsification test rather than by tuning:
   - the BDPT eye-phase sign flip (wrong for aerosol HG shells);
   - the registry success-conditioning normalization (+1/P(success)
     overcount, caught by the partition invariant E[new] = E[old] at
     4.8 sigma, fixed by one attempt-count factor, verified to 0.4 sigma);
   - the evening-solver bracket selection (last margin crossing instead
     of the peak-run crossing; pinned Mecca-June shafaq al-ahmar at the
     red-cone-gate collapse, 17.4 deg instead of 15.0; fixed and
     regression-tested, other sites bit-identical).
3. Voxel-field extension: the light subpath walks 3D cloud fields
   exactly (piecewise-constant combined DDA inversion, no majorant, no
   null collisions), so the connections now serve heterogeneous cloud
   fields; the analog field chains are unchanged.

## Final validation state

- Deep referee tier: 16 PASS / 0 FAIL / 0 LOW-POWER, ratios 0.87 to
  1.19 (1d cells at 1024 seeds; field cells at 1024/512).
- MYSTIC reference replicas: 8 fresh-seed 1e9-photon reruns; references
  reproduce within ~1 sigma except the cached tau3/103/650 case (3.3
  sigma low draw); results pass against cached AND replica-pooled refs.
- Heavy MC gate suite: 10/10 g_s gates + the independent slab referee
  PASS on the campaign tree.
- Bit identity: 861/1066 dump rows identical to the pre-campaign commit;
  all 205 differences are exactly the by-design SZA >= 99 hybrid
  surface. The field-connections branch: only deep-field rows differ.
- G-FC ladder (field connections): partition invariant 0.31 sigma with
  seed CV 6.16 -> 0.855; field-vs-1D equivalence 0.46 sigma; field
  referee cells resolve at 128-256 seeds where 512-1024 were needed
  (6-16x less variance per seed).
- GPU parity through the estimator change: Metal suite 140/140;
  G-MC-PARITY-3 clear-sky CPU/GPU ratios 0.9995 / 0.9957 / 1.0040 at
  SZA 95 / 97 / 100 (the 100-degree cell compares the CPU connection
  estimator against the GPU pre-connection estimator: same mean to 0.4
  percent, as the unbiasedness argument requires). The GPU kernels
  deliberately retain the pre-connection estimator; disclosed in paper 1.
- Criterion calibration: verified insensitive to the estimator upgrade
  and the solver fix (detection depressions unchanged within 0.02 deg at
  the anchors; A/B via the TWILIGHT_BDPT_CONN_OFF switch).

## The two papers (sources: twilight-papers repository)

Both elsarticle, single author, targeted at JQSRT as a companion pair.

Paper 1 (methods, 32 pp, 11 figures, 4 tables): "Absolute sky radiance
through deep twilight: a backward Monte Carlo model with
three-dimensional cloud transport, externally validated to solar zenith
angle 103 degrees". Validation in three rings: reference codes (DISORT
0.975-0.993 absolute at SZA 60-85; 144/144 cloud-slab points vs DISORT
and MYSTIC; 112/112 3D-cube points vs SHDOM; the deep tier above),
measured skies (zenith twilight decay to 1.2-5.5 percent per magnitude),
and human observation (sixteen campaigns, median absolute residual 0.3
deg of depression).

Paper 2 (application, 21 pp, 7 figures, 4 tables): "When does dawn
become visible? The Quranic dawn (fajr al-sadiq) and dusk (shafaq) as a
detection problem, tested against 116 years of observation". One
calibrated constant (f = 56) bridging laboratory contrast thresholds to
field detection; calibrated on the three campaigns documenting a
central-mean dawn (Riyadh, Aswan, Kottamia; per-site inversions f = 51
to 57, no latitude trend); sixteen campaigns spanning 116 years and 58
degrees of latitude at 0.3 deg median residual; the Birmingham 42-date
panel year reproduced with zero retuning (mean +0.54 deg, RMS 0.95),
where fixed angles miss by 1.8-4.4 deg RMS and the most-used one is
undefined on 20 of 42 dates. Evening events at the corrected engine:
Mecca abyad 17.33 deg (SQM end 17.99 +- 0.16), ahmar 14.96 (literature
12-15). Tubruq sea horizon: a climatological marine boundary layer (AOD
0.12) closes roughly half the +1.2 deg residual; the remainder is a
documented open row.

Calibration context that must not be lost: the desert campaign
literature mixes statistic types; several headline values are upper
confidence bounds (mean + 1-2 SD), not central means. The calibration
uses only the campaigns documenting a central mean, and paper 2's
campaign table carries a per-site "basis" column disclosing each value's
statistic type. Hail (14.01 central mean) is anomalously shallow and had
biased an earlier three-site calibration toward f = 70; the resolved
fit places f = 56 with the pristine-subset inversions at 51-57.

Submission checklist: highlights, cover letters, and declarations are in
artifacts/2026-07_deep_closure/papers/. Two items require the author
before submission: the funding statement and the generative-AI
statement in declarations.md. The "(companion paper)" references in the
manuscripts must become real citations at submission time.

## Open items (deliberate, not debt)

- The 3D-field chains remain analog; extending forced mode under fields
  is blocked by the documented per-wavelength null-ratio problem
  (ac673c7 addendum). The field connections make this mostly moot for
  the deep cells.
- GPU kernels keep the pre-connection estimator (statistical same-mean
  parity, verified; a port would be an optimization, not a correctness
  item).
- Field-path production khayt scans (broken decks at deep SZA) are now
  computationally practical and untested in production.
- SQM field campaign (user hardware) and the ahmar observation dataset
  remain the outstanding empirical extensions.

## Compute provenance

Six AWS us-east-1 c7a.24xlarge instances across 2026-07-08 to -10, all
terminated; roughly 150 USD total. Instance IDs and per-run protocols:
MANIFEST.md in this folder.
