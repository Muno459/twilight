# Deep-twilight closure campaign artifacts (2026-07-08 to 2026-07-11)

Everything in this folder was produced during the deep-regime estimator
campaign that took the deep referee tier from 7 PASS / 7 LOW-POWER to
16 PASS / 0 FAIL / 0 LOW-POWER. The full narrative and every verdict live
in `docs/BDPT_UNDER_CLOUD_PLAN.md` and `docs/FIELD_CONNECTIONS_PLAN.md`;
this folder preserves the data and the submission documents.

## papers/

Built PDFs of both manuscripts at the post-campaign state (deep tables,
figures, and prose at 16/16; corrected evening events; tested marine
sea-horizon result; sharpened titles and abstracts), plus the submission
package: highlights (within the 85-character Elsevier limit), companion
cover letters, and the shared declarations (two author-confirmation items
flagged inside: the funding statement and the generative-AI statement).
Sources live in the twilight-papers repository.

## deep_tier/

- `twilight_1d_tau1.json`, `twilight_1d_tau3.json`: per-seed radiances of
  the FINAL (registry-normalized) estimator, 1024 seeds x 16000 photons
  per (SZA, wavelength) cell, produced by the Stage A re-run on AWS
  c7a.24xlarge instances (deep_driver.py over
  tools/validate_libradtran.py, `compare --fast` ALIS protocol).
- `twilight_field_tau1.json` (1024 seeds), `twilight_field_tau3.json`
  (512 seeds): the 3D-field path per-seed radiances (polarized Stokes
  chains via the deep_referee_runner harness) that closed the two former
  LOW-POWER field cells by seed statistics alone.
- `deep_regime_results.csv`: the official tier table regenerated from
  these caches against the cached 1e9-photon spherical backward MYSTIC
  references (validation/deep/deep_mystic_*): 16 PASS / 0 FAIL /
  0 LOW-POWER, ratios 0.87 to 1.19.

## mystic_replicas/

Eight fresh-seed MYSTIC reruns of the four tau*=3 references
(SZA 101/103 at 550 nm, SZA 101 at 450 nm, SZA 103 at 650 nm; two
replicas each, distinct mc_randomseed, each at its reference's own
budget: 3e8 photons for the SZA 101 pair, 1e9 for the SZA 103 pair -
see the case.inp files), run with libRadtran 2.0.6 built
on the AWS box. (Correction 2026-08-16: this entry previously called
all eight replicas 1e9-photon.) Each directory holds the exact case.inp plus mc.rad.spc
and mc.rad.std.spc. Finding: the references reproduce within roughly one
reported sigma except the cached tau3/SZA103/650 nm case, which sits
2.8 combined sigma below the pooled value of its replicas (a low draw;
an earlier version of this entry quoted 3.3); the twilight results pass against
both the cached and the replica-pooled references.

## Verdicts preserved elsewhere (raw logs were on terminated instances)

- Heavy gate suite (10 of 10 g_s gates plus the independent slab referee,
  7057 s) and the two-tree bit-identity dumps (861/1066 rows identical
  against the pre-campaign commit; the 205 differences are exactly the
  by-design SZA >= 99 hybrid surface): recorded in
  docs/BDPT_UNDER_CLOUD_PLAN.md and the merge commit messages.
- G-FC gate ladder for the voxel-field connections (partition invariant
  0.31 sigma, field-vs-1D 0.46 sigma, field referee cells at 128 seeds):
  merge commit 4ccd737 and docs/FIELD_CONNECTIONS_PLAN.md.
- Criterion A/B (calibration insensitivity within 0.02 deg at the
  anchors), Tubruq marine three-date result (-0.54 deg mean), Mecca
  evening-solver verification (ahmar 17.40 -> 14.96): transcribed in
  docs/BDPT_UNDER_CLOUD_PLAN.md and the papers.
- Metal GPU parity through the estimator change: standard suite 140/140;
  G-MC-PARITY-3 clear-sky ratios 0.9995 / 0.9957 / 1.0040 at SZA
  95 / 97 / 100 (the SZA 100 cell compares the CPU connection estimator
  against the GPU pre-connection estimator: same mean to 0.4 percent).

## Compute provenance

AWS us-east-1 c7a.24xlarge instances, all terminated: i-0900fd9473484f396
(first 1024-seed re-run), i-03469fb9611601a2c (attribution campaign +
MYSTIC replicas), i-09adf2073a5a601a4 (criterion A/B),
i-0a1d350212aa9e4ad (README numbers, Tubruq marine, field campaign),
i-02720bc1f2bf9ecf4 (evening-solver verification), i-096f32d3d4753bc42
(heavy gates + G-FC ladder). Total compute roughly 150 USD.
