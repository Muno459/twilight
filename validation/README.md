# Validation program: index and reproduction map

Every results table in this directory states the exact command that
produced it. This file is the index: one row per document, with its
referee, its cached artifacts, and its regeneration command. All engine
runs use `--release` (the MC gates are calibrated for release builds).

External referees need a local [libRadtran](http://www.libradtran.org)
build (2.0.6; DISORT and MYSTIC) pointed to by `LIBRADTRAN_DIR`, and
SHDOM for the 3D cube tier. The cached referee outputs are committed
here, so every table can be re-derived from the caches alone without
rebuilding the referees; delete a cache file to force a fresh referee
run.

| Document | What it establishes | Referee / ground truth | Cache | Regenerate |
|---|---|---|---|---|
| `RESULTS.md` | Clear-sky transport: shape and absolute scale, SZA 30 to 106 | DISORT + MYSTIC | `tier1_*.out`, `g2_disort_*` | `python3 tools/validate_libradtran.py --tier 1a/1b/2` |
| `g2_tables.md` | Cloud-slab transport, both MC estimators, 144 points | DISORT + MYSTIC slab | `g2/` | `python3 tools/validate_libradtran.py --tier g2` |
| `RESULTS_G3_CLOUD_TWILIGHT.md` | 1D cloud decks at twilight geometry | MYSTIC | `g3/` | `python3 tools/validate_libradtran.py --tier g3` |
| `RESULTS_G3CUBE_SHDOM.md` | True-3D cloud cubes (checkerboard, broken deck) | SHDOM | `g3cube/` | `python3 tools/validate_libradtran.py --tier g3cube` |
| `RESULTS_DEEP_REGIME.md` | Deep combined regime: cloud decks past SZA 101, field-forced estimator, variance ledger | MYSTIC at 3e8 to 1e9 photons | `deep/`, `deep_regime_results.csv` | `python3 tools/validate_libradtran.py --tier deep`; gates `g_s3_*` in twilight-cpu (`cargo test -p twilight-cpu --release g_s3 -- --ignored`) |
| `RESULTS_MEASURED_SKY.md` | Absolute twilight decay vs measured skies | Patat 2008, Koomen 1952 | `measured_sky_runs/` | commands inside the document |
| `RESULTS_CRITERION_SITES.md` | The khayt criterion vs 16 observation campaigns; skyglow/aerosol response (section 9) | published campaigns + OpenFajr panel | `criterion_runs/` | commands inside the document |
| `RESULTS_EDGE_FACTOR.md` | The one calibrated constant is cross-site invariant (n = 8 inversion, literature decomposition, zodiacal cross-prediction) | campaign inversions | `criterion_runs/edge_factor/` | `python3 tools/criterion_edge_factor.py --analyze` (add `--attack3` for the zodiacal ladder) |
| `RESULTS_FORCED_CLOUD.md` | Combined-channel forced mode under decks (history + addenda) | internal gates + MYSTIC | (uses `deep/`) | gate names inside the document |
| `aod_runs/aod_runs.tsv` | Live-AOD validation table (Assiut/Tubruq/Birmingham/Mecca) | campaign observations | `aod_runs/` | `python3 tools/aod_sites.py` |

## Conventions

- **Bands are pre-registered**: every gate's tolerance is derived from
  measured seed CVs or referee SEs BEFORE the comparison, and the
  derivation is quoted in the gate's doc comment. A gate that cannot fail
  is treated as a bug (two such gates were found and rewritten by the
  adversarial review; see the RESULTS_DEEP_REGIME review section).
- **Verdict taxonomy**: PASS (statistically consistent with the referee at
  stated power), LOW-POWER (consistent but the twilight-side variance is
  too large for the band to bite; never counted as evidence), KNOWN-LIM
  (documented limitation with mechanism and follow-up), FAIL (would block
  merge).
- **Honest negatives are kept**: documents record what did NOT close
  (Tubruq sea horizon, the 550 nm heavy-tail cells, broken-deck zenith
  starvation) alongside what did.
- Bit-identity harnesses (`bitcheck_dump` in twilight-cpu) pin every
  refactor: unchanged surfaces must reproduce radiance bit-for-bit.
