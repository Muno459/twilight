# Deep-tier high-seed re-run (AWS 96-core, 2026-07-08)

Twilight-only high-seed re-run vs the SAME cached MYSTIC referees.
1d path: 1024 seeds x 16000 photons (8x the shipped 128). Field: 192 (running).

## tab:variance (tau*=3, 1d path) -- FAIL RESOLVED
Was (128 seeds): 3 PASS / 1 FAIL / ... ; the 101/550 "fail (boundary)" 0.59x
was heavy-tail UNDERSAMPLING, not a zenith bias. At 1024 seeds:

| SZA / wl | ratio (model/referee) | band/ref | verdict |
|---|---|---|---|
| 101 / 450 | 1.04 +- 0.12 | 0.42 | PASS |
| 101 / 550 | 0.82 +- 0.09 | 0.34 | PASS  (was fail-boundary 0.59) |
| 101 / 650 | 0.88 +- 0.10 | 0.37 | PASS  (was "consistent") |
| 103 / 450 | 0.86 +- 0.26 | 0.85 | LOW-POWER |
| 103 / 550 | 1.20 +- 0.53 | 1.63 | LOW-POWER |
| 103 / 650 | 1.04 +- 0.52 | 1.60 | LOW-POWER |

All three SZA-101 cells PASS (thin+thick deck); SZA-103 stays LOW-POWER
(heavy-tailed, ~1500 seeds needed to gate = infeasible). NO FAIL.

## Full 14-cell gate (1024-seed 1d + 128-seed field-tau1, cached MYSTIC)
7 PASS / 0 FAIL / 7 LOW-POWER  (was 3 PASS / 1 FAIL / 10 LOW-POWER).
All SZA-101 PASS (1d+field, tau*1+tau*3); all SZA-103 LOW-POWER.

## Pending: field tau*=3 (192 seeds, the 2 omitted appendix cells) + field
tau*=1 upgrade -> complete appendix to 16 cells, then one comprehensive
update of tab:variance + tab:deepfull + fig:deep + prose (seed counts
1024 1d / 192 field; drop "fail boundary" + "save the deepest tau*=3
zenith cell" + "omitted" claims).
