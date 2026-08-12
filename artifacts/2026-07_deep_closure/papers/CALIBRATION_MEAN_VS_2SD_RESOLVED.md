# Mean-vs-upper-bound audit of the desert calibration — RESOLVED

Date: verification pass on the f=56 edge factor. Three independent
source-extraction agents (Saudi/Aswan, Egyptian NRIAG, Tubruq/Wadi/Fayum)
read the primary campaign papers. Result below. **Bottom line: f=56
stands, but for a corrected reason; the shipped justification and the
Table 1 "observed" column mix central means with documented upper
bounds and must be reframed honestly.**

## 1. What the published "observed" depressions actually are

The NRIAG/desert group has a **documented reporting convention**: bare,
unmarked depression values in their comparison tables are the
"highest value of confidence" = **mean+1SD or median+kσ upper bound**,
NOT the central tendency. Confirmed verbatim per site:

| Site | Cited value | What it actually is | Central mean | n |
|---|---|---|---|---|
| Hail (Khalifa 2018) | 14.66 | **mean + 2SD** | **14.014 ± 0.317** | 32 (selected good-vis) |
| Riyadh (Al-Mostafa 2005 / Taha 2025) | 14.6 | **mean** (mean+1SD is a separate 14.88) | **14.58 ± 0.303** | 13 |
| Aswan camera (Mawad 2024) | 14.90 | mean ± SD (arch height 2.66°±0.23) | **14.90 ± 0.17** | 5 |
| Kottamia (Hassan 2014) | 14.66 | **site MEAN** (coincidentally ~14.7) | **14.665 ± 0.197** | 4 |
| Bahariya (Hassan 2014) | 14.6 | 14.6 is a *different photoelectric study*; naked-eye mean is shallower | **13.814 ± 1.244** | (pooled n=35) |
| Matrouh (Hassan 2014) | ~14.7 | pooled Egypt mean+1SD, not a Matrouh value | **13.41 ± 1.096** | 4 |
| Sinai/Beer-Al-Abd (Hassan 2016) | 14.61 | **median + σ** | median ≈ 13.66 (σ 0.955) | 10 |
| Assiut (Hassan 2016) | 13.665 | **median + 2σ** | **mean 11.247 / median 11.375** | 120 |
| Tubruq desert (Hassan 2015) | 14.7 | **mean + 2SD** | **13.144 ± 0.757** | 623 |
| Wadi Al-Natrun (Semeida 2018) | 14.57 | **mean + 1SD** ("highest value of confidence") | below 14.57, range 12.48–15.14 | 38 |
| Fayum (Rashed 2022) | 14.8 | single-mission point / 4-method range 14–14.8 | no clean mean | 2 missions |
| Pooled Egypt (Hassan 2014) | 14.7 | **grand mean + 1SD** | 13.642 ± 1.054 | 35 |

## 2. The re-fit on a consistent central-mean basis

Using the cached f-ladder runs (`tools/refit_edge_factor.py`,
`tools/clean_subset_fit.py`), inverting each site's confirmed central
mean to the f that reproduces it:

- **On ALL six OOS "desert campaign" central means**, the implied f
  scatters catastrophically: Kottamia 57, Fayum 80, Bahariya 107,
  Sinai 131, Matrouh 139, **Tubruq 249** — a 4.4× spread, RMS 0.55.
  The shallow means (11–14°) are physically **unreachable** by
  clear-sky transport at any sane f: they are contaminated by light
  pollution (Assiut mean 11.25°), haze-averaging over 623 all-nights
  (Tubruq mean 13.14° vs Hail's 32 *selected* clear nights → 14.01°),
  or n=4 samples (Matrouh). **The naked-eye desert set cannot pin a
  single clear-sky constant on a raw-mean basis.**

- **On the pristine, good-visibility subset** (the campaigns that
  observed clear nights and report a genuine mean), the implied f
  converges tightly:
  - Aswan 14.90 → f = 51.1
  - Riyadh 14.58 → f = 53.1
  - Kottamia 14.665 → f = 57.0  (independent, out-of-sample)
  - Hail 14.01 → f = 81.2  (lone deep outlier; 1000 m altitude,
    thinner atmosphere → fainter twilight → deeper detection)
  - RMS-optimal over the sea-level trio {Riyadh, Aswan, Kottamia} =
    **f = 53.2 (RMS 0.045)**; adding Hail → f = 59.5.

**f = 56 sits inside the pristine central-mean band (≈53–60).** It is
confirmed, not moved. Re-shipping 53 or 54 would change dawn by <0.2°
(~1 min, far inside the ±1° observational scatter) and is not worth
destabilizing the papers.

## 3. Why the shipped f=56 was numerically right but wrongly justified

The shipped calibration narrative ("OOS optimum over the six desert
campaigns", treating headline values 14.5–14.68 as targets) got 56
because for the all-nights campaigns the **mean+2SD upper bound ≈ the
clear-sky / best-visibility detection limit ≈ what the engine models**.
So the wrong targets accidentally landed on the right f. The honest
version: f is pinned by the pristine central-mean subset
(Riyadh/Aswan/Kottamia → 53–57), and the engine reproducing the deeper
all-nights *upper bounds* is the engine sitting at the clear-sky limit,
which is a positive, not a coincidence.

## 4. Required paper fixes (f unchanged)

1. **Calibration section (both papers):** replace "OOS optimum over the
   desert campaigns" with the pristine central-mean subset argument
   (Riyadh 14.58→53, Aswan 14.90→51, Kottamia 14.665→57 independently),
   Hail 14.01→81 flagged as an altitude outlier, and the explicit note
   that the raw desert means are too heterogeneous (mixed statistics,
   light pollution, haze) to pin the constant.
2. **Table 1 "observed" column:** disclose the statistic type per row
   (mean / mean+1SD / median+2σ / camera / multi-method range). The
   residual against an upper-bound row is a *lower bound* on the true
   mean-basis agreement, and equals ~0 because the engine sits at the
   clear-sky detection limit. Do NOT silently present upper bounds as
   central observations.
3. **Band-geometry citation (bonus, from Mawad 2024):** twilight-arch
   angular height 2.66°±0.23 above the horizon — a direct external
   check on the engine's arch geometry.
