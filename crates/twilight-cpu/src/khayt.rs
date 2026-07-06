//! The khayt al-abyad criterion - the Quranic definition of dawn,
//! implemented as the contrast-detection task it literally describes.
//!
//! "حتى يتبين لكم الخيط الأبيض من الخيط الأسود من الفجر" (2:187):
//! Fajr enters when the WHITE THREAD (the band of dawn light low on the
//! eastern horizon) becomes DISTINCT TO YOU from the BLACK THREAD (the
//! adjacent dark sky) - a differential, human-visual criterion, further
//! qualified by the sunnah: the true dawn spreads LATERALLY along the
//! horizon (mustatir), while the false dawn (al-fajr al-kadhib - the
//! zodiacal light cone) stands narrow and tilted "like the wolf's tail"
//! and does NOT spread.
//!
//! This module turns simulated per-direction sky luminances into those
//! judgments:
//!
//! - a FAN of sky patches: a band of directions straddling the solar
//!   azimuth just above the horizon (the candidate white thread), plus
//!   reference patches on the same elevation ring far from the dawn
//!   azimuth (the black thread);
//! - per scan step, the Weber contrast of each band patch against the
//!   reference, compared to the adaptation-dependent contrast threshold
//!   (Blackwell TVI - the same psychophysics used elsewhere in the
//!   engine, applied here to the differential task the ayah specifies);
//! - FAJR SADIQ when the contrast holds across the lateral extent
//!   (spread test); AL-FAJR AL-KADHIB when only the central patches are
//!   distinct (the zodiacal wedge passes the contrast test but fails the
//!   spread test);
//! - the mirrored disappearance criteria for Isha: shafaq al-ahmar (red
//!   band, Shafi'i/Maliki/Hanbali - the red channel of the same patches)
//!   and shafaq al-abyad (white, Hanafi).
//!
//! Being a RATIO of two simulated patches, the criterion cancels most
//! systematic errors that afflict absolute-threshold detection: a global
//! radiometric scale factor, uniform skyglow, and uniform cloud
//! attenuation multiply band and reference alike. It also subsumes the
//! high-latitude problem: contrast against tonight's actual reference
//! sky needs no "relative mode" special case.
//!
//! # Crossing resolution (the anti-quantization contract)
//!
//! The fan is scanned coarsely (1 deg SZA in the pipeline); a coarse
//! scan alone quantizes cliff-shaped crossings to bracket midpoints
//! (+-0.5 deg = minutes of clock time). [`detect_refined`] therefore
//! accepts a fan-evaluation callback and ADAPTIVELY REFINES every
//! detected crossing bracket with extra fan points (guided
//! regula-falsi with bisection fallback, proposals snapped to a
//! 1/64-deg dyadic grid so neighboring events and the two sides of the
//! night hit the pipeline's MCRT cache instead of re-simulating).
//! Refinement stops at the first of:
//! 1. bracket half-width (= worst-case midpoint recovery error) at
//!    most [`REFINE_TARGET_BRACKET_DEG`];
//! 2. the fresh point agrees with the log-secant across the bracket
//!    within its MC noise (floored at [`LN_CONSISTENCY_FLOOR`]) AND
//!    that tolerance resolves the crossing below the target: the
//!    bracket-restricted fit then interpolates below the bracket scale
//!    and further MCRT cannot improve it (never fires across a hard
//!    gate);
//! 3. both bracket endpoints statistically indistinguishable from the
//!    criterion (MC noise floor);
//! 4. the per-event budget [`REFINE_MAX_EVALS`] is spent.
//!
//! The residual bracket half-width is NEVER silently absorbed: it is
//! reported in [`KhaytCrossing::bracket_half_deg`] and folded into
//! [`KhaytCrossing::sigma_deg`] (as a uniform-distribution sigma,
//! half-width / sqrt(3)) together with the MC-noise term and a
//! first-order Jensen bias bound (see `finalize_crossing`).

use twilight_threshold::threshold::{contrast_threshold_weber, fit_crossing_loglinear};

/// Adaptive-refinement stop: worst-case midpoint recovery error, i.e.
/// the crossing-bracket HALF-width [deg SZA]. 0.05 deg is ~12-20 s of
/// clock time - below the ~1 min resolution of every field campaign
/// the engine validates against. From the 1-deg coarse fan a hard
/// cliff reaches half-width 0.03125 in 4 bisections.
pub const REFINE_TARGET_BRACKET_DEG: f64 = 0.05;

/// Maximum extra fan evaluations per event. Guided proposals converge
/// in 2-3 points on smooth (log-exponential) margin curves (then the
/// consistency stop fires); a hard cliff needs 4 bisections from the
/// 1-deg coarse grid to reach the half-width target.
pub const REFINE_MAX_EVALS: usize = 6;

/// Deterministic floor [ln units] for the log-linearity consistency
/// stop: MCRT margins are never trusted below ~2% in ln space (spectral
/// grid, celestial interpolation, and estimator systematics live at
/// that scale even when the seed-spread SE reads 0, e.g. the GPU path,
/// which derives its RNG stream from the SZA alone). The stop only
/// fires when floor/|slope| also resolves the crossing below
/// [`REFINE_TARGET_BRACKET_DEG`], and the floor is folded into the
/// reported sigma of consistency-stopped crossings.
pub const LN_CONSISTENCY_FLOOR: f64 = 0.02;

/// Cone absolute threshold [cd/m^2] gating the RED percept
/// (Hecht-Shlaer-Pirenne lineage): below it the band may be visible but
/// not COLORED - rods see no red. Shafaq al-ahmar requires the color.
const RED_CONE_GATE: f64 = 1e-3;

/// Geometry and psychophysics of the khayt detection.
///
/// Defaults are PROVISIONAL pending the observational-literature review
/// (dawn-campaign + extended-source psychophysics); each field documents
/// what should pin it down.
#[derive(Debug, Clone)]
pub struct KhaytParams {
    /// View zenith of the horizon band [deg]. The first dawn light
    /// appears in the twilight arch a few degrees above the horizon;
    /// terrain and extinction make the lowest degrees unusable.
    /// Pin: twilight-arch observations (Rozenberg/Minnaert) + campaigns.
    pub band_zenith_deg: f64,
    /// Azimuth offsets of the band patches from the solar azimuth [deg].
    /// The spread (mustatir) test acts across these.
    pub band_offsets_deg: Vec<f64>,
    /// Azimuth offsets of the reference ("black thread") patches [deg].
    /// Far enough from the dawn glow to be dark, close enough to share
    /// airmass; same zenith ring as the band.
    pub ref_offsets_deg: Vec<f64>,
    /// Multiplier on the Blackwell disc contrast threshold for the
    /// degrees-wide soft-edged band. <1 for spatial summation of large
    /// targets, >1 for gradual edges. Pin: extended-source psychophysics.
    pub k_contrast: f64,
    /// Same, for the red (shafaq al-ahmar) channel.
    pub k_contrast_red: f64,
    /// Multiplier applied on top of `k_contrast` for the MORNING task
    /// (noticing the new dawn glow). See Default for the calibration.
    pub edge_factor_appearance: f64,
    /// Multiplier for the EVENING task (tracking a fading band).
    pub edge_factor_disappearance: f64,
    /// Number of band patches that must be simultaneously distinct for
    /// the spread (true dawn) verdict.
    pub spread_required: usize,
    /// Fraction of the celestial background luminance assigned to the
    /// red channel (airglow 630 nm, red starlight) when forming red
    /// contrast. Crude; flagged for refinement.
    pub celestial_red_fraction: f64,
}

/// Calibration-analysis knob: parse an override of the APPEARANCE edge
/// factor from the `TWILIGHT_KHAYT_EDGE_APPEARANCE` environment
/// variable (raw value passed in; `None`/invalid/non-positive falls
/// through to `default`). Used by the edge-factor constraint sweep
/// (tools/criterion_edge_factor.py, validation/RESULTS_EDGE_FACTOR.md)
/// to trace a site's factor-to-depression response curve without a
/// rebuild. NOT a user-facing parameter: production semantics are the
/// calibrated default.
fn edge_appearance_override(raw: Option<&str>, default: f64) -> f64 {
    raw.and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(default)
}

impl Default for KhaytParams {
    fn default() -> Self {
        Self {
            // The twilight arch at first detection sits at 2.66 +- 0.23
            // deg altitude (Aswan calibrated-camera campaign, Adv. Space
            // Res. 2024); 3 deg keeps the patch inside the arch and
            // above near-horizon extinction/terrain.
            band_zenith_deg: 87.0,
            // True dawn at tabayyun spans ~30-40 deg of azimuth (Ilyas:
            // "whitish envelope of about 30 deg width"; the Hail
            // observers tracked 0-20 deg either side), while the false
            // dawn's zodiacal wedge is ~20 deg wide AT THE BASE
            // (Sultan, Yemen) and narrows with altitude. Outer patches
            // at +-18 deg sit beyond the wedge base but inside the
            // true band at distinctness.
            band_offsets_deg: vec![-18.0, -9.0, 0.0, 9.0, 18.0],
            ref_offsets_deg: vec![-100.0, 100.0],
            // Extended-source psychophysics (Blackwell 1946 large-disc
            // rows; Crumey 2014 asymptote): a degrees-wide band is
            // EASIER to see than the reference disc - pure size factor
            // 0.08-0.26 at scotopic adaptation, clawed back ~2x by the
            // soft edge. Recommended k ~ 0.4 (range 0.25-0.6) on the
            // disc thresholds the TVI table encodes.
            k_contrast: 0.4,
            k_contrast_red: 0.4,
            // EDGE-DISCERNIBILITY factors - the honest calibration layer.
            // Static-disc psychophysics says a 100% excess over the night
            // sky should be visible (k ~ 0.4); every field campaign says
            // the eye sees NOTHING at depression 17-18 even though SQMs
            // and cameras measure exactly such an excess there (the
            // documented ~2.5 deg instrument-vs-eye gap). The dawn at
            // that depth is a degrees-scale gradient with no border; the
            // ayah's tabayyun happens when the arch develops a
            // discernible edge - at an excess ~11x the night reference.
            // APPEARANCE (Fajr: noticing a new glow) is calibrated so
            // clear-sky Mecca lands at the desert-campaign cluster
            // (KACST 14.6+-0.3, Hail 14.0+-0.3, Aswan camera 14.9):
            // 45 x 0.4 ~ excess/L_ref ~ 10.8 at threshold.
            // DISAPPEARANCE (Isha: tracking a known fading band - an
            // easier task; classical muwaqqit mode for white-shafaq end
            // is 17 deg, SQM twilight end 17.99+-0.16) calibrates ~4x.
            // Out-of-sample check: Padborg/UK June should land at
            // OpenFajr's summer 12.3-12.7 deg without retuning.
            // RECALIBRATED 2026-07-06 on the final engine by the
            // constant's defining protocol: the calibration cluster
            // (Riyadh KACST, Hail, Aswan) rerun across an extended
            // factor ladder (40..80); the interior minimum of the
            // weighted cluster residual selects 70 (RMS 0.252 deg;
            // the curve rises on both sides: 0.264 at 65, 0.319 at
            // 75). Full record: validation/criterion_runs/
            // edge_factor_v2/RECAL_SUMMARY.json; the historical 45
            // belongs to the pre-hyperaccuracy transport frame.
            // TWILIGHT_KHAYT_EDGE_APPEARANCE overrides the default
            // for calibration analyses; production runs leave the
            // variable unset.
            edge_factor_appearance: edge_appearance_override(
                std::env::var("TWILIGHT_KHAYT_EDGE_APPEARANCE")
                    .ok()
                    .as_deref(),
                70.0,
            ),
            edge_factor_disappearance: 4.0,
            spread_required: 5,
            celestial_red_fraction: 0.25,
        }
    }
}

impl KhaytParams {
    /// Parameters adjusted for the side of the night: appearance
    /// (morning) vs disappearance (evening) edge factors.
    pub fn for_side(&self, morning: bool) -> KhaytParams {
        let f = if morning {
            self.edge_factor_appearance
        } else {
            self.edge_factor_disappearance
        };
        KhaytParams {
            k_contrast: self.k_contrast * f,
            k_contrast_red: self.k_contrast_red * f,
            ..self.clone()
        }
    }
}

/// Photometric state of one sky patch at one scan step.
#[derive(Debug, Clone, Copy, Default)]
pub struct PatchLum {
    /// Mesopic luminance [cd/m^2]: MCRT + celestial + skyglow.
    pub mesopic: f64,
    /// Red-band luminance [cd/m^2]: MCRT red + assigned celestial red.
    pub red: f64,
    /// 1-sigma RELATIVE standard error of `mesopic` from the MC seed
    /// spread (0.0 for deterministic runs). Deterministic addends
    /// (celestial background, skyglow) dilute it; the pipeline scales
    /// accordingly when composing totals.
    pub rel_se_mes: f64,
    /// Same for `red`.
    pub rel_se_red: f64,
}

/// The full fan at every scanned SZA, one side (morning or evening) at a
/// time - the celestial background differs between sides even though the
/// MCRT radiance is shared.
#[derive(Debug, Clone)]
pub struct KhaytScan {
    /// Scanned solar zenith angles [deg], ascending.
    pub szas: Vec<f64>,
    /// `band[i][j]` = band patch `j` (matching `band_offsets_deg`) at
    /// `szas[i]`.
    pub band: Vec<Vec<PatchLum>>,
    /// `refs[i][j]` = reference patch totals.
    pub refs: Vec<Vec<PatchLum>>,
}

/// Per-event solution: the SZA where the criterion is met, with the
/// honest resolution accounting.
#[derive(Debug, Clone, Copy)]
pub struct KhaytCrossing {
    pub sza_deg: f64,
    /// d(ln margin)/dSZA near the crossing (for uncertainty
    /// propagation).
    pub slope: f64,
    /// Residual crossing-bracket half-width [deg] when the crossing is
    /// cliff-shaped and only BRACKETED (0.0 when a local log-linear fit
    /// interpolated it; the fit error is then second-order in the
    /// bracket width and covered by `sigma_deg`'s MC term).
    pub bracket_half_deg: f64,
    /// Total 1-sigma uncertainty [deg]: RSS of the MC-noise term
    /// (rel SE / |slope|) and the bracket-quantization term
    /// (half-width / sqrt(3), uniform distribution), plus a first-order
    /// Jensen bias bound (rel SE^2 / (2 |slope|), the log-of-mean
    /// offset; its sign partially cancels between the band and
    /// reference logs, so it is folded into the width, not the value).
    pub sigma_deg: f64,
}

/// Outcome of the khayt analysis for one side of the night.
#[derive(Debug, Clone, Default)]
pub struct KhaytSolution {
    /// True dawn / white-thread distinctness with lateral spread.
    pub sadiq: Option<KhaytCrossing>,
    /// False dawn: central patches distinct while the spread test fails
    /// (only reported when it precedes sadiq by a meaningful margin).
    pub kadhib: Option<KhaytCrossing>,
    /// Red-band (shafaq al-ahmar) distinctness crossing.
    pub ahmar: Option<KhaytCrossing>,
    /// Contrast margin per band patch at the sadiq crossing (diagnostic).
    pub margins_at_sadiq: Vec<f64>,
}

/// Which events the adaptive refinement should spend fan evaluations
/// on. The pipeline refines only the events it reports per side
/// (morning: sadiq + kadhib; evening: abyad + ahmar).
#[derive(Debug, Clone, Copy)]
pub struct RefineEvents {
    pub spread: bool,
    pub central: bool,
    pub ahmar: bool,
}

impl RefineEvents {
    pub fn none() -> Self {
        RefineEvents {
            spread: false,
            central: false,
            ahmar: false,
        }
    }
}

/// Fan evaluator: full band + reference patch rows at one SZA, same
/// composition (MCRT + celestial + skyglow) and same K-seed protocol as
/// the coarse scan rows. `None` aborts refinement gracefully (the
/// solver then reports the residual bracket honestly).
pub type FanEval<'a> = dyn FnMut(f64) -> Option<(Vec<PatchLum>, Vec<PatchLum>)> + 'a;

/// One fan point (a scan row), coarse or refined.
#[derive(Debug, Clone)]
struct FanPoint {
    sza: f64,
    band: Vec<PatchLum>,
    refs: Vec<PatchLum>,
}

/// Margin state of the SELECTED patch for one event at one fan point.
/// Carries the smooth underlying quantities the guided refinement
/// needs when the gated margin itself has collapsed to zero.
#[derive(Debug, Clone, Copy, Default)]
struct MarginSample {
    sza: f64,
    /// The gated criterion margin (crosses 1.0 at the event).
    margin: f64,
    /// Signed excess of the selected patch above its night baseline
    /// [cd/m^2] (NOT clipped; smooth through zero).
    excess: f64,
    /// Excess required for margin = 1 [cd/m^2] (ref x k x C_thr).
    target: f64,
    /// Gating luminance for the red-cone gate (the selected patch's
    /// red); +inf for mesopic events (no gate).
    gate: f64,
    /// First-order relative SE of `margin` from the MC seed spread.
    rel_se: f64,
}

/// Per-patch margin decomposition at one step.
#[derive(Debug, Clone, Copy, Default)]
struct PatchMargin {
    margin: f64,
    excess: f64,
    target: f64,
    gate: f64,
    rel_se: f64,
}

/// The three "margin crosses 1.0" events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EventKind {
    /// k-th largest band margin (lateral spread; sadiq / abyad).
    Spread,
    /// Central patch margin (the zodiacal wedge; kadhib).
    Central,
    /// Red-channel spread (3-of-5 majority; shafaq al-ahmar).
    Ahmar,
}

/// Mean reference luminance at one step, with the propagated relative
/// SE of the mean (reference patches are independent MC estimates).
fn ref_mean(refs: &[PatchLum]) -> PatchLum {
    let n = refs.len().max(1) as f64;
    let mes_sum: f64 = refs.iter().map(|p| p.mesopic).sum();
    let red_sum: f64 = refs.iter().map(|p| p.red).sum();
    let mes_var: f64 = refs
        .iter()
        .map(|p| (p.mesopic * p.rel_se_mes) * (p.mesopic * p.rel_se_mes))
        .sum();
    let red_var: f64 = refs
        .iter()
        .map(|p| (p.red * p.rel_se_red) * (p.red * p.rel_se_red))
        .sum();
    PatchLum {
        mesopic: mes_sum / n,
        red: red_sum / n,
        rel_se_mes: if mes_sum > 0.0 {
            mes_var.sqrt() / mes_sum
        } else {
            0.0
        },
        rel_se_red: if red_sum > 0.0 {
            red_var.sqrt() / red_sum
        } else {
            0.0
        },
    }
}

/// Contrast margins for one step, decomposed per patch.
///
/// The "black thread" is BOTH spatial and temporal: the adjacent dark
/// sky (reference patches) sets the eye's adaptation, while the standing
/// night structure of the dawn direction itself - zodiacal cone,
/// Milky Way, skyglow dome - is part of the night the dawn must become
/// distinct FROM. At a desert site the zodiacal base keeps the dawn
/// azimuth photometrically brighter than the ±100° sky ALL night
/// (verified at Mecca: spatial-only margins never drop below ~4), and
/// field campaigns resolve exactly this by judging the NEW horizontal
/// light that supersedes the standing column (OpenFajr's criterion).
///
/// So: `margin[j] = (L_j - L_j^night) / L_ref / (k * C_thr(L_ref))` -
/// the GROWTH of patch j above its own deep-night baseline, judged
/// against the adaptation set by the reference sky. Margin > 1 means
/// the new light of dawn in patch `j` is distinct.
///
/// The relative SE of each margin is propagated to first order: excess
/// noise (band + baseline terms, amplified by L/excess when the excess
/// is a small difference) RSS'd with the reference-mean noise; the
/// TVI-slope sensitivity of C_thr to the reference level is below 1 in
/// magnitude and is absorbed into the reference term (a deliberate,
/// slightly conservative bound).
fn patch_margins(
    band: &[PatchLum],
    night_base: &[PatchLum],
    refs: &[PatchLum],
    params: &KhaytParams,
    red: bool,
) -> Vec<PatchMargin> {
    let r = ref_mean(refs);
    // The eye adapts to the prevailing (broadband) sky, not to one channel.
    let l_adapt = r.mesopic.max(1e-6);
    let c_thr = contrast_threshold_weber(l_adapt);
    let k = if red {
        params.k_contrast_red
    } else {
        params.k_contrast
    };
    band.iter()
        .zip(night_base.iter())
        .map(|(p, b)| {
            let (lp, lb, lr, relse_p, relse_b, relse_r) = if red {
                (
                    p.red,
                    b.red,
                    r.red.max(1e-9),
                    p.rel_se_red,
                    b.rel_se_red,
                    r.rel_se_red,
                )
            } else {
                (
                    p.mesopic,
                    b.mesopic,
                    r.mesopic.max(1e-9),
                    p.rel_se_mes,
                    b.rel_se_mes,
                    r.rel_se_mes,
                )
            };
            let excess = lp - lb;
            let target = lr * k * c_thr;
            let gate = if red { p.red } else { f64::INFINITY };
            let gated = red && p.red < RED_CONE_GATE;
            let margin = if gated || excess <= 0.0 {
                0.0
            } else {
                excess / target
            };
            let rel_se = if margin > 0.0 {
                let var_excess =
                    (lp * relse_p) * (lp * relse_p) + (lb * relse_b) * (lb * relse_b);
                let relse_excess = var_excess.sqrt() / excess;
                (relse_excess * relse_excess + relse_r * relse_r).sqrt()
            } else {
                0.0
            };
            PatchMargin {
                margin,
                excess,
                target,
                gate,
                rel_se,
            }
        })
        .collect()
}

/// The k-th largest patch margin (1-based), tie-broken by excess so the
/// guided quantities always come from a well-defined patch.
fn kth_by_margin(mut pm: Vec<PatchMargin>, k: usize) -> PatchMargin {
    pm.sort_by(|a, b| {
        b.margin
            .total_cmp(&a.margin)
            .then(b.excess.total_cmp(&a.excess))
    });
    pm[(k - 1).min(pm.len() - 1)]
}

/// Event margin sample at one fan point.
fn event_sample(
    kind: EventKind,
    point: &FanPoint,
    night_base: &[PatchLum],
    params: &KhaytParams,
    center: usize,
) -> MarginSample {
    let pm = match kind {
        EventKind::Ahmar => patch_margins(&point.band, night_base, &point.refs, params, true),
        _ => patch_margins(&point.band, night_base, &point.refs, params, false),
    };
    let sel = match kind {
        EventKind::Spread => kth_by_margin(pm, params.spread_required),
        EventKind::Central => pm[center.min(pm.len() - 1)],
        EventKind::Ahmar => kth_by_margin(pm, 3.min(params.band_offsets_deg.len())),
    };
    MarginSample {
        sza: point.sza,
        margin: sel.margin,
        excess: sel.excess,
        target: sel.target,
        gate: sel.gate,
        rel_se: sel.rel_se,
    }
}

/// Deep-night baseline per band patch: the median over the deepest
/// three COARSE-scanned SZAs. Computed once from the coarse scan so
/// refinement points near a crossing can never contaminate it. At high
/// latitudes where the night never deepens, this floor still carries
/// residual twilight - the criterion then measures brightening above
/// tonight's actual minimum, which is exactly the right high-latitude
/// semantics.
fn night_baseline(scan: &KhaytScan) -> Vec<PatchLum> {
    let n = scan.szas.len();
    let n_band = scan.band[0].len();
    let deep = n.saturating_sub(3);
    (0..n_band)
        .map(|j| {
            let median_of = |vals: Vec<(f64, f64)>| -> (f64, f64) {
                let mut v = vals;
                v.sort_by(|a, b| a.0.total_cmp(&b.0));
                v[v.len() / 2]
            };
            let (mes, relse_mes) = median_of(
                (deep..n)
                    .map(|i| (scan.band[i][j].mesopic, scan.band[i][j].rel_se_mes))
                    .collect(),
            );
            let (red, relse_red) = median_of(
                (deep..n)
                    .map(|i| (scan.band[i][j].red, scan.band[i][j].rel_se_red))
                    .collect(),
            );
            PatchLum {
                mesopic: mes,
                red,
                rel_se_mes: relse_mes,
                rel_se_red: relse_red,
            }
        })
        .collect()
}

/// Bracket of the LAST downward crossing of margin = 1.0 (curve
/// decreasing with SZA; ascending scan order): index i with
/// margin[i] >= 1 > margin[i+1].
fn bracket_index(curve: &[MarginSample]) -> Option<usize> {
    let mut bi = None;
    for i in 0..curve.len().saturating_sub(1) {
        if curve[i].margin >= 1.0 && curve[i + 1].margin < 1.0 {
            bi = Some(i);
        }
    }
    bi
}

/// Both bracket endpoints statistically indistinguishable from the
/// criterion: |ln margin| within 1 sigma of its MC noise. Further fan
/// points cannot resolve the crossing below the noise; stop and let the
/// reported sigma carry it. Never triggers on deterministic runs
/// (rel_se = 0) or hard gates (margin = 0).
fn noise_stop(a: &MarginSample, b: &MarginSample) -> bool {
    a.margin > 0.0
        && b.margin > 0.0
        && a.rel_se > 0.0
        && b.rel_se > 0.0
        && libm::log(a.margin).abs() <= a.rel_se
        && libm::log(b.margin).abs() <= b.rel_se
}

/// Next refinement SZA inside the bracket [a, b].
///
/// Guided regula-falsi: on smooth positive margins, log-secant toward
/// margin = 1 (near-exact on log-exponential curves). When the margin
/// has collapsed to zero at `b`, guide on the smooth underlying
/// quantities instead: the excess curve toward the required target, and
/// (red events) the gate luminance toward the cone threshold - the
/// event sits at whichever constraint fails first (min SZA). Bisection
/// fallback; `force_mid` breaks regula-falsi stagnation (Illinois
/// style) when the previous step failed to shrink the bracket by 30%.
fn propose_next(a: &MarginSample, b: &MarginSample, force_mid: bool) -> f64 {
    let w = b.sza - a.sza;
    let mid = a.sza + 0.5 * w;
    if force_mid {
        return mid;
    }
    let mut cands: Vec<f64> = Vec::new();
    if a.margin > 0.0 && b.margin > 0.0 {
        let la = libm::log(a.margin);
        let lb = libm::log(b.margin);
        if la - lb > 1e-12 {
            cands.push(a.sza + w * la / (la - lb));
        }
    } else {
        // Excess guide toward the target excess (margin = 1).
        if a.excess > a.target && a.target > 0.0 {
            let s = if b.excess > 0.0 && a.excess > b.excess {
                a.sza + w * libm::log(a.excess / a.target) / libm::log(a.excess / b.excess)
            } else if a.excess > b.excess {
                a.sza + w * (a.excess - a.target) / (a.excess - b.excess)
            } else {
                mid
            };
            cands.push(s);
        }
        // Red-cone gate guide toward the cone threshold.
        if a.gate.is_finite() && a.gate >= RED_CONE_GATE && b.gate > 0.0 && b.gate < RED_CONE_GATE
        {
            cands.push(a.sza + w * libm::log(a.gate / RED_CONE_GATE) / libm::log(a.gate / b.gate));
        }
    }
    cands.retain(|s| s.is_finite() && *s > a.sza && *s < b.sza);
    if cands.is_empty() {
        return mid;
    }
    // The event is where the FIRST constraint fails: the shallowest
    // candidate binds.
    cands.into_iter().fold(f64::INFINITY, f64::min)
}

/// Final crossing estimate from a (refined) margin curve with the
/// resolution accounting.
///
/// Smooth path: local log-linear fit over the points around the
/// bracket; accepted only when its crossing is consistent with the
/// bracket (within 25% of the bracket width, min 0.02 deg). The fit
/// interpolates continuously, so no quantization term remains
/// (interpolation model error is second-order in the local spacing:
/// <= |d2 ln m / ds2| w^2 / 8 / |slope|, ~4e-4 deg at w = 0.05).
///
/// Cliff path (fit rejected or margin hard-zero): the truth is only
/// BRACKETED. Log-interpolate inside the pair when both margins are
/// positive, else take the midpoint, and fold the residual half-width
/// into sigma as a uniform-distribution term (half / sqrt(3)).
///
/// Both paths carry an MC-noise sigma and a first-order Jensen bias
/// bound (rel SE^2 / (2 |slope|)): the fit takes logs of noisy MC
/// means, whose expectation sits below the log of the truth; the band
/// and reference contributions enter with opposite signs, so the
/// bound widens the interval instead of shifting the value.
///
/// The MC-noise sigma is the LARGER of two propagation routes, never
/// below the [`LN_CONSISTENCY_FLOOR`] (so a crossing can never claim
/// zero uncertainty, e.g. on the GPU path whose SZA-derived RNG makes
/// the seed-spread SE read 0):
/// - rel SE / |margin slope| (the smooth-curve route);
/// - the EVENT-LOCATION jitter of a collapse/gate crossing: near a
///   baseline collapse the margin slope is enormous, but the crossing
///   POSITION still shifts with the noise of the smooth underlying
///   quantity (excess toward zero, or red gate toward the cone
///   threshold): rel SE / |d ln(underlying)/dSZA|. The June-solstice
///   fajr cliff is the motivating case: its position noise is
///   baseline-dominated and invisible to the margin-slope route.
fn finalize_crossing(curve: &[MarginSample], i: usize) -> KhaytCrossing {
    let a = curve[i];
    let b = curve[i + 1];
    let w = b.sza - a.sza;
    let half = 0.5 * w;
    let lo = i.saturating_sub(2);
    let hi = (i + 3).min(curve.len());
    let fit_pts: Vec<(f64, f64)> = curve[lo..hi]
        .iter()
        .filter(|s| s.margin > 0.0)
        .map(|s| (s.sza, s.margin))
        .collect();
    let relse_loc = curve[lo..hi]
        .iter()
        .map(|s| s.rel_se)
        .fold(LN_CONSISTENCY_FLOOR, f64::max);
    // Event-location jitter through the smooth underlying quantity.
    // Binding constraint: the red-cone gate when it straddles the
    // threshold across the bracket, else the excess collapse.
    let rel_a = a.rel_se.max(LN_CONSISTENCY_FLOOR);
    let gate_binds = a.gate.is_finite()
        && a.gate >= RED_CONE_GATE
        && b.gate > 0.0
        && b.gate < RED_CONE_GATE;
    let sigma_loc = if gate_binds {
        let dln = libm::log(a.gate / b.gate).abs().max(1e-9);
        rel_a * w / dln
    } else if a.excess > 0.0 && a.excess > b.excess {
        // |d ln excess/ds| at the bracket, from the SIGNED excess
        // secant (smooth through zero, unlike the gated margin).
        rel_a * w * a.excess / (a.excess - b.excess)
    } else {
        0.0
    };
    let pad = (0.25 * w).max(0.02);
    if let Some((s_fit, slope)) = fit_crossing_loglinear(&fit_pts, 1.0) {
        if s_fit >= a.sza - pad && s_fit <= b.sza + pad {
            let sa = slope.abs().max(1e-6);
            let sigma_stat = (relse_loc / sa).max(sigma_loc);
            let jensen = relse_loc * relse_loc / (2.0 * sa);
            return KhaytCrossing {
                sza_deg: s_fit,
                slope,
                bracket_half_deg: 0.0,
                sigma_deg: sigma_stat + jensen,
            };
        }
    }
    // Cliff / bracket-only path.
    let (sza, slope) = if b.margin > 0.0 {
        let la = libm::log(a.margin);
        let lb = libm::log(b.margin);
        let s = if la - lb > 1e-12 {
            a.sza + w * la / (la - lb)
        } else {
            a.sza + half
        };
        (s, (lb - la) / w.max(1e-9))
    } else {
        (a.sza + half, -2.0 / w.max(1e-6))
    };
    let sa = slope.abs().max(1e-6);
    let sigma_quant = half / 3f64.sqrt();
    let sigma_stat = (relse_loc / sa).max(sigma_loc);
    let jensen = relse_loc * relse_loc / (2.0 * sa);
    KhaytCrossing {
        sza_deg: sza,
        slope,
        bracket_half_deg: half,
        sigma_deg: (sigma_stat * sigma_stat + sigma_quant * sigma_quant).sqrt() + jensen,
    }
}

/// Finalize a crossing whose bracket has been VALIDATED log-linear at
/// the `ln_floor` tolerance: fit ONLY the points inside the validated
/// span (secant-consistent by construction, so the fit's model error
/// on the crossing is bounded by `ln_floor` / |slope| and is folded
/// into sigma through the floor), leaving no quantization residual.
/// Returns None when the restricted fit cannot place a crossing inside
/// the span (the caller then keeps refining / reports the bracket).
fn finalize_validated(
    curve: &[MarginSample],
    span_lo: f64,
    span_hi: f64,
    ln_floor: f64,
) -> Option<KhaytCrossing> {
    let inside: Vec<&MarginSample> = curve
        .iter()
        .filter(|s| s.sza >= span_lo - 1e-9 && s.sza <= span_hi + 1e-9)
        .collect();
    let fit_pts: Vec<(f64, f64)> = inside
        .iter()
        .filter(|s| s.margin > 0.0)
        .map(|s| (s.sza, s.margin))
        .collect();
    let relse_loc = inside.iter().map(|s| s.rel_se).fold(ln_floor, f64::max);
    let (s_fit, slope) = fit_crossing_loglinear(&fit_pts, 1.0)?;
    if s_fit < span_lo || s_fit > span_hi {
        return None;
    }
    let sa = slope.abs().max(1e-6);
    Some(KhaytCrossing {
        sza_deg: s_fit,
        slope,
        bracket_half_deg: 0.0,
        sigma_deg: relse_loc / sa + relse_loc * relse_loc / (2.0 * sa),
    })
}

/// Solve one event on the shared fan point set, adaptively refining the
/// crossing bracket with up to `budget` extra fan evaluations.
///
/// Stop rule (first hit wins; see the module docs for the full
/// contract):
/// 1. bracket half-width <= [`REFINE_TARGET_BRACKET_DEG`] (the
///    midpoint recovery error bound);
/// 2. log-linearity validated inside the bracket at the MC noise /
///    [`LN_CONSISTENCY_FLOOR`] tolerance AND that tolerance resolves
///    the crossing below the target (bracket-restricted fit);
/// 3. both bracket endpoints within 1 sigma (MC) of margin = 1
///    (noise floor: extra points cannot resolve further);
/// 4. `budget` evaluations spent, the evaluator declined, or the
///    proposal collides with an existing point (< 0.004 deg).
///
/// The residual bracket is then reported, never absorbed.
fn solve_event(
    points: &mut Vec<FanPoint>,
    night_base: &[PatchLum],
    params: &KhaytParams,
    kind: EventKind,
    center: usize,
    budget: usize,
    eval: &mut Option<&mut FanEval<'_>>,
) -> Option<KhaytCrossing> {
    let mut evals_used = 0usize;
    let mut prev_w = f64::INFINITY;
    loop {
        let curve: Vec<MarginSample> = points
            .iter()
            .map(|p| event_sample(kind, p, night_base, params, center))
            .collect();
        let i = bracket_index(&curve)?;
        let a = curve[i];
        let b = curve[i + 1];
        let w = b.sza - a.sza;
        let refinable = evals_used < budget && eval.is_some();
        if 0.5 * w <= REFINE_TARGET_BRACKET_DEG || !refinable || noise_stop(&a, &b) {
            return Some(finalize_crossing(&curve, i));
        }
        // Illinois-style stagnation break: force bisection when the
        // last step failed to shrink the bracket by 30%.
        let force_mid = prev_w.is_finite() && w > 0.7 * prev_w;
        prev_w = w;
        let pad = (0.1 * w).max(0.005);
        let mid = a.sza + 0.5 * w;
        // Snap proposals to the 1/64-deg dyadic grid: bisection
        // midpoints of coarse-grid brackets already lie on it, and
        // snapped guided proposals from neighboring events and from
        // BOTH sides of the night coincide exactly, turning repeat
        // fan evaluations into MCRT cache hits in the pipeline (the
        // June-solstice case puts three cliffs in one scan cell).
        // Snap error (<= 1/128 deg) is far below the target.
        let raw = propose_next(&a, &b, force_mid).clamp(a.sza + pad, b.sza - pad);
        let snapped = (raw * 64.0).round() / 64.0;
        let mut s_next = if snapped > a.sza + 1e-9 && snapped < b.sza - 1e-9 {
            snapped
        } else {
            raw
        };
        if points.iter().any(|p| (p.sza - s_next).abs() < 0.004) {
            // The informative snapped point already exists; take the
            // exact midpoint instead (fresh for any dyadic bracket).
            s_next = mid;
            if points.iter().any(|p| (p.sza - s_next).abs() < 0.004) {
                return Some(finalize_crossing(&curve, i));
            }
        }
        let Some(row) = eval.as_deref_mut().and_then(|f| f(s_next)) else {
            return Some(finalize_crossing(&curve, i));
        };
        let pos = points.partition_point(|p| p.sza < s_next);
        points.insert(
            pos,
            FanPoint {
                sza: s_next,
                band: row.0,
                refs: row.1,
            },
        );
        evals_used += 1;
        // Deterministic log-linearity stop: if the fresh point agrees
        // with the log-secant across the pre-insert bracket to within
        // its own MC noise (floored at LN_CONSISTENCY_FLOOR) and that
        // tolerance resolves the crossing below the target, the
        // bracket-restricted fit already interpolates below the
        // bracket scale - further MCRT points cannot improve it.
        // Never fires across a hard gate (needs positive margins).
        if a.margin > 0.0 && b.margin > 0.0 {
            let fresh = event_sample(kind, &points[pos], night_base, params, center);
            if fresh.margin > 0.0 {
                let la = libm::log(a.margin);
                let lb = libm::log(b.margin);
                let f = (s_next - a.sza) / w;
                let pred = (1.0 - f) * la + f * lb;
                let tol = fresh.rel_se.max(LN_CONSISTENCY_FLOOR);
                let b_secant = ((lb - la) / w).abs().max(1e-9);
                let resolvable = tol / b_secant <= REFINE_TARGET_BRACKET_DEG;
                if resolvable && (libm::log(fresh.margin) - pred).abs() <= tol {
                    let curve2: Vec<MarginSample> = points
                        .iter()
                        .map(|p| event_sample(kind, p, night_base, params, center))
                        .collect();
                    if let Some(c) =
                        finalize_validated(&curve2, a.sza, b.sza, LN_CONSISTENCY_FLOOR)
                    {
                        return Some(c);
                    }
                }
            }
        }
    }
}

/// Detect the khayt events on one side's scan (no refinement: coarse
/// scan resolution only; kept for callers without a fan evaluator).
pub fn detect(scan: &KhaytScan, params: &KhaytParams) -> KhaytSolution {
    detect_refined(scan, params, None, RefineEvents::none())
}

/// Detect the khayt events on one side's scan, adaptively refining each
/// requested crossing with extra fan evaluations (see module docs for
/// the resolution contract).
///
/// All three events are "margin crosses 1.0" problems on curves that
/// decrease with SZA (the dawn brightening fades with depth);
/// morning/evening only differ later, at SZA -> clock-time conversion.
pub fn detect_refined(
    scan: &KhaytScan,
    params: &KhaytParams,
    mut eval: Option<&mut FanEval<'_>>,
    refine: RefineEvents,
) -> KhaytSolution {
    let n = scan.szas.len();
    if n < 3 || scan.band.len() != n || scan.refs.len() != n {
        return KhaytSolution::default();
    }
    let center = params
        .band_offsets_deg
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
        .map(|(i, _)| i)
        .unwrap_or(params.band_offsets_deg.len() / 2);

    let night_base = night_baseline(scan);

    let mut points: Vec<FanPoint> = (0..n)
        .map(|i| FanPoint {
            sza: scan.szas[i],
            band: scan.band[i].clone(),
            refs: scan.refs[i].clone(),
        })
        .collect();

    if std::env::var("TWILIGHT_KHAYT_DEBUG").is_ok() {
        for p in points.iter().step_by(2) {
            let sp = event_sample(EventKind::Spread, p, &night_base, params, center);
            let ce = event_sample(EventKind::Central, p, &night_base, params, center);
            let ah = event_sample(EventKind::Ahmar, p, &night_base, params, center);
            eprintln!(
                "khayt sza {:6.2}: spread {:9.3e} (+-{:.1e}) central {:9.3e} ahmar {:9.3e}",
                p.sza, sp.margin, sp.rel_se, ce.margin, ah.margin
            );
        }
    }

    let budget = |on: bool| if on { REFINE_MAX_EVALS } else { 0 };
    let sadiq = solve_event(
        &mut points,
        &night_base,
        params,
        EventKind::Spread,
        center,
        budget(refine.spread),
        &mut eval,
    );
    // Central (kadhib) refinement only when a false-dawn verdict is
    // actually at stake: coarse-solve first, refine only when the
    // coarse bracket could put the crossing PAST the sadiq gate
    // (c > s + 0.2), i.e. when the verdict could fire. A central
    // crossing that cannot fire even at its bracket extremes is not
    // reported, so extra MCRT there would sharpen nothing.
    let central0 = solve_event(
        &mut points,
        &night_base,
        params,
        EventKind::Central,
        center,
        0,
        &mut eval,
    );
    let central = match (central0, sadiq) {
        (Some(c), s)
            if refine.central
                && s.map(|s| {
                    c.sza_deg + c.bracket_half_deg + s.bracket_half_deg
                        >= s.sza_deg + 0.2
                })
                .unwrap_or(true) =>
        {
            solve_event(
                &mut points,
                &night_base,
                params,
                EventKind::Central,
                center,
                REFINE_MAX_EVALS,
                &mut eval,
            )
        }
        (c, _) => c,
    };
    let ahmar = solve_event(
        &mut points,
        &night_base,
        params,
        EventKind::Ahmar,
        center,
        budget(refine.ahmar),
        &mut eval,
    );

    // Kadhib: the central column becomes distinct DEEPER than (or
    // without) the spread verdict - an interval where only the narrow
    // wedge is visible. Central-distinct with NO spread at all is the
    // purest wolf's-tail case.
    let kadhib = match (central, sadiq) {
        (Some(c), Some(s)) if c.sza_deg > s.sza_deg + 0.2 => Some(c),
        (Some(c), None) => Some(c),
        _ => None,
    };

    // Diagnostics: margins at the fan point nearest the sadiq crossing.
    let margins_at_sadiq = sadiq
        .map(|s| {
            let p = points
                .iter()
                .min_by(|a, b| {
                    (a.sza - s.sza_deg)
                        .abs()
                        .total_cmp(&(b.sza - s.sza_deg).abs())
                })
                .expect("points non-empty");
            patch_margins(&p.band, &night_base, &p.refs, params, false)
                .iter()
                .map(|m| m.margin)
                .collect()
        })
        .unwrap_or_default();

    KhaytSolution {
        sadiq,
        kadhib,
        ahmar,
        margins_at_sadiq,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(scan_szas: &[f64], band: f64, refs: f64) -> KhaytScan {
        KhaytScan {
            szas: scan_szas.to_vec(),
            band: scan_szas
                .iter()
                .map(|_| {
                    (0..5)
                        .map(|_| PatchLum {
                            mesopic: band,
                            red: band * 0.3,
                            ..Default::default()
                        })
                        .collect()
                })
                .collect(),
            refs: scan_szas
                .iter()
                .map(|_| {
                    (0..2)
                        .map(|_| PatchLum {
                            mesopic: refs,
                            red: refs * 0.3,
                            ..Default::default()
                        })
                        .collect()
                })
                .collect(),
        }
    }

    /// Dawn-like synthetic: band brightens exponentially as SZA falls,
    /// reference stays at the night floor.
    fn dawnlike(spread: bool) -> KhaytScan {
        let szas: Vec<f64> = (0..30).map(|i| 96.0 + 0.5 * i as f64).collect();
        let floor = 2.2e-4;
        let band: Vec<Vec<PatchLum>> = szas
            .iter()
            .map(|&sza| {
                (0..5)
                    .map(|j| {
                        // central excess decays with offset unless spreading
                        let lateral = if spread {
                            1.0
                        } else {
                            // narrow wedge: only the center glows
                            if j == 2 {
                                1.0
                            } else {
                                0.02
                            }
                        };
                        // Amplitude chosen so the red channel (0.3x)
                        // clears the 1e-3 cd/m^2 cone gate near onset.
                        let glow = 1.2e-2 * libm::exp(-(sza - 96.0) / 1.2) * lateral;
                        PatchLum {
                            mesopic: floor + glow,
                            red: 0.3 * (floor + glow),
                            ..Default::default()
                        }
                    })
                    .collect()
            })
            .collect();
        let refs: Vec<Vec<PatchLum>> = szas
            .iter()
            .map(|_| {
                (0..2)
                    .map(|_| PatchLum {
                        mesopic: floor,
                        red: 0.3 * floor,
                        ..Default::default()
                    })
                    .collect()
            })
            .collect();
        KhaytScan { szas, band, refs }
    }

    #[test]
    fn edge_appearance_env_override_is_picked_up() {
        // Parse contract (pure, no process-global state).
        assert_eq!(edge_appearance_override(Some("18.0"), 45.0), 18.0);
        assert_eq!(edge_appearance_override(Some(" 60 "), 45.0), 60.0);
        assert_eq!(edge_appearance_override(Some("junk"), 45.0), 45.0);
        assert_eq!(edge_appearance_override(Some("-3"), 45.0), 45.0);
        assert_eq!(edge_appearance_override(Some("inf"), 45.0), 45.0);
        assert_eq!(edge_appearance_override(None, 45.0), 45.0);
        // End-to-end: the env var reaches KhaytParams::default().
        // Set/restore window kept minimal (env is process-global; the
        // sibling tests construct defaults from parallel threads, and
        // their assertions hold at any factor in [18, 45]).
        std::env::set_var("TWILIGHT_KHAYT_EDGE_APPEARANCE", "18.0");
        let picked = KhaytParams::default().edge_factor_appearance;
        std::env::remove_var("TWILIGHT_KHAYT_EDGE_APPEARANCE");
        assert_eq!(picked, 18.0);
        assert_eq!(KhaytParams::default().edge_factor_appearance, 70.0);
    }

    #[test]
    fn featureless_sky_yields_nothing() {
        let szas: Vec<f64> = (0..20).map(|i| 96.0 + 0.5 * i as f64).collect();
        let s = detect(&flat(&szas, 2.2e-4, 2.2e-4), &KhaytParams::default());
        assert!(s.sadiq.is_none() && s.kadhib.is_none() && s.ahmar.is_none());
    }

    #[test]
    fn spreading_dawn_is_sadiq_no_kadhib() {
        let s = detect(&dawnlike(true), &KhaytParams::default());
        let sadiq = s.sadiq.expect("spreading dawn must be detected");
        // With these synthetic numbers the crossing must be interior.
        assert!(
            (96.5..110.0).contains(&sadiq.sza_deg),
            "sadiq at {}",
            sadiq.sza_deg
        );
        // A uniformly spreading dawn has central and spread crossing
        // together: no false-dawn interval.
        assert!(s.kadhib.is_none(), "{:?}", s.kadhib);
        assert!(s.ahmar.is_some(), "red band rides the same glow");
    }

    #[test]
    fn narrow_wedge_is_kadhib_not_sadiq() {
        // Center-only glow (the wolf's tail): central contrast crosses,
        // spread does not -> kadhib with no (or much later) sadiq.
        let s = detect(&dawnlike(false), &KhaytParams::default());
        match (s.kadhib, s.sadiq) {
            (Some(k), Some(sq)) => assert!(
                k.sza_deg > sq.sza_deg + 1.0,
                "wedge must be seen well before any spread: {k:?} vs {sq:?}"
            ),
            (Some(_), None) => {} // ideal: wedge visible, never spreads
            other => panic!("expected kadhib: {other:?}"),
        }
    }

    #[test]
    fn brighter_adaptation_demands_more_contrast() {
        // Same 20% band excess: detectable at dark adaptation (Weber
        // ~0.17 at 1e-3) but NOT at bright adaptation (Weber ~0.7 floor
        // ... inverse: at very dark floors the threshold contrast is
        // HIGHER (rod noise), so the excess fails there and passes at
        // mid levels. Verify the adaptation dependence acts at all.
        let szas: Vec<f64> = (0..10).map(|i| 96.0 + 0.5 * i as f64).collect();
        let dark = flat(&szas, 1.2e-4 * 1.2, 1.2e-4);
        let mid = flat(&szas, 1e-2 * 1.2, 1e-2);
        let p = KhaytParams::default();
        // Zero night baseline isolates the adaptation dependence.
        let zero = vec![PatchLum::default(); 5];
        let m_dark = patch_margins(&dark.band[0], &zero, &dark.refs[0], &p, false);
        let m_mid = patch_margins(&mid.band[0], &zero, &mid.refs[0], &p, false);
        assert!(
            m_mid[2].margin > m_dark[2].margin,
            "20% excess is easier at mesopic than at the rod floor: {:?} vs {:?}",
            m_mid[2],
            m_dark[2]
        );
    }

    // ── Synthetic-crossing gates: the solver must recover a planted
    // crossing to <= REFINE_TARGET_BRACKET_DEG on both smooth and
    // cliff-shaped margin curves (analytic luminance, no MC). ──

    const FLOOR: f64 = 2.2e-4;

    /// Analytic fan rows at any SZA for a given band-glow law.
    fn analytic_rows(
        sza: f64,
        glow: &dyn Fn(f64) -> f64,
        red_frac: f64,
    ) -> (Vec<PatchLum>, Vec<PatchLum>) {
        let g = glow(sza);
        let band = (0..5)
            .map(|_| PatchLum {
                mesopic: FLOOR + g,
                red: red_frac * (FLOOR + g),
                ..Default::default()
            })
            .collect();
        let refs = (0..2)
            .map(|_| PatchLum {
                mesopic: FLOOR,
                red: red_frac * FLOOR,
                ..Default::default()
            })
            .collect();
        (band, refs)
    }

    fn analytic_scan(glow: &dyn Fn(f64) -> f64, red_frac: f64) -> KhaytScan {
        let szas: Vec<f64> = (0..16).map(|i| 96.0 + i as f64).collect();
        let rows: Vec<(Vec<PatchLum>, Vec<PatchLum>)> = szas
            .iter()
            .map(|&s| analytic_rows(s, glow, red_frac))
            .collect();
        KhaytScan {
            szas,
            band: rows.iter().map(|r| r.0.clone()).collect(),
            refs: rows.iter().map(|r| r.1.clone()).collect(),
        }
    }

    /// Ground truth: bisect the ANALYTIC margin curve (same night-base
    /// protocol as detect: median of the 3 deepest coarse rows) to
    /// ~1e-16 deg.
    fn true_crossing(
        scan: &KhaytScan,
        params: &KhaytParams,
        glow: &dyn Fn(f64) -> f64,
        red_frac: f64,
        red: bool,
        mut lo: f64,
        mut hi: f64,
    ) -> f64 {
        let night = night_baseline(scan);
        let margin_at = |s: f64| -> f64 {
            let (band, refs) = analytic_rows(s, glow, red_frac);
            let pm = patch_margins(&band, &night, &refs, params, red);
            if red {
                kth_by_margin(pm, 3).margin
            } else {
                kth_by_margin(pm, params.spread_required).margin
            }
        };
        assert!(margin_at(lo) >= 1.0 && margin_at(hi) < 1.0, "test bracket");
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if margin_at(mid) >= 1.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    /// Central (kadhib) refinement gate: production morning runs use
    /// central: true with a refine-only-when-the-verdict-is-at-stake
    /// condition; no other synthetic gate exercised that branch (review
    /// round 2). Plants a central-only glow whose crossing sits ~0.5 deg
    /// PAST the sadiq crossing and asserts (a) the kadhib fires, (b) the
    /// refined kadhib recovers the analytic central-crossing truth to
    /// REFINE_TARGET_BRACKET_DEG. A second scan with the central
    /// crossing only ~0.05 deg past sadiq asserts the verdict gate does
    /// NOT report a kadhib (the interval cannot fire), pinning the
    /// conditional arithmetic against sign/operand regressions.
    #[test]
    fn refined_solver_recovers_central_kadhib_crossing() {
        let params = KhaytParams::default().for_side(true);
        // Base glow shared by ALL band patches (log-curved so the
        // refinement does real work), plus a central-only brightness
        // factor (1 + amp). Through the local ln-margin slope
        // (~0.9/deg near the crossing) the central crossing lands
        // ln(1 + amp) / 0.9 deg deeper than sadiq: amp picks the
        // planted kadhib separation directly.
        let mk_rows = move |sza: f64, amp: f64| -> (Vec<PatchLum>, Vec<PatchLum>) {
            let base = 2.0e-2 * libm::exp(-((sza - 96.0) / 3.0) * ((sza - 96.0) / 3.0));
            let band = (0..5)
                .map(|i| {
                    let g = if i == 2 { base * (1.0 + amp) } else { base };
                    PatchLum {
                        mesopic: FLOOR + g,
                        red: 0.3 * (FLOOR + g),
                        ..Default::default()
                    }
                })
                .collect();
            let refs = (0..2)
                .map(|_| PatchLum {
                    mesopic: FLOOR,
                    red: 0.3 * FLOOR,
                    ..Default::default()
                })
                .collect();
            (band, refs)
        };
        let mk_scan = |amp: f64| -> KhaytScan {
            let szas: Vec<f64> = (0..16).map(|i| 96.0 + i as f64).collect();
            let rows: Vec<_> = szas.iter().map(|&s| mk_rows(s, amp)).collect();
            KhaytScan {
                szas,
                band: rows.iter().map(|r| r.0.clone()).collect(),
                refs: rows.iter().map(|r| r.1.clone()).collect(),
            }
        };

        // Case A: amp 3 plants the central crossing ~ln(4)/0.9 = 1.5
        // deg past sadiq: the kadhib must fire and refine.
        let scan = mk_scan(3.0);
        let night = night_baseline(&scan);
        // Analytic central-crossing truth via the production event
        // sample (central patch index 2 of the 5-patch fan).
        let central_margin = |s: f64| -> f64 {
            let (band, refs) = mk_rows(s, 3.0);
            let p = FanPoint {
                sza: s,
                band,
                refs,
            };
            event_sample(EventKind::Central, &p, &night, &params, 2).margin
        };
        let (mut lo, mut hi) = (96.0, 111.0);
        assert!(central_margin(lo) >= 1.0 && central_margin(hi) < 1.0, "bracket");
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if central_margin(mid) >= 1.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let truth_c = 0.5 * (lo + hi);

        let mut eval = |s: f64| Some(mk_rows(s, 3.0));
        let eval_ref: &mut FanEval<'_> = &mut eval;
        let sol = detect_refined(
            &scan,
            &params,
            Some(eval_ref),
            RefineEvents {
                spread: true,
                central: true,
                ahmar: false,
            },
        );
        let sadiq = sol.sadiq.expect("sadiq detected");
        let kadhib = sol.kadhib.expect("planted central-only wedge must fire the kadhib");
        assert!(
            kadhib.sza_deg > sadiq.sza_deg + 0.2,
            "kadhib {:.3} must sit past sadiq {:.3} + 0.2",
            kadhib.sza_deg,
            sadiq.sza_deg
        );
        assert!(
            (kadhib.sza_deg - truth_c).abs() <= REFINE_TARGET_BRACKET_DEG,
            "central: got {:.4} truth {:.4} (err {:.3}): the kadhib the user \
             sees must be refined, not coarse-grid quantized",
            kadhib.sza_deg,
            truth_c,
            (kadhib.sza_deg - truth_c).abs()
        );

        // Case B: amp 0.1 plants the central crossing ~ln(1.1)/0.9 =
        // 0.11 deg past sadiq: inside the 0.2 deg distinctness gate, so
        // no kadhib may be reported (pins the c > s + 0.2 arithmetic
        // direction).
        let scan_b = mk_scan(0.1);
        let mut eval_b = |s: f64| Some(mk_rows(s, 0.1));
        let eval_b_ref: &mut FanEval<'_> = &mut eval_b;
        let sol_b = detect_refined(
            &scan_b,
            &params,
            Some(eval_b_ref),
            RefineEvents {
                spread: true,
                central: true,
                ahmar: false,
            },
        );
        assert!(sol_b.sadiq.is_some(), "sadiq still detected in case B");
        assert!(
            sol_b.kadhib.is_none(),
            "a central crossing within 0.2 deg of sadiq must NOT report a \
             kadhib (got {:?})",
            sol_b.kadhib.map(|c| c.sza_deg)
        );
    }

    #[test]
    fn refined_solver_recovers_smooth_crossing() {
        // Log-CURVED glow (Gaussian in SZA): a straight log-linear fit
        // over the 1-deg coarse grid is biased here, so this exercises
        // the refinement, not just the fit.
        let glow = |s: f64| 2.0e-2 * libm::exp(-((s - 96.0) / 3.0) * ((s - 96.0) / 3.0));
        let params = KhaytParams::default().for_side(true);
        let scan = analytic_scan(&glow, 0.3);
        let truth = true_crossing(&scan, &params, &glow, 0.3, false, 96.0, 111.0);

        let mut n_evals = 0usize;
        let mut eval = |s: f64| {
            n_evals += 1;
            Some(analytic_rows(s, &glow, 0.3))
        };
        let eval_ref: &mut FanEval<'_> = &mut eval;
        let sol = detect_refined(
            &scan,
            &params,
            Some(eval_ref),
            RefineEvents {
                spread: true,
                central: false,
                ahmar: false,
            },
        );
        let c = sol.sadiq.expect("smooth crossing detected");
        assert!(
            (c.sza_deg - truth).abs() <= REFINE_TARGET_BRACKET_DEG,
            "smooth: got {} truth {} (err {:.3})",
            c.sza_deg,
            truth,
            (c.sza_deg - truth).abs()
        );
        assert!(
            n_evals <= 3,
            "the log-linearity consistency stop must keep smooth-curve \
             refinement cheap (deterministic floor, no MC noise): \
             {n_evals} evals"
        );
        // Sanity: the UNREFINED coarse answer is materially worse, so
        // the gate actually tests the refinement.
        let coarse = detect(&scan, &params).sadiq.expect("coarse crossing");
        assert!(
            (coarse.sza_deg - truth).abs() > (c.sza_deg - truth).abs(),
            "refinement must improve on the coarse fit: coarse err {:.4}, refined err {:.4}",
            (coarse.sza_deg - truth).abs(),
            (c.sza_deg - truth).abs()
        );
    }

    #[test]
    fn refined_solver_recovers_cliff_crossing() {
        // Hard mesopic cliff: the glow collapses to the floor at a
        // known SZA planted BETWEEN coarse grid points. The margin is a
        // step function: only bracketing can localize it.
        let s_cliff = 101.37;
        let glow = move |s: f64| if s < s_cliff { 1.5e-2 } else { 0.0 };
        let params = KhaytParams::default().for_side(true);
        let scan = analytic_scan(&glow, 0.3);

        let mut eval = |s: f64| Some(analytic_rows(s, &glow, 0.3));
        let eval_ref: &mut FanEval<'_> = &mut eval;
        let sol = detect_refined(
            &scan,
            &params,
            Some(eval_ref),
            RefineEvents {
                spread: true,
                central: false,
                ahmar: false,
            },
        );
        let c = sol.sadiq.expect("cliff crossing detected");
        assert!(
            (c.sza_deg - s_cliff).abs() <= REFINE_TARGET_BRACKET_DEG,
            "cliff: got {} truth {} (err {:.3})",
            c.sza_deg,
            s_cliff,
            (c.sza_deg - s_cliff).abs()
        );
        // The residual bracket must be REPORTED, not absorbed: the
        // half-width is nonzero and folds into sigma.
        assert!(
            c.bracket_half_deg > 0.0 && c.bracket_half_deg <= REFINE_TARGET_BRACKET_DEG,
            "bracket half {} must be honest",
            c.bracket_half_deg
        );
        assert!(
            c.sigma_deg >= c.bracket_half_deg / 3f64.sqrt() - 1e-12,
            "sigma {} must fold the bracket term {}",
            c.sigma_deg,
            c.bracket_half_deg / 3f64.sqrt()
        );
        // Without refinement the same cliff quantizes to the coarse
        // bracket midpoint, +-0.5 deg.
        let coarse = detect(&scan, &params).sadiq.expect("coarse cliff");
        assert!(
            coarse.bracket_half_deg >= 0.49,
            "coarse cliff carries the grid half-width: {:?}",
            coarse
        );
    }

    #[test]
    fn refined_solver_recovers_red_gate_cliff() {
        // The shafaq al-ahmar cone gate: the contrast margin is still
        // well above 1 when the red luminance sinks through the 1e-3
        // cd/m^2 gate, so the event IS the gate crossing - the cliff
        // class documented in the criterion-sites sweep. The gate
        // location is analytic: red = 0.3 (FLOOR + glow) = 1e-3.
        let glow = |s: f64| 8.0e-2 * libm::exp(-(s - 96.0) / 1.6);
        let params = KhaytParams::default().for_side(false);
        let scan = analytic_scan(&glow, 0.3);
        // Solve 0.3 * (FLOOR + glow(s)) = RED_CONE_GATE exactly.
        let s_gate = 96.0 - 1.6 * libm::log((RED_CONE_GATE / 0.3 - FLOOR) / 8.0e-2);

        let mut eval = |s: f64| Some(analytic_rows(s, &glow, 0.3));
        let eval_ref: &mut FanEval<'_> = &mut eval;
        let sol = detect_refined(
            &scan,
            &params,
            Some(eval_ref),
            RefineEvents {
                spread: false,
                central: false,
                ahmar: true,
            },
        );
        let c = sol.ahmar.expect("gated red crossing detected");
        assert!(
            (c.sza_deg - s_gate).abs() <= REFINE_TARGET_BRACKET_DEG,
            "red gate: got {} truth {} (err {:.3})",
            c.sza_deg,
            s_gate,
            (c.sza_deg - s_gate).abs()
        );
        assert!(
            c.bracket_half_deg > 0.0,
            "gate crossing is bracket-limited: {:?}",
            c
        );
    }

    #[test]
    fn noise_stop_halts_refinement_inside_mc_noise() {
        // A shallow crossing whose endpoint margins sit within their
        // own MC noise of 1.0: refinement must stop and the sigma must
        // carry the noise term instead of chasing phantom precision.
        let glow = |s: f64| 2.6e-3 * libm::exp(-(s - 96.0) / 8.0);
        let params = KhaytParams {
            // Bring the crossing into the scanned range for this glow.
            edge_factor_appearance: 8.0,
            ..KhaytParams::default()
        }
        .for_side(true);
        let noisy_rows = |s: f64| {
            let (mut band, refs) = analytic_rows(s, &glow, 0.3);
            for p in band.iter_mut() {
                p.rel_se_mes = 0.2; // huge MC noise
            }
            (band, refs)
        };
        let szas: Vec<f64> = (0..16).map(|i| 96.0 + i as f64).collect();
        let rows: Vec<_> = szas.iter().map(|&s| noisy_rows(s)).collect();
        let scan = KhaytScan {
            szas,
            band: rows.iter().map(|r| r.0.clone()).collect(),
            refs: rows.iter().map(|r| r.1.clone()).collect(),
        };
        let mut n_evals = 0usize;
        let mut eval = |s: f64| {
            n_evals += 1;
            Some(noisy_rows(s))
        };
        let eval_ref: &mut FanEval<'_> = &mut eval;
        let sol = detect_refined(
            &scan,
            &params,
            Some(eval_ref),
            RefineEvents {
                spread: true,
                central: false,
                ahmar: false,
            },
        );
        let c = sol.sadiq.expect("shallow crossing detected");
        assert!(
            n_evals < REFINE_MAX_EVALS,
            "noise stop must fire before the budget: {n_evals}"
        );
        assert!(
            c.sigma_deg > 0.1,
            "sigma must carry the MC noise: {}",
            c.sigma_deg
        );
    }
}
