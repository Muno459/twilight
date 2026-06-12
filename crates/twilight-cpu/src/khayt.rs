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

use twilight_threshold::threshold::{contrast_threshold_weber, fit_crossing_loglinear};

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
            edge_factor_appearance: 45.0,
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

/// Per-event solution: the SZA where the criterion is met, plus the fit
/// slope (for uncertainty propagation).
#[derive(Debug, Clone, Copy)]
pub struct KhaytCrossing {
    pub sza_deg: f64,
    pub slope: f64,
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

/// Where a decreasing margin curve crosses 1.0.
///
/// Primary: LOCAL log-linear fit over the points bracketing the
/// crossing (a global fit is biased by the bright-twilight plateau and
/// can extrapolate outside the scan - verified at Mecca). Fallback for
/// cliff-shaped curves (the red cone gate zeroes the channel abruptly):
/// log/linear interpolation across the bracketing pair itself.
fn solve_crossing(curve: &[(f64, f64)]) -> Option<KhaytCrossing> {
    let n = curve.len();
    // Bracket: last point >= 1 followed by a point < 1 (curve decreasing
    // with SZA; ascending scan order).
    let mut bi = None;
    for i in 0..n - 1 {
        if curve[i].1 >= 1.0 && curve[i + 1].1 < 1.0 {
            bi = Some(i);
        }
    }
    let i = bi?;
    // Local fit window around the bracket.
    let lo = i.saturating_sub(2);
    let hi = (i + 3).min(n);
    if let Some((s, b)) = fit_crossing_loglinear(&curve[lo..hi], 1.0) {
        return Some(KhaytCrossing { sza_deg: s, slope: b });
    }
    // Cliff fallback: interpolate inside the bracketing pair.
    let (s0, m0) = curve[i];
    let (s1, m1) = curve[i + 1];
    let sza = if m1 > 0.0 {
        // log interpolation
        s0 + (s1 - s0) * (libm::log(m0) / (libm::log(m0) - libm::log(m1)))
    } else {
        // hard gate: the event sits inside the bracket; midpoint
        0.5 * (s0 + s1)
    };
    let slope = if m1 > 0.0 {
        (libm::log(m1) - libm::log(m0)) / (s1 - s0)
    } else {
        -2.0 / (s1 - s0).max(1e-6)
    };
    Some(KhaytCrossing {
        sza_deg: sza,
        slope,
    })
}

/// k-th largest of a small slice (1-based: k=1 is the maximum).
fn kth_largest(values: &[f64], k: usize) -> f64 {
    let mut v: Vec<f64> = values.to_vec();
    v.sort_by(|a, b| b.total_cmp(a));
    v[(k - 1).min(v.len() - 1)]
}

/// Mean reference luminance at one step.
fn ref_mean(refs: &[PatchLum]) -> PatchLum {
    let n = refs.len().max(1) as f64;
    PatchLum {
        mesopic: refs.iter().map(|p| p.mesopic).sum::<f64>() / n,
        red: refs.iter().map(|p| p.red).sum::<f64>() / n,
    }
}

/// Contrast margins for one step.
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
fn margins(
    band: &[PatchLum],
    night_base: &[PatchLum],
    refs: &[PatchLum],
    k_contrast: f64,
    red: bool,
    k_red: f64,
) -> Vec<f64> {
    let r = ref_mean(refs);
    // The eye adapts to the prevailing (broadband) sky, not to one channel.
    let l_adapt = r.mesopic.max(1e-6);
    let c_thr = contrast_threshold_weber(l_adapt);
    band.iter()
        .zip(night_base.iter())
        .map(|(p, b)| {
            let (excess, lr, k) = if red {
                (p.red - b.red, r.red.max(1e-9), k_red)
            } else {
                (p.mesopic - b.mesopic, r.mesopic.max(1e-9), k_contrast)
            };
            // RED is a cone percept: below the cone threshold
            // (~1e-3 cd/m^2, Hecht-Shlaer-Pirenne lineage) the band may
            // be VISIBLE but not COLORED - rods see no red. Shafaq
            // al-ahmar requires the color, so the red channel is gated.
            if red && p.red < 1e-3 {
                return 0.0;
            }
            if excess <= 0.0 {
                0.0
            } else {
                (excess / lr) / (k * c_thr)
            }
        })
        .collect()
}

/// Detect the khayt events on one side's scan.
///
/// All three events are "margin crosses 1.0" problems on curves that
/// decrease with SZA (the dawn brightening fades with depth), so the
/// existing log-linear crossing fit applies directly; morning/evening
/// only differ later, at SZA -> clock-time conversion.
pub fn detect(scan: &KhaytScan, params: &KhaytParams) -> KhaytSolution {
    let n = scan.szas.len();
    if n < 3 || scan.band.len() != n || scan.refs.len() != n {
        return KhaytSolution::default();
    }
    let n_band = params.band_offsets_deg.len();
    let center = params
        .band_offsets_deg
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
        .map(|(i, _)| i)
        .unwrap_or(n_band / 2);

    // Deep-night baseline per band patch: the median over the deepest
    // three scanned SZAs (the scan is ascending in SZA). At high
    // latitudes where the night never deepens, this floor still carries
    // residual twilight - the criterion then measures brightening above
    // tonight's actual minimum, which is exactly the right high-latitude
    // semantics.
    let n_band = scan.band[0].len();
    let deep = n.saturating_sub(3);
    let night_base: Vec<PatchLum> = (0..n_band)
        .map(|j| {
            let mut mes: Vec<f64> = (deep..n).map(|i| scan.band[i][j].mesopic).collect();
            let mut red: Vec<f64> = (deep..n).map(|i| scan.band[i][j].red).collect();
            mes.sort_by(|a, b| a.total_cmp(b));
            red.sort_by(|a, b| a.total_cmp(b));
            PatchLum {
                mesopic: mes[mes.len() / 2],
                red: red[red.len() / 2],
            }
        })
        .collect();

    let mut spread_curve = Vec::with_capacity(n);
    let mut central_curve = Vec::with_capacity(n);
    let mut ahmar_curve = Vec::with_capacity(n);
    let mut per_step_margins = Vec::with_capacity(n);
    for i in 0..n {
        let m = margins(
            &scan.band[i],
            &night_base,
            &scan.refs[i],
            params.k_contrast,
            false,
            params.k_contrast_red,
        );
        let mr = margins(
            &scan.band[i],
            &night_base,
            &scan.refs[i],
            params.k_contrast,
            true,
            params.k_contrast_red,
        );
        spread_curve.push((scan.szas[i], kth_largest(&m, params.spread_required)));
        central_curve.push((scan.szas[i], m[center]));
        // Red band: the sunset/dawn red glow is broad but narrower than
        // the white spread - require a majority of the fan (3 of 5) so
        // the narrow zodiacal core (also reddish) cannot satisfy it.
        let _ = center;
        ahmar_curve.push((
            scan.szas[i],
            kth_largest(&mr, 3.min(n_band)),
        ));
        per_step_margins.push(m);
    }

    if std::env::var("TWILIGHT_KHAYT_DEBUG").is_ok() {
        for i in (0..n).step_by(2) {
            eprintln!(
                "khayt sza {:6.2}: spread {:9.3e} central {:9.3e} ahmar {:9.3e} margins {:?}",
                scan.szas[i],
                spread_curve[i].1,
                central_curve[i].1,
                ahmar_curve[i].1,
                per_step_margins[i]
                    .iter()
                    .map(|m| (m * 1000.0).round() / 1000.0)
                    .collect::<Vec<_>>()
            );
        }
    }

    let sadiq = solve_crossing(&spread_curve);
    let central = solve_crossing(&central_curve);
    let ahmar = solve_crossing(&ahmar_curve);

    // Kadhib: the central column becomes distinct DEEPER than (or
    // without) the spread verdict - an interval where only the narrow
    // wedge is visible. Central-distinct with NO spread at all is the
    // purest wolf's-tail case.
    let kadhib = match (central, sadiq) {
        (Some(c), Some(s)) if c.sza_deg > s.sza_deg + 0.2 => Some(c),
        (Some(c), None) => Some(c),
        _ => None,
    };

    // Diagnostics: margins at the step nearest the sadiq crossing.
    let margins_at_sadiq = sadiq
        .map(|s| {
            let i = scan
                .szas
                .iter()
                .enumerate()
                .min_by(|a, b| (a.1 - s.sza_deg).abs().total_cmp(&(b.1 - s.sza_deg).abs()))
                .map(|(i, _)| i)
                .unwrap_or(0);
            per_step_margins[i].clone()
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
                    })
                    .collect()
            })
            .collect();
        KhaytScan { szas, band, refs }
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
        let m_dark = margins(&dark.band[0], &zero, &dark.refs[0], p.k_contrast, false, 1.0);
        let m_mid = margins(&mid.band[0], &zero, &mid.refs[0], p.k_contrast, false, 1.0);
        assert!(
            m_mid[2] > m_dark[2],
            "20% excess is easier at mesopic than at the rod floor: {m_mid:?} vs {m_dark:?}"
        );
    }
}
