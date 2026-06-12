//! Ingestion of cloud3d satellite 3D cloud reconstructions.
//!
//! The Python sidecar (`tools/cloud3d_profile.py`) runs the
//! csaybar/cloud3d TorchScript model (SegFormer trained on Cloud3DTACO:
//! geostationary imagery paired with CloudSat radar profiles,
//! arXiv:2511.04773) on live GOES imagery and emits an 80-level
//! ice-water-content profile (~240 m bins, top ~18.9 km). This module
//! reads that JSON and collapses the profile into contiguous cloud
//! layers for the radiative-transfer builder - real measured vertical
//! cloud STRUCTURE instead of a single assumed slab.
//!
//! IWC -> visible extinction: for ice particles with effective radius
//! r_eff, beta = 3*IWC / (2*rho_ice*r_eff) (geometric-optics Q~2).
//! With r_eff = 30 um (midlatitude cirrus median, Heymsfield & Platt):
//! beta [1/m] = IWC [g/m^3] * 0.0545.

use std::path::Path;

use serde::Deserialize;
use twilight_data::cloud::CloudProperties;

/// Ice density [g/m^3].
const RHO_ICE: f64 = 0.917e6;
/// Ice effective radius assumed for the IWC->extinction conversion [m].
const R_EFF_ICE: f64 = 30e-6;
/// IWC below this is treated as clear (the model's clear-sky fill is
/// 1e-5 g/m^3; real cirrus is 1e-3..1e-1) [g/m^3].
const IWC_FLOOR: f64 = 5e-4;
/// Layers with less optical depth than this are dropped.
const MIN_LAYER_TAU: f64 = 0.01;
/// Keep at most this many layers (largest optical depth first).
const MAX_LAYERS: usize = 6;

#[derive(Debug, Deserialize)]
pub struct Cloud3dProfile {
    pub satellite: String,
    pub granule: String,
    pub time_utc: String,
    /// Bin-center heights [m], descending (index 0 = top ~18.9 km).
    pub heights_m: Vec<f64>,
    /// Fraction of window columns with ice water path > 1 g/m^2.
    pub cloud_fraction: f64,
    pub profiles: Cloud3dColumns,
}

#[derive(Debug, Deserialize)]
pub struct Cloud3dColumns {
    /// Column at the observer.
    pub center: Vec<f64>,
    /// Mean over the full imagery window (~256 km).
    pub window_mean: Vec<f64>,
    /// Columns sampled along the sunward azimuth.
    #[serde(default)]
    pub path: Vec<Cloud3dPathSample>,
}

#[derive(Debug, Deserialize)]
pub struct Cloud3dPathSample {
    pub km: f64,
    pub iwc_g_m3: Vec<f64>,
}

/// Load a sidecar-produced profile JSON.
pub fn load(path: &Path) -> Result<Cloud3dProfile, String> {
    let body = std::fs::read_to_string(path)
        .map_err(|e| format!("cloud3d profile read failed: {e}"))?;
    let p: Cloud3dProfile =
        serde_json::from_str(&body).map_err(|e| format!("cloud3d profile parse failed: {e}"))?;
    if p.heights_m.len() < 2 || p.profiles.center.len() != p.heights_m.len() {
        return Err("cloud3d profile: inconsistent level count".to_string());
    }
    Ok(p)
}

impl Cloud3dProfile {
    /// The IWC column used for transport: mean of the sunward-path
    /// samples when available (that is where the twilight shadow path
    /// crosses the cloud field), otherwise the window mean.
    fn effective_column(&self) -> Vec<f64> {
        let n = self.heights_m.len();
        if self.profiles.path.len() >= 2 {
            let mut acc = vec![0.0; n];
            let mut cnt = 0.0;
            for s in &self.profiles.path {
                if s.iwc_g_m3.len() == n {
                    for (a, v) in acc.iter_mut().zip(&s.iwc_g_m3) {
                        *a += v;
                    }
                    cnt += 1.0;
                }
            }
            if cnt > 0.0 {
                for a in &mut acc {
                    *a /= cnt;
                }
                return acc;
            }
        }
        self.profiles.window_mean.clone()
    }

    /// Collapse the 80-level IWC profile into contiguous cloud layers.
    ///
    /// `rescale_total_tau`: when a near-real-time measured cloud optical
    /// thickness is available (GIBS MODIS COT), the cloud3d profile
    /// supplies the vertical STRUCTURE and the measurement the AMPLITUDE:
    /// all layer optical depths are scaled so they sum to the measured
    /// value (scale clamped to 0.2..5 against retrieval disagreement).
    pub fn to_cloud_layers(&self, rescale_total_tau: Option<f64>) -> Vec<CloudProperties> {
        let iwc = self.effective_column();
        let h = &self.heights_m;
        let n = h.len();

        // Contiguous runs of cloudy bins (gaps of one clear bin are bridged).
        let mut layers: Vec<CloudProperties> = Vec::new();
        let mut run_start: Option<usize> = None;
        let mut gap = 0usize;
        let mut i = 0usize;
        while i <= n {
            let cloudy = i < n && iwc[i] >= IWC_FLOOR;
            if cloudy {
                if run_start.is_none() {
                    run_start = Some(i);
                }
                gap = 0;
            } else if let Some(s) = run_start {
                gap += 1;
                if gap > 1 || i >= n {
                    let e = i - gap; // exclusive end of the cloudy run
                    if let Some(layer) = self.run_to_layer(&iwc, s, e) {
                        layers.push(layer);
                    }
                    run_start = None;
                    gap = 0;
                }
            }
            i += 1;
        }

        layers.retain(|l| l.optical_depth >= MIN_LAYER_TAU);
        layers.sort_by(|a, b| b.optical_depth.total_cmp(&a.optical_depth));
        layers.truncate(MAX_LAYERS);

        if let Some(target) = rescale_total_tau {
            let total: f64 = layers.iter().map(|l| l.optical_depth).sum();
            if total > 0.0 && target > 0.0 {
                let scale = (target / total).clamp(0.2, 5.0);
                for l in &mut layers {
                    l.optical_depth *= scale;
                }
            }
        }
        layers
    }

    /// Bins [s, e) -> one cloud layer.
    fn run_to_layer(&self, iwc: &[f64], s: usize, e: usize) -> Option<CloudProperties> {
        if e <= s {
            return None;
        }
        let h = &self.heights_m;
        let n = h.len();
        let dz = (h[0] - h[n - 1]).abs() / (n - 1) as f64; // ~240 m
        let mut tau = 0.0;
        for &v in &iwc[s..e] {
            let beta = 3.0 * v / (2.0 * RHO_ICE * R_EFF_ICE); // 1/m
            tau += beta * dz;
        }
        // heights are descending: bin s is the TOP of the run.
        let top_km = (h[s] + dz / 2.0) / 1000.0;
        let base_km = ((h[e - 1] - dz / 2.0) / 1000.0).max(0.0);
        if top_km <= base_km {
            return None;
        }
        // Altitude-typed optics, consistent with mapping::map_cloud_satellite:
        // high ice / mid alto / low stratiform.
        let mid_km = 0.5 * (top_km + base_km);
        let (ssa, g) = if mid_km > 6.0 {
            (0.97, 0.77)
        } else if mid_km > 3.0 {
            (0.99, 0.82)
        } else {
            (0.999, 0.85)
        };
        Some(CloudProperties {
            base_km,
            top_km,
            optical_depth: tau,
            ssa,
            asymmetry: g,
        })
    }

    /// Total optical depth of the collapsed profile (no rescale).
    pub fn total_optical_depth(&self) -> f64 {
        self.to_cloud_layers(None)
            .iter()
            .map(|l| l.optical_depth)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic(iwc_fill: &[(usize, usize, f64)]) -> Cloud3dProfile {
        let n = 80usize;
        let heights: Vec<f64> = (0..n)
            .map(|i| 18945.0 * (1.0 - i as f64 / (n - 1) as f64))
            .collect();
        let mut col = vec![0.0; n];
        for &(s, e, v) in iwc_fill {
            for x in col.iter_mut().take(e).skip(s) {
                *x = v;
            }
        }
        Cloud3dProfile {
            satellite: "noaa-goes19".into(),
            granule: "test".into(),
            time_utc: "2026-06-12T08:30".into(),
            heights_m: heights,
            cloud_fraction: 0.5,
            profiles: Cloud3dColumns {
                center: col.clone(),
                window_mean: col,
                path: vec![],
            },
        }
    }

    #[test]
    fn cirrus_blob_becomes_single_ice_layer() {
        // Bins 33..40 ~ heights 11.0..9.4 km, IWC 0.02 g/m^3.
        let p = synthetic(&[(33, 40, 0.02)]);
        let layers = p.to_cloud_layers(None);
        assert_eq!(layers.len(), 1, "{layers:?}");
        let l = &layers[0];
        assert!(l.top_km > 10.5 && l.top_km < 11.5, "top {l:?}");
        assert!(l.base_km > 8.8 && l.base_km < 9.8, "base {l:?}");
        // tau = 7 bins * 240m * 0.02*0.0545 = ~1.83
        assert!(
            (1.2..2.5).contains(&l.optical_depth),
            "tau {}",
            l.optical_depth
        );
        assert!((l.asymmetry - 0.77).abs() < 1e-9, "ice g");
    }

    #[test]
    fn two_separated_decks_become_two_layers() {
        let p = synthetic(&[(30, 35, 0.05), (70, 76, 0.05)]);
        let layers = p.to_cloud_layers(None);
        assert_eq!(layers.len(), 2, "{layers:?}");
        // low deck below 3 km gets stratiform optics
        let low = layers.iter().find(|l| l.base_km < 3.0).unwrap();
        assert!((low.asymmetry - 0.85).abs() < 1e-9);
    }

    #[test]
    fn rescale_to_measured_cot() {
        let p = synthetic(&[(33, 40, 0.02)]);
        let t0 = p.total_optical_depth();
        let layers = p.to_cloud_layers(Some(2.0 * t0));
        let t1: f64 = layers.iter().map(|l| l.optical_depth).sum();
        assert!((t1 / t0 - 2.0).abs() < 1e-9);
        // clamp guards absurd scalings
        let clamped = p.to_cloud_layers(Some(100.0 * t0));
        let t2: f64 = clamped.iter().map(|l| l.optical_depth).sum();
        assert!((t2 / t0 - 5.0).abs() < 1e-9);
    }

    #[test]
    fn clear_profile_no_layers() {
        let p = synthetic(&[]);
        assert!(p.to_cloud_layers(None).is_empty());
        // sub-floor noise also stays clear
        let p = synthetic(&[(20, 60, 1e-4)]);
        assert!(p.to_cloud_layers(None).is_empty());
    }

    #[test]
    fn faint_single_bin_dropped() {
        let p = synthetic(&[(40, 41, 6e-4)]);
        // tau = 240 * 6e-4*0.0545 ~ 0.008 < MIN_LAYER_TAU
        assert!(p.to_cloud_layers(None).is_empty());
    }
}
