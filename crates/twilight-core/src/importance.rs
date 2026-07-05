//! Forward-informed importance map for deep-twilight population control.
//!
//! The weight windows in the secondary chains need an importance
//! function I(x) ~ "expected future contribution of a chain at x". The
//! historical heuristic (exponential altitude ramp x one-sided CADIS
//! lateral term) encodes the right monotonicity but not the physics; it
//! demonstrably leaves the deepest cells starved (RESULTS_DEEP_REGIME
//! addendum: the 103/650 uniform cell, the broken-deck zenith rows).
//!
//! This module builds the physical signal deterministically instead. In
//! a spherically-stratified atmosphere the solar in-scatter source
//! density at a point depends only on
//!
//!   (altitude, cos_sun) with cos_sun = pos_hat . sun_dir,
//!
//! so ONE small 2D map serves every observer, every LOS step, and every
//! SZA of a scan:
//!
//!   S(alt, cos) = beta_scat(alt) * T_sun(alt, cos)
//!
//! with T_sun the straight-line gas transmittance toward the sun
//! (ground-shadow aware). A few diffusion sweeps then spread importance
//! into the geometric shadow: a shadowed cell one scatter away from the
//! sunlit terminator inherits (damped) importance, which is exactly the
//! "the chain can still climb or drift sunward" value the windows must
//! not roulette away. The result is a smooth, deterministic CADIS-style
//! adjoint proxy: importance QUALITY affects only variance, never the
//! mean (split/roulette bookkeeping is unbiased for any positive
//! target), so approximations here are safe by construction.
//!
//! Clouds are deliberately ignored when building the map: the (alt,
//! cos) coordinates cannot represent lateral field structure, and
//! penalizing under-deck cells would starve the chains that escape a
//! deck sideways. The gas map is the conservative choice.

use crate::atmosphere::AtmosphereModel;
use crate::geometry::Vec3;

/// Altitude bands [0, ALT_SPAN_M) in ALT_BANDS uniform steps.
pub const ALT_BANDS: usize = 40;
/// cos_sun bands over [COS_MIN, COS_MAX] in COS_BANDS uniform steps.
pub const COS_BANDS: usize = 64;
/// Altitude span covered by the map [m] (the shell stack's 120 km).
const ALT_SPAN_M: f64 = 120_000.0;
/// cos_sun range: +0.5 (sun 60 deg high at the cell) down to -0.5
/// (position angle 120 deg, far into the night side). Clamped outside.
const COS_MIN: f64 = -0.5;
const COS_MAX: f64 = 0.5;
/// Relaxation sweeps propagating importance into the shadow (transport
/// operator below): enough for the longest useful corridor
/// (climb + full lateral span + descend ~ 150 hops).
const RELAX_SWEEPS: usize = 160;
/// Lateral arc length of one cos_sun bin at Earth radius [m]
/// (r * dtheta with dtheta ~ dcos / sin(theta), sin ~ 0.97 over the
/// domain: ~104 km; the constant approximation only shapes importance).
const LATERAL_HOP_M: f64 = 1.0e5;
/// Floor relative to the map maximum: bounds every window ratio (and
/// keeps roulette survival probabilities sane) even in the deepest
/// shadow cell.
const FLOOR_REL: f64 = 1e-9;

/// Deterministic solar-source importance map over (altitude, cos_sun).
///
/// ~10 KB, built once per `hybrid_scatter_radiance` call in the deep
/// regime (a shell walk per cell: sub-millisecond against minutes of
/// chain MC).
pub struct SolarImportanceMap {
    /// Normalized (max = 1) source density, row-major [alt][cos].
    data: [f32; ALT_BANDS * COS_BANDS],
}

impl SolarImportanceMap {
    /// Build the map for one atmosphere at a hero wavelength.
    ///
    /// The sun is placed along +X and the cell position in the XY
    /// plane; spherical symmetry makes that fully general.
    pub fn build(atm: &AtmosphereModel, wavelength_idx: usize) -> Self {
        let surface_radius = atm.surface_radius();
        let mut data = [0.0f32; ALT_BANDS * COS_BANDS];
        let sun_dir = Vec3::new(1.0, 0.0, 0.0);

        for ia in 0..ALT_BANDS {
            let alt = (ia as f64 + 0.5) * (ALT_SPAN_M / ALT_BANDS as f64);
            let r = surface_radius + alt;
            // Local scattering coefficient at the cell's shell.
            let beta_scat = match atm.shell_index(r) {
                Some(si) => {
                    let o = &atm.optics[si][wavelength_idx];
                    o.extinction * o.ssa
                }
                None => 0.0,
            };
            if beta_scat <= 0.0 {
                continue;
            }
            for ic in 0..COS_BANDS {
                let c = COS_MIN + (ic as f64 + 0.5) * ((COS_MAX - COS_MIN) / COS_BANDS as f64);
                let s = libm::sqrt((1.0 - c * c).max(0.0));
                let pos = Vec3::new(r * c, r * s, 0.0);
                let t_sun = gas_tau_to_sun(atm, pos, sun_dir, wavelength_idx);
                data[ia * COS_BANDS + ic] = (beta_scat * t_sun) as f32;
            }
        }

        // Transport relaxation: importance is the best achievable
        // product of hop transmittances back to a sunlit source cell,
        //
        //   I(c) = max( S(c), max_n I(n) * exp(-tau_hop(n, c)) ),
        //
        // iterated to convergence (Bellman-Ford over the grid; the
        // damping uses the mean gas extinction of the two cells over
        // the hop length: 3 km vertical, ~100 km lateral). This is the
        // single-path adjoint approximation: it discovers the physical
        // deep-twilight corridor (climb where tau is cheap, travel
        // sunward at altitude, descend), which fixed-radius diffusion
        // cannot reach 12-19 bins past the terminator where the khayt
        // regime lives. No per-hop redirect penalty: importance is
        // OVERestimated in the deep shadow, which errs toward milder
        // roulette and earlier splitting (a variance choice, never a
        // bias).
        let alt_hop_m = ALT_SPAN_M / ALT_BANDS as f64;
        let mut beta = [0.0f64; ALT_BANDS];
        for (ia, b) in beta.iter_mut().enumerate() {
            let r = surface_radius + (ia as f64 + 0.5) * alt_hop_m;
            *b = match atm.shell_index(r) {
                Some(si) => atm.optics[si][wavelength_idx].extinction,
                None => 0.0,
            };
        }
        let mut cur = data;
        for _ in 0..RELAX_SWEEPS {
            let mut changed = false;
            for ia in 0..ALT_BANDS {
                for ic in 0..COS_BANDS {
                    let mut best = cur[ia * COS_BANDS + ic];
                    let mut relax = |from_a: usize, from_c: usize, hop_m: f64| {
                        let damp =
                            libm::exp(-0.5 * (beta[ia] + beta[from_a]) * hop_m) as f32;
                        let cand = cur[from_a * COS_BANDS + from_c] * damp;
                        if cand > best {
                            best = cand;
                        }
                    };
                    if ia + 1 < ALT_BANDS {
                        relax(ia + 1, ic, alt_hop_m);
                    }
                    if ia > 0 {
                        relax(ia - 1, ic, alt_hop_m);
                    }
                    if ic + 1 < COS_BANDS {
                        relax(ia, ic + 1, LATERAL_HOP_M);
                    }
                    if ic > 0 {
                        relax(ia, ic - 1, LATERAL_HOP_M);
                    }
                    if best > cur[ia * COS_BANDS + ic] {
                        cur[ia * COS_BANDS + ic] = best;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        // Normalize to max = 1 and apply the relative floor.
        let mut maxv = 0.0f32;
        for &v in cur.iter() {
            if v > maxv {
                maxv = v;
            }
        }
        if maxv > 0.0 {
            let floor = (FLOOR_REL as f32) * maxv;
            for v in cur.iter_mut() {
                *v = (*v).max(floor) / maxv;
            }
        } else {
            // Degenerate atmosphere (no scattering anywhere): flat map,
            // windows become inert (every ratio 1).
            for v in cur.iter_mut() {
                *v = 1.0;
            }
        }

        SolarImportanceMap { data: cur }
    }

    /// Bilinear importance lookup (clamped to the domain).
    pub fn lookup(&self, alt_m: f64, cos_sun: f64) -> f64 {
        let fa = ((alt_m / (ALT_SPAN_M / ALT_BANDS as f64)) - 0.5)
            .clamp(0.0, (ALT_BANDS - 1) as f64);
        let fc = (((cos_sun - COS_MIN) / ((COS_MAX - COS_MIN) / COS_BANDS as f64)) - 0.5)
            .clamp(0.0, (COS_BANDS - 1) as f64);
        let a0 = fa as usize;
        let c0 = fc as usize;
        let a1 = (a0 + 1).min(ALT_BANDS - 1);
        let c1 = (c0 + 1).min(COS_BANDS - 1);
        let wa = fa - a0 as f64;
        let wc = fc - c0 as f64;
        let g = |a: usize, c: usize| self.data[a * COS_BANDS + c] as f64;
        (1.0 - wa) * ((1.0 - wc) * g(a0, c0) + wc * g(a0, c1))
            + wa * ((1.0 - wc) * g(a1, c0) + wc * g(a1, c1))
    }

    /// Weight-window target for a chain at (alt, cos) that was seeded at
    /// (alt_start, cos_start): w_target = I(start) / I(here), clamped so
    /// a pathological ratio can neither explode the split count nor
    /// drive roulette survival to zero.
    pub fn window_target(
        &self,
        alt_m: f64,
        cos_sun: f64,
        alt_start_m: f64,
        cos_sun_start: f64,
    ) -> f64 {
        let i_here = self.lookup(alt_m, cos_sun);
        let i_start = self.lookup(alt_start_m, cos_sun_start);
        if i_here <= 0.0 || i_start <= 0.0 {
            return 1.0;
        }
        (i_start / i_here).clamp(1e-4, 1e4)
    }
}

/// Straight-line gas transmittance from `pos` toward the sun (no
/// refraction, no cloud channel): the map's source term. Ground
/// intersection returns 0 (hard shadow); the diffusion sweeps supply
/// the soft edge.
fn gas_tau_to_sun(atm: &AtmosphereModel, pos: Vec3, sun_dir: Vec3, wavelength_idx: usize) -> f64 {
    let surface_radius = atm.surface_radius();
    let toa_radius = atm.toa_radius();

    // Ground shadow: does the ray toward the sun dip below the surface?
    // Closest approach at t* = -pos.sun_dir (only relevant if ahead).
    let b = pos.dot(sun_dir);
    if b < 0.0 {
        let p_min2 = pos.dot(pos) - b * b;
        if p_min2 < surface_radius * surface_radius {
            return 0.0;
        }
    }

    // Shell walk accumulating gas extinction to TOA.
    let mut tau = 0.0f64;
    let mut p = pos;
    let mut shell_idx = match atm.shell_index(p.length()) {
        Some(i) => i,
        None => return 1.0, // already above TOA
    };
    for _ in 0..2 * crate::atmosphere::MAX_SHELLS {
        let shell = &atm.shells[shell_idx];
        let ext = atm.optics[shell_idx][wavelength_idx].extinction;
        match crate::geometry::next_shell_boundary(p, sun_dir, shell.r_inner, shell.r_outer) {
            Some((dist, is_outward)) => {
                tau += ext * dist;
                p = p + sun_dir * dist;
                if is_outward {
                    if shell_idx + 1 >= atm.num_shells {
                        break; // exited TOA
                    }
                    shell_idx += 1;
                } else {
                    if shell_idx == 0 {
                        return 0.0; // reached the surface: shadow
                    }
                    shell_idx -= 1;
                }
            }
            None => break,
        }
        if p.length() >= toa_radius {
            break;
        }
    }
    libm::exp(-tau)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exponential Rayleigh-like atmosphere: 24 shells 0-120 km,
    /// scale height 8 km, sea-level extinction 1.2e-5 /m, ssa 1.
    fn test_atm() -> AtmosphereModel {
        let alts: [f64; 25] = core::array::from_fn(|i| i as f64 * 5.0);
        let mut atm = AtmosphereModel::new(&alts, &[450.0, 550.0, 650.0]);
        for si in 0..atm.num_shells {
            let mid_km = atm.shells[si].altitude_mid / 1000.0;
            let ext = 1.2e-5 * libm::exp(-mid_km / 8.0);
            for w in 0..atm.num_wavelengths {
                atm.optics[si][w].extinction = ext;
                atm.optics[si][w].ssa = 1.0;
                atm.optics[si][w].rayleigh_fraction = 1.0;
            }
        }
        atm
    }

    #[test]
    fn map_builds_and_normalizes() {
        let atm = test_atm();
        let m = SolarImportanceMap::build(&atm, 1);
        let mut maxv = 0.0f64;
        for ia in 0..ALT_BANDS {
            for ic in 0..COS_BANDS {
                let v = m.data[ia * COS_BANDS + ic] as f64;
                assert!(v.is_finite() && v > 0.0, "cell ({ia},{ic}) = {v}");
                maxv = maxv.max(v);
            }
        }
        assert!((maxv - 1.0).abs() < 1e-6, "max {maxv} must be 1");
    }

    /// The physical signatures the heuristic could only guess. Deep in
    /// the shadow, importance decays monotonically AWAY from the
    /// terminator (the lateral-transport corridor: CADIS quantified);
    /// vertically the night side is nearly flat (the gas column above
    /// 5 km is only tau ~ 0.05, so climbing is cheap and gains little
    /// by itself), while the day side is source-dominated at low
    /// altitude. And the high-altitude corridor must reach usefully
    /// deep: at 16 bins past the terminator the map must sit far above
    /// its floor (the fixed-radius diffusion this replaced left
    /// everything there AT the floor).
    #[test]
    fn shadow_importance_monotone_toward_terminator() {
        let atm = test_atm();
        let m = SolarImportanceMap::build(&atm, 1);
        let near = m.lookup(80_000.0, -0.10);
        let mid = m.lookup(80_000.0, -0.25);
        let deep = m.lookup(80_000.0, -0.40);
        assert!(
            near > mid && mid > deep,
            "sunward monotonicity at altitude: {near:e} > {mid:e} > {deep:e}"
        );
        assert!(
            mid > 1e-3,
            "corridor must reach 16 bins deep, far above the 1e-9 floor: {mid:e}"
        );
        let day_low = m.lookup(5_000.0, 0.4);
        let day_high = m.lookup(80_000.0, 0.4);
        assert!(
            day_low > day_high,
            "day side: low {day_low:e} must beat high {day_high:e}"
        );
        // Night side vertical: the high corridor cell must not be
        // WORSE than the low cell it feeds (descent only loses tau).
        let night_low = m.lookup(5_000.0, -0.25);
        assert!(
            mid >= night_low,
            "corridor {mid:e} must be >= the cell it feeds {night_low:e}"
        );
    }

    /// Lateral gradient: at fixed altitude in the shadow, importance
    /// must increase toward the terminator (the CADIS direction).
    #[test]
    fn importance_increases_toward_terminator() {
        let atm = test_atm();
        let m = SolarImportanceMap::build(&atm, 1);
        let deep = m.lookup(30_000.0, -0.35);
        let near = m.lookup(30_000.0, -0.10);
        assert!(
            near > deep,
            "toward terminator {near:e} must beat deep night {deep:e}"
        );
    }

    /// Window-target sanity: a chain that climbs from a dark seed into
    /// a brighter cell must see its target DROP (I rises, target =
    /// I_start/I_here < 1), which is what triggers splitting.
    #[test]
    fn window_target_drops_toward_brighter_cells() {
        let atm = test_atm();
        let m = SolarImportanceMap::build(&atm, 1);
        let t = m.window_target(80_000.0, -0.10, 5_000.0, -0.35);
        assert!(t < 1.0, "climbing sunward must lower the target: {t}");
        let t_back = m.window_target(5_000.0, -0.35, 80_000.0, -0.10);
        assert!(t_back > 1.0, "diving into the dark must raise it: {t_back}");
        assert!((1e-4..=1e4).contains(&t) && (1e-4..=1e4).contains(&t_back));
    }
}
