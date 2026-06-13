//! 3D cloud field: the measured cloud volume as a first-class transport
//! medium ("3D clouds, 1D gases").
//!
//! The field is a view over CALLER-OWNED slices (this crate is
//! no_std/no-alloc): twilight-data builds and owns the buffers,
//! twilight-gpu packs the same layout for Metal.
//!
//! Geometry: an equal-angle local grid centered on the observer
//! (longitude columns x latitude rows) with a dedicated fine vertical
//! grid (default 250 m for 0-16 km), DECOUPLED from the transport
//! shells: cloud3d's 240 m vertical resolution survives, and at grazing
//! twilight geometry a few hundred meters of cloud-top error displaces
//! the shadow edge by 10-15 km horizontally.
//!
//! Contents per voxel: the DELTA-EDDINGTON-SCALED cloud SCATTERING
//! extinction [1/m] (matching the semantics of
//! `AtmosphereModel::cloud_extinction`), spectrally gray. Asymmetry is
//! per voxel via a parallel slice (empty slice = use `g_default`).
//!
//! Ownership convention (no seams): when a `Cloud3DField` is passed to
//! transport, IT owns all cloud. The atmosphere builder zeroes the
//! per-shell cloud arrays, and queries outside the field's horizontal
//! footprint are answered by the embedded `background_column` (the
//! horizontal mean profile), so the field edge is a smooth handover,
//! not a discontinuity.
//!
//! The line integral along a ray (`tau_along`) is computed by exact
//! piecewise-constant traversal (Amanatides-Woo style DDA in
//! (r, lat, lon) coordinates: sphere, cone, and meridian-plane
//! crossings all have closed forms). Every transport consumer (shadow
//! rays, eye LOS, forced-collision scouts and their tau inversion,
//! delta-tracking flights) must use THESE routines so all integrals of
//! the same ray agree bitwise.

use crate::geometry::Vec3;

/// Earth radius used for grid georeferencing (must match transport).
use crate::atmosphere::EARTH_RADIUS_M;

const DEG: f64 = core::f64::consts::PI / 180.0;

/// core-compatible rem_euclid (f64::rem_euclid is std-only).
#[inline]
fn rem_euclid(x: f64, y: f64) -> f64 {
    let r = libm::fmod(x, y);
    if r < 0.0 {
        r + y
    } else {
        r
    }
}

/// A 3D cloud volume as a borrowed view.
#[derive(Clone, Copy)]
pub struct Cloud3DField<'a> {
    /// Delta-scaled cloud scattering extinction [1/m], laid out
    /// `[iz * nlat * nlon + ilat * nlon + ilon]`, z ascending from
    /// `z0_m`, lat ascending from `lat0_deg`, lon ascending from
    /// `lon0_deg`.
    pub sigma: &'a [f32],
    /// Per-voxel delta-scaled asymmetry g*, same layout; empty = use
    /// `g_default` everywhere.
    pub g_star: &'a [f32],
    /// Horizontal-mean profile per z level [1/m] (len = nz): the
    /// out-of-footprint background. All-zero = clear beyond the edge.
    pub background_column: &'a [f32],
    /// Macro-cell majorants: max sigma over each (z level, tile) where
    /// tiles are `tile` x `tile` voxels; laid out
    /// `[iz * ntlat * ntlon + itlat * ntlon + itlon]`. Used by delta
    /// tracking (Stage 2); may be empty until then.
    pub macrocell_max: &'a [f32],
    /// Voxels per macro-cell tile edge (e.g. 8 -> 16 km tiles at 2 km).
    pub tile: usize,

    pub nz: usize,
    pub nlat: usize,
    pub nlon: usize,
    /// Bottom of the vertical grid [m above the reference sphere].
    pub z0_m: f64,
    /// Vertical spacing [m].
    pub dz_m: f64,
    /// Grid origin (south-west corner), degrees.
    pub lat0_deg: f64,
    pub lon0_deg: f64,
    /// Angular spacing, degrees.
    pub dlat_deg: f64,
    pub dlon_deg: f64,
    /// Asymmetry when `g_star` is empty.
    pub g_default: f64,
}

/// Geodetic-ish coordinates of an ECEF point on the transport sphere.
#[inline]
fn sphere_coords(p: Vec3) -> (f64 /*r*/, f64 /*lat deg*/, f64 /*lon deg*/) {
    let r = p.length();
    let lat = libm::asin((p.z / r).clamp(-1.0, 1.0)) / DEG;
    let lon = libm::atan2(p.y, p.x) / DEG;
    (r, lat, lon)
}

impl<'a> Cloud3DField<'a> {
    /// Total voxel count expected in `sigma`.
    #[inline]
    pub fn len(&self) -> usize {
        self.nz * self.nlat * self.nlon
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Top of the vertical grid [m].
    #[inline]
    pub fn z_top_m(&self) -> f64 {
        self.z0_m + self.dz_m * self.nz as f64
    }

    /// Grid indices for a position, or None when horizontally outside
    /// the footprint (vertically outside => None as well: no cloud).
    #[inline]
    fn indices(&self, r: f64, lat: f64, lon: f64) -> Option<(usize, usize, usize)> {
        let z = r - EARTH_RADIUS_M;
        if z < self.z0_m || z >= self.z_top_m() {
            return None;
        }
        let iz = ((z - self.z0_m) / self.dz_m) as usize;
        let flat = (lat - self.lat0_deg) / self.dlat_deg;
        // Wrap-safe longitude offset in [0, 360): points west of the
        // origin wrap to large values and fall outside the footprint.
        let dlon = rem_euclid(lon - self.lon0_deg, 360.0);
        let flon = dlon / self.dlon_deg;
        if flat < 0.0 || flon < 0.0 {
            return None;
        }
        let (ilat, ilon) = (flat as usize, flon as usize);
        if ilat >= self.nlat || ilon >= self.nlon {
            return None;
        }
        Some((iz.min(self.nz - 1), ilat, ilon))
    }

    /// Cloud scattering extinction [1/m] at an ECEF position. Outside
    /// the footprint: the background column; outside the z range: 0.
    #[inline]
    pub fn sigma_at(&self, p: Vec3) -> f64 {
        let (r, lat, lon) = sphere_coords(p);
        match self.indices(r, lat, lon) {
            Some((iz, ilat, ilon)) => {
                self.sigma[(iz * self.nlat + ilat) * self.nlon + ilon] as f64
            }
            None => {
                let z = r - EARTH_RADIUS_M;
                if z < self.z0_m || z >= self.z_top_m() || self.background_column.is_empty() {
                    0.0
                } else {
                    let iz = (((z - self.z0_m) / self.dz_m) as usize).min(self.nz - 1);
                    self.background_column[iz] as f64
                }
            }
        }
    }

    /// Asymmetry g* at a position (only meaningful where sigma > 0).
    #[inline]
    pub fn g_at(&self, p: Vec3) -> f64 {
        if self.g_star.is_empty() {
            return self.g_default;
        }
        let (r, lat, lon) = sphere_coords(p);
        match self.indices(r, lat, lon) {
            Some((iz, ilat, ilon)) => {
                self.g_star[(iz * self.nlat + ilat) * self.nlon + ilon] as f64
            }
            None => self.g_default,
        }
    }

    /// Exact cloud optical depth along the segment `p0 + t*dir` for
    /// t in [0, t_max] (dir unit). Piecewise-constant traversal: the
    /// integral is exact up to the cell-boundary root finding.
    ///
    /// Boundary families in (r, lat, lon):
    /// - radial: |p + t d| = r_k          (sphere; closed-form quadratic)
    /// - latitude: z(t)^2 = tan^2(phi) rho(t)^2  (cone; quadratic)
    /// - longitude: x(t) sin(lam) - y(t) cos(lam) = 0 (plane; linear)
    ///
    /// Implementation: step through the segment, at each point compute
    /// the distance to the NEAREST cell boundary of the three families
    /// ahead, accumulate sigma * step. Falls back to a bounded minimum
    /// step so degenerate geometry cannot stall the walk.
    pub fn tau_along(&self, p0: Vec3, dir: Vec3, t_max: f64) -> f64 {
        if t_max <= 0.0 || self.is_empty() {
            return 0.0;
        }
        let mut tau = 0.0f64;
        let mut t = 0.0f64;
        // Bounded iteration: a 2,000 km path through 250 m cells crosses
        // < 20k boundaries in pathological diagonals.
        for _ in 0..40_000 {
            if t >= t_max {
                break;
            }
            let p = p0 + dir * t;
            let step = self.distance_to_next_boundary(p, dir).max(self.min_step());
            let t_next = (t + step).min(t_max);
            let mid = p0 + dir * ((t + t_next) * 0.5);
            tau += self.sigma_at(mid) * (t_next - t);
            t = t_next;
        }
        tau
    }

    /// Same traversal, but returns the parameter t at which the
    /// accumulated cloud optical depth reaches `tau_target` (or None if
    /// the segment ends first). THE inverse of `tau_along`, sharing its
    /// stepping exactly: forced-collision sampling must invert the same
    /// function it normalized with, or the sample pdf will not match
    /// the weights.
    pub fn advance_to_tau(&self, p0: Vec3, dir: Vec3, t_max: f64, tau_target: f64) -> Option<f64> {
        if tau_target <= 0.0 {
            return Some(0.0);
        }
        let mut tau = 0.0f64;
        let mut t = 0.0f64;
        for _ in 0..40_000 {
            if t >= t_max {
                return None;
            }
            let p = p0 + dir * t;
            let step = self.distance_to_next_boundary(p, dir).max(self.min_step());
            let t_next = (t + step).min(t_max);
            let mid = p0 + dir * ((t + t_next) * 0.5);
            let sigma = self.sigma_at(mid);
            let dtau = sigma * (t_next - t);
            if tau + dtau >= tau_target {
                // Constant sigma within the cell: linear inversion.
                return Some(t + (tau_target - tau) / sigma.max(1e-300));
            }
            tau += dtau;
            t = t_next;
        }
        None
    }

    /// Minimum step: a fraction of the smallest cell dimension, so the
    /// boundary-distance fallback can never stall.
    #[inline]
    fn min_step(&self) -> f64 {
        let dz = self.dz_m;
        let dxy = self.dlat_deg * DEG * EARTH_RADIUS_M;
        (dz.min(dxy)) * 0.25
    }

    /// Distance along `dir` from `p` to the nearest grid-cell boundary
    /// of the three families (radial shells of the z-grid, latitude
    /// cones, longitude planes). Conservative: returning a smaller
    /// distance than the true boundary is always safe (the traversal
    /// just takes more steps); returning a larger one is not, so each
    /// family computes its exact crossing.
    fn distance_to_next_boundary(&self, p: Vec3, dir: Vec3) -> f64 {
        let (r, lat, lon) = sphere_coords(p);
        let mut best = f64::MAX;

        // ── Radial (sphere) crossings of the z-grid ──
        // Candidate windows span floor-1 ..= floor+2 in every family:
        // the walk lands EXACTLY on a boundary each step, and fp rounding
        // decides which side the landing index falls on. A {floor,
        // floor+1} window then sometimes contains only the just-crossed
        // boundary (root ~ 0, rejected) and misses the true next one,
        // silently merging cells (verified: exactly this halved a slant
        // integral when landings alternated parity).
        let z = r - EARTH_RADIUS_M;
        let iz = libm::floor((z - self.z0_m) / self.dz_m);
        for k in [iz - 1.0, iz, iz + 1.0, iz + 2.0] {
            let rk = EARTH_RADIUS_M + self.z0_m + k * self.dz_m;
            // |p + t d|^2 = rk^2
            let b = p.dot(dir);
            let c = r * r - rk * rk;
            let disc = b * b - c;
            if disc >= 0.0 {
                let s = libm::sqrt(disc);
                for t in [-b - s, -b + s] {
                    if t > 1e-6 && t < best {
                        best = t;
                    }
                }
            }
        }
        // ── Latitude (cone) crossings ──
        let flat = (lat - self.lat0_deg) / self.dlat_deg;
        let kf = libm::floor(flat);
        for k in [kf - 1.0, kf, kf + 1.0, kf + 2.0] {
            let phi = (self.lat0_deg + k * self.dlat_deg) * DEG;
            // cone: z(t)^2 = tan^2(phi) * (x(t)^2 + y(t)^2); phi=0 is the plane z=0
            let tp = libm::tan(phi);
            let a = dir.z * dir.z - tp * tp * (dir.x * dir.x + dir.y * dir.y);
            let b = p.z * dir.z - tp * tp * (p.x * dir.x + p.y * dir.y);
            let c = p.z * p.z - tp * tp * (p.x * p.x + p.y * p.y);
            if a.abs() > 1e-30 {
                let disc = b * b - a * c;
                if disc >= 0.0 {
                    let s = libm::sqrt(disc);
                    for t in [(-b - s) / a, (-b + s) / a] {
                        // Reject the mirror cone (opposite hemisphere).
                        if t > 1e-6 && t < best {
                            let zc = p.z + t * dir.z;
                            if zc * phi >= -1e-9 {
                                best = t;
                            }
                        }
                    }
                }
            } else if b.abs() > 1e-30 {
                let t = -c / (2.0 * b);
                if t > 1e-6 && t < best {
                    best = t;
                }
            }
        }

        // ── Longitude (meridian plane) crossings ──
        let flon = rem_euclid(lon - self.lon0_deg, 360.0) / self.dlon_deg;
        let kn = libm::floor(flon);
        for k in [kn - 1.0, kn, kn + 1.0, kn + 2.0] {
            let lam = (self.lon0_deg + k * self.dlon_deg) * DEG;
            let (sl, cl) = (libm::sin(lam), libm::cos(lam));
            // plane: x sin(lam) - y cos(lam) = 0
            let denom = dir.x * sl - dir.y * cl;
            if denom.abs() > 1e-30 {
                let t = -(p.x * sl - p.y * cl) / denom;
                if t > 1e-6 && t < best {
                    best = t;
                }
            }
        }

        best
    }
}


/// Cloud optical depth along a straight sub-segment, from the 3D field
/// when one is present (the field owns ALL cloud) or from the per-shell
/// 1D array otherwise. EVERY deterministic-leg consumer goes through
/// this function so the two representations cannot be double-counted.
#[inline]
pub fn cloud_tau_segment(
    atm: &crate::atmosphere::AtmosphereModel,
    field: Option<&Cloud3DField>,
    shell_idx: usize,
    pos: Vec3,
    dir: Vec3,
    dist: f64,
) -> f64 {
    match field {
        Some(f) => f.tau_along(pos, dir, dist),
        None => atm.cloud_extinction[shell_idx] * dist,
    }
}

/// Cloud extinction [1/m] at a point, from the 3D field when one is
/// present (the field owns ALL cloud) or from the per-shell 1D array
/// otherwise. Sibling of [`cloud_tau_segment`] for fixed-step LOS loops
/// where per-step midpoint sampling is the established quadrature.
#[inline]
pub fn cloud_ext_at(
    atm: &crate::atmosphere::AtmosphereModel,
    field: Option<&Cloud3DField>,
    shell_idx: usize,
    pos: Vec3,
) -> f64 {
    match field {
        Some(f) => f.sigma_at(pos),
        None => atm.cloud_extinction[shell_idx],
    }
}

/// Outcome of racing the gray cloud channel over a straight segment
/// `pos + t*dir`, `t` in `[0, seg_len]`, against a remaining cloud
/// optical-depth budget `tau_c_remaining`.
///
/// Decomposition tracking: a chain samples gas and cloud free flights as
/// two independent Poisson processes; the shorter wins. The cloud channel
/// is GRAY (wavelength flat) and (unlike the gas channel) carries no
/// exponential transform. Its collision distance is found by EXACT
/// inversion of the piecewise-constant cloud optical depth, so no majorant
/// rejection loop is needed.
#[derive(Clone, Copy, Debug)]
pub enum CloudFlight {
    /// The cloud collision fires inside the segment, at distance `dist`.
    Collide { dist: f64 },
    /// No cloud collision in the segment; `tau_consumed` cloud optical
    /// depth was crossed (subtract from the running budget and continue
    /// into the next shell segment with the SAME budget, no re-draw).
    Pass { tau_consumed: f64 },
}

/// Race the gray cloud channel over one straight in-shell segment.
///
/// `tau_c_remaining` is the cloud optical depth still to be consumed
/// before the next cloud collision (drawn once per free flight as
/// `-ln(1-u)` and carried, undiminished by gas events, across shell
/// crossings within that flight). Returns whether the cloud collision
/// lands in this segment and where, or how much cloud tau the segment
/// consumed without colliding.
///
/// Shares `tau_along` / `advance_to_tau` (field) or the analytic per-shell
/// extinction (1D fallback) so the in-field and out-of-field cloud models
/// are one model: in chain mode there is no `T_diff` anywhere on this
/// path, the cloud is delta-tracked by exact inversion exactly like the
/// field case.
#[inline]
pub fn cloud_flight_segment(
    atm: &crate::atmosphere::AtmosphereModel,
    field: Option<&Cloud3DField>,
    shell_idx: usize,
    pos: Vec3,
    dir: Vec3,
    seg_len: f64,
    tau_c_remaining: f64,
) -> CloudFlight {
    match field {
        Some(f) => match f.advance_to_tau(pos, dir, seg_len, tau_c_remaining) {
            Some(dist) => CloudFlight::Collide { dist },
            None => CloudFlight::Pass {
                tau_consumed: f.tau_along(pos, dir, seg_len),
            },
        },
        None => {
            let sigma_c = atm.cloud_extinction[shell_idx];
            if sigma_c <= 0.0 {
                return CloudFlight::Pass { tau_consumed: 0.0 };
            }
            let dist = tau_c_remaining / sigma_c;
            if dist <= seg_len {
                CloudFlight::Collide { dist }
            } else {
                CloudFlight::Pass {
                    tau_consumed: sigma_c * seg_len,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A uniform field over a wide footprint for analytic checks.
    fn uniform_field<'a>(
        sigma_store: &'a [f32],
        bg: &'a [f32],
    ) -> Cloud3DField<'a> {
        Cloud3DField {
            sigma: sigma_store,
            g_star: &[],
            background_column: bg,
            macrocell_max: &[],
            tile: 8,
            nz: 4,
            nlat: 10,
            nlon: 10,
            z0_m: 1000.0,
            dz_m: 500.0,
            lat0_deg: -5.0,
            lon0_deg: -5.0,
            dlat_deg: 1.0,
            dlon_deg: 1.0,
            g_default: 0.46,
        }
    }

    fn ecef(lat_deg: f64, lon_deg: f64, alt_m: f64) -> Vec3 {
        let r = EARTH_RADIUS_M + alt_m;
        let (la, lo) = (lat_deg * DEG, lon_deg * DEG);
        Vec3::new(
            r * libm::cos(la) * libm::cos(lo),
            r * libm::cos(la) * libm::sin(lo),
            r * libm::sin(la),
        )
    }

    #[test]
    fn vertical_tau_matches_analytic() {
        // Uniform sigma = 1e-4/m over z 1000..3000 m: vertical tau = 0.2.
        let store = [1e-4f32; 400];
        let f = uniform_field(&store, &[]);
        let p0 = ecef(0.0, 0.0, 0.0);
        let up = p0.normalize();
        let tau = f.tau_along(p0, up, 10_000.0);
        assert!(
            (tau - 0.2).abs() < 0.2 * 1e-3,
            "vertical tau {tau} vs 0.2"
        );
    }

    /// G1 (the Stage-0 geometry gate): a horizontally uniform field's
    /// SLANT tau must equal the 1D analytic slant tau through the same
    /// layer, at twilight-relevant grazing angles.
    #[test]
    fn g1_slant_tau_matches_1d_analytic() {
        let store = [2e-4f32; 400];
        // Background continues the same value: horizontally infinite.
        let bg = [2e-4f32; 4];
        let f = uniform_field(&store, &bg);
        let p0 = ecef(0.0, 0.0, 0.0);
        let up = p0.normalize();
        // East tangent at the origin.
        let east = Vec3::new(0.0, 1.0, 0.0);
        for zen_deg in [0.0, 60.0, 80.0, 87.0] {
            let z = zen_deg * DEG;
            let dir = (up.scale(libm::cos(z)) + east.scale(libm::sin(z))).normalize();
            let tau = f.tau_along(p0, dir, 2_000_000.0);
            // 1D analytic slant tau through a spherical layer
            // [z0, z1]: sum over the chord lengths.
            let analytic = {
                let r0 = EARTH_RADIUS_M;
                let b = r0 * libm::sin(z); // impact parameter
                let chord = |radius: f64| -> f64 {
                    libm::sqrt((radius * radius - b * b).max(0.0))
                };
                let zlo = EARTH_RADIUS_M + 1000.0;
                let zhi = EARTH_RADIUS_M + 3000.0;
                let t_in = chord(zlo) - chord(r0).min(chord(zlo));
                let _ = t_in;
                // Observer at the surface looking up: path enters the
                // layer at radius zlo and exits at zhi (ascending ray):
                let l = chord(zhi) - chord(zlo);
                2e-4 * l
            };
            let rel = (tau - analytic).abs() / analytic;
            assert!(
                rel < 5e-3,
                "zen {zen_deg}: 3D tau {tau:.6} vs analytic {analytic:.6} (rel {rel:.4})"
            );
        }
    }

    #[test]
    fn advance_inverts_tau_along() {
        let store = [5e-4f32; 400];
        let f = uniform_field(&store, &[]);
        let p0 = ecef(0.0, 0.0, 500.0);
        let up = p0.normalize();
        let dir = (up.scale(0.3) + Vec3::new(0.0, 1.0, 0.0).scale(0.954)).normalize();
        let t_max = 50_000.0;
        let total = f.tau_along(p0, dir, t_max);
        for frac in [0.25, 0.5, 0.9] {
            let target = total * frac;
            let t = f.advance_to_tau(p0, dir, t_max, target).expect("inverts");
            let back = f.tau_along(p0, dir, t);
            assert!(
                (back - target).abs() < total * 1e-3,
                "frac {frac}: tau(advance(target)) {back} vs {target}"
            );
        }
        assert!(f.advance_to_tau(p0, dir, t_max, total * 1.5).is_none());
    }

    #[test]
    fn outside_footprint_uses_background() {
        let store = [3e-4f32; 400];
        let bg = [1e-4f32; 4];
        let f = uniform_field(&store, &bg);
        // 40 degrees east: far outside the 10x10 deg footprint.
        let p_out = ecef(0.0, 40.0, 1500.0);
        assert!((f.sigma_at(p_out) - 1e-4).abs() < 1e-9);
        let p_in = ecef(0.0, 0.0, 1500.0);
        assert!((f.sigma_at(p_in) - 3e-4).abs() < 1e-9);
        // Above the grid: clear.
        assert_eq!(f.sigma_at(ecef(0.0, 0.0, 5000.0)), 0.0);
    }
}
