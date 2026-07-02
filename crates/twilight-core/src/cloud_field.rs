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
    /// Implementation: repeated [`Self::next_segment`] calls, each
    /// yielding one constant-sigma span (a fine voxel span, a whole
    /// empty macro-tile, or an out-of-footprint stretch of one radial
    /// band), with tau accumulating sigma * length. `advance_to_tau`
    /// walks the SAME segments, so its inversion is exact by
    /// construction.
    pub fn tau_along(&self, p0: Vec3, dir: Vec3, t_max: f64) -> f64 {
        self.tau_along_counted(p0, dir, t_max).0
    }

    /// `tau_along` plus the number of traversal segments taken: the
    /// step count is the acceleration observable. Tests assert it stays
    /// far below fine-stepping counts on mostly-empty fields, so the
    /// empty-tile skip cannot be silently disabled by a future edit.
    fn tau_along_counted(&self, p0: Vec3, dir: Vec3, t_max: f64) -> (f64, u32) {
        if t_max <= 0.0 || self.is_empty() {
            return (0.0, 0);
        }
        let mut tau = 0.0f64;
        let mut t = 0.0f64;
        let mut steps = 0u32;
        let min_step = self.min_step();
        let has_macro = !self.macrocell_max.is_empty();
        // Bounded iteration: a 2,000 km path through 250 m cells crosses
        // < 20k boundaries in pathological diagonals. Empty macro-tiles
        // and out-of-footprint stretches are crossed in far fewer,
        // coarser steps: a deep-twilight ray spends ~95% of its length
        // in clear air.
        for _ in 0..40_000 {
            if t >= t_max {
                break;
            }
            let (t_next, sigma) = self.next_segment(p0, dir, t, t_max, min_step, has_macro);
            tau += sigma * (t_next - t);
            t = t_next;
            steps += 1;
        }
        (tau, steps)
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
        let min_step = self.min_step();
        let has_macro = !self.macrocell_max.is_empty();
        for _ in 0..40_000 {
            if t >= t_max {
                return None;
            }
            let (t_next, sigma) = self.next_segment(p0, dir, t, t_max, min_step, has_macro);
            let dtau = sigma * (t_next - t);
            if tau + dtau >= tau_target {
                // Constant sigma within the segment: linear inversion.
                // (sigma > 0 whenever this fires: zero-sigma segments
                // cannot lift tau, which stays strictly below the
                // target until a positive-sigma segment reaches it.)
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

    /// One traversal segment starting at parameter `t`: returns
    /// `(t_next, sigma)` where sigma is constant over `[t, t_next]` (up
    /// to the bounded min_step flooring described below), so `tau_along`
    /// accumulates `sigma * (t_next - t)` and `advance_to_tau` inverts
    /// linearly inside one segment. Both walk through THIS one function:
    /// forced-collision sampling must invert exactly the integral it was
    /// normalized with.
    ///
    /// Algorithm (the port reference for the GPU field DDA):
    ///
    /// 1. FINE CANDIDATE. `d_f = max(distance_to_next_boundary,
    ///    min_step)`, `t_f = min(t + d_f, t_max)`. A fine segment lies
    ///    between adjacent boundaries of the full lattice (fine radial
    ///    shells, all latitude cones, all meridian planes, which
    ///    contains every footprint surface), so it sits inside ONE
    ///    voxel, ONE macro-tile, ONE z level, and on ONE side of the
    ///    footprint edge. Fine segments therefore never straddle the
    ///    inside/outside handover and need no extra cap.
    /// 2. Without majorant data, integrate the fine segment with sigma
    ///    sampled at its midpoint (pre-macro behavior, unchanged).
    /// 3. MIDPOINT CLASSIFICATION. `maj_f = macro_majorant_at(mid)` at
    ///    the midpoint of `[t, t_f]`. The midpoint identifies the
    ///    segment's tile unambiguously. The segment ENDPOINTS land
    ///    exactly on boundaries, where fp rounding parity can index the
    ///    just-left tile: classifying at the landing point made an
    ///    empty-tile skip re-fire from an empty-to-occupied boundary and
    ///    drop the whole occupied chord (same failure class as the
    ///    Stage-0 halved-integral bug). Nothing here may classify at a
    ///    segment endpoint.
    /// 4. `maj_f > 0` (occupied tile): integrate the fine segment.
    /// 5. `maj_f <= 0` (empty tile, or outside the footprint/z range):
    ///    COARSE EXTENSION. `d_c = max(min(distance_to_next_tile_boundary,
    ///    distance_to_footprint_boundary), min_step)`, `t_c = min(t +
    ///    d_c, t_max)`, and `maj_c = macro_majorant_at` at the midpoint
    ///    of `[t, t_c]`. The footprint cap guarantees no coarse segment
    ///    straddles the inside/outside handover: an uncapped radial step
    ///    misintegrated grazing chords that cross the footprint edge
    ///    mid-segment, and an uncapped tile skip overshot the edge when
    ///    nlat/nlon is not a multiple of `tile` (the edge is then not on
    ///    the tile lattice), dropping background tau. The tile candidate
    ///    keeps the FINE radial lattice, so a coarse segment also spans
    ///    exactly one z level.
    ///    - `maj_f == 0 && maj_c == 0`: the coarse segment lies in one
    ///      provably empty tile at one z level: skip it, sigma = 0.
    ///    - `maj_f < 0 && maj_c < 0`: the segment is outside the
    ///      footprint (or the z range), where sigma is altitude-only;
    ///      it crosses no radial shell and no footprint surface, so
    ///      sigma is constant: sample it at the coarse midpoint.
    ///    - Any disagreement (reachable only through min_step flooring
    ///      near tangencies/corners or fp-degenerate landings): fall
    ///      back to the fine segment classified by ITS midpoint, which
    ///      is always valid.
    ///
    /// min_step flooring: every candidate is floored to min_step (a
    /// quarter of the smallest cell) so degenerate root geometry cannot
    /// stall the walk. A floored step can overshoot a boundary by less
    /// than min_step; the midpoint then decides the whole sliver, a
    /// quadrature error bounded by sigma_max * min_step per tangency
    /// event. It can never skip a full cell: a chord longer than
    /// min_step gets its own midpoint-classified segments, and shorter
    /// corner clips are themselves sub-min_step slivers.
    #[inline]
    fn next_segment(
        &self,
        p0: Vec3,
        dir: Vec3,
        t: f64,
        t_max: f64,
        min_step: f64,
        has_macro: bool,
    ) -> (f64, f64) {
        let p = p0 + dir * t;
        let d_fine = self.distance_to_next_boundary(p, dir).max(min_step);
        let t_fine = (t + d_fine).min(t_max);
        let mid_fine = p0 + dir * ((t + t_fine) * 0.5);
        if !has_macro {
            return (t_fine, self.sigma_at(mid_fine));
        }
        let maj_f = self.macro_majorant_at(mid_fine);
        if maj_f > 0.0 {
            // Occupied tile: integrate finely within it.
            return (t_fine, self.sigma_at(mid_fine));
        }
        // Empty tile or outside the footprint: try the coarse extension,
        // capped by the footprint surfaces so the segment cannot
        // straddle the inside/outside handover.
        let d_fp = self.distance_to_footprint_boundary(p, dir);
        let d_coarse = self
            .distance_to_next_tile_boundary(p, dir)
            .min(d_fp)
            .max(min_step);
        let t_coarse = (t + d_coarse).min(t_max);
        let mid_coarse = p0 + dir * ((t + t_coarse) * 0.5);
        let maj_c = self.macro_majorant_at(mid_coarse);
        if maj_f == 0.0 && maj_c == 0.0 {
            // Provably empty tile: cross it in one step, tau += 0.
            return (t_coarse, 0.0);
        }
        if maj_f < 0.0 && maj_c < 0.0 {
            // Outside the footprint (or z range): sigma is altitude-only
            // and constant over the capped coarse segment.
            return (t_coarse, self.sigma_at(mid_coarse));
        }
        // Fine/coarse classification disagreement (min_step flooring
        // degeneracy): the fine segment with its own midpoint is always
        // a valid constant-sigma span.
        (t_fine, self.sigma_at(mid_fine))
    }

    /// Distance along `dir` from `p` to the nearest crossing of one of
    /// the SIX bounding surfaces of the voxel grid: the `lat0` and
    /// `lat0 + nlat*dlat` cones, the `lon0` and `lon0 + nlon*dlon`
    /// meridian planes, and the spheres at `z0` and `z_top`. Caps every
    /// coarse step (empty-tile skip and out-of-footprint segment) so no
    /// coarse segment straddles the footprint edge or the z0/z_top
    /// handover. All six surfaces are fixed (absolute indices), so no
    /// floor-window is needed: every root of every surface is a
    /// candidate, and the ~0 root of a surface the walk just landed on
    /// is rejected by the same `t > 1e-6` guard as the lattice
    /// functions. The meridian planes are full great-circle planes
    /// (both halves): a crossing of the antipodal half just splits a
    /// segment in two, which is always safe.
    fn distance_to_footprint_boundary(&self, p: Vec3, dir: Vec3) -> f64 {
        let mut best = f64::MAX;

        // Spheres at z0 and z_top.
        let r = p.length();
        let b = p.dot(dir);
        for zk in [self.z0_m, self.z_top_m()] {
            let rk = EARTH_RADIUS_M + zk;
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
        // Latitude cones at lat0 and lat0 + nlat*dlat.
        for lat_k in [
            self.lat0_deg,
            self.lat0_deg + self.nlat as f64 * self.dlat_deg,
        ] {
            let phi = lat_k * DEG;
            let tp = libm::tan(phi);
            let a = dir.z * dir.z - tp * tp * (dir.x * dir.x + dir.y * dir.y);
            let bq = p.z * dir.z - tp * tp * (p.x * dir.x + p.y * dir.y);
            let c = p.z * p.z - tp * tp * (p.x * p.x + p.y * p.y);
            if a.abs() > 1e-30 {
                let disc = bq * bq - a * c;
                if disc >= 0.0 {
                    let s = libm::sqrt(disc);
                    for t in [(-bq - s) / a, (-bq + s) / a] {
                        // Reject the mirror cone (opposite hemisphere).
                        if t > 1e-6 && t < best {
                            let zc = p.z + t * dir.z;
                            if zc * phi >= -1e-9 {
                                best = t;
                            }
                        }
                    }
                }
            } else if bq.abs() > 1e-30 {
                let t = -c / (2.0 * bq);
                if t > 1e-6 && t < best {
                    best = t;
                }
            }
        }
        // Meridian planes at lon0 and lon0 + nlon*dlon.
        for lon_k in [
            self.lon0_deg,
            self.lon0_deg + self.nlon as f64 * self.dlon_deg,
        ] {
            let lam = lon_k * DEG;
            let (sl, cl) = (libm::sin(lam), libm::cos(lam));
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

    /// Max sigma over the macro-tile containing `p`. Returns 0.0 for a
    /// provably EMPTY tile (the whole tile contributes no cloud tau, so it
    /// is crossed in one coarse step), a positive value for an OCCUPIED
    /// tile (step finely), and -1.0 when there is no tile data or `p` is
    /// outside the footprint / z range (sigma is then altitude-only, so
    /// a footprint-capped coarse segment has constant sigma). Only ever
    /// evaluated at SEGMENT MIDPOINTS by the traversal (see
    /// `next_segment`): landing points sit exactly on boundaries where
    /// fp parity picks an arbitrary side. Mirrors the GPU
    /// `field_macro_majorant_at` so the two backends skip empty space
    /// identically, leaving the accumulated tau unchanged (empty cells
    /// contribute zero either way).
    #[inline]
    fn macro_majorant_at(&self, p: Vec3) -> f64 {
        if self.macrocell_max.is_empty() {
            return -1.0;
        }
        let (r, lat, lon) = sphere_coords(p);
        match self.indices(r, lat, lon) {
            Some((iz, ilat, ilon)) => {
                let ntlat = self.nlat.div_ceil(self.tile);
                let ntlon = self.nlon.div_ceil(self.tile);
                self.macrocell_max[(iz * ntlat + ilat / self.tile) * ntlon + ilon / self.tile]
                    as f64
            }
            None => -1.0,
        }
    }

    /// Distance to the nearest COARSE (macro-tile) boundary: the z-grid
    /// stays fine (sigma varies per level), lat/lon crossings use the tile
    /// spacing. Lets a provably empty tile be crossed in one step.
    /// Conservative (a smaller distance is always safe), so the
    /// floor-1..=floor+2 candidate window is kept. Mirrors the GPU
    /// `field_distance_to_next_tile_boundary`.
    fn distance_to_next_tile_boundary(&self, p: Vec3, dir: Vec3) -> f64 {
        let (r, lat, lon) = sphere_coords(p);
        let mut best = f64::MAX;
        let dlat_t = self.dlat_deg * self.tile as f64;
        let dlon_t = self.dlon_deg * self.tile as f64;

        // Radial: fine z-grid (sigma varies per level).
        let z = r - EARTH_RADIUS_M;
        let iz = libm::floor((z - self.z0_m) / self.dz_m);
        let b_r = p.dot(dir);
        for k in [iz - 1.0, iz, iz + 1.0, iz + 2.0] {
            let rk = EARTH_RADIUS_M + self.z0_m + k * self.dz_m;
            let c = r * r - rk * rk;
            let disc = b_r * b_r - c;
            if disc >= 0.0 {
                let s = libm::sqrt(disc);
                for t in [-b_r - s, -b_r + s] {
                    if t > 1e-6 && t < best {
                        best = t;
                    }
                }
            }
        }
        // Latitude cones at tile spacing.
        let flat = (lat - self.lat0_deg) / dlat_t;
        let kf = libm::floor(flat);
        for k in [kf - 1.0, kf, kf + 1.0, kf + 2.0] {
            let phi = (self.lat0_deg + k * dlat_t) * DEG;
            let tp = libm::tan(phi);
            let a = dir.z * dir.z - tp * tp * (dir.x * dir.x + dir.y * dir.y);
            let b = p.z * dir.z - tp * tp * (p.x * dir.x + p.y * dir.y);
            let c = p.z * p.z - tp * tp * (p.x * p.x + p.y * p.y);
            if a.abs() > 1e-30 {
                let disc = b * b - a * c;
                if disc >= 0.0 {
                    let s = libm::sqrt(disc);
                    for t in [(-b - s) / a, (-b + s) / a] {
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
        // Longitude planes at tile spacing.
        let flon = rem_euclid(lon - self.lon0_deg, 360.0) / dlon_t;
        let kn = libm::floor(flon);
        for k in [kn - 1.0, kn, kn + 1.0, kn + 2.0] {
            let lam = (self.lon0_deg + k * dlon_t) * DEG;
            let (sl, cl) = (libm::sin(lam), libm::cos(lam));
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
    // The crate is no_std; the test harness links std anyway, so bind
    // it here for println/Instant in the diagnostics below.
    extern crate std;

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

    // ---- 3D DDA geometry gates: checkerboard macro-tile harness ----
    //
    // nlat = nlon = 27 with tile 8 is deliberate: 27 is NOT a multiple
    // of 8, so the last tile row/column is partial and the footprint
    // edge does NOT lie on the tile lattice (BUG 3's precondition).
    // Alternating empty/occupied tiles maximize empty-to-occupied
    // boundary landings (BUG 1's precondition).
    const CB_NZ: usize = 4;
    const CB_N: usize = 27;
    const CB_TILE: usize = 8;
    const CB_NT: usize = 4; // ceil(27 / 8)
    const CB_SIGMA: f64 = 5e-4;
    const CB_BG: f32 = 1.3e-4;

    /// Majorant derivation, same as the twilight-data builder.
    fn derive_majorants(sigma: &[f32], mm: &mut [f32]) {
        for v in mm.iter_mut() {
            *v = 0.0;
        }
        for iz in 0..CB_NZ {
            for ilat in 0..CB_N {
                for ilon in 0..CB_N {
                    let v = sigma[(iz * CB_N + ilat) * CB_N + ilon];
                    let m =
                        &mut mm[(iz * CB_NT + ilat / CB_TILE) * CB_NT + ilon / CB_TILE];
                    if v > *m {
                        *m = v;
                    }
                }
            }
        }
    }

    fn fill_checkerboard(sigma: &mut [f32], mm: &mut [f32]) {
        for v in sigma.iter_mut() {
            *v = 0.0;
        }
        for iz in 0..CB_NZ {
            for ilat in 0..CB_N {
                for ilon in 0..CB_N {
                    if (ilat / CB_TILE + ilon / CB_TILE).is_multiple_of(2) {
                        sigma[(iz * CB_N + ilat) * CB_N + ilon] = CB_SIGMA as f32;
                    }
                }
            }
        }
        derive_majorants(sigma, mm);
    }

    /// z 1000..3000 m, lat/lon [-0.27, 0.27] deg in 0.02 deg (~2.2 km)
    /// cells; tile planes every 0.16 deg; min_step = 125 m (dz / 4).
    fn cb_field<'a>(sigma: &'a [f32], mm: &'a [f32], bg: &'a [f32]) -> Cloud3DField<'a> {
        Cloud3DField {
            sigma,
            g_star: &[],
            background_column: bg,
            macrocell_max: mm,
            tile: CB_TILE,
            nz: CB_NZ,
            nlat: CB_N,
            nlon: CB_N,
            z0_m: 1000.0,
            dz_m: 500.0,
            lat0_deg: -0.27,
            lon0_deg: -0.27,
            dlat_deg: 0.02,
            dlon_deg: 0.02,
            g_default: 0.46,
        }
    }

    fn east(p: Vec3) -> Vec3 {
        Vec3::new(-p.y, p.x, 0.0).normalize()
    }

    fn north(p: Vec3) -> Vec3 {
        p.normalize().cross(east(p))
    }

    /// Brute-force referee, independent of ALL DDA machinery: midpoint
    /// Riemann sum over `sigma_at`. 0.25 m steps keep the boundary
    /// misassignment noise (~sigma * step per cell crossing) far below
    /// the 1e-4 relative gate on every ray in the fan (1 m steps were
    /// marginal against 1e-4 on low-tau grazers).
    fn brute_tau(f: &Cloud3DField, p0: Vec3, dir: Vec3, t_max: f64) -> f64 {
        let n = (t_max / 0.25) as usize;
        let dt = t_max / n as f64;
        let mut tau = 0.0f64;
        for i in 0..n {
            tau += f.sigma_at(p0 + dir * ((i as f64 + 0.5) * dt)) * dt;
        }
        tau
    }

    /// The referee ray fan: axis-aligned, diagonal, near-grazing, and
    /// footprint-crossing geometries.
    fn cb_ray_fan() -> [(&'static str, Vec3, Vec3, f64); 7] {
        let p1 = ecef(0.0, -0.26, 1250.0);
        let p2 = ecef(-0.26, 0.005, 1250.0);
        let p3 = ecef(-0.252, -0.252, 1100.0);
        let p4 = ecef(0.004, -0.26, 1050.0);
        let z4 = 89.5 * DEG;
        let d4 =
            (p4.normalize().scale(libm::cos(z4)) + east(p4).scale(libm::sin(z4))).normalize();
        let p5 = ecef(0.01, -0.60, 1250.0);
        let p6 = ecef(0.0, -0.10, 5000.0);
        let d6 = (east(p6) - p6.normalize().scale(0.08)).normalize();
        let p7 = ecef(-0.02, -0.05, 0.0);
        let d7 = (p7.normalize().scale(0.5) + east(p7).scale(libm::sqrt(0.75))).normalize();
        [
            ("east along lon", p1, east(p1), 80_000.0),
            ("north along lat", p2, north(p2), 80_000.0),
            ("diagonal", p3, (east(p3) + north(p3)).normalize(), 100_000.0),
            ("grazing zen 89.5", p4, d4, 200_000.0),
            ("lateral entry from outside", p5, east(p5), 150_000.0),
            ("entry from above z_top", p6, d6, 80_000.0),
            ("from below through z0", p7, d7, 10_000.0),
        ]
    }

    /// (a) Brute-force referee on the checkerboard: the DDA must match
    /// a fixed-step integration of the SAME `sigma_at` to < 1e-4
    /// relative, and macro-skipping must match pure fine stepping to
    /// < 1e-9 relative (the skip is exact, not approximate).
    #[test]
    fn checkerboard_tau_matches_brute_force() {
        let mut sigma = [0f32; CB_NZ * CB_N * CB_N];
        let mut mm = [0f32; CB_NZ * CB_NT * CB_NT];
        fill_checkerboard(&mut sigma, &mut mm);
        let bg = [CB_BG; CB_NZ];
        let f = cb_field(&sigma, &mm, &bg);
        let f_fine = Cloud3DField {
            macrocell_max: &[],
            ..f
        };
        for (label, p0, dir, t_max) in cb_ray_fan() {
            let reference = brute_tau(&f, p0, dir, t_max);
            let tau = f.tau_along(p0, dir, t_max);
            let rel = (tau - reference).abs() / reference.max(1e-12);
            std::println!(
                "  [referee] {label}: dda {tau:.6} brute {reference:.6} rel {rel:.2e}"
            );
            assert!(
                rel < 1e-4,
                "{label}: dda {tau:.8} vs brute {reference:.8} (rel {rel:.3e})"
            );
            let tau_fine = f_fine.tau_along(p0, dir, t_max);
            let rel_fine = (tau - tau_fine).abs() / tau_fine.max(1e-12);
            assert!(
                rel_fine < 1e-9,
                "{label}: macro-skip {tau:.12} vs fine {tau_fine:.12} (rel {rel_fine:.3e})"
            );
        }
    }

    /// (b) Inverse property on the checkerboard: advance_to_tau must be
    /// THE inverse of tau_along (they share next_segment), and must
    /// return None exactly when the total tau falls short of the target.
    #[test]
    fn checkerboard_advance_inverts_tau_along() {
        let mut sigma = [0f32; CB_NZ * CB_N * CB_N];
        let mut mm = [0f32; CB_NZ * CB_NT * CB_NT];
        fill_checkerboard(&mut sigma, &mut mm);
        let bg = [CB_BG; CB_NZ];
        let f = cb_field(&sigma, &mm, &bg);
        for (label, p0, dir, t_max) in cb_ray_fan() {
            let total = f.tau_along(p0, dir, t_max);
            assert!(total > 0.05, "{label}: fan ray must see cloud (tau {total})");
            for frac in [0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 0.999] {
                let target = total * frac;
                let t_hit = f
                    .advance_to_tau(p0, dir, t_max, target)
                    .unwrap_or_else(|| panic!("{label} frac {frac}: must collide"));
                let back = f.tau_along(p0, dir, t_hit);
                assert!(
                    (back - target).abs() < 1e-9,
                    "{label} frac {frac}: tau(advance) {back:.12} vs target {target:.12}"
                );
            }
            assert!(
                f.advance_to_tau(p0, dir, t_max, total + 1e-6).is_none(),
                "{label}: a target above the total tau must be None"
            );
        }
    }

    /// (c) BUG 1 regression: coarse-skip landings exactly on
    /// empty-to-occupied tile boundaries. Pre-fix, macro_majorant_at at
    /// the LANDING point could index the just-left empty tile (fp
    /// rounding parity), and distance_to_next_tile_boundary then
    /// rejected the ~0 root and returned the FAR side of the occupied
    /// tile: the whole occupied chord was skipped with tau += 0.
    #[test]
    fn bug1_empty_to_occupied_boundary_landing_keeps_chord() {
        let mut sigma = [0f32; CB_NZ * CB_N * CB_N];
        let mut mm = [0f32; CB_NZ * CB_NT * CB_NT];
        fill_checkerboard(&mut sigma, &mut mm);
        let bg = [CB_BG; CB_NZ];
        let f = cb_field(&sigma, &mm, &bg);
        // Equator row (tile row 1): tile col 0 (lon -0.27..-0.11) is
        // empty, col 1 (-0.11..0.05) occupied. Starting mid tile 0, the
        // first skip LANDS exactly on the -0.11 tile plane; the eps fan
        // additionally pins both fp parities of a start exactly on it.
        for lon_start in [-0.19, -0.11 - 1e-9, -0.11, -0.11 + 1e-9] {
            let p0 = ecef(0.0, lon_start, 1250.0);
            let dir = east(p0);
            let t_max = 40_000.0;
            let reference = brute_tau(&f, p0, dir, t_max);
            let tau = f.tau_along(p0, dir, t_max);
            let rel = (tau - reference).abs() / reference.max(1e-12);
            std::println!(
                "  [bug1] start lon {lon_start}: dda {tau:.6} brute {reference:.6} rel {rel:.2e}"
            );
            assert!(
                rel < 1e-4,
                "start lon {lon_start}: dda {tau:.8} vs brute {reference:.8} (rel {rel:.3e})"
            );
        }
        // And the occupied chord is FULLY counted, not halved/dropped:
        // tile col 1 spans 0.16 deg (~17.8 km) at sigma 5e-4.
        let p0 = ecef(0.0, -0.19, 1250.0);
        let tau = f.tau_along(p0, east(p0), 40_000.0);
        assert!(
            tau > 0.9 * CB_SIGMA * 16_000.0,
            "occupied tile chord dropped at a boundary landing: tau {tau:.4}"
        );
    }

    /// (d) BUG 2 regression: chords crossing the footprint boundary.
    /// Pre-fix, the outside-footprint walk stepped RADIALLY only; at
    /// 1250 m altitude the next radial shell is ~56 km out for a
    /// grazing ray, so one radial segment crossed the lateral footprint
    /// edge (or the z_top handover) mid-segment and integrated a single
    /// midpoint sigma across the mix. Exactly the sun-shadow geometry
    /// that matters at twilight.
    #[test]
    fn bug2_footprint_crossing_chords_match_brute_force() {
        let mut sigma = [0f32; CB_NZ * CB_N * CB_N];
        let mut mm = [0f32; CB_NZ * CB_NT * CB_NT];
        fill_checkerboard(&mut sigma, &mut mm);
        let bg = [CB_BG; CB_NZ];
        let f = cb_field(&sigma, &mm, &bg);
        let p5 = ecef(0.01, -0.60, 1250.0);
        let p6 = ecef(0.0, -0.10, 5000.0);
        let d6 = (east(p6) - p6.normalize().scale(0.08)).normalize();
        let cases = [
            ("grazing lateral entry", p5, east(p5), 150_000.0),
            ("descent through z_top", p6, d6, 80_000.0),
        ];
        for (label, p0, dir, t_max) in cases {
            let reference = brute_tau(&f, p0, dir, t_max);
            let tau = f.tau_along(p0, dir, t_max);
            let rel = (tau - reference).abs() / reference.max(1e-12);
            std::println!("  [bug2] {label}: dda {tau:.6} brute {reference:.6} rel {rel:.2e}");
            assert!(
                rel < 1e-4,
                "{label}: dda {tau:.8} vs brute {reference:.8} (rel {rel:.3e})"
            );
        }
    }

    /// BUG 3 regression: 27 is not a multiple of tile 8, so the east
    /// footprint edge (lon 0.27) is NOT on the tile lattice (next tile
    /// plane: 0.37). Pre-fix, a skip inside the partial EMPTY edge tile
    /// overshot the footprint edge and dropped the nonzero background
    /// tau across the overshoot.
    #[test]
    fn bug3_partial_tile_skip_respects_footprint_edge() {
        let mut sigma = [0f32; CB_NZ * CB_N * CB_N];
        let mut mm = [0f32; CB_NZ * CB_NT * CB_NT];
        fill_checkerboard(&mut sigma, &mut mm);
        let bg = [CB_BG; CB_NZ];
        let f = cb_field(&sigma, &mm, &bg);
        // Tile row 2 (ilat 16..24), partial tile col 3 (ilon 24..27,
        // lon 0.21..0.27) is EMPTY there: start inside it, heading east
        // out through the edge into the nonzero background.
        let p0 = ecef(0.132, 0.22, 1250.0);
        let dir = east(p0);
        let t_max = 60_000.0;
        let reference = brute_tau(&f, p0, dir, t_max);
        let tau = f.tau_along(p0, dir, t_max);
        let rel = (tau - reference).abs() / reference.max(1e-12);
        std::println!("  [bug3] dda {tau:.6} brute {reference:.6} rel {rel:.2e}");
        assert!(
            rel < 1e-4,
            "background tau beyond the partial-tile edge dropped: dda {tau:.8} vs brute {reference:.8} (rel {rel:.3e})"
        );
    }

    /// min_step flooring near tangencies: rays whose boundary crossings
    /// cluster tighter than min_step (125 m here) force repeated
    /// flooring. The DDA may misassign a sub-min_step SLIVER (bounded
    /// by |dsigma| * min_step = 6e-4 * 125 = 0.075 per event, the
    /// documented compromise in next_segment), but must never jump an
    /// occupied CELL, whose tau is orders of magnitude larger (>= 0.4
    /// vertically, >= 2 on any grazing chord).
    #[test]
    fn min_step_flooring_cannot_skip_cells_at_tangency() {
        // Horizontally uniform, vertically CONTRASTING levels so a
        // misassigned radial sliver actually changes tau; majorants all
        // positive, so the fine path (with its midpoint probes) runs.
        let lv = [3e-4f32, 8e-4, 2e-4, 4e-4];
        let mut sigma = [0f32; CB_NZ * CB_N * CB_N];
        for iz in 0..CB_NZ {
            for i in 0..CB_N * CB_N {
                sigma[iz * CB_N * CB_N + i] = lv[iz];
            }
        }
        let mut mm = [0f32; CB_NZ * CB_NT * CB_NT];
        derive_majorants(&sigma, &mut mm);
        let bg = [0f32; CB_NZ];
        let f = cb_field(&sigma, &mm, &bg);
        // (i) start at a grazing contact 0.2 mm below the z = 2000 m
        // shell: the crossing ahead is ~50 m out, under min_step.
        let pa = ecef(0.0, -0.10, 2000.0 - 2e-4);
        let da = east(pa);
        // (ii) dip 0.1 mm through the shell and back within ~71 m: two
        // crossings inside one floored step.
        let pb = ecef(0.0, -0.10, 2000.0 + 1e-4);
        let alpha = libm::sqrt(4e-4 / pb.length());
        let db = (east(pb).scale(libm::cos(alpha)) - pb.normalize().scale(libm::sin(alpha)))
            .normalize();
        for (label, p0, dir) in [("graze from below", pa, da), ("dip through", pb, db)] {
            let t_max = 60_000.0;
            let reference = brute_tau(&f, p0, dir, t_max);
            let tau = f.tau_along(p0, dir, t_max);
            let diff = (tau - reference).abs();
            std::println!(
                "  [tangency] {label}: dda {tau:.6} brute {reference:.6} diff {diff:.4}"
            );
            assert!(
                diff < 0.08,
                "{label}: flooring error {diff:.4} exceeds the sliver bound (cell skipped?)"
            );
        }
    }

    /// The fix must not silently disable coarse skipping: on a mostly
    /// empty field the macro walk must take FAR fewer steps than pure
    /// fine stepping while agreeing to fp accuracy.
    #[test]
    fn empty_tile_skip_still_engages() {
        // Single occupied tile (row 1, col 1), no background.
        let mut sigma = [0f32; CB_NZ * CB_N * CB_N];
        for iz in 0..CB_NZ {
            for ilat in 8..16 {
                for ilon in 8..16 {
                    sigma[(iz * CB_N + ilat) * CB_N + ilon] = CB_SIGMA as f32;
                }
            }
        }
        let mut mm = [0f32; CB_NZ * CB_NT * CB_NT];
        derive_majorants(&sigma, &mut mm);
        let bg = [0f32; CB_NZ];
        let f = cb_field(&sigma, &mm, &bg);
        let f_fine = Cloud3DField {
            macrocell_max: &[],
            ..f
        };
        let p0 = ecef(0.0, -0.26, 1250.0);
        let dir = east(p0);
        let t_max = 300_000.0;
        let (tau_m, steps_m) = f.tau_along_counted(p0, dir, t_max);
        let (tau_f, steps_f) = f_fine.tau_along_counted(p0, dir, t_max);
        std::println!(
            "  [skip] macro: tau {tau_m:.6} in {steps_m} steps; fine: tau {tau_f:.6} in {steps_f} steps"
        );
        assert!(
            (tau_m - tau_f).abs() <= tau_f * 1e-9,
            "skip changed tau: {tau_m:.12} vs {tau_f:.12}"
        );
        assert!(
            steps_m * 3 < steps_f,
            "empty-tile skip no longer engages: {steps_m} macro vs {steps_f} fine steps"
        );
        // The occupied chord itself is fully integrated.
        assert!(
            tau_m > 0.9 * CB_SIGMA * 16_000.0,
            "occupied chord lost: {tau_m:.4}"
        );
    }

    /// Timing diagnostic (not a gate): pathological checkerboard,
    /// macro-skipping vs pure fine stepping, per fan ray. Run with:
    /// `cargo test -p twilight-core --release --
    ///  --ignored --nocapture checkerboard_skip_timing`.
    #[test]
    #[ignore = "timing diagnostic; run with --ignored --nocapture"]
    fn checkerboard_skip_timing() {
        use std::time::Instant;
        let mut sigma = [0f32; CB_NZ * CB_N * CB_N];
        let mut mm = [0f32; CB_NZ * CB_NT * CB_NT];
        fill_checkerboard(&mut sigma, &mut mm);
        let bg = [CB_BG; CB_NZ];
        let f = cb_field(&sigma, &mm, &bg);
        let f_fine = Cloud3DField {
            macrocell_max: &[],
            ..f
        };
        for (label, p0, dir, t_max) in cb_ray_fan() {
            let reps = 400;
            let mut acc = 0.0f64;
            let t0 = Instant::now();
            for _ in 0..reps {
                acc += f.tau_along(p0, dir, t_max);
            }
            let dt_m = t0.elapsed().as_secs_f64() / reps as f64;
            let t1 = Instant::now();
            for _ in 0..reps {
                acc += f_fine.tau_along(p0, dir, t_max);
            }
            let dt_f = t1.elapsed().as_secs_f64() / reps as f64;
            let (_, sm) = f.tau_along_counted(p0, dir, t_max);
            let (_, sf) = f_fine.tau_along_counted(p0, dir, t_max);
            std::println!(
                "  [timing] {label}: macro {:.1} us / {sm} steps vs fine {:.1} us / {sf} steps (acc {acc:.1})",
                dt_m * 1e6,
                dt_f * 1e6
            );
        }
    }
}
