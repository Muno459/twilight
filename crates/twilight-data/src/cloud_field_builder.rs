//! Builders for [`twilight_core::cloud_field::Cloud3DField`].
//!
//! twilight-core only borrows; this module owns the buffers (std side)
//! and derives the acceleration data: the horizontal-mean background
//! column (the out-of-footprint fallback) and the macro-cell majorants
//! (delta tracking, Stage 2).
//!
//! Convention reminder: voxels store the DELTA-EDDINGTON-SCALED cloud
//! SCATTERING extinction [1/m], spectrally gray, exactly matching the
//! semantics of `AtmosphereModel::cloud_extinction`. When a field is in
//! play it owns ALL cloud; the shell arrays must be zero (the pipeline
//! enforces this by building the atmosphere without cloud layers).

use twilight_core::cloud_field::Cloud3DField;

use crate::cloud::CloudProperties;

/// Default vertical grid: 250 m for 0-16 km (preserves cloud3d's 240 m
/// resolution; a few hundred meters of cloud-top error displaces the
/// grazing twilight shadow edge by 10-15 km).
pub const DEFAULT_NZ: usize = 64;
pub const DEFAULT_DZ_M: f64 = 250.0;
/// Macro-cell tile edge in voxels (8 x 2 km = 16 km tiles).
pub const DEFAULT_TILE: usize = 8;

/// An owned 3D cloud field; `view()` borrows it as the transport type.
#[derive(Debug, Clone)]
pub struct OwnedCloudField {
    pub sigma: Vec<f32>,
    pub g_star: Vec<f32>,
    pub background_column: Vec<f32>,
    pub macrocell_max: Vec<f32>,
    pub tile: usize,
    pub nz: usize,
    pub nlat: usize,
    pub nlon: usize,
    pub z0_m: f64,
    pub dz_m: f64,
    pub lat0_deg: f64,
    pub lon0_deg: f64,
    pub dlat_deg: f64,
    pub dlon_deg: f64,
    pub g_default: f64,
    /// Provenance: data timestamp (ISO) and source label, carried for
    /// honest reporting (stale fields are the caller's to refuse).
    pub timestamp: String,
    pub source: String,
}

impl OwnedCloudField {
    pub fn view(&self) -> Cloud3DField<'_> {
        Cloud3DField {
            sigma: &self.sigma,
            g_star: &self.g_star,
            background_column: &self.background_column,
            macrocell_max: &self.macrocell_max,
            tile: self.tile,
            nz: self.nz,
            nlat: self.nlat,
            nlon: self.nlon,
            z0_m: self.z0_m,
            dz_m: self.dz_m,
            lat0_deg: self.lat0_deg,
            lon0_deg: self.lon0_deg,
            dlat_deg: self.dlat_deg,
            dlon_deg: self.dlon_deg,
            g_default: self.g_default,
        }
    }

    /// Derive the background column (horizontal mean per level) and the
    /// macro-cell majorants from `sigma`. Call after any mutation.
    pub fn derive(&mut self) {
        let (nz, nlat, nlon) = (self.nz, self.nlat, self.nlon);
        self.background_column = (0..nz)
            .map(|iz| {
                let s: f64 = (0..nlat * nlon)
                    .map(|i| self.sigma[iz * nlat * nlon + i] as f64)
                    .sum();
                (s / (nlat * nlon) as f64) as f32
            })
            .collect();
        let (ntlat, ntlon) = (nlat.div_ceil(self.tile), nlon.div_ceil(self.tile));
        let mut mm = vec![0.0f32; nz * ntlat * ntlon];
        for iz in 0..nz {
            for ilat in 0..nlat {
                for ilon in 0..nlon {
                    let v = self.sigma[(iz * nlat + ilat) * nlon + ilon];
                    let m =
                        &mut mm[(iz * ntlat + ilat / self.tile) * ntlon + ilon / self.tile];
                    if v > *m {
                        *m = v;
                    }
                }
            }
        }
        self.macrocell_max = mm;
    }
}

/// Geometry for a new field around an observer.
#[derive(Debug, Clone, Copy)]
pub struct FieldGeometry {
    pub center_lat_deg: f64,
    pub center_lon_deg: f64,
    /// Half-extent of the core footprint [km].
    pub half_extent_km: f64,
    /// Horizontal resolution [km].
    pub res_km: f64,
}

impl FieldGeometry {
    fn grid(&self) -> (usize, usize, f64, f64, f64, f64) {
        let dlat = self.res_km / 111.32;
        let coslat = (self.center_lat_deg.to_radians()).cos().max(0.05);
        let dlon = self.res_km / (111.32 * coslat);
        let n = ((self.half_extent_km * 2.0) / self.res_km).ceil() as usize;
        let lat0 = self.center_lat_deg - dlat * n as f64 / 2.0;
        let lon0 = self.center_lon_deg - dlon * n as f64 / 2.0;
        (n, n, lat0, lon0, dlat, dlon)
    }
}

/// Build a HORIZONTALLY UNIFORM field from layer descriptions: the
/// Stage-0/1 equivalence workhorse (a uniform field must reproduce the
/// 1D layered transport) and the fallback when only 2D cover data
/// exists. Layer optical depths are delta-Eddington scaled here with
/// the same formulas as `builder::add_cloud_layer`.
pub fn field_from_layers(
    layers: &[CloudProperties],
    geom: FieldGeometry,
    timestamp: &str,
) -> OwnedCloudField {
    let (nlat, nlon, lat0, lon0, dlat, dlon) = geom.grid();
    let nz = DEFAULT_NZ;
    let dz = DEFAULT_DZ_M;
    let mut sigma = vec![0.0f32; nz * nlat * nlon];
    let mut level_sigma = vec![0.0f32; nz];
    let mut g_num = 0.0f64;
    let mut g_den = 0.0f64;
    for c in layers {
        if c.optical_depth <= 0.0 || c.top_km <= c.base_km {
            continue;
        }
        // Delta-Eddington similarity scaling (matches add_cloud_layer).
        let f_peak = c.asymmetry * c.asymmetry;
        let de_scale = 1.0 - c.ssa * f_peak;
        let ssa_c = ((1.0 - f_peak) * c.ssa / de_scale).clamp(0.0, 1.0);
        let g_scaled = c.asymmetry / (1.0 + c.asymmetry);
        let ext = c.optical_depth / ((c.top_km - c.base_km) * 1000.0);
        let sigma_s = ext * de_scale * ssa_c;
        g_num += c.optical_depth * c.ssa * g_scaled;
        g_den += c.optical_depth * c.ssa;
        for (iz, level) in level_sigma.iter_mut().enumerate().take(nz) {
            let z_lo = iz as f64 * dz;
            let z_hi = z_lo + dz;
            let overlap =
                (z_hi.min(c.top_km * 1000.0) - z_lo.max(c.base_km * 1000.0)).max(0.0);
            *level += (sigma_s * overlap / dz) as f32;
        }
    }
    for iz in 0..nz {
        let v = level_sigma[iz];
        if v > 0.0 {
            sigma[iz * nlat * nlon..(iz + 1) * nlat * nlon].fill(v);
        }
    }
    let mut f = OwnedCloudField {
        sigma,
        g_star: Vec::new(),
        background_column: Vec::new(),
        macrocell_max: Vec::new(),
        tile: DEFAULT_TILE,
        nz,
        nlat,
        nlon,
        z0_m: 0.0,
        dz_m: dz,
        lat0_deg: lat0,
        lon0_deg: lon0,
        dlat_deg: dlat,
        dlon_deg: dlon,
        g_default: if g_den > 0.0 { g_num / g_den } else { 0.46 },
        timestamp: timestamp.to_string(),
        source: "layers (horizontally uniform)".to_string(),
    };
    f.derive();
    f
}

/// Build a field from a regular (nz_src, ny, nx) ice-water-content grid
/// in g/m^3 (the cloud3d sidecar's full-grid output): conservative
/// vertical regrid onto the field z-grid (column tau is preserved
/// exactly because both grids integrate sigma dz piecewise), horizontal
/// taken 1:1 (the sidecar grid is already ~2-5 km).
///
/// IWC -> delta-scaled scattering extinction via the same microphysics
/// as `crate::...` ice handling: beta = 3 IWC / (2 rho_ice r_eff),
/// r_eff 30 um, then delta-Eddington with ice optics (ssa 0.97 g 0.77).
#[allow(clippy::too_many_arguments)]
pub fn field_from_iwc_grid(
    iwc: &[f32],
    nz_src: usize,
    ny: usize,
    nx: usize,
    src_top_m: f64,
    lat0: f64,
    lon0: f64,
    dlat: f64,
    dlon: f64,
    timestamp: &str,
) -> OwnedCloudField {
    const RHO_ICE: f64 = 0.917e6; // g/m^3
    const R_EFF: f64 = 30e-6; // m
    const SSA_ICE: f64 = 0.97;
    const G_ICE: f64 = 0.77;
    let f_peak = G_ICE * G_ICE;
    let de_scale = 1.0 - SSA_ICE * f_peak;
    let ssa_c = ((1.0 - f_peak) * SSA_ICE / de_scale).clamp(0.0, 1.0);
    let to_sigma = |iwc_g_m3: f64| -> f64 {
        let beta = 3.0 * iwc_g_m3 / (2.0 * RHO_ICE * R_EFF);
        beta * de_scale * ssa_c
    };

    let nz = DEFAULT_NZ;
    let dz = DEFAULT_DZ_M;
    let dz_src = src_top_m / nz_src as f64;
    let mut sigma = vec![0.0f32; nz * ny * nx];
    // Conservative vertical regrid: distribute each source slab's
    // sigma*dz into overlapping destination cells.
    for izs in 0..nz_src {
        // Source grids are TOP-DOWN (cloud3d convention); flip.
        let z_hi = src_top_m - izs as f64 * dz_src;
        let z_lo = z_hi - dz_src;
        let d0 = ((z_lo / dz).floor().max(0.0)) as usize;
        let d1 = (((z_hi / dz).ceil()) as usize).min(nz);
        for izd in d0..d1 {
            let dz_lo = izd as f64 * dz;
            let overlap = (z_hi.min(dz_lo + dz) - z_lo.max(dz_lo)).max(0.0);
            if overlap <= 0.0 {
                continue;
            }
            for iy in 0..ny {
                for ix in 0..nx {
                    let v = iwc[(izs * ny + iy) * nx + ix] as f64;
                    if v > 0.0 {
                        sigma[(izd * ny + iy) * nx + ix] +=
                            (to_sigma(v) * overlap / dz) as f32;
                    }
                }
            }
        }
    }
    let mut f = OwnedCloudField {
        sigma,
        g_star: Vec::new(),
        background_column: Vec::new(),
        macrocell_max: Vec::new(),
        tile: DEFAULT_TILE,
        nz,
        nlat: ny,
        nlon: nx,
        z0_m: 0.0,
        dz_m: dz,
        lat0_deg: lat0,
        lon0_deg: lon0,
        dlat_deg: dlat,
        dlon_deg: dlon,
        g_default: G_ICE / (1.0 + G_ICE),
        timestamp: timestamp.to_string(),
        source: "cloud3d IWC grid".to_string(),
    };
    f.derive();
    f
}

#[cfg(test)]
mod tests {
    use super::*;
    use twilight_core::geometry::Vec3;

    fn ecef(lat_deg: f64, lon_deg: f64, alt_m: f64) -> Vec3 {
        let r = twilight_core::atmosphere::EARTH_RADIUS_M + alt_m;
        let (la, lo) = (lat_deg.to_radians(), lon_deg.to_radians());
        Vec3::new(
            r * la.cos() * lo.cos(),
            r * la.cos() * lo.sin(),
            r * la.sin(),
        )
    }

    fn stratus() -> CloudProperties {
        CloudProperties {
            base_km: 0.5,
            top_km: 1.5,
            optical_depth: 10.0,
            ssa: 0.999,
            asymmetry: 0.85,
        }
    }

    /// G1 against the LAYERED builder: a uniform field's vertical cloud
    /// tau must equal the delta-scaled scattering tau the 1D builder
    /// would put into the shells for the same layer.
    #[test]
    fn uniform_field_matches_layer_tau() {
        let geom = FieldGeometry {
            center_lat_deg: 55.0,
            center_lon_deg: 9.0,
            half_extent_km: 64.0,
            res_km: 2.0,
        };
        let f = field_from_layers(&[stratus()], geom, "test");
        let v = f.view();
        let p0 = ecef(55.0, 9.0, 0.0);
        let up = p0.normalize();
        let tau = v.tau_along(p0, up, 50_000.0);
        // Expected: tau * de_scale * ssa_c for OD-10 g 0.85 ssa 0.999.
        let g: f64 = 0.85;
        let de = 1.0 - 0.999 * g * g;
        let ssa_c = (1.0 - g * g) * 0.999 / de;
        let expect = 10.0 * de * ssa_c;
        assert!(
            (tau - expect).abs() / expect < 5e-3,
            "tau {tau:.4} vs {expect:.4}"
        );
        // Background column carries the same profile beyond the edge.
        let p_far = ecef(58.0, 9.0, 0.0);
        let tau_bg = v.tau_along(p_far, p_far.normalize(), 50_000.0);
        assert!((tau_bg - expect).abs() / expect < 5e-3, "bg {tau_bg:.4}");
    }

    #[test]
    fn iwc_grid_conserves_column_tau() {
        // One source slab of IWC 0.1 g/m^3 over the 9.6-12.0 km band
        // (cloud3d-style top-down, 80 levels to 19.2 km, 240 m).
        let (nzs, ny, nx) = (80, 4, 4);
        let mut iwc = vec![0.0f32; nzs * ny * nx];
        let top = 19_200.0;
        for izs in 0..nzs {
            let z_hi = top - izs as f64 * 240.0;
            if z_hi > 9_600.0 && z_hi <= 12_000.0 {
                for i in 0..ny * nx {
                    iwc[izs * ny * nx + i] = 0.1;
                }
            }
        }
        let f = field_from_iwc_grid(&iwc, nzs, ny, nx, top, 54.0, 8.0, 0.02, 0.03, "t");
        let v = f.view();
        let p0 = ecef(54.03, 8.05, 0.0);
        let tau = v.tau_along(p0, p0.normalize(), 30_000.0);
        // Analytic: thickness 2400 m, beta = 3*0.1/(2*0.917e6*30e-6),
        // delta-scaled by ice optics.
        let beta = 3.0 * 0.1 / (2.0 * 0.917e6 * 30e-6);
        let de = 1.0 - 0.97 * 0.77f64 * 0.77;
        let ssa_c = (1.0 - 0.77 * 0.77f64) * 0.97 / de;
        let expect = beta * de * ssa_c * 2400.0;
        assert!(
            (tau - expect).abs() / expect < 1e-2,
            "tau {tau:.5} vs {expect:.5}"
        );
    }

    #[test]
    fn macrocells_bound_their_tiles() {
        let geom = FieldGeometry {
            center_lat_deg: 0.0,
            center_lon_deg: 0.0,
            half_extent_km: 32.0,
            res_km: 2.0,
        };
        let mut f = field_from_layers(&[stratus()], geom, "test");
        // Punch a hot voxel and re-derive.
        let idx = (4 * f.nlat + 3) * f.nlon + 5;
        f.sigma[idx] = 0.5;
        f.derive();
        let (ntlat, ntlon) = (
            f.nlat.div_ceil(f.tile),
            f.nlon.div_ceil(f.tile),
        );
        let m = f.macrocell_max[(4 * ntlat + 3 / f.tile) * ntlon + 5 / f.tile];
        assert!((m - 0.5).abs() < 1e-9, "majorant must see the hot voxel");
        // Every voxel <= its tile majorant.
        for iz in 0..f.nz {
            for ilat in 0..f.nlat {
                for ilon in 0..f.nlon {
                    let v = f.sigma[(iz * f.nlat + ilat) * f.nlon + ilon];
                    let mm = f.macrocell_max
                        [(iz * ntlat + ilat / f.tile) * ntlon + ilon / f.tile];
                    assert!(v <= mm);
                }
            }
        }
    }
}
