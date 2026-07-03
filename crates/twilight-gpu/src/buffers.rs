//! GPU buffer packing: convert f64 CPU structs to f32 GPU-friendly layouts.
//!
//! All GPU backends share this code. The atmosphere model, solar spectrum,
//! vision LUTs, and light sources are packed into flat `Vec<f32>` / `Vec<u32>`
//! buffers with 16-byte (vec4) alignment and a versioned header.
//!
//! # Buffer versioning
//!
//! Every buffer begins with a [`BufferHeader`] (2 x u32):
//! - `magic`: `0x54574C54` ("TWLT" in ASCII)
//! - `version`: monotonically increasing layout version
//!
//! GPU shaders check the magic/version at kernel start and bail with a
//! sentinel value if they don't match, making stale-buffer bugs obvious.

use twilight_core::atmosphere::{AtmosphereModel, MAX_SHELLS, MAX_WAVELENGTHS};

// ── Constants ───────────────────────────────────────────────────────────

/// Magic bytes identifying a Twilight GPU buffer: "TWLT" = 0x544C5754.
pub const BUFFER_MAGIC: u32 = 0x544C_5754;

/// Current buffer layout version. Increment when the packing format changes.
/// v2: added refractive_index[MAX_SHELLS] after surface_albedo.
/// v3: added cloud_extinction[MAX_SHELLS] + cloud_g_scaled (delta-scaled
///     cloud scattering for the Eddington diffuse-transmission factor).
/// v4: Stage 3 3D cloud transport. No change to the atmosphere buffer
///     layout itself, but a SEPARATE packed cloud field buffer
///     ([`PackedCloudField`], bound at index 3) now carries the voxel
///     sigma grid, g*, macrocell majorants, background column, and grid
///     geometry. The version bump forces a re-pack so a stale v3 host
///     paired with a v4 shader (or vice versa) trips the header gate
///     loudly instead of silently running the wrong transport.
pub const BUFFER_VERSION: u32 = 4;

/// Maximum number of light pollution sources in a single dispatch.
pub const MAX_LIGHT_SOURCES: usize = 2048;

// ── Buffer header ───────────────────────────────────────────────────────

/// 8-byte header at the start of every GPU buffer.
///
/// Stored as two u32 values (reinterpreted via `f32::from_bits` on GPU
/// since all our buffers are `buffer<f32>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferHeader {
    pub magic: u32,
    pub version: u32,
}

impl BufferHeader {
    pub fn current() -> Self {
        Self {
            magic: BUFFER_MAGIC,
            version: BUFFER_VERSION,
        }
    }

    /// Validate a header read back from GPU.
    pub fn validate(&self) -> bool {
        self.magic == BUFFER_MAGIC && self.version == BUFFER_VERSION
    }
}

// ── Packed atmosphere buffer ────────────────────────────────────────────

/// Packed atmosphere model for GPU upload.
///
/// Layout (all f32, 16-byte aligned where noted):
///
/// ```text
/// [0..2]   BufferHeader (magic, version) as f32::from_bits
/// [2]      num_shells (as f32, cast to u32 on GPU)
/// [3]      num_wavelengths (as f32, cast to u32 on GPU)
///
/// --- Shell geometry: 4 floats per shell, MAX_SHELLS entries ---
/// [4 .. 4+4*64]  shells[i] = (r_inner, r_outer, altitude_mid, thickness)
///
/// --- Optics: 4 floats per (shell, wavelength) pair ---
/// [260 .. 260+4*64*64]  optics[s][w] = (extinction, ssa, asymmetry, rayleigh_fraction)
///
/// --- Wavelengths: padded to vec4 alignment ---
/// [16644 .. 16644+64]  wavelengths_nm[w]
///
/// --- Surface albedo: padded to vec4 alignment ---
/// [16708 .. 16708+64]  surface_albedo[w]
///
/// --- Refractive index per shell (v2) ---
/// [16772 .. 16772+64]  refractive_index[s]
/// ```
///
/// Total: 16,836 f32 values = ~67.3 KB.
#[derive(Debug, Clone)]
pub struct PackedAtmosphere {
    /// Flat f32 buffer ready for GPU upload.
    pub data: Vec<f32>,
    /// Number of active shells (for kernel dispatch sizing).
    pub num_shells: u32,
    /// Number of active wavelengths.
    pub num_wavelengths: u32,
}

/// Offset constants into the packed atmosphere buffer.
pub mod atm_offsets {
    /// Header: magic (u32 as f32 bits)
    pub const HEADER_MAGIC: usize = 0;
    /// Header: version (u32 as f32 bits)
    pub const HEADER_VERSION: usize = 1;
    /// Number of active shells
    pub const NUM_SHELLS: usize = 2;
    /// Number of active wavelengths
    pub const NUM_WAVELENGTHS: usize = 3;
    /// Start of shell geometry array (4 f32 per shell)
    pub const SHELLS_START: usize = 4;
    /// Stride between shells (r_inner, r_outer, altitude_mid, thickness)
    pub const SHELL_STRIDE: usize = 4;
    /// Start of optics array (4 f32 per shell*wavelength)
    pub const OPTICS_START: usize = SHELLS_START + SHELL_STRIDE * super::MAX_SHELLS; // 4 + 256 = 260
    /// Stride between optics entries
    pub const OPTICS_STRIDE: usize = 4;
    /// Start of wavelengths array
    pub const WAVELENGTHS_START: usize =
        OPTICS_START + OPTICS_STRIDE * super::MAX_SHELLS * super::MAX_WAVELENGTHS; // 260 + 16384 = 16644
    /// Start of surface albedo array
    pub const ALBEDO_START: usize = WAVELENGTHS_START + super::MAX_WAVELENGTHS; // 16644 + 64 = 16708
    /// Start of refractive index array (1 f32 per shell, padded to vec4)
    pub const REFRACTIVE_INDEX_START: usize = ALBEDO_START + super::MAX_WAVELENGTHS; // 16708 + 64 = 16772
    /// Start of cloud (delta-scaled scattering) extinction array, 1 f32/shell
    pub const CLOUD_EXT_START: usize = REFRACTIVE_INDEX_START + super::MAX_SHELLS; // 16836
    /// Scalar: delta-scaled cloud asymmetry g* (0 when no cloud)
    pub const CLOUD_G_SCALED: usize = CLOUD_EXT_START + super::MAX_SHELLS; // 16900
    /// Total buffer size in f32 elements (CLOUD_G_SCALED + vec4 padding)
    pub const TOTAL_SIZE: usize = CLOUD_G_SCALED + 4; // 16904
}

impl PackedAtmosphere {
    /// Pack an `AtmosphereModel` (f64) into a flat f32 GPU buffer.
    ///
    /// The conversion is lossless for the physics (f32 rounding error is
    /// 84 trillion times smaller than MC noise -- see precision analysis).
    pub fn pack(atm: &AtmosphereModel) -> Self {
        let mut data = vec![0.0f32; atm_offsets::TOTAL_SIZE];

        // Header
        data[atm_offsets::HEADER_MAGIC] = f32::from_bits(BUFFER_MAGIC);
        data[atm_offsets::HEADER_VERSION] = f32::from_bits(BUFFER_VERSION);

        // Dimensions
        data[atm_offsets::NUM_SHELLS] = atm.num_shells as f32;
        data[atm_offsets::NUM_WAVELENGTHS] = atm.num_wavelengths as f32;

        // Shell geometry
        for s in 0..atm.num_shells {
            let base = atm_offsets::SHELLS_START + s * atm_offsets::SHELL_STRIDE;
            data[base] = atm.shells[s].r_inner as f32;
            data[base + 1] = atm.shells[s].r_outer as f32;
            data[base + 2] = atm.shells[s].altitude_mid as f32;
            data[base + 3] = atm.shells[s].thickness as f32;
        }

        // Optics: packed as [shell][wavelength] with 4 f32 per entry
        for s in 0..atm.num_shells {
            for w in 0..atm.num_wavelengths {
                let idx = s * MAX_WAVELENGTHS + w;
                let base = atm_offsets::OPTICS_START + idx * atm_offsets::OPTICS_STRIDE;
                data[base] = atm.optics[s][w].extinction as f32;
                data[base + 1] = atm.optics[s][w].ssa as f32;
                data[base + 2] = atm.optics[s][w].asymmetry as f32;
                data[base + 3] = atm.optics[s][w].rayleigh_fraction as f32;
            }
        }

        // Wavelengths
        for w in 0..atm.num_wavelengths {
            data[atm_offsets::WAVELENGTHS_START + w] = atm.wavelengths_nm[w] as f32;
        }

        // Surface albedo
        for w in 0..atm.num_wavelengths {
            data[atm_offsets::ALBEDO_START + w] = atm.surface_albedo[w] as f32;
        }

        // Refractive index per shell (v2)
        for s in 0..atm.num_shells {
            data[atm_offsets::REFRACTIVE_INDEX_START + s] = atm.refractive_index[s] as f32;
            data[atm_offsets::CLOUD_EXT_START + s] = atm.cloud_extinction[s] as f32;
        }
        data[atm_offsets::CLOUD_G_SCALED] = atm.cloud_g_scaled as f32;

        PackedAtmosphere {
            data,
            num_shells: atm.num_shells as u32,
            num_wavelengths: atm.num_wavelengths as u32,
        }
    }

    /// Unpack back to f64 for validation / round-trip testing.
    ///
    /// This is only used in tests to verify the packing is correct.
    pub fn unpack(&self) -> AtmosphereModel {
        let ns = self.num_shells as usize;
        let nw = self.num_wavelengths as usize;

        // Reconstruct altitude boundaries from shell r_inner / r_outer.
        // We need (ns + 1) altitude boundaries.
        let mut altitudes_km = vec![0.0f64; ns + 1];
        let earth_r = twilight_core::atmosphere::EARTH_RADIUS_M;
        for s in 0..ns {
            let base = atm_offsets::SHELLS_START + s * atm_offsets::SHELL_STRIDE;
            let r_inner = self.data[base] as f64;
            let r_outer = self.data[base + 1] as f64;
            altitudes_km[s] = (r_inner - earth_r) / 1000.0;
            if s == ns - 1 {
                altitudes_km[s + 1] = (r_outer - earth_r) / 1000.0;
            }
        }

        let mut wavelengths_nm = vec![0.0f64; nw];
        for (w, wl) in wavelengths_nm.iter_mut().enumerate().take(nw) {
            *wl = self.data[atm_offsets::WAVELENGTHS_START + w] as f64;
        }

        let mut atm = AtmosphereModel::new(&altitudes_km, &wavelengths_nm);

        // Restore optics
        for s in 0..ns {
            for w in 0..nw {
                let idx = s * MAX_WAVELENGTHS + w;
                let base = atm_offsets::OPTICS_START + idx * atm_offsets::OPTICS_STRIDE;
                atm.optics[s][w].extinction = self.data[base] as f64;
                atm.optics[s][w].ssa = self.data[base + 1] as f64;
                atm.optics[s][w].asymmetry = self.data[base + 2] as f64;
                atm.optics[s][w].rayleigh_fraction = self.data[base + 3] as f64;
            }
        }

        // Restore albedo
        for w in 0..nw {
            atm.surface_albedo[w] = self.data[atm_offsets::ALBEDO_START + w] as f64;
        }

        // Restore refractive indices (v2)
        for s in 0..ns {
            atm.refractive_index[s] = self.data[atm_offsets::REFRACTIVE_INDEX_START + s] as f64;
            atm.cloud_extinction[s] = self.data[atm_offsets::CLOUD_EXT_START + s] as f64;
        }
        atm.cloud_g_scaled = self.data[atm_offsets::CLOUD_G_SCALED] as f64;

        atm
    }

    /// Read the header from a packed buffer and validate it.
    pub fn validate_header(&self) -> bool {
        if self.data.len() < 2 {
            return false;
        }
        let magic = self.data[atm_offsets::HEADER_MAGIC].to_bits();
        let version = self.data[atm_offsets::HEADER_VERSION].to_bits();
        magic == BUFFER_MAGIC && version == BUFFER_VERSION
    }

    /// Total size in bytes of the GPU buffer.
    pub fn size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }
}

// ── Packed 3D cloud field buffer (v4) ───────────────────────────────────

/// Offsets into the packed cloud-field header (f32 slots).
///
/// The header is a fixed-size prefix; the four variable-length arrays
/// (sigma, g_star, macrocell_max, background_column) follow it, and their
/// start offsets are stored in the header so the shader can locate them.
pub mod field_offsets {
    pub const HEADER_MAGIC: usize = 0;
    pub const HEADER_VERSION: usize = 1;
    pub const NZ: usize = 2;
    pub const NLAT: usize = 3;
    pub const NLON: usize = 4;
    pub const TILE: usize = 5;
    pub const NTLAT: usize = 6;
    pub const NTLON: usize = 7;
    pub const G_STAR_PRESENT: usize = 8;
    pub const BG_PRESENT: usize = 9;
    pub const MACRO_PRESENT: usize = 10;
    // slot 11 reserved
    // Geometry (f32): grid spacings and origins.
    pub const Z0_M: usize = 12;
    pub const DZ_M: usize = 13;
    pub const LAT0_DEG: usize = 14;
    pub const LON0_DEG: usize = 15;
    pub const DLAT_DEG: usize = 16;
    pub const DLON_DEG: usize = 17;
    pub const G_DEFAULT: usize = 18;
    // slot 19 reserved
    // Array start offsets (f32 index into the data buffer).
    pub const SIGMA_OFFSET: usize = 20;
    pub const G_STAR_OFFSET: usize = 21;
    pub const MACRO_OFFSET: usize = 22;
    pub const BG_OFFSET: usize = 23;
    /// Header length (vec4-aligned: 24 f32 = 6 vec4).
    pub const HEADER_LEN: usize = 24;
}

/// Packed 3D cloud field for GPU upload (v4).
///
/// All voxel data is f32 (the field's owning slices already are f32). The
/// grid geometry is packed as f32; the device-side DDA runs in the same
/// f32 regime as the rest of the shader (the f32 error budget for the tau
/// prefix and crossing roots is derived in the GPU test module).
#[derive(Debug, Clone)]
pub struct PackedCloudField {
    pub data: Vec<f32>,
}

impl PackedCloudField {
    /// Pack a [`twilight_core::cloud_field::Cloud3DField`] view.
    pub fn pack(field: &twilight_core::cloud_field::Cloud3DField) -> Self {
        use field_offsets as fo;

        let nz = field.nz;
        let nlat = field.nlat;
        let nlon = field.nlon;
        let tile = field.tile.max(1);
        let ntlat = nlat.div_ceil(tile);
        let ntlon = nlon.div_ceil(tile);

        let sigma_len = field.sigma.len();
        let g_star_len = field.g_star.len();
        let macro_len = field.macrocell_max.len();
        let bg_len = field.background_column.len();

        let sigma_off = fo::HEADER_LEN;
        let g_star_off = sigma_off + sigma_len;
        let macro_off = g_star_off + g_star_len;
        let bg_off = macro_off + macro_len;
        let total = bg_off + bg_len;

        let mut data = vec![0.0f32; total.max(fo::HEADER_LEN)];

        data[fo::HEADER_MAGIC] = f32::from_bits(BUFFER_MAGIC);
        data[fo::HEADER_VERSION] = f32::from_bits(BUFFER_VERSION);
        data[fo::NZ] = nz as f32;
        data[fo::NLAT] = nlat as f32;
        data[fo::NLON] = nlon as f32;
        data[fo::TILE] = tile as f32;
        data[fo::NTLAT] = ntlat as f32;
        data[fo::NTLON] = ntlon as f32;
        data[fo::G_STAR_PRESENT] = if g_star_len > 0 { 1.0 } else { 0.0 };
        data[fo::BG_PRESENT] = if bg_len > 0 { 1.0 } else { 0.0 };
        data[fo::MACRO_PRESENT] = if macro_len > 0 { 1.0 } else { 0.0 };
        data[fo::Z0_M] = field.z0_m as f32;
        data[fo::DZ_M] = field.dz_m as f32;
        data[fo::LAT0_DEG] = field.lat0_deg as f32;
        data[fo::LON0_DEG] = field.lon0_deg as f32;
        data[fo::DLAT_DEG] = field.dlat_deg as f32;
        data[fo::DLON_DEG] = field.dlon_deg as f32;
        data[fo::G_DEFAULT] = field.g_default as f32;
        data[fo::SIGMA_OFFSET] = f32::from_bits(sigma_off as u32);
        data[fo::G_STAR_OFFSET] = f32::from_bits(g_star_off as u32);
        data[fo::MACRO_OFFSET] = f32::from_bits(macro_off as u32);
        data[fo::BG_OFFSET] = f32::from_bits(bg_off as u32);

        data[sigma_off..sigma_off + sigma_len].copy_from_slice(field.sigma);
        data[g_star_off..g_star_off + g_star_len].copy_from_slice(field.g_star);
        data[macro_off..macro_off + macro_len].copy_from_slice(field.macrocell_max);
        data[bg_off..bg_off + bg_len].copy_from_slice(field.background_column);

        PackedCloudField { data }
    }

    /// Total size in bytes of the GPU buffer.
    pub fn size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }
}

// ── Packed solar irradiance ─────────────────────────────────────────────

/// Solar irradiance LUT packed for GPU.
///
/// Layout: [magic, version, num_entries, pad, wavelengths..., irradiance...]
/// Total: 2 + 2 + 41 + (pad to vec4) + 41 + pad = ~88 f32 values.
#[derive(Debug, Clone)]
pub struct PackedSolarSpectrum {
    pub data: Vec<f32>,
    pub num_entries: u32,
}

impl PackedSolarSpectrum {
    /// Pack the solar irradiance tables from twilight-data.
    pub fn pack() -> Self {
        use twilight_data::solar_spectrum::{
            SOLAR_IRRADIANCE, SOLAR_NUM_ENTRIES, SOLAR_WAVELENGTHS_NM,
        };

        // Header (4 f32 = 16 bytes, vec4 aligned)
        // [magic, version, num_entries, pad]
        // Wavelengths (41 entries, padded to next multiple of 4 = 44)
        // Irradiance (41 entries, padded to 44)
        let wl_padded = (SOLAR_NUM_ENTRIES + 3) & !3; // round up to 4
        let total = 4 + wl_padded + wl_padded;
        let mut data = vec![0.0f32; total];

        data[0] = f32::from_bits(BUFFER_MAGIC);
        data[1] = f32::from_bits(BUFFER_VERSION);
        data[2] = SOLAR_NUM_ENTRIES as f32;
        // data[3] = padding

        for i in 0..SOLAR_NUM_ENTRIES {
            data[4 + i] = SOLAR_WAVELENGTHS_NM[i] as f32;
        }

        for i in 0..SOLAR_NUM_ENTRIES {
            data[4 + wl_padded + i] = SOLAR_IRRADIANCE[i] as f32;
        }

        PackedSolarSpectrum {
            data,
            num_entries: SOLAR_NUM_ENTRIES as u32,
        }
    }
}

// ── Packed vision LUTs ──────────────────────────────────────────────────

/// CIE photopic V(l) and scotopic V'(l) packed for GPU.
///
/// Layout: [magic, version, num_entries, pad, photopic_v..., scotopic_v...]
/// 81 entries each, padded to vec4 alignment.
#[derive(Debug, Clone)]
pub struct PackedVisionLuts {
    pub data: Vec<f32>,
    pub num_entries: u32,
}

impl PackedVisionLuts {
    /// Pack the vision LUTs from twilight-threshold.
    pub fn pack() -> Self {
        use twilight_threshold::vision::{PHOTOPIC_V, SCOTOPIC_V_PRIME};

        let n = PHOTOPIC_V.len(); // 81
        let n_padded = (n + 3) & !3; // 84
        let total = 4 + n_padded + n_padded;
        let mut data = vec![0.0f32; total];

        data[0] = f32::from_bits(BUFFER_MAGIC);
        data[1] = f32::from_bits(BUFFER_VERSION);
        data[2] = n as f32;

        for i in 0..n {
            data[4 + i] = PHOTOPIC_V[i] as f32;
        }

        for i in 0..n {
            data[4 + n_padded + i] = SCOTOPIC_V_PRIME[i] as f32;
        }

        PackedVisionLuts {
            data,
            num_entries: n as u32,
        }
    }
}

// ── Packed light source (for Garstang kernel) ───────────────────────────

/// A single light pollution source, packed for GPU.
///
/// 8 f32 values (2 x vec4):
/// vec4(distance_m, zenith_angle_rad, radiance_wm2sr, spectrum_type)
/// vec4(height_m, _pad, _pad, _pad)
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct PackedLightSource {
    /// Distance from observer to light source [m].
    pub distance_m: f32,
    /// Zenith angle of the source as seen from observer [rad].
    pub zenith_angle_rad: f32,
    /// Radiance of the source [W/m^2/sr].
    pub radiance: f32,
    /// Spectrum type: 0 = HPS, 1 = LED, 2 = generic.
    pub spectrum_type: f32,
    /// Emission height above ground [m].
    pub height_m: f32,
    pub _pad1: f32,
    pub _pad2: f32,
    pub _pad3: f32,
}

impl Default for PackedLightSource {
    fn default() -> Self {
        Self {
            distance_m: 0.0,
            zenith_angle_rad: 0.0,
            radiance: 0.0,
            spectrum_type: 0.0,
            height_m: 10.0,
            _pad1: 0.0,
            _pad2: 0.0,
            _pad3: 0.0,
        }
    }
}

// ── Dispatch parameters buffer ──────────────────────────────────────────

/// Per-dispatch parameters packed as a uniform/constant buffer.
///
/// Layout (16 f32 = 64 bytes, 4 x vec4):
/// vec4(observer_x, observer_y, observer_z, field_present)
/// vec4(view_dir_x, view_dir_y, view_dir_z, _pad)
/// vec4(sun_dir_x, sun_dir_y, sun_dir_z, _pad)
/// vec4(photons_per_wl, secondary_rays, rng_seed_lo, rng_seed_hi)
///
/// `field_present` is 1.0 when a 3D cloud field buffer is bound and the
/// chains must take the gray cloud channel (Beer-Lambert), 0.0 for the
/// legacy 1D shell-cloud path (Eddington diffuse transmittance, unchanged).
#[derive(Debug, Clone)]
pub struct PackedDispatchParams {
    pub data: [f32; 16],
}

impl PackedDispatchParams {
    /// Pack dispatch parameters from f64 vectors (no cloud field).
    pub fn new(
        observer_pos: [f64; 3],
        view_dir: [f64; 3],
        sun_dir: [f64; 3],
        photons_per_wl: u32,
        secondary_rays: u32,
        rng_seed: u64,
    ) -> Self {
        Self::new_with_field(
            observer_pos,
            view_dir,
            sun_dir,
            photons_per_wl,
            secondary_rays,
            rng_seed,
            false,
        )
    }

    /// Pack dispatch parameters, flagging whether a 3D cloud field is bound.
    pub fn new_with_field(
        observer_pos: [f64; 3],
        view_dir: [f64; 3],
        sun_dir: [f64; 3],
        photons_per_wl: u32,
        secondary_rays: u32,
        rng_seed: u64,
        field_present: bool,
    ) -> Self {
        let seed_lo = (rng_seed & 0xFFFF_FFFF) as u32;
        let seed_hi = (rng_seed >> 32) as u32;

        Self {
            data: [
                observer_pos[0] as f32,
                observer_pos[1] as f32,
                observer_pos[2] as f32,
                if field_present { 1.0 } else { 0.0 },
                view_dir[0] as f32,
                view_dir[1] as f32,
                view_dir[2] as f32,
                0.0,
                sun_dir[0] as f32,
                sun_dir[1] as f32,
                sun_dir[2] as f32,
                0.0,
                f32::from_bits(photons_per_wl),
                f32::from_bits(secondary_rays),
                f32::from_bits(seed_lo),
                f32::from_bits(seed_hi),
            ],
        }
    }

    /// Set the LOS step-window offset (params slot 7, previously pad and
    /// packed as 0.0 == from_bits(0), so existing callers are bit-identical).
    /// The hybrid v2 kernel adds this to its threadgroup y-index, letting the
    /// host split one logical (wavelength x 200 step) dispatch into several
    /// SMALLER command buffers along the step axis: a STATIC watchdog split
    /// (never adaptive) for the field path, where a full-grid buffer of
    /// chain work exceeds the macOS GPU watchdog.
    pub fn with_step_offset(mut self, step_offset: u32) -> Self {
        self.data[7] = f32::from_bits(step_offset);
        self
    }
}

// ── Utility ─────────────────────────────────────────────────────────────

/// Compute the number of workgroups needed for `total_threads` with
/// `workgroup_size` threads per group.
pub fn dispatch_groups(total_threads: u32, workgroup_size: u32) -> u32 {
    total_threads.div_ceil(workgroup_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── BufferHeader ────────────────────────────────────────────────────

    #[test]
    fn header_current_is_valid() {
        let h = BufferHeader::current();
        assert!(h.validate());
    }

    #[test]
    fn header_wrong_magic_is_invalid() {
        let h = BufferHeader {
            magic: 0xDEADBEEF,
            version: BUFFER_VERSION,
        };
        assert!(!h.validate());
    }

    #[test]
    fn header_wrong_version_is_invalid() {
        let h = BufferHeader {
            magic: BUFFER_MAGIC,
            version: 999,
        };
        assert!(!h.validate());
    }

    #[test]
    fn buffer_magic_is_twlt() {
        // "TWLT" in little-endian ASCII
        assert_eq!(BUFFER_MAGIC, 0x544C_5754);
    }

    // ── PackedAtmosphere ────────────────────────────────────────────────

    fn make_test_atm() -> AtmosphereModel {
        let altitudes_km = [0.0, 10.0, 50.0, 100.0];
        let wavelengths = [400.0, 550.0, 700.0];
        let mut atm = AtmosphereModel::new(&altitudes_km, &wavelengths);

        // Set non-trivial optics
        for s in 0..3 {
            for w in 0..3 {
                atm.optics[s][w].extinction = 1e-5 * (s as f64 + 1.0) * (4.0 - w as f64);
                atm.optics[s][w].ssa = 0.95 + 0.01 * w as f64;
                atm.optics[s][w].asymmetry = 0.1 * s as f64;
                atm.optics[s][w].rayleigh_fraction = 1.0 - 0.1 * s as f64;
            }
        }
        atm.surface_albedo[0] = 0.05; // blue
        atm.surface_albedo[1] = 0.15; // green
        atm.surface_albedo[2] = 0.25; // red

        atm
    }

    #[test]
    fn packed_atm_has_correct_size() {
        let atm = make_test_atm();
        let packed = PackedAtmosphere::pack(&atm);
        assert_eq!(packed.data.len(), atm_offsets::TOTAL_SIZE);
    }

    #[test]
    fn packed_atm_header_validates() {
        let atm = make_test_atm();
        let packed = PackedAtmosphere::pack(&atm);
        assert!(packed.validate_header());
    }

    #[test]
    fn packed_atm_dimensions_correct() {
        let atm = make_test_atm();
        let packed = PackedAtmosphere::pack(&atm);
        assert_eq!(packed.num_shells, 3);
        assert_eq!(packed.num_wavelengths, 3);
        assert_eq!(packed.data[atm_offsets::NUM_SHELLS] as u32, 3);
        assert_eq!(packed.data[atm_offsets::NUM_WAVELENGTHS] as u32, 3);
    }

    #[test]
    fn packed_atm_shell_geometry_roundtrip() {
        let atm = make_test_atm();
        let packed = PackedAtmosphere::pack(&atm);

        for s in 0..atm.num_shells {
            let base = atm_offsets::SHELLS_START + s * atm_offsets::SHELL_STRIDE;
            let r_inner = packed.data[base] as f64;
            let r_outer = packed.data[base + 1] as f64;
            let alt_mid = packed.data[base + 2] as f64;
            let thickness = packed.data[base + 3] as f64;

            // f32 relative error for Earth-radius scale numbers: ~1e-7
            let rtol = 1e-5;
            assert!(
                (r_inner - atm.shells[s].r_inner).abs() / atm.shells[s].r_inner < rtol,
                "shell[{}].r_inner: packed={}, original={}",
                s,
                r_inner,
                atm.shells[s].r_inner,
            );
            assert!(
                (r_outer - atm.shells[s].r_outer).abs() / atm.shells[s].r_outer < rtol,
                "shell[{}].r_outer: packed={}, original={}",
                s,
                r_outer,
                atm.shells[s].r_outer,
            );
            assert!(
                (alt_mid - atm.shells[s].altitude_mid).abs() < 1.0,
                "shell[{}].altitude_mid: packed={}, original={}",
                s,
                alt_mid,
                atm.shells[s].altitude_mid,
            );
            assert!(
                (thickness - atm.shells[s].thickness).abs() / atm.shells[s].thickness < rtol,
                "shell[{}].thickness: packed={}, original={}",
                s,
                thickness,
                atm.shells[s].thickness,
            );
        }
    }

    #[test]
    fn packed_atm_optics_roundtrip() {
        let atm = make_test_atm();
        let packed = PackedAtmosphere::pack(&atm);

        for s in 0..atm.num_shells {
            for w in 0..atm.num_wavelengths {
                let idx = s * MAX_WAVELENGTHS + w;
                let base = atm_offsets::OPTICS_START + idx * atm_offsets::OPTICS_STRIDE;

                let ext = packed.data[base] as f64;
                let ssa = packed.data[base + 1] as f64;
                let asym = packed.data[base + 2] as f64;
                let rfrac = packed.data[base + 3] as f64;

                let rtol = 1e-5;
                assert!(
                    (ext - atm.optics[s][w].extinction).abs()
                        < rtol * atm.optics[s][w].extinction + 1e-30,
                    "optics[{}][{}].extinction: packed={}, original={}",
                    s,
                    w,
                    ext,
                    atm.optics[s][w].extinction,
                );
                assert!(
                    (ssa - atm.optics[s][w].ssa).abs() < rtol,
                    "optics[{}][{}].ssa: packed={}, original={}",
                    s,
                    w,
                    ssa,
                    atm.optics[s][w].ssa,
                );
                assert!(
                    (asym - atm.optics[s][w].asymmetry).abs() < rtol + 1e-10,
                    "optics[{}][{}].asymmetry: packed={}, original={}",
                    s,
                    w,
                    asym,
                    atm.optics[s][w].asymmetry,
                );
                assert!(
                    (rfrac - atm.optics[s][w].rayleigh_fraction).abs() < rtol,
                    "optics[{}][{}].rayleigh_fraction: packed={}, original={}",
                    s,
                    w,
                    rfrac,
                    atm.optics[s][w].rayleigh_fraction,
                );
            }
        }
    }

    #[test]
    fn packed_atm_wavelengths_roundtrip() {
        let atm = make_test_atm();
        let packed = PackedAtmosphere::pack(&atm);

        for w in 0..atm.num_wavelengths {
            let wl = packed.data[atm_offsets::WAVELENGTHS_START + w] as f64;
            assert!(
                (wl - atm.wavelengths_nm[w]).abs() < 0.01,
                "wavelength[{}]: packed={}, original={}",
                w,
                wl,
                atm.wavelengths_nm[w],
            );
        }
    }

    #[test]
    fn packed_atm_albedo_roundtrip() {
        let atm = make_test_atm();
        let packed = PackedAtmosphere::pack(&atm);

        for w in 0..atm.num_wavelengths {
            let alb = packed.data[atm_offsets::ALBEDO_START + w] as f64;
            assert!(
                (alb - atm.surface_albedo[w]).abs() < 1e-5,
                "albedo[{}]: packed={}, original={}",
                w,
                alb,
                atm.surface_albedo[w],
            );
        }
    }

    #[test]
    fn packed_atm_unpack_roundtrip() {
        let atm = make_test_atm();
        let packed = PackedAtmosphere::pack(&atm);
        let unpacked = packed.unpack();

        assert_eq!(unpacked.num_shells, atm.num_shells);
        assert_eq!(unpacked.num_wavelengths, atm.num_wavelengths);

        for s in 0..atm.num_shells {
            // Shell geometry (f32 roundtrip has ~1m error on Earth-radius scale)
            assert!(
                (unpacked.shells[s].r_inner - atm.shells[s].r_inner).abs() < 2.0,
                "shell[{}].r_inner: {} vs {}",
                s,
                unpacked.shells[s].r_inner,
                atm.shells[s].r_inner,
            );
        }

        for s in 0..atm.num_shells {
            for w in 0..atm.num_wavelengths {
                let rtol = 1e-5;
                assert!(
                    (unpacked.optics[s][w].extinction - atm.optics[s][w].extinction).abs()
                        < rtol * atm.optics[s][w].extinction + 1e-30,
                    "optics[{}][{}].extinction roundtrip: {} vs {}",
                    s,
                    w,
                    unpacked.optics[s][w].extinction,
                    atm.optics[s][w].extinction,
                );
            }
        }
    }

    #[test]
    fn packed_atm_unused_shells_are_zero() {
        let atm = make_test_atm(); // 3 shells
        let packed = PackedAtmosphere::pack(&atm);

        // Shells 3..63 should be zeroed
        for s in atm.num_shells..MAX_SHELLS {
            let base = atm_offsets::SHELLS_START + s * atm_offsets::SHELL_STRIDE;
            for i in 0..4 {
                assert_eq!(
                    packed.data[base + i],
                    0.0,
                    "unused shell[{}] field {} should be 0",
                    s,
                    i,
                );
            }
        }
    }

    #[test]
    fn packed_atm_size_bytes() {
        let atm = make_test_atm();
        let packed = PackedAtmosphere::pack(&atm);
        assert_eq!(packed.size_bytes(), atm_offsets::TOTAL_SIZE * 4);
    }

    // ── PackedSolarSpectrum ─────────────────────────────────────────────

    #[test]
    fn packed_solar_header_validates() {
        let solar = PackedSolarSpectrum::pack();
        assert_eq!(solar.data[0].to_bits(), BUFFER_MAGIC);
        assert_eq!(solar.data[1].to_bits(), BUFFER_VERSION);
    }

    #[test]
    fn packed_solar_num_entries() {
        let solar = PackedSolarSpectrum::pack();
        assert_eq!(solar.num_entries, 41);
        assert_eq!(solar.data[2] as u32, 41);
    }

    #[test]
    #[allow(clippy::needless_range_loop)] // packed-offset indexing
    fn packed_solar_wavelengths_match() {
        use twilight_data::solar_spectrum::SOLAR_WAVELENGTHS_NM;
        let solar = PackedSolarSpectrum::pack();
        for i in 0..41 {
            let packed_wl = solar.data[4 + i] as f64;
            assert!(
                (packed_wl - SOLAR_WAVELENGTHS_NM[i]).abs() < 0.01,
                "solar wl[{}]: packed={}, original={}",
                i,
                packed_wl,
                SOLAR_WAVELENGTHS_NM[i],
            );
        }
    }

    #[test]
    fn packed_solar_irradiance_positive() {
        let solar = PackedSolarSpectrum::pack();
        let wl_padded = (41 + 3) & !3;
        for i in 0..41 {
            let irr = solar.data[4 + wl_padded + i];
            assert!(
                irr > 0.0,
                "solar irradiance[{}] = {} should be positive",
                i,
                irr,
            );
        }
    }

    // ── PackedVisionLuts ────────────────────────────────────────────────

    #[test]
    fn packed_vision_header_validates() {
        let vision = PackedVisionLuts::pack();
        assert_eq!(vision.data[0].to_bits(), BUFFER_MAGIC);
        assert_eq!(vision.data[1].to_bits(), BUFFER_VERSION);
    }

    #[test]
    fn packed_vision_num_entries() {
        let vision = PackedVisionLuts::pack();
        assert_eq!(vision.num_entries, 81);
    }

    #[test]
    fn packed_vision_photopic_peak() {
        use twilight_threshold::vision::PHOTOPIC_V;
        let vision = PackedVisionLuts::pack();
        // V(555nm) is at index 35
        let v_555 = vision.data[4 + 35] as f64;
        assert!(
            (v_555 - PHOTOPIC_V[35]).abs() < 1e-5,
            "V(555nm) packed={}, expected={}",
            v_555,
            PHOTOPIC_V[35],
        );
    }

    #[test]
    #[allow(clippy::needless_range_loop)] // packed-offset indexing
    fn packed_vision_scotopic_values_present() {
        use twilight_threshold::vision::SCOTOPIC_V_PRIME;
        let vision = PackedVisionLuts::pack();
        let n_padded = (81 + 3) & !3; // 84
        for i in 0..81 {
            let v = vision.data[4 + n_padded + i] as f64;
            assert!(
                (v - SCOTOPIC_V_PRIME[i]).abs() < 1e-5,
                "scotopic[{}] packed={}, expected={}",
                i,
                v,
                SCOTOPIC_V_PRIME[i],
            );
        }
    }

    // ── PackedDispatchParams ────────────────────────────────────────────

    #[test]
    fn dispatch_params_observer_pos() {
        let p = PackedDispatchParams::new(
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            10000,
            100,
            42,
        );
        assert_eq!(p.data[0], 1.0f32);
        assert_eq!(p.data[1], 2.0f32);
        assert_eq!(p.data[2], 3.0f32);
        assert_eq!(p.data[3], 0.0f32); // pad
    }

    #[test]
    fn dispatch_params_view_dir() {
        let p =
            PackedDispatchParams::new([0.0, 0.0, 0.0], [0.5, 0.6, 0.7], [0.0, 0.0, 0.0], 0, 0, 0);
        assert!((p.data[4] - 0.5).abs() < 1e-6);
        assert!((p.data[5] - 0.6).abs() < 1e-6);
        assert!((p.data[6] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn dispatch_params_sun_dir() {
        let p =
            PackedDispatchParams::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.2, 0.3], 0, 0, 0);
        assert!((p.data[8] - 0.1).abs() < 1e-6);
        assert!((p.data[9] - 0.2).abs() < 1e-6);
        assert!((p.data[10] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn dispatch_params_seed_roundtrip() {
        let seed: u64 = 0xDEAD_BEEF_CAFE_BABE;
        let p = PackedDispatchParams::new([0.0; 3], [0.0; 3], [0.0; 3], 0, 0, seed);
        let lo = p.data[14].to_bits() as u64;
        let hi = p.data[15].to_bits() as u64;
        let reconstructed = lo | (hi << 32);
        assert_eq!(reconstructed, seed);
    }

    #[test]
    fn dispatch_params_photons_and_rays() {
        let p = PackedDispatchParams::new([0.0; 3], [0.0; 3], [0.0; 3], 10_000, 100, 0);
        assert_eq!(p.data[12].to_bits(), 10_000);
        assert_eq!(p.data[13].to_bits(), 100);
    }

    // ── dispatch_groups ─────────────────────────────────────────────────

    #[test]
    fn dispatch_groups_exact_multiple() {
        assert_eq!(dispatch_groups(256, 256), 1);
        assert_eq!(dispatch_groups(512, 256), 2);
        assert_eq!(dispatch_groups(1024, 256), 4);
    }

    #[test]
    fn dispatch_groups_with_remainder() {
        assert_eq!(dispatch_groups(257, 256), 2);
        assert_eq!(dispatch_groups(1, 256), 1);
        assert_eq!(dispatch_groups(255, 256), 1);
    }

    #[test]
    fn dispatch_groups_zero_threads() {
        assert_eq!(dispatch_groups(0, 256), 0);
    }

    // ── PackedLightSource ───────────────────────────────────────────────

    #[test]
    fn packed_light_source_default() {
        let src = PackedLightSource::default();
        assert_eq!(src.distance_m, 0.0);
        assert_eq!(src.height_m, 10.0);
        assert_eq!(src.spectrum_type, 0.0);
    }

    #[test]
    fn packed_light_source_is_32_bytes() {
        assert_eq!(
            std::mem::size_of::<PackedLightSource>(),
            32,
            "PackedLightSource must be 32 bytes (2 x vec4)",
        );
    }

    // ── Offset constants ────────────────────────────────────────────────

    #[test]
    fn atm_offset_optics_start() {
        // 4 (header + dims) + 4 * 64 (shells) = 260
        assert_eq!(atm_offsets::OPTICS_START, 260);
    }

    #[test]
    fn atm_offset_wavelengths_start() {
        // 260 + 4 * 64 * 64 = 260 + 16384 = 16644
        assert_eq!(atm_offsets::WAVELENGTHS_START, 16644);
    }

    #[test]
    fn atm_offset_albedo_start() {
        // 16644 + 64 = 16708
        assert_eq!(atm_offsets::ALBEDO_START, 16708);
    }

    #[test]
    fn atm_offset_refractive_index_start() {
        // 16708 + 64 = 16772
        assert_eq!(atm_offsets::REFRACTIVE_INDEX_START, 16772);
    }

    #[test]
    fn atm_offset_total_size() {
        // v3: 16772 + 64 (refr) + 64 (cloud ext) + 4 (g* + pad) = 16904
        assert_eq!(atm_offsets::TOTAL_SIZE, 16904);
    }

    #[test]
    fn atm_total_size_in_bytes() {
        // 16904 * 4 = 67616 bytes ~ 66 KB
        assert_eq!(atm_offsets::TOTAL_SIZE * 4, 67616);
    }

    #[test]
    fn packed_atm_refractive_index_default_roundtrip() {
        let atm = make_test_atm(); // default n=1.0 everywhere
        let packed = PackedAtmosphere::pack(&atm);
        for s in 0..atm.num_shells {
            let n = packed.data[atm_offsets::REFRACTIVE_INDEX_START + s] as f64;
            assert!(
                (n - 1.0).abs() < 1e-5,
                "default refractive_index[{}] should be 1.0, got {}",
                s,
                n,
            );
        }
    }

    #[test]
    fn packed_atm_refractive_index_computed_roundtrip() {
        let mut atm = make_test_atm();
        // Set Rayleigh extinction so compute_refractive_indices works
        atm.optics[0][0].extinction = 1.3e-5;
        atm.optics[0][0].rayleigh_fraction = 1.0;
        atm.optics[1][0].extinction = 1.3e-6;
        atm.optics[1][0].rayleigh_fraction = 1.0;
        atm.optics[2][0].extinction = 1.3e-8;
        atm.optics[2][0].rayleigh_fraction = 1.0;
        atm.compute_refractive_indices();

        let packed = PackedAtmosphere::pack(&atm);
        let unpacked = packed.unpack();

        for s in 0..atm.num_shells {
            let rel_err = (unpacked.refractive_index[s] - atm.refractive_index[s]).abs();
            assert!(
                rel_err < 1e-5,
                "refractive_index[{}] roundtrip: packed={}, original={}, err={}",
                s,
                unpacked.refractive_index[s],
                atm.refractive_index[s],
                rel_err,
            );
        }
    }

    // ── Shader offset coupling ──────────────────────────────────────────
    //
    // The shader hardcodes the atm_offsets values as `constant uint ATM_*`.
    // There is no codegen tying the two together, so this test parses the
    // MSL source and compares numerically -- a cheap guard against silent
    // layout drift.

    /// Extract the value of `constant uint <name> = <value>;` from MSL
    /// source. Handles decimal and 0x-hex literals with an optional `u`
    /// suffix and trailing `//` comments.
    fn shader_uint_const(source: &str, name: &str) -> u32 {
        for line in source.lines() {
            let Some(rest) = line.trim_start().strip_prefix("constant uint ") else {
                continue;
            };
            let Some((lhs, rhs)) = rest.split_once('=') else {
                continue;
            };
            if lhs.trim() != name {
                continue;
            }
            let value = rhs
                .split(';')
                .next()
                .unwrap()
                .trim()
                .trim_end_matches('u');
            let parsed = match value.strip_prefix("0x") {
                Some(hex) => u32::from_str_radix(hex, 16),
                None => value.parse(),
            };
            return parsed
                .unwrap_or_else(|e| panic!("bad literal for {} ({:?}): {}", name, value, e));
        }
        panic!("constant uint {} not found in twilight.metal", name);
    }

    #[test]
    fn shader_constants_match_buffers_rs() {
        let src = include_str!("../shaders/twilight.metal");

        let expected: &[(&str, usize)] = &[
            ("ATM_HEADER_MAGIC", atm_offsets::HEADER_MAGIC),
            ("ATM_HEADER_VERSION", atm_offsets::HEADER_VERSION),
            ("ATM_NUM_SHELLS", atm_offsets::NUM_SHELLS),
            ("ATM_NUM_WAVELENGTHS", atm_offsets::NUM_WAVELENGTHS),
            ("ATM_SHELLS_START", atm_offsets::SHELLS_START),
            ("ATM_SHELL_STRIDE", atm_offsets::SHELL_STRIDE),
            ("ATM_OPTICS_START", atm_offsets::OPTICS_START),
            ("ATM_OPTICS_STRIDE", atm_offsets::OPTICS_STRIDE),
            ("ATM_ALBEDO_START", atm_offsets::ALBEDO_START),
            ("ATM_REFRACTIVE_INDEX_START", atm_offsets::REFRACTIVE_INDEX_START),
            ("ATM_CLOUD_EXT_START", atm_offsets::CLOUD_EXT_START),
            ("ATM_CLOUD_G_SCALED", atm_offsets::CLOUD_G_SCALED),
            ("MAX_WAVELENGTHS", MAX_WAVELENGTHS),
        ];
        for &(name, rust_value) in expected {
            let shader_value = shader_uint_const(src, name) as usize;
            assert_eq!(
                shader_value, rust_value,
                "shader {} = {} disagrees with buffers.rs value {}",
                name, shader_value, rust_value,
            );
        }

        assert_eq!(
            shader_uint_const(src, "BUFFER_MAGIC"),
            BUFFER_MAGIC,
            "shader BUFFER_MAGIC disagrees with buffers.rs",
        );
        assert_eq!(
            shader_uint_const(src, "BUFFER_VERSION"),
            BUFFER_VERSION,
            "shader BUFFER_VERSION disagrees with buffers.rs",
        );
    }

    #[test]
    fn packed_atm_refractive_index_decreases_with_altitude() {
        let mut atm = make_test_atm();
        atm.compute_refractive_indices_from_altitude();

        let packed = PackedAtmosphere::pack(&atm);
        for s in 0..(atm.num_shells - 1) {
            let n_low = packed.data[atm_offsets::REFRACTIVE_INDEX_START + s];
            let n_high = packed.data[atm_offsets::REFRACTIVE_INDEX_START + s + 1];
            assert!(
                n_low > n_high,
                "n[{}]={} should be > n[{}]={} (denser air below)",
                s,
                n_low,
                s + 1,
                n_high,
            );
        }
    }
}
